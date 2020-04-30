import torch
from torch import nn, optim
from torch.nn import functional as F
from transformers import AlbertTokenizer, AlbertModel, AdamW, BertPreTrainedModel, get_linear_schedule_with_warmup
from transformers.modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits
from transformers.modeling_albert import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
import joblib
import click
import random
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy

MAX_LEN = 100
cls_token_id = 2
pad_token_id = 0
sep_token_id = 3


def collect_func(records):
    inputs = []
    starts = []
    ends = []
    offsets = []
    texts = []
    gts = []
    for rec in records:
        inputs.append(torch.LongTensor([cls_token_id, rec['sentiment'], sep_token_id] + rec['tokens_id'][:MAX_LEN] + [sep_token_id]))
        starts.append(rec['start']+3)
        ends.append(rec['end']+3)
        offsets.append(rec['offsets'])
        texts.append(rec['text'])
        gts.append(rec['gt'])
    return pad_sequence(inputs, batch_first=True, padding_value=pad_token_id).cuda(), torch.LongTensor(starts).cuda(), torch.LongTensor(ends).cuda(), \
        offsets, texts, gts


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, optimizer, lr_scheduler, dataiter, accumulate_step):
    model.train()
    sample_num = 0
    cum_loss = 0
    step = 0
    for inputs, starts, ends, _, _, _ in tqdm(dataiter):
        step += 1
        sample_num += inputs.shape[0]
        loss = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long(), start_positions=starts, end_positions=ends)
        cum_loss += loss.sum().detach().cpu().data.numpy()
        loss.mean().backward()
        clip_grad_norm_(model.parameters(), 2)
        if step % accumulate_step == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    return cum_loss/sample_num


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def validate_epoch(model, dataiter):
    model.eval()
    score = 0
    sample_counts = 0
    with torch.no_grad():
        for inputs, _, _, offsets, texts, gts in dataiter:
            span_logits = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long())
            span = span_logits.max(-1)[1].cpu().numpy()  # b x idx
            bsz, slen = inputs.shape
            sample_counts += bsz
            for i in range(bsz):
                try:
                    start, end = divmod(span[i], slen)
                    predict = texts[i][offsets[i][start-3][0]: offsets[i][end-3][1]]
                except IndexError:
                    print(span_logits[i], inputs[i], offsets[i], start, end)
                if inputs[i][1] == 8387:
                    predict = texts[i]
                score += jaccard(gts[i], predict)
    return score/sample_counts


def deepcopy_state_dict_to_cpu(model):
    output = {}
    for k, v in model.state_dict().items():
        output[k] = deepcopy(v.cpu())
    return output


def fold_train(model, optimizer, lr_scheduler, epoch, train_dataiter, val_dataiter, accumulate_step):
    best_score = float('-inf')
    best_model = deepcopy_state_dict_to_cpu(model)
    for e in range(epoch):
        loss = train_epoch(model, optimizer, lr_scheduler, train_dataiter, accumulate_step)
        score = validate_epoch(model, val_dataiter)
        print(f'epoch {e} loss {loss:.6f} score: {score:.6f}')
        if score > best_score:
            best_score = score
            best_model = deepcopy_state_dict_to_cpu(model)
    return best_score, best_model


class TaskLayer(nn.Module):

    def __init__(self, hidden_size):
        super(TaskLayer, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.key = nn.Linear(self.hidden_size*2, self.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        bsz, slen, hsz = hidden_states.shape
        query = self.query(hidden_states)
        key = self.key(hidden_states)  # b x s_len x h
        logits = torch.matmul(query, key.transpose(-1, -2)) # b x s_len x s_len
        logits = logits/np.sqrt(self.hidden_size)
        if attention_mask is not None:  # 1 for available, 0 for unavailable
            attention_mask = attention_mask[:, :, None].expand(-1, -1, slen) * attention_mask[:, None, :].expand(-1, slen, -1)
        else:
            attention_mask = torch.ones(bsz, slen, slen)
            if hidden_states.is_cuda:
                attention_mask = attention_mask.cuda()
        attention_mask = torch.triu(attention_mask)
        logits = logits*attention_mask - 1e6*(1-attention_mask)
        logits = logits.view(bsz, -1)  # b x slen*slen
        return logits


class AlbertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForQuestionAnswering, self).__init__(config)
        setattr(config, 'output_hidden_states', True)
        self.albert = AlbertModel(config)
        self.task = TaskLayer(config.hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()
        # torch.nn.init.normal_(self.logits.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        token_type_ids = torch.ones(input_ids.shape).long().cuda()
        token_type_ids[:,:3] = 0
        outputs = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = torch.cat([outputs[0], outputs[1].unsqueeze(1).expand_as(outputs[0])], dim=-1)
        p_mask = attention_mask.float() * (input_ids != sep_token_id).float()
        p_mask[:,:2] = 0
        span_logits = self.task(self.dropout(hidden_states), attention_mask=p_mask)  # b x slen*slen
        bsz, slen = input_ids.shape
        if start_positions is not None and end_positions is not None:
            span = start_positions * slen + end_positions
            loss = F.cross_entropy(span_logits, span, reduction='none')
            return loss
        else:
            return span_logits


@click.command()
@click.option('--data', default='albert.input.joblib')
@click.option('--pretrained', default='../model/albert.base/')
@click.option('--lr', default=5e-5)
@click.option('--batch-size', default=32)
@click.option('--epoch', default=3)
@click.option('--accumulate-step', default=1)
@click.option('--seed', default=9895)
def main(data, pretrained, lr, batch_size, epoch, accumulate_step, seed):
    seed_everything(seed)
    data = joblib.load(data)
    best_models = []
    best_scores = []
    k = 0
    for train_idx, val_idx in KFold(n_splits=5, random_state=seed).split(data, [i['sentiment'] for i in data]):
        k += 1
        print(f'---- {k} Fold ---')
        # train = [data[i] for i in train_idx]
        train = [data[i] for i in train_idx if data[i]['sentiment'] != 8387]
        val = [data[i] for i in val_idx]
        model = AlbertForQuestionAnswering.from_pretrained(pretrained).cuda()
        no_decay = ['.bias', 'full_layer_layer_norm.weight', 'LayerNorm.weight', ]
        optimizer = AdamW([{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                            "lr": lr, 'weight_decay': 1e-2},
                           {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                            "lr": lr, 'weight_decay': 0}], betas=(0.9, 0.98), eps=1e-6)
        train_steps = np.ceil(len(train) / batch_size / accumulate_step * epoch)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.06*train_steps, num_training_steps=train_steps)
        trainiter = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collect_func)
        valiter = DataLoader(val, batch_size=batch_size*2, shuffle=False, collate_fn=collect_func)
        best_score, best_model = fold_train(model, optimizer, lr_scheduler, epoch, trainiter, valiter, accumulate_step)
        best_scores.append(best_score)
        best_models.append(best_model)
    print(f'final cv {np.mean(best_scores)}')
    torch.save({'models': best_models, 'score': np.mean(best_scores)}, 'trained.models')


if __name__ == '__main__':
    main()
