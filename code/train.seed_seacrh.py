import torch
from torch import nn, optim
from torch.nn import functional as F
from transformers import *
from transformers.modeling_bert import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import joblib
import click
import random
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy

MAX_LEN = 200
cls_token_id = 0
pad_token_id = 1
sep_token_id = 2


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
        # clip_grad_norm_(model.parameters(), 2)
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
            start_logits, end_logits = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long())
            s_top_probs, s_top_idxs = torch.topk(start_logits.softmax(-1), 5)
            e_top_probs, e_top_idxs = torch.topk(end_logits.softmax(-1), 5)
            batch_size = inputs.shape[0]
            sample_counts += batch_size
            for i in range(batch_size):
                c = []
                for _i, s_idx in enumerate(s_top_idxs[i]):
                    for _j, e_idx in enumerate(e_top_idxs[i]):
                        if s_idx <= e_idx:
                            c.append((s_top_probs[i][_i]*e_top_probs[i][_j], s_idx, e_idx))
                _, start, end = sorted(c)[-1]
                predict = texts[i][offsets[i][start-3][0]: offsets[i][end-3][1]]
                if inputs[i][1] == 7974:
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


class RobertaForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        setattr(config, 'output_hidden_states', True)
        self.roberta = RobertaModel(config)
        self.logits = nn.Linear(config.hidden_size*2, 2)
        # self.logits = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size), nn.Tanh(), nn.Dropout(0.5), nn.Linear(config.hidden_size, 2))
        # self.decoder_config = BertConfig(num_hidden_layers=2, hidden_size=128, num_attention_heads=8)
        # self.fc = nn.Linear(config.hidden_size*2, 128)
        # self.decoder = BertEncoder(self.decoder_config)
        # self.logits = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)
        # nn.init.xavier_uniform_(self.logits.weight)
        self.init_weights()
        # torch.nn.init.normal_(self.logits.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # hidden_states = torch.cat([torch.stack(outputs[-1][-6:], dim=-1).max(-1)[0], outputs[1][:,None,:].expand_as(outputs[0])], dim=-1)
        # hidden_states = torch.stack(outputs[-1][-6:], dim=-1).max(-1)[0]
        # hidden_states = torch.cat([torch.stack(outputs[-1][-4:], dim=-1).mean(-1), outputs[1][:,None,:].expand_as(outputs[0])], dim=-1)
        # hidden_states = torch.cat([torch.cat(outputs[-1][-4:], dim=-1), outputs[1][:,None,:].expand_as(outputs[0])], dim=-1)
        hidden_states = torch.cat([outputs[0], outputs[1][:,None,:].expand_as(outputs[0])], dim=-1)
        # hidden_states = outputs[0]
        # start_end_logits = self.logits(self.dropout(self.bn(hidden_states.reshape(-1, hidden_states.shape[-1])).reshape_as(hidden_states)))
        # decoder_att = attention_mask.float() * (input_ids != sep_token_id).float()
        # decoder_att[:,:2] = .0
        # decoder_att = decoder_att.unsqueeze(1).unsqueeze(2)
        # decoder_att = (1.0 - decoder_att) * -10000.0
        # outputs = self.decoder(self.fc(hidden_states), attention_mask=decoder_att, head_mask=[None]* self.decoder_config.num_hidden_layers)
        # hidden_states = outputs[0]
        start_end_logits = self.logits(self.dropout(hidden_states))
        start_logits, end_logits = start_end_logits.split(1, dim=-1)

        if start_positions is not None and end_positions is not None:
            for x in (start_positions, end_positions):
                if x.dim() > 1:
                    x.squeeze_(-1)
            start_loss = F.cross_entropy(start_logits.squeeze(-1), start_positions, reduction='none')
            end_loss = F.cross_entropy(end_logits.squeeze(-1), end_positions, reduction='none')
            return start_loss + end_loss

        if not self.training:
            p_mask = attention_mask.float() * (input_ids != sep_token_id).float()  # 1 for available 0 for unavailable
            p_mask[:, :3] = 0.0
            start_logits = start_logits.squeeze(-1) * p_mask - 1e30 * (1 - p_mask)
            end_logits = end_logits.squeeze(-1) * p_mask - 1e30 * (1- p_mask)
            return start_logits, end_logits


@click.command()
@click.option('--data', default='roberta.input.joblib')
@click.option('--pretrained', default='../model/roberta-l12/')
@click.option('--lr', default=5e-5)
@click.option('--batch-size', default=32)
@click.option('--epoch', default=3)
@click.option('--accumulate-step', default=1)
@click.option('--seed', default=9895)
def main(data, pretrained, lr, batch_size, epoch, accumulate_step, seed):
    output = []
    seeds = np.arange(500)
    np.random.shuffle(seeds)
    data = joblib.load(data)
    for seed in seeds[:20]:
        seed_everything(seed)
        best_models = []
        best_scores = []
        k = 0
        for train_idx, val_idx in StratifiedKFold(n_splits=5, random_state=seed).split(data, [i['sentiment'] for i in data]):
            k += 1
            print(f'---- {k} Fold ---')
            train = [data[i] for i in train_idx if data[i]['sentiment'] != 7974]
            # train = [data[i] for i in train_idx]
            val = [data[i] for i in val_idx]
            model = RobertaForQuestionAnswering.from_pretrained(pretrained).cuda()
            no_decay = ['.bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer = AdamW([{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                "lr": lr, 'weight_decay': 1e-2},
                               {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                "lr": lr, 'weight_decay': 0}])
            train_steps = np.ceil(len(train) / batch_size / accumulate_step * epoch)
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*train_steps, num_training_steps=train_steps)
            trainiter = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collect_func)
            valiter = DataLoader(val, batch_size=batch_size*2, shuffle=False, collate_fn=collect_func)
            best_score, best_model = fold_train(model, optimizer, lr_scheduler, epoch, trainiter, valiter, accumulate_step)
            best_scores.append(best_score)
            best_models.append(best_model)
        print(f'final cv {np.mean(best_scores)}')
        mean_score = np.mean(best_scores)
        output.append((seed, mean_score))
        if mean_score > 0.711:
            torch.save({'models': best_models, 'score': mean_score}, f'trained.models.{seed}.{int(mean_score*10000)}')
    with open('report.txt', 'w') as f:
        for i, j in output:
            f.writelines(f'{i},{j}\n')


if __name__ == '__main__':
    main()
