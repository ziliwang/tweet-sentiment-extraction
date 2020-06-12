import torch
from torch import nn, optim
from torch.nn import functional as F
from transformers import *
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

MAX_LEN = 100
cls_token_id = 0
pad_token_id = 1
sep_token_id = 2
positive_token_id = 1313
negative_token_id = 2430
neutral_token_id = 7974

# epoch 5 lr 3e-5 seed  cv 7149 7089,7192,7184,7111,7171
# ratio: 0.3 or 0.2, loss kl_div pytorch, train set > 0.5 and remove neutral


def collect_func0(records):
    inputs = []
    starts = []
    ends = []
    offsets = []
    texts = []
    gts = []
    for rec in records:
        inputs.append(torch.LongTensor([cls_token_id, rec['sentiment'], sep_token_id] + rec['tokens_id'][:MAX_LEN] + [sep_token_id]))
        if 'start' in rec:
            starts.append(rec['start']+3)
            ends.append(rec['end']+3)
            gts.append(rec['gt'])
        elif rec['sentiment'] == neutral_token_id:
            starts.append(3)
            ends.append(3+len(rec['tokens_id'][:MAX_LEN])-1)
            gts.append('')
        else:
            starts.append(0)
            ends.append(0)
            gts.append('')
        offsets.append(rec['offsets'])
        texts.append(rec['text'])
    return pad_sequence(inputs, batch_first=True, padding_value=pad_token_id).cuda(), torch.LongTensor(starts).cuda(), torch.LongTensor(ends).cuda(), \
        offsets, texts, gts


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


def train_epoch(model, optimizer, lr_scheduler, dataiter, accumulate_step, step, alpha):
    model.train()
    sample_num = 0
    cum_loss = 0
    for inputs, starts, ends, _, _, _ in tqdm(dataiter):
        step += 1

        if step < alpha[0]:
            alpha_t = 0.0
        elif step >= alpha[1]:
            alpha_t = 3.0
        else:
            alpha_t = (step - alpha[0]) / (alpha[1]-alpha[0]) * 3.0

        bsz, slen = inputs.shape
        sample_num += bsz
        pseudo_mask = (starts == 0).long()

        if pseudo_mask.sum() > 0 and alpha_t > .0:
            with torch.no_grad():
                p = model(inputs, attention_mask=(inputs != pad_token_id).long()).max(-1)[1]
                p_start = (p // slen).long()
                p_end = (p % slen).long()
                starts = (1-pseudo_mask)*starts + pseudo_mask*p_start
                ends = (1-pseudo_mask)*ends+pseudo_mask*p_end
        loss = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long(), start_positions=starts, end_positions=ends)
        coefs = 1.0 - pseudo_mask.float() + alpha_t * pseudo_mask.float()
        loss = coefs * loss
        cum_loss += loss.sum().detach().cpu().data.numpy()
        loss.mean().backward()
        # cum_loss += loss.detach().cpu().data.numpy()
        # loss.backward()
        # clip_grad_norm_(model.parameters(), 2)
        if step % accumulate_step == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    return cum_loss/sample_num, step
    # return cum_loss/step


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
            # probs, spans = torch.topk(span_logits.softmax(-1), k=2, dim=-1)
            # spans = spans.cpu().numpy()
            bsz, slen = inputs.shape
            sample_counts += bsz
            # for i in range(bsz):
            #     if inputs[i][1] == neutral_token_id or len(texts[i].split()) == 1:
            #         predict = [(texts[i], 1.0)]
            #     else:
            #         predict = []
            #         for j in range(2):
            #             try:
            #                 start, end = divmod(spans[i][j], slen)
            #                 predict.append((texts[i][offsets[i][start-3][0]: offsets[i][end-3][1]], probs[i][j]))
            #             except IndexError:
            #                 # predict = texts[i]
            #                 print(span_logits[i], inputs[i], offsets[i], start, end)
            #     if len(predict) == 2 and predict[0][1] / predict[1][1] < 1.05:
            #         predict[0] = (predict[0][0] + ' ' + predict[1][0], 1.0)
                # if jaccard(gts[i], predict[0][0]) < 0.5 and len(predict) > 1:
                #     print(f'gt: {gts[i]}\np1: {predict[0]} p2: {predict[1]}')
                # score += jaccard(gts[i], predict[0][0])
            for gt, p, text, offset, inp in zip(gts, span, texts, offsets, inputs):
                predict = text
                if inp[1] != neutral_token_id:
                    start, end = divmod(p, slen)
                    if start -3 >= 0 and start -3 < len(offset) and end -3 >= 0 and end - 3 < len(offset):
                        predict = text[offset[start-3][0]: offset[end-3][1]]
                score += jaccard(gt, predict)
    return score/sample_counts


def deepcopy_state_dict_to_cpu(model):
    output = {}
    for k, v in model.state_dict().items():
        output[k] = deepcopy(v.cpu())
    return output


def fold_train(model, optimizer, lr_scheduler, epoch, train_dataiter, val_dataiter, accumulate_step):
    best_score = float('-inf')
    best_model = deepcopy_state_dict_to_cpu(model)
    step = 0
    for e in range(epoch):
        loss, step = train_epoch(model, optimizer, lr_scheduler, train_dataiter, accumulate_step, step, (2000, 4000))
        score = validate_epoch(model, val_dataiter)
        print(f'epoch {e} loss {loss:.6f} score: {score:.6f}')
        if score > best_score:
            best_score = score
            best_model = deepcopy_state_dict_to_cpu(model)
    return best_score, best_model


def loose_dist(start, end, slen, alpha=0.3, threshold=1e-6, min_size=2):
    start_loose = torch.zeros(slen)
    end_loose = torch.zeros(slen)
    start_loose[start] = 1
    end_loose[end] = 1
    # w = (end -start + 1).cpu().numpy()
    for i in range(1, max(min_size, end-start+1)):
        p = alpha**i
        # p = np.exp(-i)
        if p < threshold:
            break
        if start + i < slen:
            start_loose[start+i] = p
        if end - i >= 0:
            end_loose[end-i] = p
        if start-i >= 0:
            start_loose[start-i] = p  # 0.8*p
        if end + i < slen:
            end_loose[end+i] = p  # 0.8*p
    return start_loose, end_loose


def loss_func(predicts, start_positions, end_positions, position_mask):
    bsz, slen = position_mask.shape
    start_dist = []
    end_dist = []
    for i, start in enumerate(start_positions):
        sd, ed = loose_dist(start, end_positions[i], slen)
        start_dist.append(sd)
        end_dist.append(ed)
    start_dist = torch.stack(start_dist)
    end_dist = torch.stack(end_dist)
    if predicts.is_cuda:
        start_dist = start_dist.cuda()
        end_dist = end_dist.cuda()
    start_dist = start_dist*position_mask
    end_dist = end_dist*position_mask
    dist = start_dist[:, :, None].expand(-1, -1, slen) * end_dist[:, None, :].expand(-1, slen, -1)
    dist = torch.triu(dist)
    dist = dist.view(bsz, -1)
    # dist = dist.view(bsz, -1) + 1e-6
    predicts = predicts + 1e-6
    # return (dist*(dist/predicts).log()).sum(-1)
    return -(dist*predicts.log()).sum(-1)
    # return F.kl_div((predicts+1e-6).log(), (dist+1e-6), reduction='batchmean')


class TaskLayer(nn.Module):

    def __init__(self, hidden_size):
        super(TaskLayer, self).__init__()
        self.hidden_size = hidden_size
        # self.query = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size))
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        # self.key = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size))
        self.key = nn.Linear(self.hidden_size, self.hidden_size)

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
        return logits, attention_mask.view(bsz, -1)


class TweetSentiment(BertPreTrainedModel):
    def __init__(self, config):
        super(TweetSentiment, self).__init__(config)
        # setattr(config, 'output_hidden_states', True)
        self.roberta = RobertaModel(config)
        self.task = TaskLayer(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # hidden_states = torch.cat([outputs[0], outputs[1].unsqueeze(1).expand_as(outputs[0])], dim=-1)
        hidden_states = outputs[0]
        p_mask = attention_mask.float() * (input_ids != sep_token_id).float()
        p_mask[:, :2] = 0
        span_logits, _ = self.task(self.dropout(hidden_states), attention_mask=p_mask)  # b x slen*slen
        # span_logits, p_mask = self.task(self.dropout(hidden_states), attention_mask=p_mask)
        bsz, slen = input_ids.shape
        if start_positions is not None and end_positions is not None:
            span = start_positions * slen + end_positions
            loss = F.cross_entropy(span_logits, span, reduction='none')
            # alpha = 0.1
            # smooth_loss = loss_func(span_logits.softmax(-1), start_positions, end_positions, position_mask=p_mask)
            # smooth_prob = (1 / p_mask.sum(-1)).unsqueeze(-1).expand_as(p_mask) * p_mask
            # smooth_loss = -(span_logits.log_softmax(-1) * smooth_prob).sum(-1)
            # return (1-alpha)*loss + alpha*smooth_loss
            return loss
        else:
            return span_logits


class TweetSentiment_C(BertPreTrainedModel):
    def __init__(self, config):
        super(TweetSentiment_C, self).__init__(config)
        # setattr(config, 'output_hidden_states', True)
        self.roberta = RobertaModel(config)
        self.task1 = TaskLayer(128+128+128)
        # self.task2 = TaskLayer(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(1, config.hidden_size))
        self.conv2 = nn.Conv2d(1, 128, kernel_size=(3, config.hidden_size), padding=(1, 0))
        self.conv3 = nn.Conv2d(1, 128, kernel_size=(5, config.hidden_size), padding=(2, 0))
        self.layernorm = nn.LayerNorm(256+256+128)
        self.init_weights()
        # torch.nn.init.normal_(self.logits.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # hidden_states = torch.cat([outputs[0], outputs[1].unsqueeze(1).expand_as(outputs[0])], dim=-1)
        hidden_states = outputs[0]
        bsz, slen, hdz = hidden_states.shape
        x1 = self.conv1(hidden_states[:, None, :, :]).squeeze(-1).transpose(-1, -2)
        x2 = self.conv2(hidden_states[:, None, :, :]).squeeze(-1).transpose(-1, -2)
        x3 = self.conv3(hidden_states[:, None, :, :]).squeeze(-1).transpose(-1, -2)
        x = torch.cat([x1, x2, x3], dim=-1)
        # start_logits, end_logits = self.task2(x).split(1, dim=-1)
        p_mask = attention_mask.float() * (input_ids != sep_token_id).float()
        p_mask[:, :2] = 0
        span_logits = self.task1(self.dropout(x), attention_mask=p_mask)  # b x slen*slen
        if start_positions is not None and end_positions is not None:
            span = start_positions * slen + end_positions
            loss = F.cross_entropy(span_logits, span, reduction='none')
            # loss = loss_func(span_logits.softmax(-1), start_positions, end_positions, position_mask=p_mask)
            return loss
        else:
            return span_logits


def lr_decay_adamw(model, rank_layers, no_weight_decay_layers, lr, decay_rate=0.97, **argv):
    layers_para_no_weight_decay = [[p for n, p in model.named_parameters() if layer in n and any(no_wd in n for no_wd in no_weight_decay_layers)]
                                   for layer in rank_layers]
    layers_para_with_weight_decay = [[p for n, p in model.named_parameters() if layer in n and not any(no_wd in n for no_wd in no_weight_decay_layers)]
                                     for layer in rank_layers]
    optimizer = AdamW([{'params': ps, 'lr': lr*(decay_rate**idx), 'weight_decay': .0} for idx, ps in enumerate(layers_para_no_weight_decay[::-1])] +
                      [{'params': ps, 'lr': lr*(decay_rate**idx)} for idx, ps in enumerate(layers_para_with_weight_decay)], **argv)
    return optimizer


@click.command()
@click.option('--data', default='roberta.input.joblib')
@click.option('--pretrained', default='../model/roberta-l12/')
@click.option('--lr', default=3e-5)
@click.option('--batch-size', default=32)
@click.option('--epoch', default=4)
@click.option('--accumulate-step', default=1)
@click.option('--seed', default=9895)
@click.option('--lr-decay-rate', default=1.0)
@click.option('--beta1', default=0.9)
@click.option('--beta2', default=0.98)
@click.option('--model-type', default='default')
def main(data, pretrained, lr, batch_size, epoch, accumulate_step, seed, lr_decay_rate, beta1, beta2, model_type):
    seed_everything(seed)
    data = joblib.load(data)
    train_data = data['train']
    test_data = data['test']
    best_models = []
    best_scores = []
    k = 0
    for train_idx, val_idx in KFold(n_splits=5, random_state=9895).split(train_data):  # StratifiedKFold(n_splits=5, random_state=seed).split(data, [i['sentiment'] for i in data]):
        k += 1
        print(f'---- {k} Fold ---')
        # train = [train_data[i] for i in train_idx if train_data[i]['sentiment'] != neutral_token_id]
        # train = [train_data[i] for i in train_idx if not train_data[i]['bad']]
        # train = [train_data[i] for i in train_idx if train_data[i]['score'] > 0.1 and train_data[i]['sentiment'] != neutral_token_id] + test_data
        # train = [train_data[i] for i in train_idx if train_data[i]['score'] > 0.5]
        train = [train_data[i] for i in train_idx if train_data[i]['score'] > 0.1] + test_data
        val = [train_data[i] for i in val_idx]
        print(f"val best score is {np.mean([i['score'] for i in val])}")
        if model_type == 'c':
            model = TweetSentiment_C.from_pretrained(pretrained).cuda()
        else:
            model = TweetSentiment.from_pretrained(pretrained).cuda()
        no_decay = ['.bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer = AdamW([{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #                     "lr": lr, 'weight_decay': 1e-2},
        #                    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #                     "lr": lr, 'weight_decay': 0}], betas=(0.9, 0.98), eps=1e-6)
        # optimizer = AdamW(model.parameters(), weight_decay=1e-2, lr=lr, eps=1e-6)
        layers_group = ['roberta.embeddings.'] + [f'roberta.encoder.layer.{i}.' for i in range(12)] + ['task.']
        optimizer = lr_decay_adamw(model, rank_layers=layers_group, no_weight_decay_layers=no_decay,
                                       decay_rate=lr_decay_rate, lr=lr, eps=1e-6, weight_decay=1e-2, betas=(beta1, beta2))
        train_steps = np.ceil(len(train) / batch_size / accumulate_step * epoch)
        # lr_scheduler = get_constant_schedule(optimizer)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*train_steps, num_training_steps=train_steps)
        trainiter = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collect_func0)
        valiter = DataLoader(val, batch_size=batch_size*2, shuffle=False, collate_fn=collect_func)
        best_score, best_model = fold_train(model, optimizer, lr_scheduler, epoch, trainiter, valiter, accumulate_step)
        best_scores.append(best_score)
        best_models.append(best_model)
    print(f'final cv {np.mean(best_scores)}')
    torch.save({'type': model_type, 'models': best_models, 'score': np.mean(best_scores), 'scores': best_scores}, 'trained.models')


if __name__ == '__main__':
    main()
