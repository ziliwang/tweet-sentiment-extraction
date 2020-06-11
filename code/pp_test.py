import torch
from torch import nn, optim
from torch.nn import functional as F
from transformers import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, KFold
import joblib
import click
import random
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy
from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import re


MAX_LEN = 100
cls_token_id = 0
pad_token_id = 1
sep_token_id = 2
positive_token_id = 1313
negative_token_id = 2430
neutral_token_id = 7974


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def validate_epoch(model, dataiter):
    model.eval()
    score = 0
    sample_counts = 0
    cache = []
    with torch.no_grad():
        for inputs, _, _, offsets, texts, gts in dataiter:
            bsz, slen = inputs.shape
            span_logits = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long())
            span_probs = span_logits.softmax(-1).view(bsz, slen, slen)
            span_probs = torch.avg_pool1d(span_probs, kernel_size=5, stride=1, padding=2, count_include_pad=False)
            span_probs = span_probs.view(bsz, -1)
            # span = span_logits.max(-1)[1].cpu().numpy()  # b x idx
            probs, spans = torch.topk(span_probs, k=2, dim=-1)
            spans = spans.cpu().numpy()
            bsz, slen = inputs.shape
            sample_counts += bsz
            for gt, ps, pbs, text, offset, inp in zip(gts, spans, probs, texts, offsets, inputs):
                predict = text
                if inp[1] != neutral_token_id:
                    item = {}
                    item['gt'] = gt
                    item['text'] = text
                    for i, p in enumerate(ps):
                        start, end = divmod(p, slen)
                        if start -3 >= 0 and start -3 < len(offset) and end -3 >= 0 and end - 3 < len(offset):
                            predict_ = text[offset[start-3][0]: offset[end-3][1]]
                            if i == 0:
                                item['first'] = predict_
                                predict = predict_
                            else:
                                item['second'] = predict_
                    cache.append(item)
                score += jaccard(gt, predict)
    import joblib
    joblib.dump(cache, 'tmp')
    # print(collections.Counter(gt_s_word), collections.Counter(s_word_pt), collections.Counter(s_word_pn))
    return score/sample_counts


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
        return logits


class TweetSentiment(BertPreTrainedModel):
    def __init__(self, config):
        super(TweetSentiment, self).__init__(config)
        # setattr(config, 'output_hidden_states', True)
        self.roberta = RobertaModel(config)
        self.task = TaskLayer(config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.init_weights()
        # torch.nn.init.normal_(self.logits.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # hidden_states = torch.cat([outputs[0], outputs[1].unsqueeze(1).expand_as(outputs[0])], dim=-1)
        hidden_states = outputs[0]
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


def pridect_epoch(model, dataiter, span_logits_bagging):
    model.eval()
    with torch.no_grad():
        for ids, inputs, _, _ in dataiter:
            span_logits = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long())
            span_prob = span_logits.softmax(-1)
            for i, v in zip(ids, span_prob):
                if i in span_logits_bagging:
                    span_logits_bagging[i] += v
                else:
                    span_logits_bagging[i] = v


@click.command()
@click.option('--data', default='roberta.input.joblib')
@click.option('--models', default='trained.models')
@click.option('--config', default='../model/roberta-l12/config.json')
def main(data, models, config):
    data = joblib.load(data)
    best_models = torch.load(models)
    scores = []
    k = 0
    for train_idx, val_idx in KFold(n_splits=5, random_state=9895).split(data):  # StratifiedKFold(n_splits=5, random_state=seed).split(data, [i['sentiment'] for i in data]):
        k += 1
        # if k in [1, 3]:
        #     continue
        print(f'---- {k} Fold ---')
        model_config = RobertaConfig.from_json_file(config)
        model = TweetSentiment(model_config).cuda()
        model.load_state_dict(best_models['models'][k-1])
        val = [data[i] for i in val_idx]
        valiter = DataLoader(val, batch_size=32*2, shuffle=False, collate_fn=collect_func)
        score = validate_epoch(model, valiter)
        print(f'fold {k} {score}')
        scores.append(score)

    print(f'mean {np.mean(scores)}')


if __name__ == "__main__":
    main()
