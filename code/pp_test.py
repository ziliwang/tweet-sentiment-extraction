import torch
from torch import nn, optim
from torch.nn import functional as F
from transformers import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
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
    with torch.no_grad():
        for inputs, gt_s, gt_e, offsets, texts, gts in dataiter:
            start_logits, end_logits = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long())
            start_probs = start_logits.softmax(-1)
            end_probs = end_logits.softmax(-1)
            _, idx = torch.topk(start_probs + end_probs, 2, dim=-1)
            s_idxs, e_idxs = idx.split(1, dim=-1)
            # start_probs = torch.avg_pool1d(start_logits[:, None, :], kernel_size=3, stride=1, padding=1, count_include_pad=False).squeeze(1)
            # end_probs = torch.avg_pool1d(end_logits[:, None, :], kernel_size=3, stride=1, padding=1, count_include_pad=False).squeeze(1)
            batch_size = inputs.shape[0]
            sample_counts += batch_size
            s_idxs = start_probs.max(-1)[1]
            e_idxs = end_probs.max(-1)[1]
            for inp, text, g_s, gt, offset, s, e in zip(inputs, texts, gt_s, gts, offsets, start_probs, end_probs):
                predict = text
                s = s[3: 3+len(offset)]
                e = e[3:3+len(offset)]
                # s = torch.avg_pool1d(s[None,None,:], kernel_size=3, stride=1, padding=1, count_include_pad=False).squeeze(0).squeeze(0)
                # e = torch.avg_pool1d(e[None,None,:], kernel_size=3, stride=1, padding=1, count_include_pad=False).squeeze(0).squeeze(0)
                s = s.max(-1)[1]
                e = e.max(-1)[1]
                if inp[1] != 7974 and s <= e:
                    if s != g_s:
                        print(s, g_s)
                        break
                    predict = text[offset[s][0]:offset[e][1]]
                score += jaccard(gt, predict)
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


class RobertaForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        setattr(config, 'output_hidden_states', True)
        self.roberta = RobertaModel(config)
        self.logits = nn.Linear(config.hidden_size, 2)
        # self.logits = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size), nn.Tanh(), nn.Linear(config.hidden_size, 2))
        self.dropout = nn.Dropout(0.2)
        # nn.init.xavier_uniform_(self.logits.weight)
        self.init_weights()
        # torch.nn.init.normal_(self.logits.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # hidden_states = torch.cat([torch.stack(outputs[-1][-6:], dim=-1).max(-1)[0], outputs[1][:,None,:].expand_as(outputs[0])], dim=-1)
        # hidden_states = torch.stack(outputs[-1][-6:], dim=-1).max(-1)[0]
        # hidden_states = torch.cat([torch.stack(outputs[-1][-4:], dim=-1).mean(-1), outputs[1][:,None,:].expand_as(outputs[0])], dim=-1)
        # hidden_states = torch.cat([torch.cat(outputs[-1][-4:], dim=-1), outputs[1][:,None,:].expand_as(outputs[0])], dim=-1)
        # hidden_states = torch.cat([outputs[0], outputs[1][:,None,:].expand_as(outputs[0])], dim=-1)
        hidden_states = self.dropout(outputs[0])
        # start_end_logits = self.logits(self.dropout(self.bn(hidden_states.reshape(-1, hidden_states.shape[-1])).reshape_as(hidden_states)))
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
    for train_idx, val_idx in StratifiedKFold(n_splits=5, random_state=100).split(data, [i['sentiment'] for i in data]):
        k += 1
        # if k in [1, 3]:
        #     continue
        print(f'---- {k} Fold ---')
        model_config = RobertaConfig.from_json_file(config)
        model = RobertaForQuestionAnswering(model_config).cuda()
        model.load_state_dict(best_models['models'][k-1])
        val = [data[i] for i in val_idx]
        valiter = DataLoader(val, batch_size=32*2, shuffle=False, collate_fn=collect_func)
        score = validate_epoch(model, valiter)
        print(f'fold {k} {score}')
        scores.append(score)

    print(f'mean {np.mean(scores)}')


if __name__ == "__main__":
    main()
