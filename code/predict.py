import torch
from torch import nn, optim
from torch.nn import functional as F
from transformers import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import joblib
import click
import random
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy
from tokenizers import ByteLevelBPETokenizer
import pandas as pd
from itertools import accumulate


MAX_LEN = 200
cls_token_id = 2
pad_token_id = 0
sep_token_id = 3


def preprocess(tokenizer, df):
    output = []
    sentiment_hash = dict(zip(['positive', 'negative', 'neutral'], tokenizer.convert_tokens_to_ids(['▁positive', '▁negative', '▁neutral'])))
    for line, row in df.iterrows():
        if pd.isna(row.text): continue
        text = row.text
        text = ' ' + ' '.join(text.split())
        tokens = tokenizer.tokenize(text)
        offsets = list(accumulate(map(len, tokens)))
        offsets = list(zip([0] + offsets, offsets))
        record = {}
        record['tokens_id'] = tokenizer.convert_tokens_to_ids(tokens)
        record['sentiment'] = sentiment_hash[row.sentiment]
        record['offsets'] = offsets
        record['text'] = text
        record['id'] = row.textID
        output.append(record)
    return output


def collect_func(records):
    ids = []
    inputs = []
    offsets = []
    texts = []
    for rec in records:
        ids.append(rec['id'])
        inputs.append(torch.LongTensor([cls_token_id, rec['sentiment'], sep_token_id] + rec['tokens_id'][:MAX_LEN] + [sep_token_id]))
        offsets.append(rec['offsets'])
        texts.append(rec['text'])
    return ids, pad_sequence(inputs, batch_first=True, padding_value=pad_token_id).cuda(), offsets, texts


class AlbertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForQuestionAnswering, self).__init__(config)
        self.albert = AlbertModel(config)
        # self.logits = nn.Linear(config.hidden_size*2, 2)
        self.logits = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size), nn.Tanh(), nn.Linear(config.hidden_size, 2))
        self.dropout = nn.Dropout(0.5)
        self.init_weights()
        # torch.nn.init.normal_(self.logits.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        token_type_ids = torch.ones(input_ids.shape).long().cuda()
        token_type_ids[:,:3] = 0
        outputs = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = torch.cat([outputs[0], outputs[1][:,None,:].expand_as(outputs[0])], dim=-1)
        # hidden_states = outputs[0]
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
            p_mask = attention_mask.float()  # 1 for available 0 for unavailable
            p_mask[:, :3] = 0.0
            start_logits = start_logits.squeeze(-1) * p_mask - 1e30 * (1 - p_mask)
            end_logits = end_logits.squeeze(-1) * p_mask - 1e30 * (1- p_mask)
            return start_logits, end_logits


def pridect_epoch(model, dataiter, starts_logits_5cv, ends_logits_5cv):
    model.eval()
    with torch.no_grad():
        for ids, inputs, _, _ in dataiter:
            start_logits, end_logits = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long())
            for i, j, k in zip(ids, start_logits, end_logits):
                if i in starts_logits_5cv:
                    starts_logits_5cv[i] += j
                else:
                    starts_logits_5cv[i] = j
                if i in ends_logits_5cv:
                    ends_logits_5cv[i] += k
                else:
                    ends_logits_5cv[i] = k


@click.command()
@click.option('--test-path', default='../input/test.csv')
@click.option('--sp_model', default='../model/albert.base/spiece.model')
@click.option('--models', default='trained.models')
@click.option('--config', default='../model/albert.base/config.json')
def main(test_path, sp_model, models, config):
    tokenizer = AlbertTokenizer(sp_model, do_lower_case=True)
    test_df = pd.read_csv(test_path)
    test = preprocess(tokenizer, test_df)
    model_config = BertConfig.from_json_file(config)
    saved_models = torch.load(models)
    model = AlbertForQuestionAnswering(model_config).cuda()
    testiter = DataLoader(test, batch_size=32, shuffle=False, collate_fn=collect_func)
    print(f"5cv {saved_models['score']}")
    starts_logits_5cv = {}
    ends_logits_5cv = {}
    for state_dict in saved_models['models']:
        model.load_state_dict(state_dict)
        pridect_epoch(model, testiter, starts_logits_5cv, ends_logits_5cv)
    test = pd.read_csv(test_path)
    submit = pd.DataFrame()
    submit['textID'] = test['textID']
    submit['selected_text'] = [''] * submit.shape[0]
    for ids, _, offsets, texts in testiter:
        for id, offset, text in zip(ids, offsets, texts):
            start = end = None
            for i_s in torch.argsort(starts_logits_5cv[id], descending=True):
                if i_s > 1 and i_s < len(offset)+3:
                    start = i_s
                    break
            for i_e in torch.argsort(ends_logits_5cv[id], descending=True):
                if i_e >= start and i_e < len(offset)+3:
                    end = i_e
                    break
            assert start is not None
            assert end is not None
            submit.selected_text[submit.textID == id] = text[offset[start-3][0]: offset[end-3][1]]
    submit.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
