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
from tokenizers import BertWordPieceTokenizer
import pandas as pd


MAX_LEN = 200
cls_token_id = 101
pad_token_id = 0
sep_token_id = 102


def preprocess(tokenizer, df):
    output = []
    sentiment_hash = dict((v, tokenizer.token_to_id(v)) for v in ('positive', 'negative', 'neutral'))
    for line, row in df.iterrows():
        if pd.isna(row.text): continue
        record = {}
        encoding = tokenizer.encode(row.text)
        record['tokens_id'] = encoding.ids[1:-1]
        record['sentiment'] = sentiment_hash[row.sentiment]
        record['offsets'] = encoding.offsets[1:-1]
        record['text'] = row.text
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
        inputs.append(torch.LongTensor([rec['sentiment'], sep_token_id] + rec['tokens_id'][:MAX_LEN] + [sep_token_id]))
        offsets.append(rec['offsets'])
        texts.append(rec['text'])
    return ids, pad_sequence(inputs, batch_first=True, padding_value=pad_token_id).cuda(), offsets, texts


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.logits = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AvgPool1d(3, stride=1, padding=1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        hidden_states = outputs[0]
        start_end_logits = self.logits(self.dropout(hidden_states))
        start_logits, end_logits = start_end_logits.split(1, dim=-1)

        if start_positions is not None and end_positions is not None:
            for x in (start_positions, end_positions):
                if x.dim() > 1:
                    x.squeeze_(-1)
            # start_logits = self.avgpool(torch.cat(start_logits))
            # end_logits = self.avgpool(torch.cat(end_logits))
            start_loss = F.cross_entropy(start_logits.squeeze(-1), start_positions, reduction='none')
            end_loss = F.cross_entropy(end_logits.squeeze(-1), end_positions, reduction='none')
            return start_loss + end_loss

        if not self.training:
            p_mask = attention_mask.float()  # 1 for available 0 for unavailable
            p_mask[:, :2] = 0.0
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
@click.option('--vocab', default='../model/bert-l12/vocab.txt')
@click.option('--models', default='trained.models')
@click.option('--config', default='../model/bert-l12/config.json')
def main(test_path, vocab, models, config):
    tokenizer = BertWordPieceTokenizer(vocab)
    test_df = pd.read_csv(test_path)
    test = preprocess(tokenizer, test_df)
    model_config = BertConfig.from_json_file(config)
    saved_models = torch.load(models)
    model = BertForQuestionAnswering(model_config).cuda()
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
                if i_s > 1 and i_s < len(offset)+2:
                    start = i_s
                    break
            for i_e in torch.argsort(ends_logits_5cv[id], descending=True):
                if i_e >= start and i_e < len(offset)+2:
                    end = i_e
                    break
            assert start is not None
            assert end is not None
            submit.selected_text[submit.textID == id] = text[offset[start-2][0]: offset[end-2][1]]
    submit.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
