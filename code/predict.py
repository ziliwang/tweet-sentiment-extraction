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
import re
from tqdm import tqdm
from copy import deepcopy
from tokenizers import ByteLevelBPETokenizer
import pandas as pd


MAX_LEN = 200
cls_token_id = 0
pad_token_id = 1
sep_token_id = 2


def preprocess(tokenizer, df):
    output = []
    sentiment_hash = dict((v[1:], tokenizer.token_to_id(v)) for v in ('Ġpositive', 'Ġnegative', 'Ġneutral'))
    for line, row in df.iterrows():
        if pd.isna(row.text): continue
        text = row.text
        if not text.startswith(' '): text = ' ' + text
        record = {}
        encoding = tokenizer.encode(text.replace('`', "'"))
        record['tokens_id'] = encoding.ids
        record['sentiment'] = sentiment_hash[row.sentiment]
        record['offsets'] = encoding.offsets
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


class RobertaForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.logits = nn.Linear(config.hidden_size*2, 2)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_states = torch.cat([outputs[0], outputs[1][:,None,:].expand_as(outputs[0])], dim=-1)
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
@click.option('--vocab', default='../model/roberta-l12/vocab.json')
@click.option('--merges', default='../model/roberta-l12/merges.txt')
@click.option('--models', default='trained.models')
@click.option('--config', default='../model/roberta-l12/config.json')
def main(test_path, vocab, merges, models, config):
    tokenizer = ByteLevelBPETokenizer(vocab, merges, lowercase=True, add_prefix_space=True)
    test_df = pd.read_csv(test_path)
    test = preprocess(tokenizer, test_df)
    model_config = BertConfig.from_json_file(config)
    saved_models = torch.load(models)
    model = RobertaForQuestionAnswering(model_config).cuda()
    testiter = DataLoader(test, batch_size=32, shuffle=False, collate_fn=collect_func)
    print(f"5cv {saved_models['score']}")
    starts_logits_5cv = {}
    ends_logits_5cv = {}
    for state_dict in saved_models['models']:
        model.load_state_dict(state_dict)
        pridect_epoch(model, testiter, starts_logits_5cv, ends_logits_5cv)
    submit = pd.DataFrame()
    submit['textID'] = test_df['textID']
    submit['selected_text'] = test_df['text']
    id2sentiment = dict((r.textID, r.sentiment) for _, r in test_df.iterrows())
    filter_reg = re.compile(r'(\?|\!|\.|)\1{1,}$')
    predicts = []
    for ids, _, offsets, texts in testiter:
        for id, offset, text in zip(ids, offsets, texts):
            if id2sentiment[id] == 'neutral':
                predicts.append(text)
                continue
            s_top_probs, s_top_idxs = torch.topk(starts_logits_5cv[id].softmax(-1).log(), 5)
            e_top_probs, e_top_idxs = torch.topk(ends_logits_5cv[id].softmax(-1).log(), 5)
            c = []
            for _i, s_idx in enumerate(s_top_idxs):
                if s_idx < 3: continue
                for _j, e_idx in enumerate(e_top_idxs):
                    if s_idx <= e_idx and e_idx -3 < len(offset):
                        c.append((s_top_probs[_i]+e_top_probs[_j], s_idx, e_idx))
            _, start, end = sorted(c)[-1]
            predicts.append(filter_reg.sub(r'\1', text[offset[start-3][0]: offset[end-3][1]]))
    submit = pd.DataFrame()
    submit['textID'] = test_df['textID']
    submit['selected_text'] = predicts
    submit.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
