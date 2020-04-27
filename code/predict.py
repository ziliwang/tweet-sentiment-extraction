import torch
from torch import nn, optim
from torch.nn import functional as F
from transformers import *
from transformers.modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits
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
import collections


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
        encoding = tokenizer.encode(text)
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
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.start_n_top = 5
        self.end_n_top = 5
        # nn.init.xavier_uniform_(self.logits.weight)
        self.init_weights()
        # torch.nn.init.normal_(self.logits.weight, std=0.02)

    def inference_start(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        p_mask = 1 - attention_mask.float() * (input_ids != sep_token_id).float()
        p_mask[:,:2] = 1.0
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        return start_logits
        #
        # if not self.training:
        #     bsz, slen, hsz = hidden_states.size()
        #     start_log_probs = start_logits.softmax(-1).log()  # shape (bsz, slen)
        #
        #     start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)  # shape (bsz, start_n_top)
        #     # start_top_log_probs = start_top_log_probs - start_log_probs[:, 0:1].expand(-1, self.start_n_top)
        #     start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
        #     start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
        #     start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)
        #
        #     hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states)  # shape (bsz, slen, start_n_top, hsz)
        #     p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
        #     end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
        #     end_log_probs = F.softmax(end_logits, dim=1).log()  # shape (bsz, slen, start_n_top)
        #
        #     end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=1)  # shape (bsz, end_n_top, start_n_top)
        #     # end_top_log_probs = end_top_log_probs - end_log_probs[:, 0:1, :].expand(-1, self.end_n_top, -1)
        #     end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
        #     end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)
        #
        #     return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index

    def inference_end(self, input_ids, attention_mask, start_logits):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        p_mask = 1 - attention_mask.float() * (input_ids != sep_token_id).float()
        p_mask[:,:2] = 1.0
        bsz, slen, hsz = hidden_states.size()
        start_log_probs = start_logits.softmax(-1).log()  # shape (bsz, slen)
        start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)  # shape (bsz, start_n_top)
        start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)
        start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
        start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

        hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states)  # shape (bsz, slen, start_n_top, hsz)
        p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
        end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)

        return end_logits


def inference_start(model, dataiter, bagging_start_logits):
    model.eval()
    with torch.no_grad():
        for ids, inputs, _, _ in dataiter:
            start_logits = model.inference_start(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long())
            for i, logits in zip(ids, start_logits):
                if i in bagging_start_logits:
                    bagging_start_logits[i] += logits
                else:
                    bagging_start_logits[i] = logits


def inference_end(model, dataiter, bagging_start_logits, bagging_end_logits):
    model.eval()
    with torch.no_grad():
        for ids, inputs, _, _ in dataiter:
            start_logits = torch.stack([bagging_start_logits[i] for i in ids])  # bsz, start_n_top
            end_logits = model.inference_end(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long(), start_logits=start_logits)
            for i, logits in zip(ids, end_logits):
                if i in bagging_end_logits:
                    bagging_end_logits[i] += logits
                else:
                    bagging_end_logits[i] = logits


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
    print('inference start')
    bagging_start_logits = {}
    for state_dict in saved_models['models']:
        model.load_state_dict(state_dict)
        inference_start(model, testiter, bagging_start_logits)
    print('inference end')
    bagging_end_logits = {}
    for state_dict in saved_models['models']:
        model.load_state_dict(state_dict)
        inference_end(model, testiter, bagging_start_logits, bagging_end_logits)
    print('inference answer')
    id2sentiment = dict((r.textID, r.sentiment) for _, r in test_df.iterrows())
    predict = []
    for ids, _, offsets, texts in testiter:

        for id, offset, text in zip(ids, offsets, texts):
            if id2sentiment[id] == 'neutral':
                predict.append(text)
                continue
            start_logits = bagging_start_logits[id]
            end_logits = bagging_end_logits[id]
            start_log_probs = start_logits.softmax(-1).log()  # slen
            start_top_log_probs, start_top_index = torch.topk(start_log_probs, model.start_n_top, dim=-1)  # shape (start_n_top)
            end_log_probs = F.softmax(end_logits, dim=0).log()  # shape (slen, start_n_top)
            end_top_log_probs, end_top_index = torch.topk(end_log_probs, model.end_n_top, dim=0)  # shape (end_n_top, start_n_top)
            c = []
            for _i, s_idx in enumerate(start_top_index):
                if s_idx < 3:
                    continue
                for _j, e_idx in enumerate(end_top_index[:, _i]):
                    if s_idx <= e_idx and e_idx -3 < len(offset):
                        c.append((start_top_log_probs[_i]+end_top_log_probs[_j, _i], s_idx, e_idx))
            if len(c) == 0:
                print(start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, len(offset))
            _, start, end = sorted(c)[-1]
            predict.append(text[offset[start-3][0]: offset[end-3][1]])

    submit = pd.DataFrame()
    submit['textID'] = test_df['textID']
    submit['selected_text'] = predict
    submit.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
