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


MAX_LEN = 100
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


class TaskLayer(nn.Module):

    def __init__(self, hidden_size):
        super(TaskLayer, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
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
@click.option('--test-path', default='../input/test.csv')
@click.option('--vocab', default='../model/roberta-l12/vocab.json')
@click.option('--merges', default='../model/roberta-l12/merges.txt')
@click.option('--models', default='trained.models')
@click.option('--config', default='../model/roberta-l12/config.json')
def main(test_path, vocab, merges, models, config):
    tokenizer = ByteLevelBPETokenizer(vocab, merges, lowercase=True, add_prefix_space=True)
    test_df = pd.read_csv(test_path)
    test = preprocess(tokenizer, test_df)
    model_config = RobertaConfig.from_json_file(config)
    saved_models = torch.load(models)
    model = TweetSentiment(model_config).cuda()
    testiter = DataLoader(test, batch_size=32, shuffle=False, collate_fn=collect_func)
    print(f"5cv {saved_models['score']}")
    span_logits_bagging = {}
    for state_dict in saved_models['models']:
        model.load_state_dict(state_dict)
        pridect_epoch(model, testiter, span_logits_bagging)
    id2sentiment = dict((r.textID, r.sentiment) for _, r in test_df.iterrows())
    predicts = {}
    for ids, inputs, offsets, texts in testiter:
        bsz, slen = inputs.shape
        for id, offset, text in zip(ids, offsets, texts):
            if id2sentiment[id] == 'neutral' and len(text.split()) == 1:
                predicts[id] = text
                continue
            prob, idxs = torch.topk(span_logits_bagging[id], 2, dim=-1)
            if prob[0]/prob[1] < 1.05:
                predict = ''
                for idx in idxs.cpu().numpy():
                    start, end = divmod(idx, slen)
                    predict += ' ' + text[offset[start-3][0]: offset[end-3][1]]
            else:
                start, end = divmod(idxs[0].cpu().numpy(), slen)
                predict = text[offset[start-3][0]: offset[end-3][1]]
            # span = span_logits_bagging[id].max(-1)[1].cpu().numpy()
            # start, end = divmod(span, slen)
            # predicts[id] = text[offset[start-3][0]: offset[end-3][1]]
            predicts[id] = predict
    submit = pd.DataFrame()
    submit['textID'] = test_df['textID']
    submit['selected_text'] = [predicts.get(r.textID, r.text) for _, r in test_df.iterrows()]
    submit.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
