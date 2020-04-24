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
        self.logits = nn.Linear(config.hidden_size*2, 2)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.start_n_top = 5
        self.end_n_top = 5
        # nn.init.xavier_uniform_(self.logits.weight)
        self.init_weights()
        # torch.nn.init.normal_(self.logits.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        p_mask = 1 - attention_mask.float() * (input_ids != sep_token_id).float()
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            for x in (start_positions, end_positions):
                if x.dim() > 1:
                    x.squeeze_(-1)
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)
            start_loss = F.cross_entropy(start_logits.squeeze(-1), start_positions, reduction='none')
            end_loss = F.cross_entropy(end_logits.squeeze(-1), end_positions, reduction='none')
            return start_loss + end_loss

        if not self.training:
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = start_logits.softmax(-1).log()  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)  # shape (bsz, start_n_top)
            # start_top_log_probs = start_top_log_probs - start_log_probs[:, 0:1].expand(-1, self.start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states)  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1).log()  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=1)  # shape (bsz, end_n_top, start_n_top)
            # end_top_log_probs = end_top_log_probs - end_log_probs[:, 0:1, :].expand(-1, self.end_n_top, -1)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index


def pridect_epoch(model, dataiter, bagging_cache):
    model.eval()
    with torch.no_grad():
        for ids, inputs, offsets, texts in dataiter:
            s_top_probs, s_top_idxs, e_top_probs, e_top_idxs = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long())
            for i in range(inputs.shape[0]):
                if inputs[i][1] == 7974:
                    continue
                for _i, s_idx in enumerate(s_top_idxs[i]):
                    if s_idx < 3: continue
                    for _j, e_idx in enumerate(e_top_idxs[i]):
                        if _j % model.end_n_top != _i: continue
                        if s_idx <= e_idx and e_idx -3 < len(offsets[i]):
                            cand_text = texts[i][offsets[i][s_idx-3][0]: offsets[i][e_idx-3][1]]
                            bagging_cache[ids[i]][cand_text].append((s_top_probs[i][_i]+e_top_probs[i][_j]).detach().cpu().numpy())


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
    bagging_cache = collections.defaultdict(lambda: collections.defaultdict(list))
    for state_dict in saved_models['models']:
        model.load_state_dict(state_dict)
        pridect_epoch(model, testiter, bagging_cache)
    submit = pd.DataFrame()
    submit['textID'] = test_df['textID']
    submit['selected_text'] = test_df['text']

    ans_mapping = {}

    for item_id in bagging_cache:
        ans, logprob = sorted(bagging_cache[item_id].items(), key=lambda x: np.mean(x[1]))[-1]
        ans_mapping[item_id] = ans

    submit['selected_text'] = submit.apply(lambda x: ans_mapping.get(x.textID, x.selected_text), axis=1)
    submit.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
