import joblib
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from transformers import XLMTokenizer
import click
import os
import numpy as np
import re


def get_moses_offsets(tokens, text):
    offsets = []
    p = 0
    for i, t in enumerate(tokens):
        if i < len(tokens) -1:
            check_text = t + tokens[i+1]
            offsets.append((p, p+len(t)))
            if text[p:p+len(check_text)] == check_text:
                p += len(t)
            else:
                p += len(t) +1
        else:
            offsets.append((p, len(text)))
    return offsets


def get_offsets(moses_tokenize, t, text):
    moses_tokens = moses_tokenize(text, lang='en')
    out_tokens =[]
    out_token_ids = []
    out_offsets = []
    moses_offsets = get_moses_offsets(moses_tokens, text)
    for i, (m_t_s, m_t_e) in enumerate(moses_offsets):
        c_token = moses_tokens[i]
        has_space_inhead = False
        if m_t_s == 0 or text[m_t_s-1] == ' ':
            has_space_inhead = True
            c_token = ' ' + c_token
        encoding = t.encode(c_token)
        for j, (r_t_s, r_t_e) in enumerate(encoding.offsets):
            if has_space_inhead:
                t_s = m_t_s + max(0, r_t_s -1)
                t_e = m_t_s + max(0, r_t_e -1)
            else:
                t_s = m_t_s + r_t_s
                t_e = m_t_s + r_t_e
            # if encoding.tokens[j].startswith('Ġ'):
            #     assert encoding.tokens[j][1:] == text[t_s:t_e], (encoding.tokens[j][1:], text[t_s:t_e], text)
            # else:
            #     assert encoding.tokens[j] == text[t_s:t_e], (encoding.tokens[j], text[t_s:t_e], text)
            out_offsets.append((t_s, t_e))
            out_tokens.append(encoding.tokens[j])
            out_token_ids.append(encoding.ids[j])
    return out_tokens, out_offsets, out_token_ids, moses_offsets


def keep_most(moses_offsets, pos, start=True):
    for i, (s, e) in enumerate(moses_offsets):
        if s <= pos and pos <= e:
            if start:
                if (e - pos)/(e-s) >= 0.5 or i == len(moses_offsets) -1:
                    return s
                else:
                    return moses_offsets[i+1][0]
            else:
                if i == 0 or (pos-s)/(e-s) >= 0.5:
                    return e
                else:
                    return moses_offsets[i-1][1]


def annotate(moses_tokenize, tokenizer, text, selected_text):
    text = ' '.join(text.split()).lower()
    selected = ' '.join(selected_text.split()).lower()
    start = text.find(selected)
    assert start != -1
    end = start+len(selected)
    tokens, offsets, tokens_ids, moses_offsets = get_offsets(moses_tokenize, tokenizer, text.replace('`', "'").replace("ï¿½", "***"))
    start = keep_most(moses_offsets, start)
    end = keep_most(moses_offsets, end, start=False)
    s_i = None
    e_i = None
    for i, (m, n) in enumerate(offsets):
        if s_i is None and n > start: s_i = i
        if e_i is None and n >= end: e_i = i
    if e_i is None: e_i = len(offsets) -1
    decode = text[offsets[s_i][0]:offsets[e_i][1]]
    if set(decode.lower().split()) != set(selected_text.lower().split()):
        raise AssertionError
    # if jaccard(decode, selected_text) < .1:
    #     print(f'o:{text}\ns:{selected_text}\na:{decode}')
    #     print(tokens)
    #     print(offsets)
    # if len(selected_text) < 3:
    #     print(f'o: {text}\nd: {decode}\ns: {selected_text}')
    return {'text': text, 'offsets': offsets, 'tokens_id': tokens_ids, 'start': s_i, 'end': e_i, 'gt': selected_text,
            'score': jaccard(decode, selected_text)}


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


@click.command()
@click.option('--vocab', default='../model/roberta-l12/vocab.json')
@click.option('--merges', default='../model/roberta-l12/merges.txt')
@click.option('--data-path', default='../input')
@click.option('--lower', is_flag=True)
@click.option('--save-path', default='roberta.input.joblib')
def main(vocab, merges, data_path, lower, save_path):
    moses_token_func = XLMTokenizer(vocab, merges, lowercase=True).moses_tokenize
    tokenizer = ByteLevelBPETokenizer(vocab, merges, lowercase=lower)
    sentiment_hash = dict((v[1:], tokenizer.token_to_id(v)) for v in ('Ġpositive', 'Ġnegative', 'Ġneutral'))
    print(sentiment_hash)
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    dataset = []
    n = nm = 0
    score = 0
    for line, row in train.iterrows():
        if pd.isna(row.text) and pd.isna(row.selected_text): continue
        try:
            ann = annotate(moses_token_func, tokenizer, row.text, row.selected_text.strip(' '))
        except AssertionError as s:
            print(s)
        ann['sentiment'] = sentiment_hash[row.sentiment]
        ann['id'] = row.textID
        dataset.append(ann)
        decode = ann['text'][ann['offsets'][ann['start']][0]:ann['offsets'][ann['end']][1]]
        if set(decode.lower().split()) != set(ann['gt'].lower().split()):
            nm+=1
        score += jaccard(decode, ann['gt'])
        n+=1
    print(f'not match {nm/n}\nBest score {score/n}')
    if not lower: save_path = 'cased_' + save_path
    joblib.dump(dataset, save_path, compress='zlib')


if __name__ == '__main__':
    main()
