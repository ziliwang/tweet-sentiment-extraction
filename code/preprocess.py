import joblib
import pandas as pd
from transformers import AlbertTokenizer
import click
import os
import numpy as np
from itertools import accumulate


def annotate(tokenizer, text, selected_text):
    text = ' ' + ' '.join(text.split())
    selected = ' ' + ' '.join(selected_text.split())

    start = None
    for i, v in enumerate(text):
        if v == selected[1] and text[i:i+len(selected)-1] == selected[1:]:
            start = i
    assert start is not None
    # start = text.find(selected_text)
    # assert start != -1
    tokens = tokenizer.tokenize(text)
    offsets = list(accumulate(map(len, tokens)))
    offsets = list(zip([0] + offsets, offsets))
    s_i = None
    e_i = None
    for i, (m, n) in enumerate(offsets):
        if s_i is None and n > start: s_i = i
        if e_i is None and n >= start + len(selected)-1: e_i = i
        # if e_i is None and n >= idx1: e_i = i
    if e_i is None: e_i = i
    assert s_i is not None, (tokens, offsets)
    decode = text[offsets[s_i][0]:offsets[e_i][1]]
    if set(decode.lower().split()) != set(selected_text.lower().split()):
        print(f'o:{text}\ns:{selected_text}\na:{decode}')
        print(tokens)
        print(offsets)
    return {'text': text, 'offsets': offsets, 'tokens_id': tokenizer.convert_tokens_to_ids(tokens), 'start': s_i, 'end': e_i, 'gt': selected_text}


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


@click.command()
@click.option('--sp-model', default='../model/albert.base/spiece.model')
@click.option('--data-path', default='../input')
@click.option('--lower', is_flag=True)
@click.option('--save-path', default='alberta.input.joblib')
def main(sp_model, data_path, lower, save_path):
    tokenizer = AlbertTokenizer(sp_model, do_lower_case=lower)
    sentiment_hash = dict(zip(['positive', 'negative', 'neutral'], tokenizer.convert_tokens_to_ids(['▁positive', '▁negative', '▁neutral'])))
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    dataset = []
    n = nm = 0
    score = 0
    for line, row in train.iterrows():
        if pd.isna(row.text) and pd.isna(row.selected_text): continue
        try:
            ann = annotate(tokenizer, row.text, row.selected_text.strip(' '))
        except AssertionError:
            print(row.text, row.selected_text.strip(' '))
            continue
        ann['sentiment'] = sentiment_hash[row.sentiment]
        ann['id'] = row.textID
        dataset.append(ann)
        decode = ann['text'][ann['offsets'][ann['start']][0]:ann['offsets'][ann['end']][1]]
        if set(decode.split()) != set(ann['gt'].split()):
            nm+=1
        if jaccard(decode, ann['gt']) < 0.7:
            print(f"a1:{decode}\na2:{ann['gt']}")
        score += jaccard(decode, ann['gt'])
        n+=1
    print(f'not match {nm/n}\nBest score {score/n}')
    if not lower: save_path = 'cased_' + save_path
    joblib.dump(dataset, save_path, compress='zlib')


if __name__ == '__main__':
    main()
