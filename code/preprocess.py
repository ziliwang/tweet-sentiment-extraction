import joblib
import pandas as pd
from transformers import AlbertTokenizer
import click
import os
import numpy as np
from itertools import accumulate
import re


def annotate(tokenizer, text, selected_text):
    text = ' ' + ' '.join(text.split())
    selected = ' '.join(selected_text.split())

    start = text.find(selected)
    # for i, v in enumerate(text):
    #     if v == selected[1] and text[i:i+len(selected)-1] == selected[1:]:
    #         start = i
    if start == -1:
        print(f'{text}\n{select}')
        raise ValueError
    end = start+len(selected)
    if text[start-1] == ' ':
        start = start -1
    else:  # constraint the start is space, without this constraint parsing score 0.974, and add this 0.972
        new_start = start
        for i in range(1, len(text)):
            if start+i < len(text):
                if text[start+i] == ' ':
                    new_start = start + i
                    break
            if start - i >= 0:
                if text[start-i] == ' ':
                    new_start = start - i
                    break
                if re.match(r'\W$', text[start-i], re.I):
                    new_start = start - i + 1
                    break
        if start != new_start:
            start = new_start
            # print(f't: {text}\ns: {selected_text}\no: {text[start:end]}\nc: {text[new_start:end]}\n')
    tokens = tokenizer.tokenize(text.replace('`', "'"))
    split_in_head = [1 if i.startswith('▁') or re.match(r'\W+$', i, re.I) else 0 for i in tokens]
    offsets = list(accumulate(map(len, tokens)))
    offsets = list(zip([0] + offsets, offsets))
    bad_ann = False
    s_i = None
    e_i = None
    for i, (m, n) in enumerate(offsets):
        if s_i is None and n > start: s_i = i
        if e_i is None and n >= end: e_i = i
        # if e_i is None and n >= idx1: e_i = i
    if e_i is None: e_i = len(offsets) -1
    if offsets[e_i][1] > end:  # shrink end
        i = e_i
        while i > 1 and not split_in_head[i]:
            i -= 1
        e_i = i - 1
    # e_i = max(s_i, e_i)
    if e_i < s_i:  # shrink end failed
        i = s_i + 1
        while i < len(offsets) and not split_in_head[i]:
            i += 1
        e_i = i - 1
        bad_ann = True
    decode = text[offsets[s_i][0]:offsets[e_i][1]]
    # if set(decode.lower().split()) != set(selected_text.lower().split()):
    #     print(f'o:{text}\ns:{selected_text}\na:{decode}')
    #     print(tokens)
    #     print(offsets)
    #     print(start, end, s_i, e_i)
    return {'text': text, 'offsets': offsets, 'tokens_id': tokenizer.convert_tokens_to_ids(tokens),
            'start': s_i, 'end': e_i, 'gt': selected_text,
            'score': jaccard(decode, selected_text), 'bad': bad_ann}


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
    print(sentiment_hash)
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
        score += jaccard(decode, ann['gt'])
        n+=1
    print(f'not match {nm/n}\nBest score {score/n}')
    if not lower: save_path = 'cased_' + save_path
    joblib.dump(dataset, save_path, compress='zlib')


if __name__ == '__main__':
    main()
