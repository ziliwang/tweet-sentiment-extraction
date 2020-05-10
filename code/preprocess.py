import joblib
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
import click
import os
import numpy as np
import re


def annotate(tokenizer, text, selected_text):
    text = text.strip()
    selected = selected_text.strip()
    if not text.startswith(' '):
        text = ' ' + text
    if not selected.startswith(' '):
        selected = ' ' + selected
    start = text.find(selected)
    if start == -1:  # head broken
        selected = selected.split(' ', 2)[-1]
        start = text.find(selected)
    end = start+len(selected)
    assert start != -1
    encoding = tokenizer.encode(text.replace('`', "'"))
    tokens = encoding.tokens
    split_in_head = [1 if i.startswith('Ġ') or re.match(r'\W$', i, re.I) else 0 for i in tokens]
    offsets = encoding.offsets
    bad_ann = False
    # TODO: split pucnt
    s_i = None
    e_i = None
    for i, (m, n) in enumerate(offsets):
        if s_i is None and n > start: s_i = i
        if e_i is None and n >= end: e_i = i
    if e_i is None: e_i = len(offsets) -1
    if offsets[e_i][1] > end:
        i = e_i
        while i > 1 and not split_in_head[i]:
            i -= 1
        e_i = i - 1
    # e_i = max(s_i, e_i)
    if e_i < s_i:
        i = s_i + 1
        while i < len(offsets) and not split_in_head[i]:
            i += 1
        e_i = i - 1
        bad_ann = True
        # print(f'o:{text}\ns:{selected_text}\na:{text[offsets[s_i][0]:offsets[max(e_i, s_i)][1]]}\n{tokens}')
    decode = text[offsets[s_i][0]:offsets[e_i][1]]
    # if set(decode.lower().split()) != set(selected_text.lower().split()):
    #     print(f'o:{text}\ns:{selected_text}\na:{decode}')
    #     print(tokens)
    #     print(offsets)
    # if len(selected_text) < 3:
    #     print(f'o: {text}\nd: {decode}\ns: {selected_text}')
    return {'text': text, 'offsets': offsets, 'tokens_id': encoding.ids, 'start': s_i, 'end': e_i, 'gt': selected_text,
            'score': jaccard(decode, selected_text), 'bad': bad_ann}


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
    tokenizer = ByteLevelBPETokenizer(vocab, merges, lowercase=lower, add_prefix_space=True)
    sentiment_hash = dict((v[1:], tokenizer.token_to_id(v)) for v in ('Ġpositive', 'Ġnegative', 'Ġneutral'))
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
