import joblib
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
import click
import os
import numpy as np


def annotate(tokenizer, text, selected_text):
    # text = " " + " ".join(text.split())
    # selected = " " + " ".join(selected_text.split())
    #
    # len_st = len(selected) - 1
    # idx0 = None
    # idx1 = None
    #
    # for ind in (i for i, e in enumerate(text) if e == selected[1]):
    #     if " " + text[ind: ind+len_st] == selected:
    #         idx0 = ind
    #         idx1 = ind + len_st - 1
    #         break
    if not text.startswith(' '):
        text = ' ' + text
    start = text.find(selected_text.strip())
    assert start != -1
    # assert idx0 is not None
    # assert idx1 is not None
    # start = idx0
    encoding = tokenizer.encode(text.replace('`', "'"))
    tokens = encoding.tokens
    offsets = encoding.offsets
    s_i = None
    e_i = None
    for i, (m, n) in enumerate(offsets):
        if s_i is None and n > start: s_i = i
        if e_i is None and n >= start + len(selected_text): e_i = i
        # if e_i is None and n >= idx1: e_i = i
    if s_i is None: s_i = 0
    if e_i is None: e_i = len(offsets) -1
    # a = np.zeros(len(text))
    # a[start:start+len(selected_text)] = 1
    # b = []
    # for i, v in enumerate(offsets):
    #     if sum(a[v[0]:v[1]]) > 0:
    #         b.append(i)
    # s_i = b[0]
    # e_i = b[-1]
    decode = text[offsets[s_i][0]:offsets[e_i][1]]
    # if set(decode.lower().split()) != set(selected_text.lower().split()):
    #     print(f'o:{text}\ns:{selected_text}\na:{decode}')
    #     print(tokens)
    #     print(offsets)
    #     print(start, s_i, e_i)
    return {'text': text, 'offsets': offsets, 'tokens_id': encoding.ids, 'start': s_i, 'end': e_i, 'gt': selected_text,
            'note_score': jaccard(decode, selected_text)}


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
