import joblib
import pandas as pd
from transformers import XLMTokenizer
import click
import os
import numpy as np
import re


# xlm tokenizer: 1. skip multi-space, 2. space is </w>, 3. space in the end, 3. ` to '
# xlm first clean the text （t.moses_punct_norm(text, lang='en')）, the ` will be convert to '
# then use the languague tokenizer to tokenize the string, i.e. english tokenizer, chinese tokenizer
# then use the bpe tokenize the tokens, each tokens will add '</w>' before bep tokenize

task_id_map = {'positive': 2258, 'negative': 3683, 'neutral': 8323}


def xlm_offsets(tokens, text):
    offsets = []
    p = 0
    clean = [i.replace('</w>', '') for i in tokens]
    for i, t in enumerate(clean):
        if i < len(clean) -1:
            check_text = t + clean[i+1]
            if text[p:p+len(check_text)] == check_text:
                offsets.append((p, p+len(t)))
                p += len(t)
            else:
                offsets.append((p, p+len(t)+1))
                p += len(t) +1
        else:
            offsets.append((p, len(text)))
    return offsets


def annotate(tokenizer, text, selected_text):
    text = ' '.join(text.strip().split())
    selected = ' '.join(selected_text.strip().split())
    start = text.find(selected)
    assert start != -1
    end = start+len(selected)
    normed_text = tokenizer.moses_punct_norm(text, lang='en').lower()
    tokens = tokenizer.tokenize(normed_text)
    offsets = xlm_offsets(tokens, normed_text)
    # TODO: split pucnt
    s_i = None
    e_i = None
    for i, (m, n) in enumerate(offsets):
        if s_i is None and n > start: s_i = i
        if e_i is None and n >= end: e_i = i
    if e_i is None: e_i = len(offsets) -1
    decode = text[offsets[s_i][0]:offsets[e_i][1]]
    if set(decode.lower().split()) != set(selected_text.lower().split()):
        print(f'o:{text}\ns:{selected_text}\na:{decode}')
        print(tokens)
        print([text[_i:_j] for _i, _j in offsets])
        print(offsets)
        print(start, end)
    # if len(selected_text) < 3:
    #     print(f'o: {text}\nd: {decode}\ns: {selected_text}')
    return {'text': text, 'offsets': offsets, 'tokens_id': tokenizer.convert_tokens_to_ids(tokens), 'start': s_i, 'end': e_i, 'gt': selected_text,
            'score': jaccard(decode, selected_text)}


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


@click.command()
@click.option('--vocab', default='../model/xlm/xlm-mlm-en-2048-vocab.json')
@click.option('--merges', default='../model/xlm/xlm-mlm-en-2048-merges.txt')
@click.option('--data-path', default='../input')
@click.option('--save-path', default='xlm.input.joblib')
def main(vocab, merges, data_path, save_path):
    tokenizer = XLMTokenizer(vocab, merges)
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
        ann['sentiment'] = task_id_map[row.sentiment]
        ann['id'] = row.textID
        dataset.append(ann)
        decode = ann['text'][ann['offsets'][ann['start']][0]:ann['offsets'][ann['end']][1]]
        if set(decode.split()) != set(ann['gt'].split()):
            nm+=1
        score += jaccard(decode, ann['gt'])
        n+=1
    print(f'not match {nm/n}\nBest score {score/n}')
    joblib.dump(dataset, save_path, compress='zlib')


if __name__ == '__main__':
    main()
