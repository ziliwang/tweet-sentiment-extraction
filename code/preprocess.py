import joblib
import pandas as pd
from tokenizers import BertWordPieceTokenizer
import click
import os


def annotate(tokenizer, text, selected_text):
    start = text.find(selected_text)
    assert start != -1
    encoding = tokenizer.encode(text)
    tokens = encoding.tokens[1:-1]
    offsets = encoding.offsets[1:-1]
    s_i = None
    e_i = None
    for i, (m, n) in enumerate(offsets):
        if s_i is None and n > start and not tokens[i].startswith('##'): s_i = i
        if e_i is None and n >= start + len(selected_text): e_i = i
    if e_i is None: e_i = i
    return {'text': text, 'offsets': offsets, 'tokens_id': encoding.ids[1:-1], 'start': s_i, 'end': e_i, 'gt': selected_text}


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


@click.command()
@click.option('--vocab', default='../model/bert-l12/vocab.txt')
@click.option('--data-path', default='../input')
@click.option('--lower', is_flag=True)
@click.option('--save-path', default='eletra.input.joblib')
def main(vocab, data_path, lower, save_path):
    tokenizer = BertWordPieceTokenizer(vocab, lowercase=lower)
    sentiment_hash = dict((v, tokenizer.token_to_id(v)) for v in ('positive', 'negative', 'neutral'))
    print(sentiment_hash)
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    dataset = []
    n = nm = 0
    score = 0
    for line, row in train.iterrows():
        if pd.isna(row.text) and pd.isna(row.selected_text): continue
        ann = annotate(tokenizer, row.text, row.selected_text.strip(' '))
        ann['sentiment'] = sentiment_hash[row.sentiment]
        ann['id'] = row.textID
        dataset.append(ann)
        decode = ann['text'][ann['offsets'][ann['start']][0]:ann['offsets'][ann['end']][1]]
        if set(decode.split()) != set(ann['gt'].split()):
            nm+=1
        # if jaccard(decode, ann['gt']) < 0.7:
        #     print(f"a1:{decode}\na2:{ann['gt']}")
        score += jaccard(decode, ann['gt'])
        n+=1
    print(f'not match {nm/n}\nBest score {score/n}')
    if not lower: save_path = 'cased_' + save_path
    joblib.dump(dataset, save_path, compress='zlib')


if __name__ == '__main__':
    main()
