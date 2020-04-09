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
    # decode = text[offsets[s_i][0]:offsets[e_i][1]]
    # if decode != selected_text:
    #     print(f'o:{text}\ns:{selected_text}\na:{decode}')
    #     print(tokens)
    #     print(offsets)
    return {'text': text, 'offsets': offsets, 'tokens_id': encoding.ids[1:-1], 'start': s_i, 'end': e_i, 'gt': selected_text}


@click.command()
@click.option('--vocab', default='../model/bert-l12/vocab.txt')
@click.option('--data-path', default='../input')
@click.option('--save-path', default='bert.input.joblib')
def main(vocab, data_path, save_path):
    tokenizer = BertWordPieceTokenizer(vocab, )
    sentiment_hash = dict((v, tokenizer.token_to_id(v)) for v in ('positive', 'negative', 'neutral'))
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    dataset = []
    for line, row in train.iterrows():
        if pd.isna(row.text) and pd.isna(row.selected_text): continue
        ann = annotate(tokenizer, row.text, row.selected_text.strip(' '))
        ann['sentiment'] = sentiment_hash[row.sentiment]
        ann['id'] = row.textID
        dataset.append(ann)
    joblib.dump(dataset, save_path, compress='zlib')


if __name__ == '__main__':
    main()
