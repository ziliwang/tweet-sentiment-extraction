import joblib
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
import click
import os


def annotate(tokenizer, text, selected_text):
    start = text.find(selected_text)
    assert start != -1
    encoding = tokenizer.encode(text)
    tokens = encoding.tokens
    offsets = encoding.offsets
    s_i = None
    e_i = None
    for i, (m, n) in enumerate(offsets):
        if s_i is None and n > start: s_i = i
        if e_i is None and n >= start + len(selected_text): e_i = i
    if e_i is None: e_i = i
    decode = text[offsets[s_i][0]:offsets[e_i][1]]
    # if set(decode.split()) != set(selected_text.split()):
    #     print(f'o:{text}\ns:{selected_text}\na:{decode}')
    #     print(tokens)
    #     print(offsets)
    return {'text': text, 'offsets': offsets, 'tokens_id': encoding.ids, 'start': s_i, 'end': e_i, 'gt': selected_text}


@click.command()
@click.option('--vocab', default='../model/roberta-l12/vocab.json')
@click.option('--merges', default='../model/roberta-l12/merges.txt')
@click.option('--data-path', default='../input')
@click.option('--lower', is_flag=True)
@click.option('--save-path', default='roberta.input.joblib')
def main(vocab, merges, data_path, lower, save_path):
    tokenizer = ByteLevelBPETokenizer(vocab, merges, lowercase=lower, add_prefix_space=True)
    sentiment_hash = dict((v, tokenizer.token_to_id(v)) for v in ('positive', 'negative', 'neutral'))
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    dataset = []
    for line, row in train.iterrows():
        if pd.isna(row.text) and pd.isna(row.selected_text): continue
        ann = annotate(tokenizer, row.text, row.selected_text.strip(' '))
        ann['sentiment'] = sentiment_hash[row.sentiment]
        ann['id'] = row.textID
        dataset.append(ann)
    if not lower: save_path = 'cased_' + save_path
    joblib.dump(dataset, save_path, compress='zlib')


if __name__ == '__main__':
    main()
