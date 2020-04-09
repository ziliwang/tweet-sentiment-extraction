import torch
from torch import nn, optim
from torch.nn import functional as F
from transformers import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import joblib
import click
import random
import numpy as np
import os
from tqdm import tqdm

MAX_LEN = 200
cls_token_id = 101
pad_token_id = 0
sep_token_id = 102


def collect_func(records):
    inputs = []
    starts = []
    ends = []
    offsets = []
    texts = []
    gts = []
    for rec in records:
        inputs.append(torch.LongTensor([rec['sentiment'], sep_token_id] + rec['tokens_id'][:MAX_LEN] + [sep_token_id]))
        starts.append(rec['start']+2)
        ends.append(rec['end']+2)
        offsets.append(rec['offsets'])
        texts.append(rec['text'])
        gts.append(rec['gt'])
    return pad_sequence(inputs, batch_first=True, padding_value=pad_token_id).cuda(), torch.LongTensor(starts).cuda(), torch.LongTensor(ends).cuda(), \
        offsets, texts, gts


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, optimizer, lr_scheduler, dataiter, accumulate_step):
    model.train()
    step = 0
    cum_loss = 0
    for inputs, starts, ends, _, _, _ in tqdm(dataiter):
        step += 1
        loss = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long(), start_positions=starts, end_positions=ends)[0]
        cum_loss += loss.detach().cpu().data.numpy()
        loss.backward()
        if step % accumulate_step == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    return cum_loss/step


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def validate_epoch(model, dataiter):
    model.eval()
    score = 0
    sample_counts = 0
    for inputs, _, _, offsets, texts, gts in dataiter:
        start_logits, end_logits = model(input_ids=inputs, attention_mask=(inputs!=pad_token_id).long())[:2]
        _, start_top5 = torch.topk(start_logits, 20)
        _, end_top5 = torch.topk(end_logits, 20)
        batch_size = inputs.shape[0]
        sample_counts += batch_size
        for i in range(batch_size):
            start = None
            for i_s in start_top5[i]:
                if i_s > 1 and i_s < len(offsets[i])+2:
                    start = i_s
                    break
            end = None
            for i_e in end_top5[i]:
                if i_e >= i_s and i_e < len(offsets[i])+2:
                    end = i_e
                    break
            assert start is not None
            assert end is not None
            predict = texts[i][offsets[i][start-2][0]: offsets[i][end-2][1]]
            score += jaccard(gts[i], predict)
    return score/sample_counts


def fold_train(model, optimizer, lr_scheduler, epoch, train_dataiter, val_dataiter, accumulate_step):
    for e in range(epoch):
        loss = train_epoch(model, optimizer, lr_scheduler, train_dataiter, accumulate_step)
        score = validate_epoch(model, val_dataiter)
        print(f'epoch {e} loss {loss:.6f} score: {score:.6f}')


@click.command()
@click.option('--data', default='bert.input.joblib')
@click.option('--pretrained', default='../model/bert-l12')
@click.option('--lr', default=3e-5)
@click.option('--batch-size', default=32)
@click.option('--epoch', default=5)
@click.option('--accumulate-step', default=1)
@click.option('--seed', default=9895)
def main(data, pretrained, lr, batch_size, epoch, accumulate_step, seed):
    seed_everything(seed)
    data = joblib.load(data)
    train, val = train_test_split(data, test_size=0.2, random_state=seed)
    model = BertForQuestionAnswering.from_pretrained(pretrained, num_labels=2).cuda()
    no_decay = ['.bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer = AdamW([{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "lr": lr, 'weight_decay': 1e-3},
                       {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "lr": lr, 'weight_decay': 0}])
    train_steps = np.ceil(len(train) / batch_size / accumulate_step * epoch)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*train_steps, num_training_steps=train_steps)
    trainiter = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collect_func)
    valiter = DataLoader(val, batch_size=batch_size*2, shuffle=False, collate_fn=collect_func)
    fold_train(model, optimizer, lr_scheduler, epoch, trainiter, valiter, accumulate_step)


if __name__ == '__main__':
    main()
