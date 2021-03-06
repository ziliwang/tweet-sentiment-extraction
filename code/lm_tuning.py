import torch
from torch import nn, optim
from transformers import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import click
import os
import joblib
import numpy as np
import logging
import random
import re

MAX_LEN = 100


def mask_tokens(inputs: torch.Tensor, tokenizer, mlm_probability):

    if tokenizer.mask_token is None:
        raise ValueError("This tokenizer does not have a mask token")

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@click.command()
@click.option('--data', default='../ext/cleaned.tweets500k')
@click.option('--pretrained', default='../model/roberta-l12/')
@click.option('--lr', default=1e-5)
@click.option('--batch-size', default=64)
@click.option('--epoch', default=2)
@click.option('--accumulate-step', default=2)
@click.option('--show-every', default=1000)
@click.option('--output-dir', default='roberta-base-tuning')
@click.option('--seed', default=9895)
def main(data, pretrained, lr, batch_size, epoch, accumulate_step, show_every, output_dir, seed):
    logging.basicConfig(format='%(asctime)-8s %(message)s')
    logging.error('seed everything')
    seed_everything(seed)
    tokenizer = RobertaTokenizer(os.path.join(pretrained, 'vocab.json'), os.path.join(pretrained, 'merges.txt'))
    logging.error('tokenizing')
    raw_data = [re.sub(r'@\w+', '', i).strip() for i in joblib.load(data)]
    train_data = [tokenizer.encode(i, max_length=MAX_LEN) for i in raw_data if len(i.split()) > 3]
    logging.error('training')

    def collate_func(records):
        output = []
        for r in records:
            output.append(torch.LongTensor(r))
        return pad_sequence(output, batch_first=True, padding_value=tokenizer.pad_token_id)

    trainiter = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_func)
    model = RobertaForMaskedLM.from_pretrained(pretrained).cuda()
    no_decay = ['.bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer = AdamW([{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "lr": lr, 'weight_decay': 1e-2},
                       {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "lr": lr, 'weight_decay': 0}], betas=(0.9, 0.98), eps=1e-6)
    train_steps = np.ceil(len(train_data) / batch_size / accumulate_step * epoch)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*train_steps, num_training_steps=train_steps)
    step = 0
    cum_loss = 0
    for e in range(epoch):
        for input in trainiter:
            step += 1
            input, label = mask_tokens(input, tokenizer, 0.15)
            input, label = input.cuda(), label.cuda()
            loss = model(input, attention_mask=(input!=tokenizer.pad_token_id).long(), masked_lm_labels=label)[0]
            loss.backward()
            cum_loss += loss.detach().cpu().data.numpy()
            if step % show_every == 0:
                logging.error(f'epoch {e} setp {step} avg loss: {cum_loss/show_every:.6f}')
                cum_loss = 0
            if step % accumulate_step == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    os.system(f"cp {os.path.join(pretrained, 'vocab.json')} {os.path.join(pretrained, 'merges.txt')} {output_dir}")


if __name__ == "__main__":
    main()
