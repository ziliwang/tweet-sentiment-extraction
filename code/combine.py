import torch
import click
import numpy as np


@click.command()
@click.argument('src', nargs=-1)
def main(src):
    best_scores = [0] * 5
    best_models = [None] * 5
    best_types = [None] * 5
    for fn in src:
        print(f'loading {fn}')
        tmp = torch.load(fn)
        for i, v in enumerate(tmp['scores']):
            if v > best_scores[i]:
                best_scores[i] = v
                best_models[i] = tmp['models'][i]
                best_types[i] = tmp['type']
    mean_score = np.mean(best_scores)
    print(f'mean score {mean_score}')
    out = f'combined-roberta-att-{int(mean_score*1e4)}.models'
    print(f'saving to {out}')
    torch.save({'score': mean_score, 'scores': best_scores, "models": best_models, 'types': best_types}, out)


if __name__ == '__main__':
    main()
