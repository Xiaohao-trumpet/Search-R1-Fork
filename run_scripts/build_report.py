#!/usr/bin/env python3
"""Assemble the 2x2 (reward x algo) comparison + H4 interaction from the 4 logs.

Cells:  baseline = EM+vanilla,  reward = F1+vanilla,  algo = EM+A+,  both = F1+A+
Reports, per cell: mean over steps of the key training-dynamics metrics, plus the
final/best val EM. Then the main effects and the H4 interaction on the primary
metric grpo/nonuniform_group_frac (frac of GRPO groups that produce a gradient).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from parse_metrics import parse_log

LOGDIR = '/mnt/backup1/lgc/search-r1-data/exp_logs'
CELLS = {'baseline': 'EM + vanilla', 'reward': 'F1 + vanilla',
         'algo': 'EM + A+', 'both': 'F1 + A+'}
METRICS = ['reward/nonzero_frac', 'grpo/nonuniform_group_frac',
           'dynamic_sampling/kept_frac', 'actor/entropy_loss',
           'actor/grad_norm', 'response_length/mean', 'env/number_of_valid_search']

def cell_stats(name):
    path = f'{LOGDIR}/{name}.log'
    if not os.path.exists(path):
        return None
    steps, val = parse_log(path)
    if not steps:
        return None
    out = {'n_steps': len([s for s in steps if s > 0])}
    for m in METRICS:
        vals = [steps[s][m] for s in steps if m in steps[s]]
        out[m] = sum(vals) / len(vals) if vals else None
    # val: use the embedded ones too
    out['val_final'] = val[max(val)] if val else None
    out['val_best'] = max(val.values()) if val else None
    return out

def fmt(x):
    return '  n/a' if x is None else f'{x:6.3f}'

def main():
    stats = {c: cell_stats(c) for c in CELLS}
    print('=' * 78)
    print('2x2 matrix (mean over training steps; val = EM on n=8, coarse)')
    print('=' * 78)
    hdr = ['cell (reward+algo)'] + [m.split('/')[-1][:11] for m in METRICS] + ['val_best']
    print('  '.join(f'{h:>11}' for h in hdr))
    for c, desc in CELLS.items():
        s = stats[c]
        if s is None:
            print(f'{desc:>18}  -- not finished --'); continue
        row = [f'{desc:>18}'] + [f'{fmt(s[m]):>11}' for m in METRICS] + [f'{fmt(s["val_best"]):>11}']
        print('  '.join(row))

    # main effects & interaction on the primary metric
    key = 'grpo/nonuniform_group_frac'
    if all(stats[c] and stats[c][key] is not None for c in CELLS):
        b, r, a, ba = (stats['baseline'][key], stats['reward'][key],
                       stats['algo'][key], stats['both'][key])
        print('\n' + '-' * 78)
        print(f'PRIMARY METRIC: {key} (higher = more groups produce gradient)')
        print('-' * 78)
        print(f'  baseline(EM+vanilla)={b:.3f}   reward(F1+vanilla)={r:.3f}')
        print(f'  algo(EM+A+)         ={a:.3f}   both(F1+A+)        ={ba:.3f}')
        print(f'  reward main effect  (r - b)         = {r-b:+.3f}')
        print(f'  algo   main effect  (a - b)         = {a-b:+.3f}')
        print(f'  H4 interaction (ba - r) - (a - b)   = {(ba-r)-(a-b):+.3f}')
        print(f'     -> negative interaction = substitutes (dense reward reduces')
        print(f'        the marginal value of the A+/dynamic-sampling machinery).')
    print()

if __name__ == '__main__':
    main()
