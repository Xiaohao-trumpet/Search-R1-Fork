#!/usr/bin/env python3
"""Parse a Search-R1 experiment log into a clean per-step metrics table.

Usage: python parse_metrics.py <variant1.log> [<variant2.log> ...]
Prints a CSV-ish table per variant and a compact summary (means over steps).
"""
import re, sys, os

KEYS = [
    'reward/mean', 'reward/nonzero_frac', 'grpo/nonuniform_group_frac',
    'dynamic_sampling/kept_frac', 'actor/entropy_loss', 'actor/pg_loss',
    'actor/grad_norm', 'actor/ppo_kl', 'response_length/mean',
    'env/number_of_valid_search', 'val/test_score/nq',
]

def strip_ansi(s):
    return re.sub(r'\x1b\[[0-9;]*m', '', s)

def parse_log(path):
    steps = {}
    val_scores = {}
    with open(path, errors='ignore') as f:
        for line in f:
            line = strip_ansi(line)
            m = re.search(r'step:(\d+) -(.*)', line)
            if m:
                step = int(m.group(1)); body = m.group(2)
                d = {}
                for k in KEYS:
                    mm = re.search(re.escape(k) + r':([-0-9.]+)', body)
                    if mm:
                        try: d[k] = float(mm.group(1))
                        except ValueError: pass
                steps[step] = d
            mv = re.search(r"val/test_score/nq[\"']?:\s*([-0-9.]+)", line)
            if mv:
                ms = re.search(r'step:(\d+)', line)
                if ms: val_scores[int(ms.group(1))] = float(mv.group(1))
    return steps, val_scores

def main():
    for path in sys.argv[1:]:
        name = os.path.basename(path).replace('.log', '')
        steps, val = parse_log(path)
        if not steps:
            print(f"\n### {name}: no steps parsed\n"); continue
        cols = ['step', 'reward/mean', 'reward/nonzero_frac', 'grpo/nonuniform_group_frac',
                'dynamic_sampling/kept_frac', 'actor/entropy_loss', 'actor/pg_loss',
                'response_length/mean', 'env/number_of_valid_search', 'val/test_score/nq']
        print(f"\n### {name}  ({len(steps)} steps)")
        print(' | '.join(c.split('/')[-1][:10] for c in cols))
        agg = {c: [] for c in cols[1:]}
        for st in sorted(steps):
            d = steps[st]
            if st in val: d = {**d, 'val/test_score/nq': val[st]}
            row = [str(st)]
            for c in cols[1:]:
                v = d.get(c)
                row.append('' if v is None else f'{v:.3f}')
                if v is not None: agg[c].append(v)
            print(' | '.join(row))
        print('--- mean over steps ---')
        print('mean | ' + ' | '.join(f'{sum(agg[c])/len(agg[c]):.3f}' if agg[c] else '' for c in cols[1:]))
        # last non-empty val score
        if val:
            last = max(val)
            print(f'final val/test_score/nq (step {last}): {val[last]:.4f}   best: {max(val.values()):.4f}')

if __name__ == '__main__':
    main()
