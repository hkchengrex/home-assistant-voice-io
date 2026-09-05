"""Plot descriptive distance/margin PR envelopes from saved random holdouts.

Evaluation labels select the envelope: these are exploratory separation plots,
not validation of a selected deployment threshold. No recordings are exported.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def operating_points(predictions):
    """Best precision attainable at each observed TP count by two thresholds.

    Accept distance <= d and signed competitor margin >= m, with m >= 0.
    Incorrect positive labels and accepted negatives both count as false accepts.
    All positive queries remain in the recall denominator, including unsupported
    or non-finite queries (which cannot be accepted).
    """
    positives = sum(p['target'] is not None for p in predictions)
    if not positives:
        raise ValueError('At least one positive observation is required')
    valid = [p for p in predictions if np.isfinite(p['distance']) and np.isfinite(p['margin'])
             and p['margin'] >= 0]
    empty = dict(tp=0, fp=0, recall=0., precision=1., distance=None, margin=None)
    if not valid:
        return [empty]
    distances = np.unique([p['distance'] for p in valid])
    margins = np.unique([0.] + [p['margin'] for p in valid])[::-1]
    di = np.searchsorted(distances, [p['distance'] for p in valid])
    mi = len(margins) - 1 - np.searchsorted(margins[::-1], [p['margin'] for p in valid])
    correct = np.array([p['target'] is not None and p['target'] == p['predicted'] for p in valid])
    tp = np.zeros((len(distances), len(margins)), dtype=np.int32)
    fp = np.zeros_like(tp)
    np.add.at(tp, (di[correct], mi[correct]), 1)
    np.add.at(fp, (di[~correct], mi[~correct]), 1)
    tp = tp.cumsum(0).cumsum(1).ravel()
    fp = fp.cumsum(0).cumsum(1).ravel()
    # For a fixed true-positive count, the minimum FP dominates other choices.
    order = np.lexsort((fp, tp))
    _, first = np.unique(tp[order], return_index=True)
    chosen = order[first]
    points = [empty]
    for i in chosen:
        t, f = int(tp[i]), int(fp[i])
        if t == 0:
            continue
        d, m = divmod(int(i), len(margins))
        points.append(dict(tp=t, fp=f, recall=t / positives, precision=t / (t + f),
                           distance=float(distances[d]), margin=float(margins[m])))
    return points


def frontier(points):
    """Remove points dominated by equal/better precision at higher recall."""
    kept, best_precision = [], -1.
    for point in reversed(points):
        if point['precision'] > best_precision:
            kept.append(point)
            best_precision = point['precision']
    kept.reverse()
    if not kept or kept[0]['recall'] > 0:
        kept.insert(0, points[0])
    return kept


def summarize(predictions, points):
    positives = sum(p['target'] is not None for p in predictions)
    def best(candidates):
        return max(candidates, key=lambda p: (p['tp'], -p['fp']))
    return {
        'observations': len(predictions), 'positives': positives,
        'negatives': len(predictions) - positives,
        'correct_top_rank': sum(p['target'] is not None and p['target'] == p['predicted'] for p in predictions),
        'at_precision': {str(limit): best([p for p in points if p['precision'] >= limit])
                         for limit in (1., .995, .99, .98, .95)},
        'at_wrong_accept_budget': {str(limit): best([p for p in points if p['fp'] <= limit])
                                   for limit in (0, 1, 2, 3, 5, 10)},
    }


LABELS = {'pcen': 'PCEN (baseline)', 'mfcc': 'MFCC', 'pncc': 'PNCC',
          'mfcc_floor': 'MFCC + variance floor', 'pcen_floor': 'PCEN + variance floor',
          'pcen24_floor': 'PCEN 24 + floor', 'mfcc_cosine': 'MFCC / cosine',
          'pcen_cosine': 'PCEN / cosine', 'fusion_mfcc50': '50/50 distance fusion',
          'fusion_mfcc75': '75/25 distance fusion'}
COLORS = {'pcen': '#1959a6', 'mfcc': '#d16a13', 'pncc': '#15906e',
          'mfcc_floor': '#a76029', 'pcen_floor': '#8b67ad', 'pcen24_floor': '#808080',
          'mfcc_cosine': '#d16a13', 'pcen_cosine': '#15906e',
          'fusion_mfcc50': '#a4418c', 'fusion_mfcc75': '#7254a3'}


def plot(curves, output, head, k, profiles, suffix, minimum_precision):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), sharey=True)
    for ax, scope, title in zip(axes, ('active', 'all'), ('Current bank', 'Entire labeled bank')):
        for profile in profiles:
            item = curves[(scope, profile, head, k)]
            points = item['frontier']
            ax.plot([p['recall'] for p in points], [p['precision'] for p in points],
                    label=(LABELS[profile].replace(' (baseline)', '') +
                           (' (baseline)' if profile == ('pcen' if head == 'command' else 'mfcc') else '')),
                    color=COLORS[profile], linewidth=2,
                    linestyle='--' if 'cosine' in profile or 'fusion' in profile else '-',
                    marker='.', markersize=3)
            zero = item['summary']['at_wrong_accept_budget']['0']
            if zero['tp']:
                ax.scatter([zero['recall']], [1.], color=COLORS[profile], s=32, zorder=5)
        summary = curves[(scope, profiles[0], head, k)]['summary']
        ax.set_title(f"{title}\n{summary['positives']} positive / {summary['negatives']} negative observations", fontsize=11)
        ax.set(xlim=(0, 1), ylim=(minimum_precision, 1.005), xlabel='Correct-accept recall')
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Precision: correct / all accepts')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .085), ncol=3, frameon=False)
    title = 'Feature representations' if suffix == 'features' else 'Feature and matching alternatives'
    fig.suptitle(f'{head.capitalize()} precision–recall: {title.lower()}', fontsize=17, fontweight='bold', y=.98)
    fig.text(.5, .905, f'Identical random holdouts · DTW · top-k = {k} · ' +
             ('Euclidean frame cost fixed' if suffix == 'features' else 'matching changes shown dashed'), ha='center', color='#444444')
    fig.text(.04, .025, 'Exploratory distance/margin threshold envelope; evaluation labels select the frontier. Dots at 100% = zero observed wrong accepts.\n'
             'Three repeats over the same clips; not independent observations. Connecting lines are guides. Full template search; live shortlist differs.',
             fontsize=8, color='#444444')
    fig.subplots_adjust(top=.79, bottom=.24, wspace=.09, left=.075, right=.98)
    name = f'{head}-pr-{suffix}' + ('-full' if minimum_precision == 0 else '')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'{name}.{ext}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    curves, alignments = {}, {}
    for source in args.input:
        report = json.loads(source.read_text())
        for comparison in report['comparisons']:
            scope, profile = comparison['scope'], comparison['profile']
            for evaluation in comparison['evaluations']:
                head, k = evaluation['head'], evaluation['top_k']
                if k != (2 if head == 'command' else 3):
                    continue
                predictions = evaluation['heldout_predictions']
                identities = sorted((p['repeat'], p['fold'], p['id'], p['target']) for p in predictions)
                alignment_key = (scope, head, k)
                if alignment_key in alignments and identities != alignments[alignment_key]:
                    raise ValueError(f'Mismatched query folds/labels for {scope}/{profile}/{head}')
                alignments[alignment_key] = identities
                points = operating_points(predictions)
                key = (scope, profile, head, k)
                if key in curves:
                    raise ValueError(f'Duplicate comparison: {key}')
                curves[key] = {'points': points, 'frontier': frontier(points),
                               'summary': summarize(predictions, points),
                               'per_repeat': {str(r): summarize(subset, operating_points(subset))
                                              for r in sorted({p['repeat'] for p in predictions})
                                              if (subset := [p for p in predictions if p['repeat'] == r])}}
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = {'method': 'Descriptive Pareto envelope of distance <= d and signed margin >= m, m >= 0',
                 'limitations': ['Thresholds selected using evaluation labels; not independent deployment validation.',
                                 'Three random repeats reuse clips; counts are not independent samples.',
                                 'Full search differs from the live template shortlist.',
                                 'Corpus precision is conditional on sampled class prevalence, not a household false-accept rate.'],
                 'comparisons': [{'scope': key[0], 'profile': key[1], 'head': key[2], 'top_k': key[3],
                                  **{k: v for k, v in value.items() if k != 'points'}} for key, value in curves.items()]}
    (args.output / 'pr-summary.json').write_text(json.dumps(aggregate, indent=2) + '\n')
    with (args.output / 'pr-operating-points.csv').open('w', newline='') as stream:
        fields = ['scope', 'profile', 'head', 'top_k', 'tp', 'fp', 'recall', 'precision', 'distance', 'margin']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, value in curves.items():
            for point in value['points']:
                writer.writerow(dict(zip(fields[:4], key)) | point)
    for head, k in [('command', 2), ('wake', 3)]:
        for suffix, profiles in [('features', ['pcen', 'mfcc', 'pncc', 'mfcc_floor', 'pcen_floor', 'pcen24_floor']),
                                 ('matching', ['pcen', 'mfcc', 'pncc', 'mfcc_cosine', 'pcen_cosine', 'fusion_mfcc75'])]:
            for minimum in (.95, 0):
                plot(curves, args.output, head, k, profiles, suffix, minimum)
    print(f'Wrote {len(curves)} aggregate comparisons to {args.output}')


if __name__ == '__main__':
    main()
