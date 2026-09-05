"""Reproducible PCEN sweep with fixed Euclidean DTW and nested parameter selection."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .bank_experiment import inventory, summarize
from .bank_pr import operating_points, summarize as pr_summary
from .bank_random import evaluate_random, random_splits
from .config import load_config
from .features import _delta, extract_command_features, extract_pcen_cepstra
from .matcher import dtw_distance, _native_accumulate_distance


BASELINE = dict(smoothing=.05, alpha=.98, root=.5, delta=2., epsilon=1e-6)


def candidates():
    """Predeclared grid: adaptation interaction, then one-factor compression trials."""
    result = [('baseline', BASELINE.copy())]
    for smoothing in (.01, .025, .05, .1):
        for alpha in (.8, .9, .95, .98):
            params = BASELINE | dict(smoothing=smoothing, alpha=alpha)
            if params != BASELINE:
                result.append((f's{smoothing}_a{alpha}', params))
    for key, values in [('root', (.25, .75)), ('delta', (.5, 1., 4.)),
                        ('epsilon', (1e-8, 1e-4))]:
        for value in values:
            result.append((f'{key}{value}', BASELINE | {key: value}))
    return result


def extract(samples, params):
    cepstra = extract_pcen_cepstra(samples, **params)
    return np.concatenate((cepstra, _delta(cepstra)), axis=1).astype(np.float32)


def compute_matrix(job):
    name, params, samples, cache = job
    path = Path(cache) / f'{name}.npy'
    if path.exists():
        return name
    if _native_accumulate_distance is None:
        raise RuntimeError('Build the native matcher before running this sweep')
    started = time.perf_counter()
    vectors = [extract(x, params) for x in samples]
    if name == 'baseline':
        for x, vector in zip(samples, vectors):
            np.testing.assert_array_equal(vector, extract_command_features(x))
    matrix = np.zeros((len(vectors), len(vectors)), dtype=np.float32)
    # Ordered pairs preserve the production recurrence's directional tie handling.
    for i, first in enumerate(vectors):
        for j, second in enumerate(vectors):
            if i != j:
                matrix[i, j] = dtw_distance(first, second)
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite distance')
    temporary = path.with_suffix('.tmp.npy')
    np.save(temporary, matrix)
    temporary.replace(path)
    print(f'{name}: matrix complete in {time.perf_counter()-started:.1f}s', flush=True)
    return name


def selection_key(summary):
    """False accepts have priority; then maximize correct accepts. Baseline wins ties."""
    return -summary['wrong_accepts'], summary['correct_accepts']


def assess_scope(scope, indices, rows, config, cache, output, top_k):
    subset = [rows[i] for i in indices]
    splits = random_splits(subset)
    comparisons = []
    inner_scores = []
    outer_predictions = []
    split_ids = [dict(repeat=s['repeat'], seed=s['seed'], fold=s['fold'],
                      train_ids=[subset[i]['id'] for i in s['train']],
                      test_ids=[subset[i]['id'] for i in s['test']]) for s in splits]
    for name, params in candidates():
        matrix = np.load(Path(cache) / f'{name}.npy')[np.ix_(indices, indices)]
        outer = evaluate_random(matrix, subset, config, 'command', top_k, splits)
        descriptive = pr_summary(outer['heldout_predictions'], operating_points(outer['heldout_predictions']))
        inner = []
        for split in splits:
            train = split['train']
            training_rows = [subset[i] for i in train]
            inner_splits = random_splits(training_rows, seeds=(9173,), folds=3)
            result = evaluate_random(matrix[np.ix_(train, train)], training_rows,
                                     config, 'command', top_k, inner_splits)
            inner.append(result['random_holdout'])
        inner_scores.append(inner)
        outer_predictions.append(outer['heldout_predictions'])
        comparisons.append(dict(name=name, parameters=params,
                                calibrated=outer['random_holdout'], folds=outer['folds'],
                                repeats=outer['repeats'], exploratory=descriptive,
                                inner_validation=inner))
        print(f'{scope} {name}: {json.dumps(outer["random_holdout"])}', flush=True)
    selected_predictions, selections = [], []
    for f, split in enumerate(splits):
        winner = max(range(len(comparisons)), key=lambda c: selection_key(inner_scores[c][f]))
        predictions = [p for p in outer_predictions[winner]
                       if p['repeat'] == split['repeat'] and p['fold'] == split['fold']]
        selected_predictions.extend(predictions)
        selections.append(dict(repeat=split['repeat'], fold=split['fold'],
                               candidate=comparisons[winner]['name'],
                               inner=inner_scores[winner][f], outer=summarize(predictions)))
    report = dict(scope=scope, unique_audio=len(subset), splits=split_ids,
                  comparisons=comparisons,
                  nested_selection=dict(summary=summarize(selected_predictions), folds=selections))
    (output / f'{scope}.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--archive', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=2)
    args = parser.parse_args()
    config = load_config(args.config)
    rows, manifest = inventory(args.recordings, config, set(args.archive))
    if len(rows) < 3:
        parser.error('Insufficient audio')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    modules = ('bank_pcen.py', 'features.py', 'matcher.py', '_dtw_native.c',
               'bank_experiment.py', 'bank_random.py', 'bank_pr.py', 'audio.py', 'config.py')
    provenance = dict(config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                      modules={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in modules}, numpy=np.__version__,
                      candidates=candidates(), top_k=args.top_k, archives=args.archive,
                      audio=[dict(id=r['id'], label=r['label'], paths=r['paths'],
                                  trimmed_sha256=hashlib.sha256(r['samples'].tobytes()).hexdigest()) for r in rows])
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = output / 'cache' / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    (output / '.gitignore').write_text('cache/\n')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'inventory.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(unique_audio=len(rows), labels=dict(Counter(r['label'] for r in rows)),
                          candidates=len(candidates()), cache=str(cache))), flush=True)
    jobs = [(name, params, [r['samples'] for r in rows], str(cache)) for name, params in candidates()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(compute_matrix, jobs))
    active = [i for i, r in enumerate(rows) if any(len(Path(p).parts) == 2 for p in r['paths'])]
    for scope, indices in [('active', active), ('all', list(range(len(rows))))]:
        assess_scope(scope, indices, rows, config, cache, output, args.top_k)


if __name__ == '__main__':
    main()
