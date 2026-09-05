"""Offline full-bank feature and template-scoring comparison; never invokes actions."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .audio import load_wav, trim_silence, trim_spoken_phrase
from .config import load_config
from .features import _frame_signal, _mel_filter_bank, _dct_basis, _delta, extract_mfcc, extract_command_features
from .matcher import dtw_distance

PROFILES = ('mfcc', 'pcen', 'mfcc_floor', 'pcen_floor', 'pcen24_floor')


def features(samples, profile):
    """Baseline features or experimental variance-floor variants, no fitted data."""
    if profile == 'mfcc':
        return extract_mfcc(samples)
    if profile == 'pcen':
        return extract_command_features(samples)
    is_mfcc = profile.startswith('mfcc')
    count = 24 if profile == 'pcen24_floor' else 13
    x = np.asarray(samples, dtype=np.float32)
    if is_mfcc:
        x = np.concatenate((x[:1], x[1:] - .97*x[:-1]))
    frames = _frame_signal(x, 400, 160)
    frames *= (np.hamming(400) if is_mfcc else np.hanning(400)).astype(np.float32)
    power = (np.abs(np.fft.rfft(frames, n=512, axis=1))**2 / 512).astype(np.float32)
    energy = np.maximum(power @ _mel_filter_bank(16000, 512, 26 if is_mfcc else 40).T, 1e-10)
    if is_mfcc:
        compressed = np.log(energy)
    else:
        smoother = np.empty_like(energy)
        smoother[0] = energy[0]
        for i in range(1, len(energy)):
            smoother[i] = .95*smoother[i-1] + .05*energy[i]
        compressed = (energy / (1e-6+smoother)**.98 + 2)**.5 - 2**.5
    cepstra = compressed @ _dct_basis(energy.shape[1], count).T
    centered = cepstra - cepstra.mean(axis=0, keepdims=True)
    variance = cepstra.var(axis=0, keepdims=True)
    # Retain relative reliability of low-variance coefficients instead of
    # amplifying each to unit variance. Floor is relative to this utterance.
    normalized = centered / np.sqrt(np.maximum(variance, max(float(np.median(variance))*.25, 1e-8)))
    return np.concatenate((normalized, _delta(normalized)), axis=1).astype(np.float32)


def inventory(root, config, archives):
    known = set(config.commands) | {config.start_phrase.name, config.calibration.name, '_not_command', '_not_start_phrase'}
    grouped = defaultdict(list)
    manifest = []
    for path in sorted(root.rglob('*.wav')):
        relative = path.relative_to(root)
        parts = relative.parts
        label = parts[0] if len(parts) == 2 and parts[0] in known else None
        if len(parts) == 3 and parts[0] in archives and parts[1] in known:
            label = parts[1]
        item = {'path': relative.as_posix(), 'label': label, 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest.append(item)
        if label is None:
            item['status'] = 'unlabeled_or_withdrawn_not_scored'
            continue
        audio = load_wav(path, 16000)
        digest = hashlib.sha256(audio.samples.astype('<f4').tobytes()).hexdigest()
        item['audio_sha256'] = digest
        item['status'] = 'labeled'
        grouped[digest].append((item, audio.samples))
    rows = []
    for digest, copies in grouped.items():
        labels = {item['label'] for item, _ in copies}
        if len(labels) != 1:
            for item, _ in copies:
                item['status'] = 'conflicting_labels_not_scored'
            continue
        item, samples = copies[0]
        try:
            day = datetime.strptime(Path(item['path']).stem[:8], '%Y%m%d').date().isoformat()
        except ValueError:
            day = 'unknown'
        clip = trim_spoken_phrase(samples, 16000) if item['label'] == config.start_phrase.name else trim_silence(samples, 16000)
        if len(clip) == 0:
            item['status'] = 'empty_after_fixed_trim'
            continue
        rows.append({'id': digest, 'label': item['label'], 'day': day, 'samples': clip, 'paths': [c[0]['path'] for c in copies]})
        for duplicate, _ in copies[1:]:
            duplicate['status'] = 'duplicate_audio'
    return rows, manifest


def scores(matrix, query, train, rows, head, config, k):
    by_class = defaultdict(list)
    for j in train:
        label = rows[j]['label']
        if head == 'wake':
            label = 'wake' if label == config.start_phrase.name else '_negative'
        elif label not in config.commands and label != '_not_command':
            continue
        by_class[label].append(float(matrix[query, j]))
    means = {label: float(np.mean(sorted(ds)[:k])) for label, ds in by_class.items()}
    if head == 'wake':
        if 'wake' not in means or '_negative' not in means:
            return None
        best, competitor = means['wake'], means['_negative']
        return 'wake', best, (competitor-best)/max(competitor, 1e-9)
    intents = {}
    for label, score in means.items():
        if label in config.commands:
            intent = config.commands[label].intent_group
            intents[intent] = min(intents.get(intent, float('inf')), score)
    if not intents:
        return None
    ranking = sorted(intents.items(), key=lambda x: x[1])
    predicted, best = ranking[0]
    competitors = [score for _, score in ranking[1:]]
    if '_not_command' in means:
        competitors.append(means['_not_command'])
    if not competitors:
        return None
    competitor = min(competitors)
    return predicted, best, (competitor-best)/max(competitor, 1e-9)


def target(row, head, config):
    if head == 'wake':
        return 'wake' if row['label'] == config.start_phrase.name else None
    command = config.commands.get(row['label'])
    return command.intent_group if command else None


def relevant(row, head, config):
    # A not-wake clip is not necessarily a not-command clip.
    return head == 'wake' or row['label'] in set(config.commands) | {'_not_command', config.start_phrase.name, config.calibration.name}


def threshold_fit(predictions):
    """Select on training-only LOO decisions, with zero observed wrong accepts."""
    if not predictions:
        return -1., 1.
    distances = np.unique(np.quantile([p['distance'] for p in predictions], np.linspace(0, 1, 31)))
    margins = (0., .01, .02, .03, .04, .06, .08, .12, .2, .3, .5)
    winner, quality = (-1., 1.), (-1, -float('inf'))
    for distance in distances:
        for margin in margins:
            accepted = [p for p in predictions if p['distance'] <= distance and p['margin'] >= margin]
            wrong = sum(p['target'] is None or p['predicted'] != p['target'] for p in accepted)
            correct = len(accepted)-wrong
            # Conservative tie break prefers larger margin, then lower distance.
            candidate = (correct, margin-distance*1e-6)
            if wrong == 0 and candidate > quality:
                winner, quality = (float(distance), float(margin)), candidate
    return winner


def prediction(matrix, i, train, rows, head, config, k):
    result = scores(matrix, i, train, rows, head, config, k)
    if result is None:
        return None
    predicted, distance, margin = result
    expected = target(rows[i], head, config)
    represented = expected is None or any(target(rows[j], head, config) == expected for j in train)
    return {'id': rows[i]['id'], 'day': rows[i]['day'], 'target': expected, 'predicted': predicted,
            'distance': distance, 'margin': margin, 'represented': represented}


def summarize(predictions):
    valid = [p for p in predictions if p['represented']]
    positive = [p for p in valid if p['target'] is not None]
    negative = [p for p in valid if p['target'] is None]
    correct = sum(p.get('accepted', False) and p['target'] is not None and p['predicted'] == p['target'] for p in valid)
    wrong = sum(p.get('accepted', False) and (p['target'] is None or p['predicted'] != p['target']) for p in valid)
    return {'queries': len(valid), 'positive_queries': len(positive), 'negative_queries': len(negative),
            'unsupported_positive_queries': sum(not p['represented'] for p in predictions),
            'rank_correct_positive': sum(p['predicted'] == p['target'] and p['margin'] > 0 for p in positive),
            'correct_accepts': correct, 'wrong_accepts': wrong,
            'negative_false_accepts': sum(p.get('accepted', False) for p in negative),
            'recall': correct/len(positive) if positive else None,
            'precision': correct/(correct+wrong) if correct+wrong else None}


def evaluate(matrix, rows, config, head, k):
    all_ids = list(range(len(rows)))
    loo = []
    for i, row in enumerate(rows):
        if relevant(row, head, config):
            p = prediction(matrix, i, [j for j in all_ids if j != i], rows, head, config, k)
            if p:
                loo.append(p)
    heldout, folds = [], []
    for day in sorted({row['day'] for row in rows} - {'unknown'}):
        train = [i for i in all_ids if rows[i]['day'] != day and rows[i]['day'] != 'unknown']
        training = []
        for i in train:
            if relevant(rows[i], head, config):
                p = prediction(matrix, i, [j for j in train if j != i], rows, head, config, k)
                if p and p['represented']:
                    training.append(p)
        distance, margin = threshold_fit(training)
        fold = []
        for i in all_ids:
            if rows[i]['day'] == day and relevant(rows[i], head, config):
                p = prediction(matrix, i, train, rows, head, config, k)
                if p:
                    p['accepted'] = p['distance'] <= distance and p['margin'] >= margin
                    fold.append(p)
        heldout.extend(fold)
        folds.append({'day': day, 'max_distance': distance, 'min_margin': margin, **summarize(fold)})
    return {'head': head, 'top_k': k, 'leave_one_audio_out_ranking': summarize(loo),
            'heldout_day': summarize(heldout), 'folds': folds, 'heldout_predictions': heldout}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--archive', action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = load_config(args.config)
    rows, manifest = inventory(args.recordings, config, set(args.archive))
    if len(rows) < 3:
        p.error('Insufficient labeled recordings')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    inventory_summary = {'files': len(manifest), 'statuses': dict(Counter(m['status'] for m in manifest)),
                         'unique_usable_audio': len(rows), 'labels': dict(Counter(r['label'] for r in rows)),
                         'days': dict(Counter(r['day'] for r in rows))}
    print(json.dumps(inventory_summary), flush=True)
    all_results = []
    for profile in PROFILES:
        began = time.perf_counter()
        vectors, extraction_ms = [], []
        for row in rows:
            start = time.perf_counter()
            vectors.append(features(row['samples'], profile))
            extraction_ms.append((time.perf_counter()-start)*1000)
        n = len(rows)
        matrix = np.zeros((n, n), dtype=np.float32)
        # Evaluate ordered pairs: the existing recurrence's tie handling can be
        # asymmetric. Avoid silently assuming a symmetric distance implementation.
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i,j] = dtw_distance(vectors[i], vectors[j])
            if i % 50 == 0:
                print(f'{profile}: distances {i}/{n}', flush=True)
        np.save(args.output/f'{profile}-distances.npy', matrix)
        evaluations = [evaluate(matrix, rows, config, head, k) for head in ('wake','command') for k in (1,2,3)]
        result = {'profile': profile, 'feature_dimensions': vectors[0].shape[1],
                  'extraction_ms_p50': float(np.median(extraction_ms)), 'extraction_ms_p95': float(np.percentile(extraction_ms,95)),
                  'elapsed_seconds': time.perf_counter()-began, 'evaluations': evaluations}
        (args.output/f'{profile}.json').write_text(json.dumps(result, indent=2)+'\n')
        all_results.append(result)
        print(json.dumps({'profile': profile, 'evaluations': [{k:v for k,v in e.items() if k not in ('folds','heldout_predictions')} for e in evaluations]}), flush=True)
    report = {'inventory': inventory_summary, 'profiles': all_results, 'limitations': [
        'Recording day is a conservative session proxy, not verified speaker/session metadata.',
        'Unknown-day audio participates in LOO only. Byte-identical decoded audio is deduplicated; near-duplicates are not automatically detected.',
        'Archived folder labels are assumed valid; conflicts are excluded. Unlabeled diagnostic and withdrawn clips are inventoried but not assigned model-generated truth.',
        'Fixed existing trimming; no VAD or duration-gate changes. Clip recognition does not imply complete-request recall.',
        'All eligible templates are scored: this isolates representation and aggregation and is not the production shortlist baseline.',
        'Thresholds are fitted on training-only LOO, targeting zero training wrong accepts; reported precision/recall uses outer heldout days.',
        'Choosing a profile from these results makes the bank development data; validate the frozen choice on a fresh recording session.',
        'Extraction time is measured on this host; total experiment time is not per-request latency.'
    ]}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')

if __name__ == '__main__':
    main()
