"""Offline cosine-cost DTW experiment using the existing native recurrence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .bank_experiment import evaluate, features, inventory
from .config import load_config
from .matcher import _native_accumulate_distance


def cosine_dtw(first, second, band_ratio=.25):
    if first.ndim!=2 or second.ndim!=2 or first.shape[1]!=second.shape[1] or min(len(first),len(second))==0:
        raise ValueError('Non-empty feature matrices with equal widths required')
    if _native_accumulate_distance is None:
        raise RuntimeError('This target-hardware experiment requires native DTW')
    a=first / np.maximum(np.linalg.norm(first,axis=1,keepdims=True),1e-8)
    b=second / np.maximum(np.linalg.norm(second,axis=1,keepdims=True),1e-8)
    distances=np.ascontiguousarray(1-np.clip(a @ b.T,-1,1),dtype=np.float32)
    band=max(abs(len(first)-len(second)),int(max(len(first),len(second))*band_ratio),2)
    return float(_native_accumulate_distance(distances,band))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--archive',action='append',default=[])
    args=p.parse_args()
    config=load_config(args.config)
    rows,_=inventory(args.recordings,config,set(args.archive))
    original=json.loads((args.results/'manifest.json').read_text())
    if [r['id'] for r in rows] != [m['audio_sha256'] for m in original if m['status']=='labeled']:
        raise ValueError('Bank changed')
    active=[i for i,r in enumerate(rows) if any(len(Path(path).parts)==2 for path in r['paths'])]
    reports=[]
    timings=[]
    for profile in ('mfcc','pcen'):
        vectors=[features(row['samples'],profile) for row in rows]
        n=len(rows)
        matrix=np.zeros((n,n),dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i!=j:
                    matrix[i,j]=cosine_dtw(vectors[i],vectors[j])
        np.save(args.results/f'{profile}_cosine-distances.npy',matrix)
        for scope,indices in [('all',list(range(n))),('active',active)]:
            subset=[rows[i] for i in indices]
            reduced=matrix[np.ix_(indices,indices)]
            reports.append({'profile':profile+'_cosine','scope':scope,
                            'evaluations':[evaluate(reduced,subset,config,head,k) for head in ('wake','command') for k in (1,2,3)]})
            print(f'Rescored {profile} cosine {scope}',flush=True)
        for head in ('wake','command'):
            eligible=[i for i,r in enumerate(rows) if (r['label']==config.start_phrase.name if head=='wake' else r['label'] in config.commands)]
            eligible.sort(key=lambda i:len(rows[i]['samples']))
            chosen=[eligible[i] for i in np.linspace(0,len(eligible)-1,8).astype(int)]
            elapsed=[]
            for repeat in range(3):
                for query in chosen:
                    start=time.perf_counter()
                    query_features=features(rows[query]['samples'],profile)
                    distances=[]
                    for i,row in enumerate(rows):
                        if i!=query and (head=='wake' or row['label'] in config.commands or row['label']=='_not_command'):
                            distances.append(cosine_dtw(query_features,vectors[i]))
                    # Include class scoring as in matrix-based evaluation.
                    from .bank_experiment import scores
                    scores(matrix,query,[i for i in range(n) if i!=query],rows,head,config,3 if head=='wake' else 2)
                    elapsed.append((time.perf_counter()-start)*1000)
            timing={'profile':profile+'_cosine','head':head,'mode':'full','queries':8,'repeats':3,
                    'p50_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95))}
            timings.append(timing)
            print(json.dumps(timing),flush=True)
    (args.results/'cosine.json').write_text(json.dumps({'comparisons':reports,'timings':timings,
        'limitations':['Same fixed bank, day folds and calibration as the first experiment.',
                       'Cosine costs replace Euclidean costs; the band, recurrence and path normalization are unchanged.',
                       'A zero feature frame has cosine distance one even to another zero frame; this is not a silence detector.',
                       'Full-search timing includes extraction and class scoring. No shortlist, capture, actions or playback.']},indent=2)+'\n')

if __name__=='__main__':
    main()
