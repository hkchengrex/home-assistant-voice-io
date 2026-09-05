"""Offline PNCC representation comparison; existing DTW and random splits stay fixed."""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .bank_experiment import inventory
from .bank_random import evaluate_random, random_splits
from .config import load_config
from .features import _delta, _frame_signal, extract_mfcc, extract_command_features
from .matcher import dtw_distance


@lru_cache(maxsize=1)
def auditory_filters():
    from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks
    return gammatone_filter_banks(nfilts=40,nfft=512,fs=16000,low_freq=0,high_freq=8000)[0]


def extract_pncc(samples):
    from scipy.fft import dct
    from spafe.features.pncc import (medium_time_power_calculation,
        asymmetric_noise_suppression_with_temporal_masking,weight_smoothing,mean_power_normalization)
    x=np.asarray(samples,dtype=np.float64)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all():
        raise ValueError('Finite non-empty mono audio required')
    emphasized=np.concatenate((x[:1],x[1:]-.97*x[:-1]))
    frames=_frame_signal(emphasized,400,160)*np.hamming(400)
    power=np.abs(np.fft.rfft(frames,n=512,axis=1))**2/512
    auditory=power @ auditory_filters().T
    medium=medium_time_power_calculation(auditory)
    suppressed=asymmetric_noise_suppression_with_temporal_masking(medium)
    # SPAFE's weight ratio divides by zero on silent channels. Stabilize the
    # denominator before division, rather than hiding NaNs in output features.
    floor=max(float(medium.max())*1e-12,1e-30)
    weights=weight_smoothing(suppressed,np.maximum(medium,floor),nfilts=40)
    normalized_power=mean_power_normalization(auditory*weights,nfilts=40)
    raw=dct(np.maximum(normalized_power,0)**(1/15),type=2,axis=1,norm='ortho')[:,:13]
    # Match the existing per-coefficient utterance normalization and delta rule.
    normalized=(raw-raw.mean(axis=0,keepdims=True))/np.maximum(raw.std(axis=0,keepdims=True),1e-4)
    result=np.concatenate((normalized,_delta(normalized)),axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite PNCC features; do not silently score the clip')
    return result


def benchmark_extraction(rows, repeats=3):
    """Warm single-thread extraction on the same duration-stratified clips."""
    ordered=sorted(rows,key=lambda row:len(row['samples']))
    selected=[ordered[i] for i in np.unique(np.linspace(0,len(rows)-1,8).astype(int))]
    report={}
    for name,extractor in [('mfcc',extract_mfcc),('pcen',extract_command_features),('pncc',extract_pncc)]:
        extractor(selected[0]['samples'])
        elapsed=[]
        for _ in range(repeats):
            for row in selected:
                start=time.perf_counter()
                extractor(row['samples'])
                elapsed.append((time.perf_counter()-start)*1000)
        report[name]={'p50_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),
                      'measurements':len(elapsed)}
    return {'method':'Warm extraction, 8 duration-stratified clips, 3 repeats, same clips per frontend',
            'duration_seconds':[len(r['samples'])/16000 for r in selected], 'frontends':report,
            'limitations':['Excludes matching, audio buffering, VAD and endpoint delay.',
                           'Small microbenchmark on the evaluation host; not end-to-end service latency.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--archive',action='append',default=[])
    p.add_argument('--timing-only',action='store_true')
    args=p.parse_args()
    config=load_config(args.config)
    rows,_=inventory(args.recordings,config,set(args.archive))
    manifest=json.loads((args.results/'manifest.json').read_text())
    if [r['id'] for r in rows] != [m['audio_sha256'] for m in manifest if m['status']=='labeled']:
        raise ValueError('Bank changed; original random splits cannot be reused')
    if args.timing_only:
        (args.results/'feature-extraction-timing.json').write_text(json.dumps(benchmark_extraction(rows),indent=2)+'\n')
        return
    vectors=[]
    extraction=[]
    for row in rows:
        start=time.perf_counter()
        vectors.append(extract_pncc(row['samples']))
        extraction.append((time.perf_counter()-start)*1000)
    n=len(rows)
    matrix=np.zeros((n,n),dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i!=j:
                matrix[i,j]=dtw_distance(vectors[i],vectors[j])
        if i%50==0:
            print(f'PNCC distances {i}/{n}',flush=True)
    np.save(args.results/'pncc-distances.npy',matrix)
    active=[i for i,r in enumerate(rows) if any(len(Path(path).parts)==2 for path in r['paths'])]
    comparisons=[]
    for scope,indices in [('all',list(range(n))),('active',active)]:
        subset=[rows[i] for i in indices]
        reduced=matrix[np.ix_(indices,indices)]
        splits=random_splits(subset)
        comparisons.append({'scope':scope,'profile':'pncc','evaluations':[
            evaluate_random(reduced,subset,config,head,k,splits) for head in ('wake','command') for k in (1,2,3)]})
        print(f'PNCC evaluated {scope}',flush=True)
    import scipy,spafe
    report={'profile':'pncc','comparisons':comparisons,
            'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'ordered_audio_ids_sha256':hashlib.sha256('\n'.join(r['id'] for r in rows).encode()).hexdigest(),
            'extraction_ms_p50':float(np.median(extraction)), 'extraction_ms_p95':float(np.percentile(extraction,95)),
            'scipy':scipy.__version__,'spafe':'0.3.3','numpy':np.__version__,
            'settings':{'sample_rate':16000,'filters':40,'coefficients':13,'delta_coefficients':13,'frame_ms':25,'hop_ms':10,'n_fft':512,
                        'normalization':'per-coefficient utterance CMVN, same delta rule as existing frontends'},
            'limitations':['PNCC using SPAFE components with a relative denominator floor and common baseline framing; not a claim of reference-implementation parity.',
                           'Fixed trimming and Euclidean DTW; identical random splits and top-k comparisons.',
                           'Feature extraction timing excludes matching and endpoint delay.']}
    (args.results/'pncc.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
