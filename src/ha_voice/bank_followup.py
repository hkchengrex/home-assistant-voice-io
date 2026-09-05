"""Rescore existing full-bank matrices and benchmark matching on the target host."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .bank_experiment import PROFILES, evaluate, features, inventory
from .config import load_config
from .matcher import Template, classify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--archive',action='append',default=[])
    args=p.parse_args()
    config=load_config(args.config)
    rows, manifest=inventory(args.recordings,config,set(args.archive))
    original=json.loads((args.results/'manifest.json').read_text())
    original_ids=[m['audio_sha256'] for m in original if m['status']=='labeled']
    if [r['id'] for r in rows] != original_ids:
        raise ValueError('Bank or ordering changed; cached matrices cannot be reused')
    matrices={profile:np.load(args.results/f'{profile}-distances.npy') for profile in PROFILES}
    active=[i for i,r in enumerate(rows) if any(len(Path(path).parts)==2 for path in r['paths'])]
    active_rows=[rows[i] for i in active]
    comparisons=[]
    combinations={**matrices,
                  'fusion_mfcc50':.5*matrices['mfcc']+.5*matrices['pcen'],
                  'fusion_mfcc75':.75*matrices['mfcc']+.25*matrices['pcen']}
    for name,matrix in combinations.items():
        for scope,indices,subset in [('active',active,active_rows),('all',list(range(len(rows))),rows)]:
            if scope=='all' and name in PROFILES:
                continue
            reduced=matrix[np.ix_(indices,indices)]
            result={'profile':name,'scope':scope,'evaluations':[evaluate(reduced,subset,config,head,k) for head in ('wake','command') for k in (1,2,3)]}
            comparisons.append(result)
            print(f'Rescored {name} {scope}',flush=True)
    (args.results/'followup.json').write_text(json.dumps({'active_audio':len(active),'comparisons':comparisons},indent=2)+'\n')
    timings=[]
    for profile in ('mfcc','pcen','mfcc_floor'):
        vectors=[features(row['samples'],profile) for row in rows]
        for head in ('wake','command'):
            bank=[]
            eligible=[]
            for i,row in enumerate(rows):
                label=row['label']
                if head=='wake':
                    label=config.start_phrase.name if label==config.start_phrase.name else '_not_start_phrase'
                elif label not in config.commands and label!='_not_command':
                    continue
                bank.append((i,Template(label,args.recordings/row['paths'][0],vectors[i])))
                if (head=='wake' and row['label']==config.start_phrase.name) or (head=='command' and row['label'] in config.commands):
                    eligible.append(i)
            eligible.sort(key=lambda i:len(rows[i]['samples']))
            chosen=[eligible[i] for i in np.linspace(0,len(eligible)-1,8).astype(int)]
            for mode in ('full','shortlist'):
                elapsed=[]
                for repeat in range(3):
                    for query in chosen:
                        templates=[t for i,t in bank if i!=query]
                        start=time.perf_counter()
                        query_features=features(rows[query]['samples'],profile)
                        classify(query_features,templates,max_distance=100,min_margin=0,top_k=3 if head=='wake' else 2,
                                 default_template_limit=8 if mode=='shortlist' else None)
                        elapsed.append((time.perf_counter()-start)*1000)
                result={'profile':profile,'head':head,'mode':mode,'queries':8,'repeats':3,
                        'template_count':len(bank)-1,'p50_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95))}
                timings.append(result)
                print(json.dumps(result),flush=True)
    (args.results/'timings.json').write_text(json.dumps({'timings':timings,'limitations':[
        'Warm template features; eight duration-stratified clips repeated three times, not independent accuracy samples.',
        'Measures feature extraction, shortlist when enabled, DTW and class aggregation; excludes trimming, intent regrouping, capture, endpoint wait, actions and playback.',
        'Uses the combined current-plus-archive bank; production has fewer templates. Single BLAS/OMP thread is set by the runner.'
    ]},indent=2)+'\n')

if __name__=='__main__':
    main()
