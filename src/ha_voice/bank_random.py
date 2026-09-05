"""Repeated stratified random holdouts using the existing bank distance caches."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .bank_experiment import PROFILES, inventory, relevant, summarize, target, threshold_fit
from .config import load_config


def random_splits(rows, seeds=(20260905,20260906,20260907), folds=5):
    if folds<2 or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('At least two folds and unique audio identities required')
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        groups[row['label']].append(i)
    splits=[]
    for repeat,seed in enumerate(seeds):
        rng=np.random.default_rng(seed)
        buckets=[[] for _ in range(folds)]
        for label in sorted(groups):
            indices=rng.permutation(groups[label])
            for fold,part in enumerate(np.array_split(indices,folds)):
                buckets[fold].extend(int(i) for i in part)
        for fold,test in enumerate(buckets):
            test=sorted(test)
            held=set(test)
            splits.append({'repeat':repeat,'seed':seed,'fold':fold,
                           'train':[i for i in range(len(rows)) if i not in held],'test':test})
    return splits


def batch_predictions(matrix, queries, train, rows, config, head, k):
    """Vectorize the existing scorer; an infinite diagonal excludes self matches."""
    labels=defaultdict(list)
    for j in train:
        label=rows[j]['label']
        if head=='wake':
            label='wake' if label==config.start_phrase.name else '_negative'
        elif label not in config.commands and label!='_not_command':
            continue
        labels[label].append(j)
    means={}
    for label,indices in labels.items():
        nearest=np.sort(matrix[np.ix_(queries,indices)],axis=1)[:,:k]
        finite=np.isfinite(nearest)
        counts=finite.sum(axis=1)
        means[label]=np.divide(np.where(finite,nearest,0).sum(axis=1),counts,
                              out=np.full(len(queries),np.inf),where=counts>0)
    if head=='wake':
        if 'wake' not in means or '_negative' not in means:
            return []
        intents={'wake':means['wake']}
        competitor=means['_negative']
        best=means['wake']
        predicted=np.full(len(queries),'wake',dtype=object)
    else:
        intents={}
        for label,distances in means.items():
            if label in config.commands:
                intent=config.commands[label].intent_group
                intents[intent]=np.minimum(intents.get(intent,np.inf),distances)
        if not intents:
            return []
        names=list(intents)
        values=np.stack(list(intents.values()),axis=1)
        winners=values.argmin(axis=1)
        best=values[np.arange(len(queries)),winners]
        predicted=np.asarray(names,dtype=object)[winners]
        competitors=values.copy()
        competitors[np.arange(len(queries)),winners]=np.inf
        competitor=competitors.min(axis=1)
        if '_not_command' in means:
            competitor=np.minimum(competitor,means['_not_command'])
    predictions=[]
    for offset,i in enumerate(queries):
        if not relevant(rows[i],head,config) or not np.isfinite(best[offset]) or not np.isfinite(competitor[offset]):
            continue
        expected=target(rows[i],head,config)
        represented=expected is None or (expected in intents and np.isfinite(intents[expected][offset]))
        predictions.append({'id':rows[i]['id'],'target':expected,'predicted':str(predicted[offset]),
                            'distance':float(best[offset]),'margin':float((competitor[offset]-best[offset])/max(competitor[offset],1e-9)),
                            'represented':bool(represented)})
    return predictions


def evaluate_random(matrix,rows,config,head,k,splits):
    # Float64 means preserve the original scorer's Python-float accumulation.
    matrix=matrix.astype(np.float64,copy=True)
    np.fill_diagonal(matrix,np.inf)
    predictions=[]
    folds=[]
    for split in splits:
        train=split['train']
        training=[p for p in batch_predictions(matrix,train,train,rows,config,head,k) if p['represented']]
        distance,margin=threshold_fit(training)
        test=batch_predictions(matrix,split['test'],train,rows,config,head,k)
        for p in test:
            p.update(accepted=p['distance']<=distance and p['margin']>=margin,
                     repeat=split['repeat'],fold=split['fold'])
        predictions.extend(test)
        folds.append({key:split[key] for key in ('repeat','seed','fold')} | {'max_distance':distance,'min_margin':margin,**summarize(test)})
    repeats=[{'repeat':repeat,**summarize([p for p in predictions if p['repeat']==repeat])}
             for repeat in sorted({s['repeat'] for s in splits})]
    return {'head':head,'top_k':k,'random_holdout':summarize(predictions),'repeats':repeats,
            'folds':folds,'heldout_predictions':predictions}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--archive',action='append',default=[])
    args=p.parse_args()
    config=load_config(args.config)
    rows,_=inventory(args.recordings,config,set(args.archive))
    manifest=json.loads((args.results/'manifest.json').read_text())
    if [r['id'] for r in rows] != [m['audio_sha256'] for m in manifest if m['status']=='labeled']:
        raise ValueError('Bank changed; cached distances cannot be reused')
    matrices={profile:np.load(args.results/f'{profile}-distances.npy') for profile in (*PROFILES,'mfcc_cosine','pcen_cosine')}
    matrices['fusion_mfcc50']=.5*matrices['mfcc']+.5*matrices['pcen']
    matrices['fusion_mfcc75']=.75*matrices['mfcc']+.25*matrices['pcen']
    active=[i for i,r in enumerate(rows) if any(len(Path(path).parts)==2 for path in r['paths'])]
    comparisons=[]
    split_manifest={}
    for scope,indices in [('all',list(range(len(rows)))),('active',active)]:
        subset=[rows[i] for i in indices]
        splits=random_splits(subset)
        split_manifest[scope]=[{key:value for key,value in s.items() if key not in ('train','test')} |
                               {'train_ids':[subset[i]['id'] for i in s['train']], 'test_ids':[subset[i]['id'] for i in s['test']]} for s in splits]
        for profile,matrix in matrices.items():
            reduced=matrix[np.ix_(indices,indices)]
            evaluations=[evaluate_random(reduced,subset,config,head,k,splits) for head in ('wake','command') for k in (1,2,3)]
            comparisons.append({'scope':scope,'profile':profile,'evaluations':evaluations})
            print(json.dumps({'scope':scope,'profile':profile,'evaluations':[{'head':e['head'],'k':e['top_k'],**e['random_holdout']} for e in evaluations]}),flush=True)
    (args.results/'random-splits.json').write_text(json.dumps(split_manifest,indent=2)+'\n')
    report={'method':'Three repeats of stratified random five-fold holdout; dates are not used.',
            'seeds':[20260905,20260906,20260907],'folds_per_repeat':5,
            'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'comparisons':comparisons,'limitations':[
                'Each recording is tested once per repeat. Aggregate counts are repeated test observations, not distinct recordings.',
                'All variants use identical splits within each corpus scope. Duplicate decoded audio is deduplicated before splitting.',
                'Thresholds use training-only leave-one-out decisions with the same fitting rule as the earlier experiment; outer test labels never tune thresholds.',
                'Dates do not represent environments and are not used for these splits.',
                'Uncertain archive labels remain uncertain. Current-only comparisons remain available.',
                'This reruns accuracy scoring from unchanged distance matrices; hardware latency measurements are unchanged.'
            ]}
    (args.results/'random-holdout.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
