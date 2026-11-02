"""Merge frozen-checkpoint audits and expose paired operating-point diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
from run_temporal_study import summarize

BASE=Path(__file__).resolve().parents[1]/'paper'/'data'/'temporal_study'


def interval(values):
    x=np.asarray(values,float)
    if not len(x):return None
    rng=np.random.default_rng(20261102)
    b=rng.choice(x,size=(10000,len(x))).mean(1)
    return [float(x.mean()),*np.quantile(b,[.025,.975]).tolist()]


def merge(name,start):
    target=BASE/(name+'_canonical')
    target.mkdir(parents=True,exist_ok=True)
    for seed in range(start,start+10):
        data=json.loads((BASE/(name+'_final')/f'seed_{seed}.json').read_text())
        extra=json.loads((BASE/(name+'_risk')/f'seed_{seed}.json').read_text())
        if data['jpda_config']!=extra['jpda_config']:raise ValueError('validation configuration changed')
        for method in ['history_gnn','history_jpda']:
            for metric in ['known_correct','known_rejected','novel_to_seeded_false_merge','born_track_count']:
                if data['methods'][method][metric]!=extra['methods'][method][metric]:
                    raise ValueError((name,seed,method,metric,'audit replay changed'))
            data['methods'][method]=extra['methods'][method]
            data['operating_curves'][method].update(extra['operating_curves'][method])
        (target/f'seed_{seed}.json').write_text(json.dumps(data,indent=2))
    summarize(target,list(range(start,start+10)))
    return target


def interpolate(points,target):
    by_rate={}
    for point in points.values():
        rate=point['novel_seeded_false_merge_rate']
        by_rate[rate]=max(by_rate.get(rate,-1),point['known_correct_per_detection'])
    x=np.array(sorted(by_rate))
    if len(x)<2 or target<x.min() or target>x.max():return None
    return float(np.interp(target,x,[by_rate[k] for k in x]))


def diagnostics(path):
    data=json.loads((path/'summary.json').read_text())
    runs=data['runs']
    results={}
    for method in ['history_gnn','history_jpda']:
        rows=[r['methods'][method] for r in runs]
        results[method]=dict(total_novel_future_correct=sum(x['novel_track_future_correct'] for x in rows),
            total_novel_future_wrong=sum(x['novel_track_future_wrong'] for x in rows),
            mixed_identity_confirmations=sum(x['birth_confirmation_impure_tracks'] for x in rows),
            total_confirmed_tracks=sum(x['born_track_count'] for x in rows),
            seed_mean_track_purity=interval([x['born_track_mean_purity'] for x in rows
                                           if x['born_track_mean_purity'] is not None]))
    results['graph_ablations']={}
    for method in ['history_no_messages','current_frame_graph','history_tabular']:
        results['graph_ablations'][method]=interval([
            r['methods']['history_gnn']['known_correct_per_detection']-
            r['methods'][method]['known_correct_per_detection'] for r in runs])
    results['retrospective_matched_test_risk']={}
    for risk in [.02,.05]:
        differences=[]
        seeds=[]
        for r in runs:
            a=interpolate(r['operating_curves']['history_gnn'],risk)
            b=interpolate(r['operating_curves']['history_jpda'],risk)
            if a is not None and b is not None:
                seeds.append(r['seed']);differences.append(a-b)
        results['retrospective_matched_test_risk'][str(risk)]=dict(
            bracketed_seeds=seeds,differences=differences,interval=interval(differences),
            note='retrospective trajectory interpolation, no extrapolation or deployed test-label tuning')
    (path/'diagnostics.json').write_text(json.dumps(results,indent=2))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=BASE)
    p.add_argument('--cohort',choices=['ordinary','crowded','crowded_growth_training','crowded_gaussian','both'],default='both')
    a=p.parse_args()
    BASE=a.root
    for name,start in [('ordinary',201),('crowded',211),('crowded_growth_training',211),('crowded_gaussian',211)]:
        if name=='crowded_gaussian' and a.cohort=='both':continue
        if name=='crowded_growth_training' and a.cohort=='both' and not (BASE/(name+'_final')/'seed_220.json').exists():continue
        if a.cohort not in ['both',name]:continue
        target=merge(name,start)
        d=diagnostics(target)
        print(name,json.dumps(d,indent=2))
