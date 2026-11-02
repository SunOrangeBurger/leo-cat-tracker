"""Replay frozen temporal checkpoints with scale-independent strict cutoffs.

The original +1e-6 threshold increment is retained in historical runs. This
audit uses the next representable score above the validation quantile, avoiding
an arbitrary absolute probability floor for JPDA. Every operating point gets
its own complete online test replay, including state and initiation feedback.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run_temporal_probe as probe
from temporal_association import HistoryGraph
from kepler_dynamics import propagate_kepler


def strict_threshold(prob, truth, catalog_ids, target=.02):
    novel=~np.isin(truth,catalog_ids)
    if not novel.any(): return float('inf')
    values=np.asarray(prob)
    cutoff=np.quantile(values[novel].max(1),1-target,method='higher')
    return float(np.nextafter(np.asarray(cutoff,dtype=values.dtype),
                              np.asarray(np.inf,dtype=values.dtype)))


def evaluate(root, seed, output, targets=(.01,.02,.05), risk_extra=False, frozen=False,normalized=False):
    library_limits=threadpool_limits(limits=2)
    torch.set_num_threads(2)
    probe.threshold_for_novel_fpr=strict_threshold
    source=root/'streams'/f'seed_{seed}'
    original=json.loads((root/f'seed_{seed}.json').read_text())
    predictor=propagate_kepler if original.get('propagator','circular')=='kepler' else probe.propagate_circular
    update=original['update'] and not frozen
    births=original['births'] and not frozen
    initial_data=np.load(root/f'seed_{seed}_initial.npz')
    initial,ids,scale=[initial_data[k] for k in ['catalog','catalog_ids','scale']]
    df=probe.load_data(source/'orbital_elements.csv',source/'observations.jsonl.truth.jsonl')
    raw=df[['x_km','y_km','z_km','vx_km_s','vy_km_s','vz_km_s']].to_numpy(float)
    times,sensors,truth=df.t_s.to_numpy(),df.sensor_id.to_numpy(),df.truth_id.to_numpy()
    val=(times>=1200)&(times<1600)
    test=times>=1600
    models={}
    for name in ['history_gnn','history_no_messages','current_frame_graph']:
        checkpoint=torch.load(root/f'seed_{seed}_{name}.pt',weights_only=True)
        model=HistoryGraph().eval()
        model.load_state_dict(checkpoint['state_dict'])
        model.use_history=checkpoint['use_history']
        model.use_messages=checkpoint['use_messages']
        models[name]=model
    for name in ['history_tabular','physics_tabular']:
        models[name]=joblib.load(root/f'seed_{seed}_{name}.joblib')
    models['jpda']=None
    _,_,_,validation=probe.replay(raw[val],times[val],sensors[val],initial,scale,
        'physics_tabular',models['physics_tabular'],.5,propagator=predictor,update=update)
    config=probe.tune_history_jpda(validation[0],validation[1],validation[2],
                                  times[val],sensors[val],truth[val],ids,normalized=normalized)
    models['history_jpda']=config
    if risk_extra:
        models={k:v for k,v in models.items() if k in ['history_gnn','history_jpda']}
    primary,curves={},{}
    for name,model in models.items():
        _,_,_,validation=probe.replay(raw[val],times[val],sensors[val],initial,scale,
                name,model,.5,propagator=predictor,candidate_gate_config=config,update=update)
        points={}
        replay_cache={}
        for target in (targets if name in ['history_gnn','history_jpda'] else [.02]):
            threshold=strict_threshold(validation[-1],truth[val],ids,target)
            if threshold in replay_cache:
                result=dict(replay_cache[threshold],validation_target=target)
                points[str(target)]=result
                if target==.02:primary[name]=result
                continue
            birth_log={}
            decisions,pools,fallback,arrays=probe.replay(raw[test],times[test],sensors[test],
                initial,scale,name,model,threshold,propagator=predictor,
                candidate_gate_config=config,births=births,birth_log=birth_log,update=update)
            result=probe.score_online(decisions,pools,truth[test],ids,{},fallback)
            correct,wrong=0,0
            novel_correct,novel_wrong=0,0
            impure_confirmations=0
            for track,birth in birth_log.items():
                labels=truth[test][birth['rows']]
                values,counts=np.unique(labels,return_counts=True)
                impure_confirmations+=int(len(values)>1)
                anchor=values[counts.argmax()]
                for row,prediction in decisions.items():
                    if prediction==track and times[test][row]>birth['confirmed_at']:
                        is_novel=anchor not in ids
                        if truth[test][row]==anchor:
                            correct+=1
                            if is_novel:novel_correct+=1
                        else:
                            wrong+=1
                            if is_novel:novel_wrong+=1
            result.update(threshold=threshold,validation_target=target,
                confirmed_track_future_correct=correct,confirmed_track_future_wrong=wrong,
                confirmed_track_future_assignments=correct+wrong,
                novel_track_future_correct=novel_correct,novel_track_future_wrong=novel_wrong,
                birth_confirmation_impure_tracks=impure_confirmations)
            points[str(target)]=result
            replay_cache[threshold]=result
            if target==.02:primary[name]=result
        curves[name]=points
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(seed=seed,calibration='next representable score above validation quantile',
        checkpoint_source=str(root),births=births,update=update,methods=primary,
        operating_curves=curves,jpda_config=config),indent=2))
    print('audited',seed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--primary-only',action='store_true')
    p.add_argument('--risk-extra',action='store_true')
    p.add_argument('--frozen',action='store_true')
    p.add_argument('--normalized',action='store_true')
    a=p.parse_args()
    evaluate(a.study,a.seed,a.output,(.02,.1,.2) if a.risk_extra else ((.02,) if a.primary_only else (.01,.02,.05)),a.risk_extra,a.frozen,a.normalized)
