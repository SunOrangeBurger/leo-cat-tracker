"""Development probe for causal history GNN versus same-history baselines.

Candidate histories for training are harvested from a fixed teacher policy;
test histories evolve exclusively from each method's own accepted decisions.
No outcome of this development probe is a confirmatory paper result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import joblib
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier

from online_catalog_benchmark import prepare_models, physics_tabular_inputs, score_online
from rust_noise_benchmark import load_data, threshold_for_novel_fpr
from rust_noise_orbit_benchmark import retrieve, propagate_circular, jpda_probabilities
from temporal_association import TrackHistory, train_graph, probabilities, summary_features
from tracklet_initiation import TrackletInitiator


def history_jpda_geometry(history, local, config):
    mask = history[..., 7:8]
    count = np.maximum(mask.sum(2), 1.)
    mean = (history[..., :6]*mask).sum(2)/count
    variance = (((history[..., :6]-mean[:,:,None])**2)*mask).sum(2)/count
    variance[..., [1,4]] *= config['along']**2
    inflation = np.ones(6)
    inflation[[1,4]] = config['along']**2
    covariance=inflation+variance
    return local[..., :6]/np.sqrt(covariance),np.exp(-.5*np.log(covariance).sum(-1))


def history_jpda_inputs(history, local, config):
    return history_jpda_geometry(history,local,config)[0]


def tune_history_jpda(h, local, positions, times, sensors, truth, ids, normalized=False):
    groups = {key: np.flatnonzero((times == key[0]) & (sensors == key[1])).tolist()
              for key in set(zip(times, sensors))}
    best = None
    for along in [1.,2.,4.]:
        for gate in [16.,25.,49.]:
            for clutter in ([2e-8,2e-6,2e-4,.02] if normalized else [.002,.02,.2]):
                config = dict(along=along,gate=gate,clutter=clutter)
                if normalized:config['normalized']=True
                candidate,normalizer = history_jpda_geometry(h, local, config)
                prob,_ = jpda_probabilities(np.zeros((len(h),6)),candidate,positions,groups,
                    gate_sq=gate,clutter_weight=clutter,max_exact_detections=12,
                    max_exact_tracks=32,fallback_policy='map',
                    likelihood_scale=normalizer if normalized else None)
                threshold = threshold_for_novel_fpr(prob,truth,ids)
                accepted = prob.max(1) >= threshold
                correct = ids[positions[np.arange(len(h)),prob.argmax(1)]] == truth
                known = np.isin(truth,ids)
                recall = float((accepted & correct & known).sum()/max(known.sum(),1))
                if best is None or recall > best[0]:
                    best = (recall,config)
    return best[1]


def replay(raw, times, sensors, initial, scale, method, model, threshold,
           collect=False, update=True, propagator=propagate_circular, births=False,
           candidate_gate_config=None, birth_log=None):
    if method not in {'history_gnn','history_no_messages','current_frame_graph',
                      'history_tabular','physics_tabular','jpda','history_jpda'}:
        raise ValueError('unknown association method: '+str(method))
    history = TrackHistory(initial, propagator=propagator)
    initiator = TrackletInitiator(scale,propagator)
    catalog = initial.copy()
    decisions, pools = {}, {}
    gathered = []
    fallback = 0
    for epoch in np.unique(times):
        rows = np.flatnonzero(times == epoch)
        q, candidates, positions = retrieve(raw[rows], times[rows], catalog, scale,
                                           propagator=propagator)
        h, predicted = history.features(raw[rows], epoch, positions, scale)
        local = physics_tabular_inputs(q, candidates, scale).reshape(len(rows), 5, -1)
        if method in ('history_gnn','history_no_messages','current_frame_graph'):
            prob = probabilities(model, h, local)
        elif method == 'history_tabular':
            features = summary_features(h, local)
            prob = model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1].reshape(-1, 5)
        elif method == 'physics_tabular':
            prob = model.predict_proba(local.reshape(-1, local.shape[-1]))[:, 1].reshape(-1, 5)
        else:
            groups = {int(s): np.flatnonzero(sensors[rows] == s).tolist()
                      for s in np.unique(sensors[rows])}
            jq, jc = q, candidates
            likelihood_scale=None
            if method == 'history_jpda':
                # Candidate-specific diagonal uncertainty in the RTN frame.
                # Scatter is computed from assigned public history, never truth.
                jc,normalizer = history_jpda_geometry(h, local, model)
                if model.get('normalized',False):likelihood_scale=normalizer
                jq = np.zeros_like(q)
            prob, degraded = jpda_probabilities(jq, jc, positions, groups,
                    gate_sq=model['gate'] if method == 'history_jpda' else 25.,
                    clutter_weight=model['clutter'] if method == 'history_jpda' else .02,
                    max_exact_detections=12, max_exact_tracks=32, fallback_policy='map',
                    likelihood_scale=likelihood_scale)
            fallback += degraded
        if candidate_gate_config is not None:
            gate_residual = history_jpda_inputs(h,local,candidate_gate_config)
            gate = np.sum(gate_residual**2,-1) <= candidate_gate_config['gate']
        else:
            gate = np.sum(((jc-jq[:, None]) if method == 'history_jpda'
                           else (candidates-q[:, None]))**2, -1) <= (model['gate'] if method == 'history_jpda' else 25.)
        prob = np.where(gate, prob, 0.)
        best = prob.argmax(1)
        assignment = np.where(prob.max(1) >= threshold, positions[np.arange(len(rows)), best], -1)
        gathered.append((rows, h, local, positions, prob))
        for i, row in enumerate(rows):
            decisions[int(row)] = int(assignment[i])
            pools[int(row)] = positions[i]
        history.add_scan(assignment, raw[rows], epoch)
        if births:
            # Defer ambiguous near-catalog rejects; only isolated rejects can
            # seed a provisional tracklet. Every scorer uses this same rule.
            isolated = (assignment < 0) & (np.sum((candidates[:,0]-q)**2,1) >= 9.)
            confirmed = initiator.process_scan(rows[isolated],raw[rows][isolated],epoch)
            for chain in confirmed:
                track = len(catalog)
                states = np.asarray(chain['states'])
                ts = np.asarray(chain['times'])
                center = np.median(propagator(states,-ts),axis=0)
                catalog = np.vstack([catalog,center])
                if birth_log is not None:
                    birth_log[track] = dict(rows=list(chain['rows']),confirmed_at=float(epoch))
                for row,state,t in zip(chain['rows'],states,ts):
                    history.rows[track].append((float(t),state.copy()))
                    decisions[int(row)] = track
        if update:
            for track in set(assignment[assignment >= 0]):
                back = propagator(raw[rows][assignment == track],
                                          -times[rows][assignment == track])
                catalog[track] = .9*catalog[track] + .1*np.median(back, axis=0)
    # Epoch grouping preserves chronological order; explicitly restore row order.
    order = np.concatenate([x[0] for x in gathered]).argsort()
    arrays = [np.concatenate([x[j] for x in gathered])[order] for j in range(1, 5)]
    return decisions, pools, fallback, arrays


def run(source, output, seed, epochs=35, update=True, kepler=False, births=False,
        training_births=False):
    library_limits=threadpool_limits(limits=2)
    code_sha = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path(__file__),Path(__file__).with_name('temporal_association.py'),
        Path(__file__).with_name('tracklet_initiation.py'),
        Path(__file__).with_name('kepler_dynamics.py')]}
    from kepler_dynamics import propagate_kepler
    propagator = propagate_kepler if kepler else propagate_circular
    df = load_data(source/'orbital_elements.csv', source/'observations.jsonl.truth.jsonl')
    ids, initial, scale, old_gnn, tabular, physics, old_thresholds = prepare_models(df, seed, propagator)
    raw = df[['x_km','y_km','z_km','vx_km_s','vy_km_s','vz_km_s']].to_numpy(float)
    times, sensors, truth = df.t_s.to_numpy(), df.sensor_id.to_numpy(), df.truth_id.to_numpy()
    train = (times >= 600) & (times < 1200)
    val = (times >= 1200) & (times < 1600)
    test = times >= 1600
    # Fixed teacher threshold is specified before calibration, avoiding a
    # validation-label path into the training histories.
    training_birth_log={}
    _, _, _, (h, local, positions, _) = replay(raw[train], times[train], sensors[train],
            initial, scale, 'physics_tabular', physics, .5, collect=True, update=update,
            propagator=propagator,births=training_births,birth_log=training_birth_log)
    training_identity=list(ids)
    for track,birth in sorted(training_birth_log.items()):
        labels_at_confirmation=np.unique(truth[train][birth['rows']])
        # Supervision can identify pure provisional tracks; the inference
        # engine never receives this identity map. Mixed confirmations have
        # no coherent target and all their candidate edges are negatives.
        if track != len(training_identity):raise ValueError('noncontiguous birth indices')
        training_identity.append(int(labels_at_confirmation[0]) if len(labels_at_confirmation)==1 else -1)
    labels = (np.asarray(training_identity)[positions] == truth[train, None]).astype(np.float32)
    _,_,_, validation = replay(raw[val], times[val], sensors[val],initial,scale,
        'physics_tabular',physics,.5,update=update,propagator=propagator)
    val_labels = (ids[validation[2]] == truth[val,None]).astype(np.float32)
    selection = (validation[0],validation[1],val_labels,np.isin(truth[val],ids))
    graph = train_graph(h, local, labels, seed, epochs,validation=selection)
    no_messages = train_graph(h,local,labels,seed,epochs,validation=selection,messages=False)
    no_history = train_graph(h,local,labels,seed,epochs,validation=selection,use_history=False)
    features = summary_features(h, local)
    control = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
            learning_rate=.05, random_state=seed, early_stopping=False)
    positive = max(float(labels.sum()), 1.)
    weights = np.where(labels.reshape(-1) > 0, max(1., (labels.size-positive)/positive)/3., 1.)
    control.fit(features.reshape(-1, features.shape[-1]), labels.reshape(-1), sample_weight=weights)
    jpda_config = tune_history_jpda(validation[0],validation[1],validation[2],
                                  times[val],sensors[val],truth[val],ids)
    results = {}
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_name(output.stem+'_initial.npz'),
                        catalog=initial,catalog_ids=ids,scale=scale)
    for name, model in [('history_gnn',graph),('history_no_messages',no_messages),
                        ('current_frame_graph',no_history), ('history_tabular',control),
                        ('physics_tabular',physics), ('jpda',None), ('history_jpda',jpda_config)]:
        _, _, _, arrays = replay(raw[val], times[val], sensors[val], initial, scale,
                                 name, model, .5, update=update, propagator=propagator,
                                 candidate_gate_config=jpda_config)
        threshold = threshold_for_novel_fpr(arrays[-1], truth[val], ids)
        birth_log = {}
        decisions, pools, fallback, arrays = replay(raw[test], times[test], sensors[test],
                initial, scale, name, model, threshold, update=update, propagator=propagator,
                births=births,candidate_gate_config=jpda_config,birth_log=birth_log)
        result = score_online(decisions, pools, truth[test], ids, {}, fallback)
        reidentified, wrong_revisits, future_assignments = 0,0,0
        for track,birth in birth_log.items():
            initial_labels = truth[test][birth['rows']]
            values,counts = np.unique(initial_labels,return_counts=True)
            anchor = values[counts.argmax()]
            for row,prediction in decisions.items():
                if prediction == track and times[test][row] > birth['confirmed_at']:
                    future_assignments += 1
                    if truth[test][row] == anchor: reidentified += 1
                    else: wrong_revisits += 1
        result['confirmed_track_future_correct'] = reidentified
        result['confirmed_track_future_wrong'] = wrong_revisits
        result['confirmed_track_future_assignments'] = future_assignments
        result['threshold'] = threshold
        result['shared_candidate_gate_config'] = jpda_config
        if name == 'history_jpda': result['validation_selected_config'] = jpda_config
        result['history_gate_calibration'] = 'one causal validation replay at fixed pilot threshold 0.5'
        results[name] = result
        if isinstance(model,torch.nn.Module):
            torch.save(dict(state_dict=model.state_dict(),
                use_history=getattr(model,'use_history',True),
                use_messages=getattr(model,'use_messages',True)),
                output.with_name(output.stem+'_'+name+'.pt'))
        elif name.endswith('tabular'):
            joblib.dump(model,output.with_name(output.stem+'_'+name+'.joblib'))
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output.with_name(output.stem+'_'+name+'_scores.npz'),
                scores=arrays[-1], candidates=arrays[-2], truth=truth[test], catalog_ids=ids)
    output.write_text(json.dumps({'seed':seed,'development_only':True,'epochs':epochs,
            'code_sha256':code_sha,
            'update':update,'births':births,'propagator':'kepler' if kepler else 'circular',
            'training_births':training_births,'training_promotions':len(training_birth_log),
            'methods':results}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed',type=int,default=51)
    parser.add_argument('--epochs',type=int,default=35)
    parser.add_argument('--no-update',action='store_true')
    parser.add_argument('--kepler',action='store_true')
    parser.add_argument('--births',action='store_true')
    parser.add_argument('--training-births',action='store_true')
    args = parser.parse_args()
    run(args.source,args.output,args.seed,args.epochs,not args.no_update,args.kepler,args.births,args.training_births)
