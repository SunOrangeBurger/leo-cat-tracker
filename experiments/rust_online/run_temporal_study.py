"""Run fresh paired temporal association scenarios and report every seed."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import itertools
from pathlib import Path
import subprocess
import sys

import numpy as np
from celestrak_synthetic import generate


def execute(snapshot, root, seed, objects, crowding, births, noise,training_births):
    source = root/'streams'/f'seed_{seed}'
    source.mkdir(parents=True,exist_ok=True)
    # Generation is deterministic for the frozen snapshot and listed parameters.
    generate(snapshot,source,seed,objects,pos_noise=noise,vel_noise=noise/100,crowding=crowding)
    output = root/f'seed_{seed}.json'
    command = [sys.executable,str(Path(__file__).with_name('run_temporal_probe.py')),
               str(source),str(output),'--seed',str(seed),'--kepler']
    if births: command.append('--births')
    if training_births:command.append('--training-births')
    with (root/f'seed_{seed}.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    return seed


def summarize(root,seeds):
    runs = [json.loads((root/f'seed_{seed}.json').read_text()) for seed in seeds]
    rng = np.random.default_rng(20261102)
    results = {}
    for name in runs[0]['methods']:
        recall = np.array([r['methods'][name]['known_correct_per_detection'] for r in runs])
        baseline = np.array([r['methods']['history_jpda']['known_correct_per_detection'] for r in runs])
        difference = recall-baseline
        fmr=np.array([r['methods'][name]['novel_seeded_false_merge_rate'] for r in runs])
        baseline_fmr=np.array([r['methods']['history_jpda']['novel_seeded_false_merge_rate'] for r in runs])
        fmr_difference=fmr-baseline_fmr
        fmr_boot=rng.choice(fmr_difference,size=(10000,len(seeds))).mean(1)
        boot = rng.choice(difference,size=(10000,len(seeds))).mean(1)
        observed=abs(difference.mean())
        signflip=np.array([abs(np.mean(difference*np.array(signs)))
                          for signs in itertools.product([-1,1],repeat=len(seeds))])
        results[name] = dict(seed_recall=recall.tolist(),mean_recall=float(recall.mean()),
            differences_from_tuned_jpda=difference.tolist(),
            mean_difference=float(difference.mean()),
            difference_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
            seeds_favoring_method=int((difference>0).sum()),
            paired_signflip_p=float(np.mean(signflip>=observed-1e-12)),
            mean_false_merge=float(fmr.mean()),
            mean_false_merge_difference=float(fmr_difference.mean()),
            false_merge_difference_bootstrap_95=np.quantile(fmr_boot,[.025,.975]).tolist(),
            total_known=sum(r['methods'][name]['known_observations'] for r in runs),
            total_novel=sum(r['methods'][name]['novel_observations'] for r in runs),
            total_confirmed_tracks=sum(r['methods'][name]['born_track_count'] for r in runs),
            total_future_correct=sum(r['methods'][name]['confirmed_track_future_correct'] for r in runs),
            total_future_wrong=sum(r['methods'][name]['confirmed_track_future_wrong'] for r in runs),
            seed_novel_false_merge=[r['methods'][name]['novel_seeded_false_merge_rate'] for r in runs],
            seed_birth_purity=[r['methods'][name]['born_track_mean_purity'] for r in runs])
    (root/'summary.json').write_text(json.dumps(dict(seeds=seeds,methods=results,runs=runs),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--start-seed',type=int,default=201)
    p.add_argument('--seeds',type=int,default=10)
    p.add_argument('--objects',type=int,default=1000)
    p.add_argument('--crowding',type=int,default=1)
    p.add_argument('--noise',type=float,default=.5)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--births',action='store_true')
    p.add_argument('--training-births',action='store_true')
    p.add_argument('--analyze-only',action='store_true')
    p.add_argument('--resume',action='store_true',help='skip completed seed JSONs; use only with the same configuration')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    seeds=list(range(a.start_seed,a.start_seed+a.seeds))
    if not a.analyze_only:
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures=[pool.submit(execute,a.snapshot,a.output,s,a.objects,a.crowding,a.births,a.noise,a.training_births)
                     for s in seeds if not (a.resume and (a.output/f'seed_{s}.json').exists())]
            for f in as_completed(futures): print('completed',f.result(),flush=True)
    summarize(a.output,seeds)
