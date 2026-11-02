"""One-command temporal training and numerically audited online evaluation.

Input is a public state CSV and separate truth sidecar produced by either the
Rust exporter/bridge or the CelesTrak-informed generator. Labels are confined
to the experiment wrapper; the replay engine receives no identities.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--predictor',choices=['auto','circular','kepler'],default='auto')
    p.add_argument('--births',action='store_true')
    p.add_argument('--training-births',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args()
    source=a.source.resolve()
    for name in ['orbital_elements.csv','observations.jsonl.truth.jsonl']:
        if not (source/name).is_file():p.error('missing input '+str(source/name))
    root=a.output.resolve()
    if (root/f'seed_{a.seed}.json').exists():p.error('choose a fresh output directory; checkpoints already exist')
    target=root/'streams'/f'seed_{a.seed}'
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source,target_is_directory=True)
    elif target.resolve()!=source:p.error('existing source link points to a different stream')
    predictor=a.predictor
    if predictor=='auto':
        manifest=source/'manifest.json'
        predictor='kepler' if manifest.exists() and json.loads(manifest.read_text()).get('propagator')=='SGP4' else 'circular'
    script=Path(__file__).with_name('run_temporal_probe.py')
    command=[sys.executable,str(script),str(target),str(root/f'seed_{a.seed}.json'),
             '--seed',str(a.seed)]
    if predictor=='kepler':command.append('--kepler')
    if a.births:command.append('--births')
    if a.training_births:command.append('--training-births')
    subprocess.run(command,check=True)
    subprocess.run([sys.executable,str(script.with_name('audit_temporal_calibration.py')),
        str(root),str(root/'audited'/f'seed_{a.seed}.json'),'--seed',str(a.seed),'--normalized'],check=True)
    print('Audited result:',root/'audited'/f'seed_{a.seed}.json')
