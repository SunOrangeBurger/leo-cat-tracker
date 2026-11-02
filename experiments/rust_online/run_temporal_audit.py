"""Run complete online calibration-audit replays for saved paired checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess
import sys

from run_temporal_study import summarize


def run_one(source,output,seed,primary_only,risk_extra,frozen,normalized):
    command=[sys.executable,str(Path(__file__).with_name('audit_temporal_calibration.py')),
             str(source),str(output/f'seed_{seed}.json'),'--seed',str(seed)]
    if primary_only: command.append('--primary-only')
    if risk_extra: command.append('--risk-extra')
    if frozen: command.append('--frozen')
    if normalized:command.append('--normalized')
    with (output/f'seed_{seed}.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    return seed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--start-seed',type=int,required=True)
    p.add_argument('--seeds',type=int,default=10)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--primary-only',action='store_true')
    p.add_argument('--analyze-only',action='store_true')
    p.add_argument('--risk-extra',action='store_true')
    p.add_argument('--frozen',action='store_true')
    p.add_argument('--normalized',action='store_true')
    p.add_argument('--resume',action='store_true',help='skip completed audit JSONs with the same configuration')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    seeds=list(range(a.start_seed,a.start_seed+a.seeds))
    if not a.analyze_only:
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures=[pool.submit(run_one,a.study,a.output,s,a.primary_only,a.risk_extra,a.frozen,a.normalized)
                     for s in seeds if not (a.resume and (a.output/f'seed_{s}.json').exists())]
            for f in as_completed(futures):print('audited',f.result(),flush=True)
    if not a.risk_extra:summarize(a.output,seeds)
