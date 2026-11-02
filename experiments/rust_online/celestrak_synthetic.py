"""Generate synthetic full-state observations from a frozen CelesTrak GP snapshot.

An empirical joint bootstrap preserves element correlations without pretending
one catalog snapshot learns sensor behaviour. Source identities and names never
enter public measurements. SGP4 yields TEME states; station visibility uses an
explicit approximate Earth rotation model. This is not a raw radar dataset.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec, SatrecArray, jday
from sgp4 import omm


def generate(snapshot, output, seed=101, objects=1000, steps=80, sensors=24,
             pos_noise=.5, vel_noise=.005, crowding=1):
    source_bytes = snapshot.read_bytes()
    records = json.loads(source_bytes)
    eligible = [x for x in records if float(x['MEAN_MOTION']) > 11
                and float(x['ECCENTRICITY']) < .1]
    if len(eligible) < 5:
        raise ValueError('insufficient LEO GP records')
    rng = np.random.default_rng(seed)
    centers = rng.choice(len(eligible), (objects+crowding-1)//crowding,
                         replace=objects > len(eligible))
    selected = np.repeat(centers,crowding)[:objects]
    sats = []
    for j, index in enumerate(selected):
        record = dict(eligible[index])
        # Small within-neighborhood perturbations; preserve correlated source
        # rows rather than sampling element marginals independently.
        record['MEAN_MOTION'] = float(record['MEAN_MOTION'])*np.exp(rng.normal(0, 2e-5))
        record['ECCENTRICITY'] = np.clip(float(record['ECCENTRICITY'])*np.exp(rng.normal(0,.02)),1e-8,.1)
        for key in ['INCLINATION','RA_OF_ASC_NODE','ARG_OF_PERICENTER','MEAN_ANOMALY']:
            value = float(record[key])+rng.normal(0,.002)
            record[key] = np.clip(value,0,180) if key == 'INCLINATION' else value % 360
        if crowding > 1:
            record['MEAN_ANOMALY'] = (record['MEAN_ANOMALY']+.02*(j%crowding))%360
        sat = Satrec()
        omm.initialize(sat, record)
        sats.append(sat)
    epoch = max(datetime.fromisoformat(x['EPOCH']).replace(tzinfo=timezone.utc) for x in eligible)
    times = [epoch+timedelta(seconds=30*i) for i in range(steps)]
    dates = np.asarray([jday(t.year,t.month,t.day,t.hour,t.minute,t.second+t.microsecond/1e6)
                        for t in times])
    error, r, v = SatrecArray(sats).sgp4(np.ascontiguousarray(dates[:,0]),
                                      np.ascontiguousarray(dates[:,1]))
    latitude = np.arcsin(rng.uniform(-1,1,sensors))
    longitude = rng.uniform(0,2*np.pi,sensors)
    station = 6378.137*np.column_stack([np.cos(latitude)*np.cos(longitude),
        np.cos(latitude)*np.sin(longitude),np.sin(latitude)])
    public, truth, csv = [], [], []
    for i,t in enumerate(times):
        # GMST approximation stated explicitly in the manifest.
        jd = dates[i].sum()
        theta = np.deg2rad((280.46061837+360.98564736629*(jd-2451545.0))%360)
        rotation = np.array([[np.cos(theta),-np.sin(theta),0],
                             [np.sin(theta),np.cos(theta),0],[0,0,1]])
        stations = station@rotation.T
        for sensor, location in enumerate(stations):
            relative = r[:,i]-location
            elevation_sine = relative@location/(np.linalg.norm(relative,axis=1)*np.linalg.norm(location))
            visible = (elevation_sine > np.sin(np.deg2rad(10))) & (error[:,i] == 0)
            detected = np.flatnonzero(visible & (rng.random(objects) < .35))
            for object_id in detected:
                state = np.r_[r[object_id,i]+rng.normal(0,pos_noise,3),
                              v[object_id,i]+rng.normal(0,vel_noise,3)]
                stamp = int(t.timestamp()*1000)
                public.append(dict(timestamp_ms=stamp,sensor_id=sensor,
                    position_km=state[:3].tolist(),velocity_km_s=state[3:].tolist(),frame='TEME'))
                truth.append(dict(object_index=int(object_id)))
                csv.append([stamp,sensor,*state])
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('observations.jsonl',public),('observations.jsonl.truth.jsonl',truth)]:
        (output/name).write_text(''.join(json.dumps(x)+'\n' for x in rows))
    pd.DataFrame(csv,columns=['timestamp_ms','sensor_id','x_km','y_km','z_km',
                             'vx_km_s','vy_km_s','vz_km_s']).to_csv(output/'orbital_elements.csv',index=False)
    manifest = dict(seed=seed,objects=objects,steps=steps,sensors=sensors,
        observation_count=len(csv),frame='TEME',propagator='SGP4',
        population_model='empirical joint bootstrap with stated local perturbations',
        crowding=crowding,crowding_is_engineered_stress=crowding>1,
        snapshot_sha256=hashlib.sha256(source_bytes).hexdigest(),snapshot_records=len(records),
        eligible_records=len(eligible),source='https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json',
        source_selection_indices=selected.tolist(),epoch=epoch.isoformat(),
        position_noise_km=pos_noise,velocity_noise_km_s=vel_noise,
        synthetic_sensor_assumptions=dict(elevation_deg=10,detection_probability=.35,
                station_distribution='uniform sphere',earth_rotation='linear GMST approximation',
                clutter=False,measurement='noisy full Cartesian state'))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_selection_indices'},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--objects',type=int,default=1000)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--sensors',type=int,default=24)
    p.add_argument('--crowding',type=int,default=1)
    a=p.parse_args()
    generate(a.snapshot,a.output,a.seed,a.objects,a.steps,a.sensors,crowding=a.crowding)
