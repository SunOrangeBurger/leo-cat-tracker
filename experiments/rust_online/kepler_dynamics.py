"""Elliptic two-body state propagation for the SGP4-generated sensor probe."""
import numpy as np

MU = 398600.4418


def propagate_kepler(states, dt_seconds):
    states = np.asarray(states, float)
    r, v = states[:, :3], states[:, 3:]
    radius = np.linalg.norm(r, axis=1)
    angular = np.cross(r, v)
    normal = angular/np.linalg.norm(angular, axis=1)[:,None]
    eccentric = np.cross(v, angular)/MU-r/radius[:,None]
    e = np.linalg.norm(eccentric,axis=1)
    a = 1/(2/radius-np.sum(v*v,axis=1)/MU)
    if np.any((a <= 0) | (e >= 1)):
        raise ValueError('elliptic propagation requires bound nonsingular states')
    direction = eccentric/np.maximum(e[:,None],1e-12)
    near = e < 1e-8
    direction[near] = r[near]/radius[near,None]
    tangent = np.cross(normal,direction)
    b = np.sqrt(1-e*e)
    anomaly = np.arctan2(np.sum(r*tangent,axis=1)/(a*b),
                        np.sum(r*direction,axis=1)/a+e)
    mean = anomaly-e*np.sin(anomaly)+np.sqrt(MU/a**3)*np.asarray(dt_seconds)
    mean = (mean+np.pi)%(2*np.pi)-np.pi
    anomaly = mean.copy()
    for _ in range(12):
        anomaly -= (anomaly-e*np.sin(anomaly)-mean)/(1-e*np.cos(anomaly))
    c,s = np.cos(anomaly),np.sin(anomaly)
    new_r = a[:,None]*((c-e)[:,None]*direction+(b*s)[:,None]*tangent)
    new_v = (np.sqrt(MU*a)/(a*(1-e*c)))[:,None]*(
        -s[:,None]*direction+(b*c)[:,None]*tangent)
    return np.column_stack([new_r,new_v])
