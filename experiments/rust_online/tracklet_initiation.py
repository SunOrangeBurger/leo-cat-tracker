"""Conservative public-measurement tracklet confirmation, shared by scorers.

A tentative chain needs four distinct scans over at least 120 seconds and
must predict each newly appended sighting before using it for fitting. This
is an experimental initiation policy; purity must be measured after replay.
"""
import numpy as np
from scipy.spatial import cKDTree


class TrackletInitiator:
    def __init__(self, scale, propagator, gate=4., min_scans=4, min_span=120.):
        self.scale = np.asarray(scale)
        self.propagator = propagator
        self.gate, self.min_scans, self.min_span = gate,min_scans,min_span
        self.pending = []

    def process_scan(self, rows, raw, epoch):
        self.pending = [x for x in self.pending if epoch-x['times'][-1] <= 600]
        tree = None
        if self.pending:
            centers = []
            for chain in self.pending:
                back = self.propagator(np.asarray(chain['states']),-np.asarray(chain['times']))
                centers.append(np.median(back,axis=0))
            predicted = self.propagator(np.asarray(centers),np.full(len(centers),epoch))
            tree = cKDTree(predicted/self.scale)
        proposals = []
        for row,state in zip(rows,raw):
            if tree is not None:
                d,indices = tree.query(state/self.scale,k=min(2,len(self.pending)))
                distances = np.atleast_1d(d)**2
                indices = np.atleast_1d(indices)
            else:
                distances,indices = [],[]
            if len(indices) and distances[0] <= self.gate and (
                    len(indices)==1 or distances[1]-distances[0] >= 1.):
                proposals.append((distances[0],int(indices[0]),int(row),state))
            else:
                proposals.append((float('inf'),-1,int(row),state))
        used = set()
        for distance,index,row,state in sorted(proposals,key=lambda x:x[0]):
            if index < 0 or index in used:
                self.pending.append(dict(rows=[row],states=[state.copy()],times=[epoch]))
                continue
            chain = self.pending[index]
            if chain['times'][-1] >= epoch:
                continue
            used.add(index)
            chain['rows'].append(row)
            chain['states'].append(state.copy())
            chain['times'].append(epoch)
        confirmed = [x for x in self.pending if len(x['times']) >= self.min_scans
                     and x['times'][-1]-x['times'][0] >= self.min_span]
        confirmed_ids = {id(x) for x in confirmed}
        self.pending = [x for x in self.pending if id(x) not in confirmed_ids]
        return confirmed
