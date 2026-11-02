"""Causal track-history encoder and contextual candidate graph.

History contains accepted PUBLIC measurements only. Labels are permitted in
the experiment wrapper for training/scoring, never in this module's buffer.
The five candidate nodes exchange learned messages after a GRU encodes each
candidate's last eight sightings relative to the current observation.
"""
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn

from online_catalog_benchmark import physics_tabular_inputs
from rust_noise_orbit_benchmark import propagate_circular


class TrackHistory:
    def __init__(self, initial, length=8, propagator=propagate_circular):
        self.length = length
        self.propagator = propagator
        self.rows = defaultdict(lambda: deque(maxlen=length))
        for j, state in enumerate(initial):
            self.rows[j].append((0., np.asarray(state).copy()))

    def add_scan(self, assignments, raw, epoch):
        # One entry per track per scan avoids duplicating simultaneous sensors.
        for j in set(assignments[assignments >= 0]):
            state = np.median(raw[assignments == j], axis=0)
            self.rows[int(j)].append((float(epoch), state.copy()))

    def features(self, raw, epoch, positions, scale):
        n, k = positions.shape
        values = np.zeros((n, k, self.length, 8), np.float32)
        physical_query = raw / scale
        candidates = np.zeros((n, k, 6), np.float32)
        for i in range(n):
            for j, track in enumerate(positions[i]):
                history = list(self.rows[int(track)])
                times = np.asarray([x[0] for x in history])
                if np.any(times > epoch):
                    raise ValueError('future history cannot be used for association')
                states = np.asarray([x[1] for x in history])
                predicted = self.propagator(states, epoch - times)
                candidates[i, j] = np.median(predicted, axis=0) / scale
                # Rotate every residual into the query's orbital frame.
                projected = physics_tabular_inputs(
                    np.broadcast_to(physical_query[i:i+1], (len(predicted), 6)),
                    np.broadcast_to(predicted[:, None, :] / scale,
                                    (len(predicted), 5, 6)), scale)
                residual = projected.reshape(len(predicted), 5, -1)[:, 0, :6]
                length = len(history)
                values[i, j, -length:, :6] = np.clip(residual, -30, 30)
                values[i, j, -length:, 6] = (epoch - times) / 600.
                values[i, j, -length:, 7] = 1.
        return values, candidates


class HistoryGraph(nn.Module):
    def __init__(self, feature_dim=20, width=48):
        super().__init__()
        self.history = nn.GRU(8, width, batch_first=True)
        self.local = nn.Sequential(nn.Linear(feature_dim, width), nn.SiLU())
        self.messages = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, history, local):
        n, k, length, _ = history.shape
        if not getattr(self,'use_history',True):
            history = torch.zeros_like(history)
        _, h = self.history(history.reshape(n*k, length, 8))
        # Distances can span many orders of magnitude for open-set queries.
        # Preserve signs and compress magnitudes before neural layers.
        transformed = torch.sign(local)*torch.log1p(local.abs())
        nodes = h[0].reshape(n, k, -1) + self.local(transformed)
        messages, _ = self.messages(nodes, nodes, nodes, need_weights=False)
        if not getattr(self,'use_messages',True):
            messages = torch.zeros_like(messages)
        context = self.norm(nodes + messages)
        return self.output(torch.cat([nodes, context], -1)).squeeze(-1)


def summary_features(history, local):
    mask = history[..., 7:8]
    count = np.maximum(mask.sum(2), 1)
    mean = (history[..., :7] * mask).sum(2) / count
    variance = (((history[..., :7] - mean[:, :, None])**2) * mask).sum(2) / count
    return np.concatenate([local, mean, np.sqrt(variance), count / 8], -1)


def train_graph(history, local, labels, seed=101, epochs=60, validation=None, width=48,
                messages=True, use_history=True):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = HistoryGraph(local.shape[-1],width)
    model.use_messages = messages
    model.use_history = use_history
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    arrays = [torch.as_tensor(x, dtype=torch.float32) for x in (history, local, labels)]
    rng = np.random.default_rng(seed)
    positive = max(float(labels.sum()), 1.)
    weight = torch.tensor(max(1., (labels.size-positive)/positive)/3.)
    best_state, best_metric = None, -1.
    for epoch in range(epochs):
        model.train()
        for rows in np.array_split(rng.permutation(len(labels)), max(1, (len(labels)+127)//128)):
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(arrays[0][rows], arrays[1][rows]), arrays[2][rows], pos_weight=weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if validation is not None and ((epoch+1)%5 == 0 or epoch+1 == epochs):
            vh,vl,vy,known = validation
            model.eval()
            score = probabilities(model,vh,vl)
            threshold = np.quantile(score[~known].max(1),.98)+1e-6
            chosen = score.argmax(1)
            metric = float(((score.max(1)>=threshold) &
                            (vy[np.arange(len(vy)),chosen]>0)).sum())
            if metric > best_metric:
                best_metric = metric
                best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def probabilities(model, history, local):
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(history[i:i+256]),
                                torch.from_numpy(local[i:i+256])).sigmoid()
                          for i in range(0, len(history), 256)]).numpy()
