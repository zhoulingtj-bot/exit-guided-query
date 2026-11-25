"""Actual TabPFN calls and query assembly from the frozen study."""
import time

import numpy as np
import torch
from tabpfn import TabPFNRegressor

from .prefix_features import features as d1_prefix_features
from .rules import policy, QUERY, TARGET
from .settings import TASK


def features(task, observed, composition, scale, voltage, positions):
    assert observed.shape[1:] == (task.prefix, 2)
    state = d1_prefix_features(observed, composition, task.prefix, scale)
    t = np.asarray(positions) - 1
    v = voltage[t]
    query = np.column_stack((t/499., v/1.5, v*v/2.25, np.where(t < 350, -1., 1.), (t >= 350).astype(float)))
    return np.column_stack((np.repeat(state, len(t), axis=0), np.tile(query, (len(observed),1)))).astype('float32')


def interpolate(anchors, targets, values):
    """Interpolate predicted currents at original one-based sample indices."""
    assert len(anchors) == values.shape[1] and anchors[0] == targets[0] and anchors[-1] == targets[-1]
    return np.stack([np.interp(targets, anchors, row) for row in values])


def timed(action):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = action()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def model_fit(task, x, y):
    model = TabPFNRegressor(n_estimators=task.estimators, model_path=str(task.weight),
        device='cuda', inference_precision=torch.float32, fit_mode=task.fit_mode, random_state=task.seed)
    return model.fit(x,y)


def predict_blocks(model, x, batch):
    return np.concatenate([model.predict(x[i:i+batch]) for i in range(0,len(x),batch)])


def extra_features(mask, prefix, groups, scale, voltage):
    records = np.flatnonzero(mask.any(axis=1))
    if not len(records):
        return np.empty((0,TASK.features),dtype=np.float32)
    return np.concatenate([features(TASK,prefix[i:i+1],groups[i:i+1],scale,voltage,TARGET[mask[i]]) for i in records])


def infer(method, model, prefix, groups, scale, voltage):
    """One end-to-end timed request, querying real model outputs in every branch."""
    n = len(prefix)
    start = time.perf_counter()
    indices = TARGET if method=='C' else QUERY
    x = features(TASK,prefix,groups,scale,voltage,indices)
    first = predict_blocks(model,x,TASK.batch).reshape(n,len(indices))
    first_seconds = time.perf_counter()-start
    stats = dict(first_stage_seconds=first_seconds,selection_seconds=0.0,second_stage_seconds=0.0,
                 reconstruction_seconds=0.0,query_rows=len(x),first_predict_calls=(len(x)+TASK.batch-1)//TASK.batch,
                 second_predict_calls=0,extra_query_rows=0)
    if method=='C':
        return first,None,stats
    if method=='B':
        t = time.perf_counter()
        curve = interpolate(QUERY,TARGET,first)
        stats['reconstruction_seconds'] = time.perf_counter()-t
        return curve,first,stats
    t = time.perf_counter()
    selected = policy(first)[method]
    stats['selection_seconds'] = time.perf_counter()-t
    if selected.any():
        t = time.perf_counter()
        x_extra = extra_features(selected,prefix,groups,scale,voltage)
        extra = predict_blocks(model,x_extra,TASK.batch)
        assert len(extra)==int(selected.sum()) and np.isfinite(extra).all()
        stats['second_stage_seconds'] = time.perf_counter()-t
        stats['second_predict_calls'] = (len(extra)+TASK.batch-1)//TASK.batch
        stats['extra_query_rows'] = len(extra)
        stats['query_rows'] += len(extra)
    t = time.perf_counter()
    trajectory = interpolate(QUERY,TARGET,first)
    if selected.any():
        trajectory[selected] = extra
    stats['reconstruction_seconds'] = time.perf_counter()-t
    return trajectory,first,stats
