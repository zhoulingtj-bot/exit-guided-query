"""Batched refinement and six query controls from the unified CUDA run."""
import time
import numpy as np
from .settings import TASK, THRESHOLD
from .rules import policy


def bisect_brackets(plan, query):
    left, right = plan["left"].copy(), plan["right"].copy()
    shape = (len(left), len(TASK.targets))
    query_round = np.zeros(shape, dtype=np.uint8)
    extra_current = np.full(shape, np.nan)
    round_id = 0
    while (active := np.flatnonzero(plan["trigger"] & (right - left > 1))).size:
        round_id += 1
        mid = (left[active] + right[active]) // 2
        offsets = mid - TASK.targets[0]
        assert (query_round[active, offsets] == 0).all()
        currents = query(active, offsets)
        assert currents.shape == active.shape and np.isfinite(currents).all()
        query_round[active, offsets] = round_id
        extra_current[active, offsets] = currents
        below_threshold = abs(currents) < THRESHOLD
        right[active[below_threshold]] = mid[below_threshold]
        left[active[~below_threshold]] = mid[~below_threshold]
    return query_round, extra_current, right


def reconstruct(anchor_current, query_round, extra_current):
    anchor_mask = np.isin(TASK.targets, TASK.anchors)
    assert not query_round[:, anchor_mask].any()
    curve = np.empty(query_round.shape)
    for i, anchors in enumerate(anchor_current):
        used = anchor_mask | (query_round[i] > 0)
        known = extra_current[i].copy()
        known[anchor_mask] = anchors
        curve[i] = np.interp(TASK.targets, TASK.targets[used], known[used])
    return curve


class CountedPredictor:
    def __init__(self, model):
        self.model = model
        self.rows = []

    def predict(self, inputs):
        self.rows.append(len(inputs))
        return self.model.predict(inputs)


def bisect_inference(kernels, model, task, prefix, composition, scale, voltage):
    import numpy as np

    start = time.perf_counter()
    inputs = kernels.features(task, prefix, composition, scale, voltage, task.anchors)
    anchor = kernels.predict_blocks(model, inputs, task.batch).reshape(len(prefix), len(task.anchors))
    first_seconds = time.perf_counter() - start
    n_first = len(model.rows)
    start = time.perf_counter()
    plan = policy(anchor)
    select_seconds = time.perf_counter() - start

    def query(active, offsets):
        mask = np.zeros(plan["Exit"].shape, dtype=bool)
        mask[active, offsets] = True
        block = kernels.extra_features(mask, prefix, composition, scale, voltage)
        return kernels.predict_blocks(model, block, task.batch)

    start = time.perf_counter()
    rounds, values, index = bisect_brackets(plan, query)
    second_seconds = time.perf_counter() - start
    start = time.perf_counter()
    curve = reconstruct(anchor, rounds, values)
    reconstruct_seconds = time.perf_counter() - start
    counters = dict(first_stage_seconds=first_seconds, selection_seconds=select_seconds,
                    second_stage_seconds=second_seconds, reconstruction_seconds=reconstruct_seconds,
                    query_rows=sum(model.rows), first_predict_calls=n_first,
                    second_predict_calls=len(model.rows)-n_first,
                    extra_query_rows=int(np.count_nonzero(rounds)))
    return curve, anchor, counters, rounds, index


def uniform_mask(targets, offset, n):
    import numpy as np

    mask = np.zeros((n, len(targets)), dtype=bool)
    for i in range(n):
        j = offset + i
        count = 20 + ((j + 1) * 28 // 83 - j * 28 // 83)
        indices = np.rint(np.linspace(0, len(targets) - 1, count)).astype(int)
        mask[i, indices] = True
        assert mask[i].sum() == count
    return mask


def spread_mask(task, plan):
    import numpy as np

    mask = np.zeros_like(plan["Exit"])
    for i, count in enumerate(plan["Exit"].sum(axis=1)):
        locations = task.anchors.tolist()
        for _ in range(count):
            left = int(np.diff(locations).argmax())
            middle = (locations[left] + locations[left + 1]) // 2
            assert locations[left] < middle < locations[left + 1]
            mask[i, middle - task.targets[0]] = True
            locations.insert(left + 1, middle)
    return mask


def live_prediction(kernels, task, predictor, method, prefix, groups, scale, voltage, offset):
    import numpy as np

    shape = (len(prefix), len(task.targets))
    anchors = np.broadcast_to(np.isin(task.targets, task.anchors), shape)
    rounded = np.zeros(shape, dtype=np.uint8)
    if method == "Bisection":
        curve, first, stats, rounded, _ = bisect_inference(kernels, predictor, task, prefix, groups, scale, voltage)
        queried = anchors | (rounded > 0)
    elif method in ("Uniform", "Spread"):
        select = time.perf_counter()
        if method == "Uniform":
            queried = uniform_mask(task.targets, offset, len(prefix))
            selected = queried
            first = None
            first_calls = 0
            first_seconds = 0.0
        else:
            block = kernels.features(task, prefix, groups, scale, voltage, task.anchors)
            first = kernels.predict_blocks(predictor, block, task.batch).reshape(len(prefix), len(task.anchors))
            first_calls = len(predictor.rows)
            first_seconds = time.perf_counter() - select
            select = time.perf_counter()
            selected = spread_mask(task, policy(first))
            queried = anchors | selected
            rounded = selected.astype(np.uint8)
        selection = time.perf_counter() - select
        start = time.perf_counter()
        features = kernels.extra_features(selected, prefix, groups, scale, voltage)
        values = kernels.predict_blocks(predictor, features, task.batch) if len(features) else np.empty(0)
        query_seconds = time.perf_counter() - start
        start = time.perf_counter()
        if method == "Uniform":
            lattice = np.full(shape, np.nan)
            lattice[queried] = values
            curve = np.stack([np.interp(task.targets, task.targets[mask], row[mask])
                              for mask, row in zip(queried, lattice)])
            first_calls = len(predictor.rows)
            first_seconds, query_seconds = query_seconds, 0.0
        else:
            curve = kernels.interpolate(task.anchors, task.targets, first)
            curve[selected] = values
        stats = dict(first_stage_seconds=first_seconds, selection_seconds=selection,
                     second_stage_seconds=query_seconds, reconstruction_seconds=time.perf_counter() - start,
                     first_predict_calls=first_calls, second_predict_calls=len(predictor.rows) - first_calls,
                     extra_query_rows=int(rounded.astype(bool).sum()), query_rows=sum(predictor.rows))
    else:
        curve, first, stats = kernels.infer(method, predictor, prefix, groups, scale, voltage)
        if method in ("Exit", "Start"):
            rounded = policy(first)[method].astype(np.uint8)
        queried = np.ones(shape, dtype=bool) if method == "C" else anchors | (rounded > 0)
    return curve, first, queried, rounded, stats


def run_path(kernels, model, task, method, prefix, groups, scale, voltage, offset):
    import numpy as np

    predictor = CountedPredictor(model)
    (curve, first, queried, rounded, stats), seconds = kernels.timed(
        lambda: live_prediction(kernels, task, predictor, method, prefix, groups, scale, voltage, offset))
    assert np.isfinite(curve).all() and curve.shape == queried.shape == rounded.shape
    assert queried.sum() == stats["query_rows"] == sum(predictor.rows)
    assert np.count_nonzero(rounded) == stats["extra_query_rows"]
    assert len(predictor.rows) == stats["first_predict_calls"] + stats["second_predict_calls"]
    assert all(0 < n <= task.batch for n in predictor.rows)
    timing = dict(seconds=seconds, actual_predict_calls=len(predictor.rows), actual_predict_call_rows=predictor.rows,
                  extra_serial_rounds=int(rounded.max()),
                  extra_rows_by_round=[int((rounded == r).sum()) for r in range(1, int(rounded.max()) + 1)], **stats)
    return dict(curve=curve, first=first, queried=queried, rounded=rounded, timing=timing)
