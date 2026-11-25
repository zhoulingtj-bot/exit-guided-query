"""Run the frozen P70 seven-path comparison with synchronized CUDA timings."""
import argparse
from dataclasses import replace
import gc
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import run_path
from .prefix_features import context
from .rules import policy
from .settings import (CONFIG, FOLDS, METHODS, TASK, array_sha256, read_config, sha256, write_json)


def fit_context(kernels, task, source, fold, weights):
    spec = CONFIG["splits"][fold]
    with np.load(source, allow_pickle=False) as inputs:
        train, train_groups, train_ids = (inputs[k] for k in
            ("train_curves_V_A", "train_composition", "train_ids"))
        prefix, groups, ids = (inputs[k] for k in ("observed_prefix_V_A", "composition", "record_ids"))
    assert train_ids.tolist() == spec["training_record_ids"]
    assert ids.tolist() == spec["evaluation_record_ids"]
    assert set(train_ids).isdisjoint(ids)
    scale, voltage = context(train)
    x = kernels.features(task, train[:, :task.prefix], train_groups, scale, voltage, task.anchors)
    y = (train[:, task.anchors - 1, 1] * 1e6).ravel()
    dense_x = kernels.features(task, prefix, groups, scale, voltage, task.targets)
    for value, key in ((x, "training_features_sha256"), (y, "training_labels_sha256"),
                       (dense_x, "evaluation_features_sha256")):
        if array_sha256(value) != spec[key]:
            raise ValueError(f"{fold} differs from the archived numerical input for {key}.")
    model, elapsed = kernels.timed(lambda: kernels.model_fit(replace(task, weight=weights), x, y))
    return model, elapsed, prefix, groups, ids, train_ids, scale, voltage


def run_fold(kernels, data, output, fold, offset, weights, repeats, runtime):
    source = data / f"{fold}.npz"
    seal = output / f"{fold}_seal.json"
    checkpoint = output / f"{fold}.npz"
    timing_csv = output / f"{fold}_timing.csv"
    run_id = dict(runtime=runtime, input_sha256=sha256(source))
    if seal.is_file():
        existing = read_config(seal)
        assert existing["configuration"] == run_id, "Use a new output directory for different inputs."
        assert existing["predictions_sha256"] == sha256(checkpoint)
        assert existing["timing_sha256"] == sha256(timing_csv)
        print(f"{fold}: verified existing outputs", flush=True)
        return

    model, fit_seconds, prefix, groups, ids, train_ids, scale, voltage = fit_context(
        kernels, TASK, source, fold, weights)
    print(f"{fold}: numerical input hashes match; warming seven paths", flush=True)
    for method in METHODS:
        run_path(kernels, model, TASK, method, prefix, groups, scale, voltage, offset)

    saved, traces = {}, []
    for trial in range(repeats):
        shift = (int(fold[-1]) * repeats + trial) % len(METHODS)
        order = METHODS[shift:] + METHODS[:shift]
        times = []
        for position, method in enumerate(order, 1):
            item = run_path(kernels, model, TASK, method, prefix, groups, scale, voltage, offset)
            old = saved.setdefault(method, item)
            np.testing.assert_array_equal(old["queried"], item["queried"])
            np.testing.assert_array_equal(old["rounded"], item["rounded"])
            traces.append(dict(fold=fold, repeat=trial+1, order_in_repeat=position, method=method,
                               n_evaluated=len(ids), shared_fit_seconds=fit_seconds, **item["timing"]))
            times.append(f"{method} {item['timing']['seconds']:.3f} s")
        print(f"{fold} trial {trial+1}: {', '.join(times)}", flush=True)

    first = saved["B"]["first"]
    plan = policy(first)
    for method in ("Exit", "Start", "Bisection", "Spread"):
        np.testing.assert_array_equal(first, saved[method]["first"])
    for method in ("Exit", "Start", "Spread"):
        np.testing.assert_array_equal(saved[method]["rounded"].sum(axis=1), plan["Exit"].sum(axis=1))
    assert not ((saved["Bisection"]["rounded"] > 0) & ~plan["Exit"]).any()
    payload = dict(record_ids=ids, train_ids=train_ids, composition=groups, sample_index=TASK.targets,
                   anchors=TASK.anchors, anchor_current_uA=first, trigger=plan["trigger"],
                   bracket_left=plan["left"], bracket_right=plan["right"])
    for method, item in saved.items():
        payload[f"prediction_{method}"] = item["curve"]
        payload[f"queries_{method}"] = item["queried"]
        payload[f"refinement_{method}"] = item["rounded"]
    temporary = checkpoint.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(checkpoint)
    pd.DataFrame(traces).to_csv(timing_csv, index=False)
    write_json(seal, dict(configuration=run_id, predictions_sha256=sha256(checkpoint), timing_sha256=sha256(timing_csv)))
    del model
    gc.collect()
    kernels.torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("work/data"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("work/live"))
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least three repeats for the timing protocol.")
    weights = args.weights.resolve(strict=True)
    weight_sha = sha256(weights)
    if weight_sha != CONFIG["engine"]["weight_sha256"]:
        raise ValueError("The checkpoint SHA256 must match protocol.json.")
    from . import kernels
    if not kernels.torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-capable PyTorch installation.")
    if version("tabpfn") != "2.0.9":
        raise RuntimeError("Use TabPFN 2.0.9 from the pinned source in the README.")
    kernels.torch.set_num_threads(TASK.cpu_threads)
    runtime = dict(weight_sha256=weight_sha, torch=kernels.torch.__version__,
                   cuda=kernels.torch.version.cuda, gpu=kernels.torch.cuda.get_device_name(),
                   tabpfn=version("tabpfn"), repeats=args.repeats)
    args.output.mkdir(parents=True, exist_ok=True)
    offset = 0
    for fold in FOLDS:
        if not args.fold or args.fold == fold:
            run_fold(kernels, args.data, args.output, fold, offset, weights, args.repeats, runtime)
        offset += len(CONFIG["splits"][fold]["evaluation_record_ids"])


if __name__ == "__main__":
    main()
