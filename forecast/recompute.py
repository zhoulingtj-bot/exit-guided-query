"""Recompute the main P70 tables from saved predictions and measured targets."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import aggregate, pairwise, score_fold
from .settings import CONFIG, FOLDS, METHODS, ROOT, TASK, read_config, sha256, write_json


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def timings(folder, live):
    if live:
        frames = []
        for fold in FOLDS:
            seal = read_config(folder / f"{fold}_seal.json")
            if sha256(folder / f"{fold}.npz") != seal["predictions_sha256"]:
                raise ValueError(f"Prediction hash mismatch for {fold}.")
            csv_path = folder / f"{fold}_timing.csv"
            if sha256(csv_path) != seal["timing_sha256"]:
                raise ValueError(f"Timing hash mismatch for {fold}.")
            frames.append(pd.read_csv(csv_path))
        return pd.concat(frames, ignore_index=True)
    for entry in (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = entry.split("  ", 1)
        if sha256(folder / name) != expected:
            raise ValueError(f"Archived file hash mismatch: {name}")
    return pd.read_csv(folder / "timing_trials.csv")


def table_from_trials(trials):
    keys = ["fold", "method"]
    groups = trials.groupby(keys, sort=False)
    assert set(trials.fold) == set(FOLDS) and set(trials.method) == set(METHODS)
    counts = groups.size()
    assert len(counts) == len(FOLDS) * len(METHODS) and counts.min() >= 3
    assert not trials.duplicated(["fold", "method", "repeat"]).any()
    assert not trials.duplicated(["fold", "order_in_repeat", "repeat"]).any()
    assert trials.groupby("fold").shared_fit_seconds.nunique().eq(1).all()
    for name in ("query_rows", "actual_predict_calls", "extra_serial_rounds"):
        assert groups[name].nunique().eq(1).all(), name
    fold = groups.agg(repeats=("seconds", "size"), query_rows=("query_rows", "first"),
                      predict_calls=("actual_predict_calls", "first"),
                      serial_rounds=("extra_serial_rounds", "first"),
                      median_seconds=("seconds", "median"),
                      shared_fit_seconds=("shared_fit_seconds", "first")).reset_index()
    fold["fit_plus_inference_s"] = fold.shared_fit_seconds + fold.median_seconds
    overall = fold.groupby("method", sort=False).agg(
        prediction_calls=("predict_calls", "sum"), query_rows=("query_rows", "sum"),
        max_serial_rounds=("serial_rounds", "max"), sum_serial_rounds=("serial_rounds", "sum"),
        warmed_inference_s=("median_seconds", "sum"), context_fit_s=("shared_fit_seconds", "sum"),
        fit_plus_inference_s=("fit_plus_inference_s", "sum")).reset_index()
    return fold, overall


def score(data, predictions, output, live):
    trials = timings(predictions, live)
    records = []
    for fold in FOLDS:
        trajectory = load_arrays(predictions / f"{fold}.npz")
        measured = load_arrays(data / f"{fold}_measured.npz")
        for key in ("record_ids", "composition", "sample_index"):
            np.testing.assert_array_equal(trajectory[key], measured[key])
        np.testing.assert_array_equal(trajectory["record_ids"], CONFIG["splits"][fold]["evaluation_record_ids"])
        np.testing.assert_array_equal(trajectory["sample_index"], TASK.targets)
        np.testing.assert_array_equal(trajectory["anchors"], TASK.anchors)
        y = measured["measured_uA"]
        curves = {key: trajectory[f"prediction_{key}"] for key in METHODS}
        masks = {key: trajectory[f"queries_{key}"] for key in METHODS}
        assert all(x.shape == y.shape and np.isfinite(x).all() for x in curves.values())
        for key, mask in masks.items():
            assert mask.shape == y.shape and mask.dtype == bool
            budget = trials.loc[(trials.fold == fold) & (trials.method == key), "query_rows"]
            assert (budget == mask.sum()).all(), (fold, key)
        records.append(score_fold(fold, y, curves, masks, trajectory))
    records = pd.concat(records, ignore_index=True)
    fold_timing, total = table_from_trials(trials)
    summary = aggregate(records, ["method"]).merge(total, on="method", validate="one_to_one")
    assert len(records) == 83 * len(METHODS)
    assert summary.query_rows.equals(summary.total_queries)
    paired = pairwise(records)
    if not live:
        verify_reference(summary, predictions / "table2_reference.csv", ["method"])
        verify_reference(paired, predictions / "paired_reference.csv", ["reference", "comparison", "cohort", "metric"])
    output.mkdir(parents=True, exist_ok=True)
    frames = {"table2": summary, "s3c_paired": paired, "s3_timing_by_fold": fold_timing,
              "record_metrics": records}
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    result = summary.set_index("method")
    stats = {"timing_source": "new CUDA run" if live else "archived unified CUDA trials"}
    for name, baseline in (("dense", "C"), ("bisection", "Bisection")):
        for label, field in (("warm", "warmed_inference_s"), ("fit_plus_inference", "fit_plus_inference_s")):
            stats[f"{label}_percent_saved_vs_{name}"] = float(100 * (1 - result.at["Exit", field] / result.at[baseline, field]))
    write_json(output / "timing_summary.json", stats)
    print(summary[["method", "mean_queries", "macro_rmse_uA", "exact_dense_indices",
                   "localization_common7_mae_samples", "warmed_inference_s", "fit_plus_inference_s"]]
          .round(3).to_string(index=False))


def verify_reference(calculated, ref, keys):
    expected = pd.read_csv(ref).set_index(keys).sort_index()
    actual = calculated.set_index(keys).sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    pd.testing.assert_index_equal(actual.columns.sort_values(), expected.columns.sort_values())
    for name in expected.columns:
        np.testing.assert_allclose(actual[name].to_numpy(), expected[name].to_numpy(), rtol=0, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("work/data"))
    parser.add_argument("--live", type=Path, help="Score a new five-fold CUDA run instead of the archived outputs.")
    parser.add_argument("--output", type=Path, default=Path("work/summary"))
    args = parser.parse_args()
    score(args.data, args.live or ROOT / "evidence", args.output, args.live is not None)


if __name__ == "__main__":
    main()
