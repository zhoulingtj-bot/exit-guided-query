"""Record-level trajectory, detection and localization statistics."""
import numpy as np
import pandas as pd
from .settings import TASK, METHODS
from .rules import exit_index

KEYS = ["fold", "record_id", "composition"]


def bootstrap_paired(values,groups):
    if not len(values):
        return [np.nan,np.nan]
    generator = np.random.default_rng(TASK.seed)
    total = np.zeros(20000)
    for group in sorted(set(groups)):
        x = values[groups==group]
        total += x[generator.integers(0,len(x),(20000,len(x)))].sum(axis=1)
    return np.quantile(total/len(values),[.025,.975])


def score_fold(fold, measured, curves, masks, archived):
    y = exit_index(measured)
    predicted = {method: exit_index(curves[method]) for method in METHODS}
    dense = predicted["C"]
    common = (y >= 0) & np.logical_and.reduce([idx >= 0 for idx in predicted.values()])
    rows = []
    for method in METHODS:
        m = predicted[method]
        true_positive = (m >= 0) & (y >= 0)
        d = curves[method] - measured
        status = np.select([true_positive, (m >= 0) & (y < 0), (m < 0) & (y >= 0)],
                           ["TP", "FP", "FN"], default="TN")
        for i, rid in enumerate(archived["record_ids"]):
            idx = TASK.targets[masks[method][i]]
            rows.append(dict(
                fold=fold, record_id=rid, composition=archived["composition"][i], method=method,
                query_count=len(idx), queried_indices="|".join(map(str, idx)),
                RMSE_uA=float(np.sqrt(np.mean(d[i] ** 2))), MAE_uA=float(np.mean(np.abs(d[i]))),
                measured_exit_index=int(y[i]) if y[i] >= 0 else None,
                predicted_exit_index=int(m[i]) if m[i] >= 0 else None,
                dense_exit_index=int(dense[i]) if dense[i] >= 0 else None,
                exact_dense_exit=bool(dense[i] >= 0 and dense[i] == m[i]),
                dense_false_positive=bool(dense[i] >= 0 and y[i] < 0),
                event_status=status[i], common_measured_all7=bool(common[i]),
                absolute_exit_error=int(abs(m[i] - y[i])) if true_positive[i] else None,
                signed_exit_error=int(m[i] - y[i]) if true_positive[i] else None))
    return pd.DataFrame(rows)


def aggregate(records, groups):
    summaries = []
    for labels, rows in records.groupby(groups, sort=False):
        values = dict(zip(groups, labels if isinstance(labels, tuple) else (labels,)))
        common = rows[rows.common_measured_all7]
        tp = rows[rows.event_status == "TP"]
        counts = rows.event_status.value_counts()
        med = rows[rows.measured_exit_index.notna() & rows.dense_exit_index.notna() & rows.predicted_exit_index.notna()]
        summaries.append(dict(**values, N=len(rows),
                              mean_queries=rows.query_count.mean(), total_queries=int(rows.query_count.sum()),
                              macro_rmse_uA=rows.RMSE_uA.mean(), macro_mae_uA=rows.MAE_uA.mean(),
                              dense_exits=rows.dense_exit_index.notna().sum(),
                              dense_false_positives=rows.dense_false_positive.sum(),
                              exact_dense_indices=rows.exact_dense_exit.sum(),
                              **{k: int(counts.get(k, 0)) for k in ("TP", "FP", "FN", "TN")},
                              measured_exits=rows.measured_exit_index.notna().sum(),
                              localization_common7_n=len(common), localization_common7_mae_samples=common.absolute_exit_error.mean(),
                              localization_common7_bias_samples=common.signed_exit_error.mean(),
                              localization_matched_n=len(tp), localization_matched_mae_samples=tp.absolute_exit_error.mean(),
                              localization_common_with_dense_n=len(med),
                              localization_common_with_dense_mae_samples=med.absolute_exit_error.mean()))
    return pd.DataFrame(summaries)


def pairwise(records):
    exit_rows = records[records.method == "Exit"].set_index(KEYS)
    assert exit_rows.index.is_unique
    results = []
    for method in ("B", "Uniform", "Spread", "Start", "Bisection", "C"):
        ref = records[records.method == method].set_index(KEYS).loc[exit_rows.index]
        pair = ref.event_status.eq("TP") & exit_rows.event_status.eq("TP")
        for metric, mask, cohort in (
            ("absolute_exit_error", pair, "jointly detected measured exits"),
            ("absolute_exit_error", ref.common_measured_all7, "measured exits detected by all seven"),
            ("RMSE_uA", np.ones(len(ref), dtype=bool), "all development records"),
        ):
            x, a = ref.loc[mask, metric], exit_rows.loc[mask, metric]
            gains = (x - a).to_numpy()
            groups = x.index.get_level_values("composition").to_numpy()
            low, high = bootstrap_paired(gains, groups)
            results.append(dict(reference=method, comparison="Exit", cohort=cohort, metric=metric,
                                n=len(gains), reference_mean=x.mean(), exit_mean=a.mean(),
                                mean_gain=gains.mean(), CI95_low=low, CI95_high=high,
                                improved=(gains > 0).sum(), worsened=(gains < 0).sum(), tied=(gains == 0).sum()))
    return pd.DataFrame(results)
