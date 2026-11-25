"""Recover the fixed P70 records from the two publisher workbooks."""
import argparse
import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from openpyxl import load_workbook

from .settings import CONFIG, FOLDS, TASK


def workbook_bytes(source, name):
    if source.is_dir():
        return (source / name).read_bytes()
    if hashlib.md5(source.read_bytes()).hexdigest() != CONFIG["dataset_zip_md5"]:
        raise ValueError("Use the original Zenodo 17799630 v2 dataset.zip.")
    with ZipFile(source) as archive:
        names = [entry for entry in archive.namelist() if PurePosixPath(entry).name == name]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one {name} in the dataset.")
        return archive.read(names[0])


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_source(source):
    wanted = {rid for item in CONFIG["splits"].values()
              for rid in item["evaluation_record_ids"]}
    curves, composition = {}, {}
    for name, meta in CONFIG["workbooks"].items():
        data = workbook_bytes(source, name)
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise ValueError(f"Workbook content differs from the study source: {name}")
        wb = load_workbook(BytesIO(data), read_only=True, data_only=True)
        try:
            matrix = np.array([[numeric(value) for value in row]
                               for row in wb["Sheet1"].values], dtype=float)
        finally:
            wb.close()
        for rid in sorted(key for key in wanted if key.split("|", 1)[0] == name):
            ordinal = int(rid.rsplit("_", 1)[1])
            channels = matrix[:, 2 * (ordinal - 1):2 * ordinal]
            curve = channels[np.isfinite(channels).all(axis=1)]
            if curve.shape != (500, 2):
                raise ValueError(f"Unexpected dimensions for {rid}: {curve.shape}")
            curves[rid] = curve
            composition[rid] = meta["composition"]
    if len(curves) != 83 or set(curves) != wanted:
        raise ValueError("The workbook and frozen evaluation records do not align.")
    return curves, composition


def build(source, output):
    curves, labels = read_source(source)
    output.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        spec = CONFIG["splits"][fold]
        train_ids, ids = spec["training_record_ids"], spec["evaluation_record_ids"]
        assert set(train_ids).isdisjoint(ids) and set(train_ids + ids) == set(curves)
        train = np.stack([curves[key] for key in train_ids])
        evaluation = np.stack([curves[key] for key in ids])
        groups = [labels[key] for key in ids]
        np.savez_compressed(output / f"{fold}.npz", train_curves_V_A=train, train_ids=train_ids,
                            train_composition=[labels[key] for key in train_ids],
                            observed_prefix_V_A=evaluation[:, :TASK.prefix],
                            record_ids=ids, composition=groups)
        np.savez_compressed(output / f"{fold}_measured.npz", record_ids=ids, composition=groups,
                            sample_index=TASK.targets, measured_uA=evaluation[:, TASK.targets - 1, 1] * 1e6)
        print(f"{fold}: {len(train_ids)} training, {len(ids)} held out")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Zenodo dataset.zip or the directory containing both original workbooks")
    parser.add_argument("--output", type=Path, default=Path("work/data"))
    args = parser.parse_args()
    build(args.dataset, args.output)


if __name__ == "__main__":
    main()
