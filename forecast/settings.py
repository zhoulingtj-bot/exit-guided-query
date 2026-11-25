"""Portable bindings for the archived P70 protocol."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_config(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


@dataclass(frozen=True)
class Task:
    prefix: int
    anchors: np.ndarray
    targets: np.ndarray
    features: int
    fit_mode: str
    estimators: int
    seed: int
    batch: int
    cpu_threads: int
    weight: Path | None = None

    @property
    def anchor_offsets(self):
        return self.anchors - self.targets[0]


CONFIG = read_config(CONFIG_PATH)
ENGINE = CONFIG["engine"]
TASK = Task(CONFIG["observed_1based"][-1], np.asarray(CONFIG["anchors_1based"]),
            np.asarray(CONFIG["target_1based"]), ENGINE["feature_count"], ENGINE["cache_mode"],
            ENGINE["n_estimators"], ENGINE["seed"], ENGINE["batch_limit_rows"], ENGINE["cpu_threads"])
THRESHOLD = CONFIG["threshold_uA"]
METHODS = ("B", "Exit", "Start", "Bisection", "C", "Uniform", "Spread")
FOLDS = tuple(CONFIG["splits"])
