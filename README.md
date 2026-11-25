# Exit-guided batch querying

Code for the P70 electroforming benchmark in *Exit-guided batch querying for current forecasting in perovskite memristors*.

**Built with PriorLabs-TabPFN.**

The algorithm evaluates 16 fixed anchors, identifies the first candidate plateau exit from the predicted currents, and pools the interior positions of the triggered intervals for one refinement round. It updates those positions with newly queried currents. A round can use multiple model calls when the batch limit is reached.

## Recompute Table 2 and the P70 paired comparisons

Use Python 3.12. Download [dataset.zip from Zenodo 17799630, version 2](https://zenodo.org/records/17799630). Save it at the repository root. Alternatively, pass the folder containing the two original workbooks to `--dataset`.

```sh
python -m pip install -r requirements.txt
python -m forecast.prepare_data --dataset dataset.zip
python -m forecast.recompute
```

The first command after installation prepares the original five development folds from `Figure 3(a).xlsx` and `Figure 3(d).xlsx` using the record IDs and workbook checksums in `protocol.json`. It writes the observed inputs and measured targets to separate files. The second command uses the archived out-of-fold predictions and timing trials in `evidence/` to regenerate:

| Output under `work/summary/` | Contents |
| --- | --- |
| `table2.csv` | Query counts, trajectory errors, event counts, localization and timing |
| `s3c_paired.csv` | Full-precision paired differences and 20,000-draw descriptive intervals |
| `s3_timing_by_fold.csv` | Fold-level query calls and median timing |
| `record_metrics.csv` | Queried indices, exit positions and errors by record and method |
| `timing_summary.json` | Same-run time savings with and without context fitting |

The archived run has 83 held-out records across five folds. At 1127 μA, exit-guided batch querying averages 20.34 queries per record and matches 40 of 40 dense-predicted exit indices, including one dense false positive. Its measured-exit MAE is 11.205 samples on the 39 events detected by all seven query paths. The archived fit-plus-inference times are 41.21 s for exit-guided batch querying, 54.21 s for bisection and 127.63 s for dense querying. Recomputing these timings uses the recorded trials. Use the CUDA path below to obtain new timings.

## Run the live CUDA benchmark

The recorded run used Python 3.12.7, PyTorch 2.11.0+cu128 and TabPFN 2.0.9 on an NVIDIA GeForce RTX 5070 Ti Laptop GPU. Install a CUDA-enabled PyTorch build that matches your hardware. These commands use CUDA 12.8 and install the archived TabPFN source revision via Git.

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-cuda.txt
```

Download the [official regression checkpoint at the frozen revision](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/4972a65a1b30806315c6f92499959ffbfc69a673/tabpfn-v2-regressor.ckpt) and place it at the repository root or pass its local path to `--weights`. The runner verifies its SHA-256 as `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`.

```sh
python -m forecast.run_cuda --weights tabpfn-v2-regressor.ckpt
python -m forecast.recompute --live work/live --output work/live-summary
```

Add `--fold fold0` for a one-fold check. A complete scoring run needs all five fold outputs. Each fold checks the archived feature and label hashes, fits one context, warms all seven paths, then times each path three times in rotating order with CUDA synchronization. Existing sealed fold outputs are checked before reuse. Saved predictions are generated from the training curves, observed prefixes and training-derived voltage program. Measured evaluation targets are loaded by the separate scoring command.

## Repository contents

- `forecast/rules.py` defines the exit and bracket rules.
- `forecast/kernels.py` builds model inputs and queries the anchor and refinement positions.
- `forecast/paths.py` implements the remaining P70 controls and records query calls.
- `forecast/prepare_data.py`, `forecast/run_cuda.py` and `forecast/recompute.py` implement the three reproduction steps.
- `protocol.json` and `evidence/` contain the fixed experiment settings and the P70 reference outputs. `B`, `Exit`, `Start`, `Bisection`, `C`, `Uniform` and `Spread` identify the seven query paths.

The reproduction scope is the primary P70 unified experiment. The other datasets and secondary comparisons are described in the manuscript and its accompanying material.

## Sources

- Kim, S.-Y., Rivera-Sierra, G., Shiferaw Mengesha, B., Iñiguez, B., and Bisquert, J. Electroforming measurement dataset, version 2. [Zenodo 17799630](https://doi.org/10.5281/zenodo.17799630).
- Hollmann, N. et al. Accurate predictions on small data with a tabular foundation model. *Nature* 637, 319–326. [Article](https://doi.org/10.1038/s41586-024-08328-6).
- The [pinned TabPFN source](https://github.com/PriorLabs/TabPFN/tree/9b8f498fef327548c51c17edb747f705fc7dcaac) and [checkpoint license](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/4972a65a1b30806315c6f92499959ffbfc69a673/LICENSE.txt) are maintained upstream.
