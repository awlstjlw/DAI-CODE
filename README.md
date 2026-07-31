# Anonymous Submission Code

This folder contains the minimal code needed to reproduce the neural-network + simulation pipeline.

## Contents

- `code/` — core pipeline:
  - `oracle/fair_pivot_bi.py` — exact backward-induction oracle
  - `model/backbone_flex.py` — learned group-selection model
  - `train/` — losses, data normalization, training utilities
  - `eval/` — simulation drivers for m=2 and m=3
  - `scripts/` — end-to-end entry points
  - `configs/experiments_1to9.json` — experiment configurations
- `code_dp/` — DP baseline / simulation environment used by the evaluators
- `run_pipeline.sh` / `run_pipeline.bat` — full pipeline
- `run_smoke.sh` / `run_smoke.bat` — quick sanity check (4 settings, 2 epochs, 2 runs)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Quick smoke test

```bash
# Linux / macOS
bash run_smoke.sh

# Windows
run_smoke.bat
```

This creates a tiny `a_eps5` dataset, trains for 2 epochs, evaluates 2 simulation runs, and exports CSVs under `results/smoke/eval/`.

### 3. Full pipeline

```bash
# Linux / macOS
bash run_pipeline.sh

# Windows
run_pipeline.bat
```

1. Oracle data generation (`data_cache/1to9_full_raw/`)
2. State sampling (`data_cache/1to9_sampled_raw/`)
3. Unified shuffle (`data_cache/1to9_unified_shuffled/`)
4. Model training (`results/ckpt/unified_1to9.pth`)
5. Simulation evaluation (`results/eval_all/`)
6. CSV export (`results/csv/sampled_raw/`)

By default group 3 (the m=3 experiment) is skipped because it is much slower; pass `--skip-groups ""` to include it, or `--skip-groups "3,4,5,7,8,9"` to run only group 1.

## Useful options

| Option | Description |
|--------|-------------|
| `--skip-build` | Skip oracle data generation (use existing `data_cache/1to9_full_raw/`) |
| `--skip-sample` | Skip sampling (use existing `data_cache/1to9_sampled_raw/`) |
| `--skip-train` | Skip training (use existing checkpoint) |
| `--only-eval --ckpt PATH` | Run evaluation only using the given checkpoint |
| `--num-runs=N` | Simulation runs per config (default 500) |
| `--epochs=N` | Training epochs (default 50) |
| `--skip-groups=G` | Comma-separated group IDs to skip (default `3`) |

## Notes

- The code does **not** include data, checkpoints, or result CSVs. Running the scripts will regenerate them.
- On CPU the full pipeline is slow; the oracle generation and training steps benefit from a CUDA GPU.
- `PYTHONHASHSEED=0` is set in the scripts for reproducibility.
