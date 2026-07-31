# Fair Dynamic Auctions via Equity-Constrained Pivot Mechanisms

We study repeated allocation of a scarce indivisible resource in dynamic auctions where strategic agents' private valuations evolve as a Markov process and the population is partitioned into exogenous groups. Although the dynamic pivot mechanism is efficient and satisfies periodic ex-post incentive compatibility and individual rationality, it can create persistent inter-group disparities when valuation processes differ structurally. We introduce \emph{$\epsilon$-periodic ex-post group fairness} ($\epsilon$-GF) to bound gaps in groups' expected discounted social welfare from every time step onward, and propose a fairness-aware mechanism that decouples equity control from truthful allocation: a state-dependent policy randomizes group selection to satisfy $\epsilon$-GF, while an adapted pivot rule runs within the selected group. To limit manipulation of the equity controller, group selection follows a report-independent virtual-state trajectory. We provide an exact dynamic-programming solution for the equity-constrained policy and a scalable neural approximation, $\mathsf{FairPivotNet}$. Experiments show that our approach attains near-target fairness with modest welfare and revenue loss relative to the unconstrained pivot benchmark.

This project is an implementation of the research paper. It includes the **Pivot (Pivot)** mechanism, the **FairPivotBI** method and the **FairPivotNet** method (an adapted dynamic pivot mechanism).

---

## File Structure

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

## Environment Setup & Installation

To run this project, you need a Python environment (3.11.9 recommended) with PyTorch and standard scientific computing libraries.

## Experimental Setup & Hyperparameters

### Neural Network Configuration

Our neural network employs a Transformer-based backbone with per-group encoding and multi-head self-attention. Key hyperparameters are configured as follows:

| Hyperparameter | Value |
| :--- | :--- |
| **Architecture** | Transformer (2-layer, 4-head self-attention) |
| **Hidden Dimension** | 128 |
| **Total Epochs** | 50 |
| **Batch Size** | 256 |
| **Optimizer** | AdamW |
| **Base Learning Rate** | $5 \times 10^{-4}$ |
| **Weight Decay** | $1 \times 10^{-3}$ |
| **Learning Rate Scheduler** | Cosine Annealing |
| **Loss Function** | MSE (Mean Squared Error) |
| **Gradient Clipping** | Max Norm 1.0 |
| **Dropout Rate** | 0.1 |
| **Early Stopping** | Patience 15, Min Delta $1 \times 10^{-5}$ |

*Note: Only the model weights with the lowest validation loss are saved during training to ensure optimal performance.*

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
