"""Generate full, raw Oracle data for 1-9 experiments."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, PROJECT_DIR)

from code.oracle.fair_pivot_bi import fair_pivot_bi


def generate_matrix(K: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Generate one KxK transition matrix according to mode."""
    if mode == "self":
        # p11, p22 in [0.7, 1.0]; off-diagonal = 1 - diagonal
        if K != 2:
            raise NotImplementedError("matrix_mode='self' only implemented for K=2")
        p11 = rng.uniform(0.7, 1.0)
        p22 = rng.uniform(0.7, 1.0)
        M = np.array([[p11, 1.0 - p11],
                      [1.0 - p22, p22]], dtype=np.float64)
        return M
    elif mode == "sym":
        # p00 = p11 = x; p01 = p10 = 1 - x
        if K != 2:
            raise NotImplementedError("matrix_mode='sym' only implemented for K=2")
        x = rng.uniform(0.0, 1.0)
        M = np.array([[x, 1.0 - x],
                      [1.0 - x, x]], dtype=np.float64)
        return M
    elif mode == "random":
        M = rng.random((K, K))
        M /= M.sum(axis=1, keepdims=True)
        return M.astype(np.float64)
    else:
        raise ValueError(f"Unknown matrix_mode: {mode}")


def build_one_setting(args) -> Tuple[np.ndarray, ...]:
    """Worker: return one setting's unnormalised Oracle inputs and labels."""
    si, config, seed_of_matrix, seed_of_value = args

    # Re-create the same RNGs that generate_market_setting would use.
    mr = np.random.default_rng(int(seed_of_matrix))
    vr = np.random.default_rng(int(seed_of_value))

    m = len(config["group_sizes"])
    K = config["num_states"]
    mode = config.get("matrix_mode", "random")

    draw_kind = config.get("valuation_draw", "uniform")
    valuation_ranges = config["valuation_ranges"]
    if len(valuation_ranges) < m:
        valuation_ranges = [valuation_ranges[k % len(valuation_ranges)] for k in range(m)]
    valuations = []
    for k in range(m):
        low, high = valuation_ranges[k]
        if draw_kind == "normal":
            # Match the original 1-9 truncated-normal generator: ``high`` is
            # exclusive, so [360, 401) has an actual maximum of 400.
            mean = (low + high - 1) / 2.0
            std = (high - low) / 6.0
            vals = np.unique(
                np.clip(vr.normal(mean, std, K * 3), low, high - 1).astype(int)
            )
            while len(vals) < K:
                supplement = vr.integers(low, high, size=K)
                vals = np.unique(np.concatenate([vals, supplement]))
            vals = np.sort(vals[:K]).astype(np.float64)
        else:
            vals = np.sort(
                vr.choice(range(low, high), K, replace=False).astype(np.float64)
            )
        valuations.append(vals)
    valuations = np.vstack(valuations)

    matrix0_list = []
    matrix1_list = []
    for _ in range(m):
        matrix0_list.append(generate_matrix(K, mode, mr))
        matrix1_list.append(generate_matrix(K, mode, mr))

    oracle = fair_pivot_bi(
        group_sizes=config["group_sizes"],
        num_states=config["num_states"],
        T=config["T"],
        delta=config["delta"],
        epsilon=config["epsilon"],
        valuations=valuations,
        matrix0=matrix0_list,
        matrix1=matrix1_list,
        # Training learns phi only.  Exact VCG payments remain available from
        # the Oracle's default mode and from the online auction simulation.
        compute_payment=False,
    )
    return (
        si,
        oracle["states"].astype(np.int16, copy=False),          # (S, m*K)
        valuations.astype(np.float32, copy=False),               # (m, K)
        np.asarray(matrix0_list, dtype=np.float32),               # (m, K, K)
        np.asarray(matrix1_list, dtype=np.float32),               # (m, K, K)
        oracle["phi"].astype(np.float32, copy=False),            # (S, T, m)
        oracle["V"].astype(np.float32, copy=False),              # (S, T)
    )


def _save_setting_checkpoint(ckpt_dir: str, item) -> None:
    """Persist one payment-free training setting for future resume."""
    si = int(item[0])
    tmp = os.path.join(ckpt_dir, f".tmp_done_{si}.npz")
    final = os.path.join(ckpt_dir, f"done_{si}.npz")
    np.savez(
        tmp,
        states=item[1],
        valuations=item[2],
        matrix0=item[3],
        matrix1=item[4],
        phi=item[5],
        V=item[6],
        data_schema="setting_raw_v2",
    )
    os.replace(tmp, final)


def _load_done_indices(ckpt_dir: str) -> set:
    """Return the set of setting indices that already have a checkpoint."""
    done = set()
    if not os.path.isdir(ckpt_dir):
        return done
    for name in os.listdir(ckpt_dir):
        if name.startswith("done_") and name.endswith(".npz"):
            try:
                idx = int(name[len("done_"):-len(".npz")])
                done.add(idx)
            except ValueError:
                continue
    return done


def _reassemble_from_checkpoints(ckpt_dir: str, n_settings: int,
                                 expected_state_grid_shape) -> Dict[str, np.ndarray]:
    """Combine per-setting checkpoint NPZs back into the canonical schema."""
    items = []
    for si in range(n_settings):
        path = os.path.join(ckpt_dir, f"done_{si}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint for setting {si} missing in {ckpt_dir}")
        z = np.load(path)
        if str(z["data_schema"]) != "setting_raw_v2":
            raise ValueError(f"Stale checkpoint schema in {path}")
        items.append((si, z["states"], z["valuations"], z["matrix0"],
                      z["matrix1"], z["phi"], z["V"]))
    state_grid = items[0][1]
    if state_grid.shape != expected_state_grid_shape:
        raise ValueError(
            f"State grid shape mismatch: expected {expected_state_grid_shape},"
            f" got {state_grid.shape}")
    for item in items[1:]:
        if not np.array_equal(item[1], state_grid):
            raise ValueError("State grids differ across settings in one config")
    return {
        "states": state_grid,
        "setting_idx": np.asarray([item[0] for item in items], dtype=np.int32),
        "valuations": np.stack([item[2] for item in items], axis=0),
        "matrix0":    np.stack([item[3] for item in items], axis=0),
        "matrix1":    np.stack([item[4] for item in items], axis=0),
        "phi":        np.stack([item[5] for item in items], axis=0),
        "V":          np.stack([item[6] for item in items], axis=0),
    }


def build_dataset_for_config(config: Dict, n_jobs: int = 1,
                             ckpt_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Generate a compact full-raw dataset for one experiment config,
    resuming from per-setting checkpoints when ckpt_dir is provided."""
    n_settings = config["num_settings"]

    # Generate per-setting seeds up front using mother seed (matches 1-9)
    mother_rng = np.random.default_rng(config["seed"])
    seeds = []
    for si in range(n_settings):
        seed_of_matrix = mother_rng.integers(0, 2**32 - 1, dtype=np.uint32)
        seed_of_value = mother_rng.integers(0, 2**32 - 1, dtype=np.uint32)
        seeds.append((si, config, seed_of_matrix, seed_of_value))

    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        done = _load_done_indices(ckpt_dir)
        if done:
            print(f"    resume: skipping {len(done)} already-done setting(s)",
                  flush=True)
    else:
        done = set()

    report_every = max(1, min(n_settings // 20, 32))
    completed_count = len(done)
    progress_target = n_settings

    def _consume(item):
        nonlocal completed_count
        completed_count += 1
        if ckpt_dir is not None:
            _save_setting_checkpoint(ckpt_dir, item)
        if completed_count % report_every == 0 or completed_count == progress_target:
            print(f"    progress: {completed_count}/{n_settings} settings",
                  flush=True)

    pending = [a for a in seeds if a[0] not in done]
    if not pending:
        print(f"    progress: {n_settings}/{n_settings} settings (all resumed)",
              flush=True)
    elif n_jobs == 1:
        for a in pending:
            item = build_one_setting(a)
            _consume(item)
    else:
        from multiprocessing import Pool
        # ``imap_unordered`` preserves worker utilisation but does not give
        # us the original setting index ordering.  When checkpointing is
        # enabled we still want to flush per setting, which ``imap`` would
        # do but ``imap_unordered`` does not; ``imap`` is fine here because
        # ``ckpt_dir`` removes the need to keep an in-memory list.
        with Pool(processes=n_jobs) as pool:
            for item in pool.imap(build_one_setting, pending, chunksize=1):
                _consume(item)

    if ckpt_dir is None:
        # Fallback: behave exactly like the old in-memory path.
        items = []
        for a in pending:
            items.append(build_one_setting(a))
        items.sort(key=lambda item: item[0])
        state_grid = items[0][1]
        for item in items[1:]:
            if not np.array_equal(item[1], state_grid):
                raise ValueError(
                    "State grids differ across settings in one config")
        return {
            "states": state_grid,
            "setting_idx": np.asarray([item[0] for item in items], dtype=np.int32),
            "valuations": np.stack([item[2] for item in items], axis=0),
            "matrix0":    np.stack([item[3] for item in items], axis=0),
            "matrix1":    np.stack([item[4] for item in items], axis=0),
            "phi":        np.stack([item[5] for item in items], axis=0),
            "V":          np.stack([item[6] for item in items], axis=0),
        }

    # With checkpoints enabled, rebuild the final dictionary from disk.  We
    # may have just produced all the checkpoints (so ``done`` is still the
    # initial empty set); always re-scan the directory to learn the state
    # grid shape.
    final_done = _load_done_indices(ckpt_dir)
    if not final_done:
        raise RuntimeError(
            f"No checkpoints found under {ckpt_dir} after a run that was "
            f"supposed to checkpoint every setting")
    sample_path = os.path.join(ckpt_dir, f"done_{next(iter(final_done))}.npz")
    sample_states = np.load(sample_path)["states"]
    expected_shape = sample_states.shape
    return _reassemble_from_checkpoints(ckpt_dir, n_settings, expected_shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_json", type=str, default="code/configs/experiments_1to9.json")
    p.add_argument("--tag", type=str, default=None, help="Generate only one config by tag")
    p.add_argument("--start_idx", type=int, default=None, help="Start from this config index (inclusive)")
    p.add_argument("--end_idx", type=int, default=None, help="Stop before this config index (exclusive)")
    p.add_argument("--reverse", action="store_true", help="Process configs in reverse order")
    p.add_argument("--out_dir", type=str, default="data_cache/1to9_full_raw")
    p.add_argument("--n_jobs", type=int, default=1, help="Parallel processes per config")
    p.add_argument("--skip_existing", action="store_true", help="Skip if output .npz exists")
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Resume from v2 payment-free per-setting checkpoints (default)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Ignore any per-setting checkpoints and rerun from scratch")
    args = p.parse_args()

    with open(args.config_json, "r", encoding="utf-8") as f:
        all_configs = json.load(f)["configs"]

    if args.reverse:
        all_configs = list(reversed(all_configs))

    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx or 0
        end = args.end_idx or len(all_configs)
        all_configs = all_configs[start:end]

    if args.tag:
        all_configs = [c for c in all_configs if c["tag"] == args.tag]
        if not all_configs:
            raise ValueError(f"Tag {args.tag} not found")

    os.makedirs(args.out_dir, exist_ok=True)

    import csv
    summary_path = os.path.join(args.out_dir, "_generation_summary.csv")
    summary_fields = ["tag", "group_sizes", "num_states", "T", "delta", "epsilon",
                      "matrix_mode", "num_settings", "seed", "states_per_setting",
                      "total_states", "file_size_mb", "time_s", "status"]
    summary_rows = []

    for cfg in all_configs:
        tag = cfg["tag"]
        out_path = os.path.join(args.out_dir, f"{tag}.npz")

        if args.skip_existing and os.path.exists(out_path):
            print(f"[SKIP] {tag}: {out_path} exists")
            summary_rows.append({
                "tag": tag, "group_sizes": cfg["group_sizes"], "num_states": cfg["num_states"],
                "T": cfg["T"], "delta": cfg["delta"], "epsilon": cfg["epsilon"],
                "matrix_mode": cfg.get("matrix_mode", "random"),
                "num_settings": cfg["num_settings"], "seed": cfg["seed"],
                "states_per_setting": "", "total_states": "", "file_size_mb": "",
                "time_s": "", "status": "skipped"
            })
            continue

        print(f"\n{'='*60}")
        print(f"Generating: {tag}")
        print(f"  group_sizes={cfg['group_sizes']} K={cfg['num_states']} T={cfg['T']} "
              f"delta={cfg['delta']} epsilon={cfg['epsilon']} matrix_mode={cfg.get('matrix_mode','random')}")
        print(f"  settings={cfg['num_settings']} raw_full=True seed={cfg['seed']}")
        sys.stdout.flush()

        t0 = time.time()
        try:
            # New root intentionally prevents reuse of pre-fix/payment-bearing
            # checkpoints produced by the v1 data pipeline.
            ckpt_dir = os.path.join(args.out_dir, "_checkpoints_v2", tag) if args.resume else None
            if not args.resume and ckpt_dir and os.path.isdir(ckpt_dir):
                import shutil
                print(f"    no-resume: deleting existing checkpoints at {ckpt_dir}")
                shutil.rmtree(ckpt_dir)
            # Auto-cap worker count for m=3 (LP linprog is memory+cache heavy)
            cfg_m = len(cfg["group_sizes"])
            cfg_n_jobs = args.n_jobs
            if cfg_m >= 3 and cfg_n_jobs > 4:
                print(f"    auto-cap: m={cfg_m} -> n_jobs {cfg_n_jobs} -> 4 (LP linprog memory/cache safety)")
                cfg_n_jobs = 4
            raw = build_dataset_for_config(cfg, n_jobs=cfg_n_jobs, ckpt_dir=ckpt_dir)
            elapsed = time.time() - t0

            np.savez(
                out_path,
                **raw,
                group_sizes=np.asarray(cfg["group_sizes"], dtype=np.int16),
                T=np.int16(cfg["T"]),
                delta=np.float32(cfg["delta"]),
                epsilon=np.float32(cfg["epsilon"]),
                config=json.dumps(cfg),
                labels=np.asarray(["phi", "V"]),
                payment_source="online_simulation_or_oracle_on_demand",
                data_schema="full_raw_v2",
            )

            states_per_setting = raw["states"].shape[0]
            total_states = states_per_setting * cfg["num_settings"]
            file_size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"  Saved {out_path} ({file_size_mb:.1f} MB)")
            print(f"  settings={cfg['num_settings']} states/setting={states_per_setting} "
                  f"total_states={total_states} time={elapsed:.1f}s")
            sys.stdout.flush()

            summary_rows.append({
                "tag": tag, "group_sizes": cfg["group_sizes"], "num_states": cfg["num_states"],
                "T": cfg["T"], "delta": cfg["delta"], "epsilon": cfg["epsilon"],
                "matrix_mode": cfg.get("matrix_mode", "random"),
                "num_settings": cfg["num_settings"], "seed": cfg["seed"],
                "states_per_setting": states_per_setting, "total_states": total_states,
                "file_size_mb": f"{file_size_mb:.1f}", "time_s": f"{elapsed:.1f}",
                "status": "done"
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            sys.stdout.flush()
            summary_rows.append({
                "tag": tag, "group_sizes": cfg["group_sizes"], "num_states": cfg["num_states"],
                "T": cfg["T"], "delta": cfg["delta"], "epsilon": cfg["epsilon"],
                "matrix_mode": cfg.get("matrix_mode", "random"),
                "num_settings": cfg["num_settings"], "seed": cfg["seed"],
                "states_per_setting": "", "total_states": "", "file_size_mb": "",
                "time_s": f"{elapsed:.1f}", "status": f"error: {e}"
            })

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
