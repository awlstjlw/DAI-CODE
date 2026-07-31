"""Sample raw states from full Oracle files."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, PROJECT_DIR)

from code.train.train_loop_flex import sample_state_indices

FULL_SCHEMA = "full_raw_v2"
# Per-current-time-query layout.  Each row in `phi` / `V` is one
# independent (state, t) policy query; ``T`` is no longer a sample axis.
SAMPLED_SCHEMA = "sampled_raw"


def load_full_raw(path: str) -> Dict[str, object]:
    """Load and validate one full-raw Oracle file."""
    z = np.load(path, allow_pickle=True)
    required = {
        "states", "setting_idx", "valuations", "matrix0", "matrix1",
        "phi", "V", "group_sizes", "T", "delta", "epsilon",
        "config", "labels", "payment_source", "data_schema",
    }
    missing = sorted(required.difference(z.files))
    if missing:
        raise ValueError(f"{path}: missing full-raw fields {missing}")
    schema = str(z["data_schema"])
    if schema != FULL_SCHEMA:
        raise ValueError(f"{path}: expected {FULL_SCHEMA}, got {schema!r}")

    cfg = json.loads(str(z["config"]))
    states = z["states"]
    setting_idx = z["setting_idx"]
    valuations = z["valuations"]
    matrix0 = z["matrix0"]
    matrix1 = z["matrix1"]
    phi = z["phi"]
    V = z["V"]

    n_settings = int(cfg["num_settings"])
    m = len(cfg["group_sizes"])
    K = int(cfg["num_states"])
    T = int(cfg["T"])
    S = states.shape[0]

    expected_shapes = {
        "states": (S, m * K),
        "setting_idx": (n_settings,),
        "valuations": (n_settings, m, K),
        "matrix0": (n_settings, m, K, K),
        "matrix1": (n_settings, m, K, K),
        "phi": (n_settings, S, T, m),
        "V": (n_settings, S, T),
    }
    arrays = {
        "states": states,
        "setting_idx": setting_idx,
        "valuations": valuations,
        "matrix0": matrix0,
        "matrix1": matrix1,
        "phi": phi,
        "V": V,
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(
                f"{path}: {name} must have shape {expected}, "
                f"got {arrays[name].shape}"
            )
    if not np.array_equal(setting_idx, np.arange(n_settings)):
        raise ValueError(f"{path}: setting_idx must be ordered 0..{n_settings - 1}")

    return {"z": z, "config": cfg, **arrays}


def sample_full_raw(path: str, out_path: str, n_sample: int, seed: int,
                     n_extreme_high: Optional[int] = None,
                     n_extreme_low: Optional[int] = None,
                     n_random: Optional[int] = None,
                     n_settings_sample: Optional[int] = None,
                     settings_seed: int = 12345,
                     n_extreme: Optional[int] = None) -> None:
    """Sample one full-raw file and save raw selected data.

    Args:
        path:               path to full_raw_v2 NPZ.
        out_path:           destination NPZ path.
        n_sample:           number of states per setting to keep.
        seed:               per-setting RNG seed (combined with config seed).
        n_extreme_high:     highest-phi states per setting.
        n_extreme_low:      lowest-phi states per setting.
        n_random:           uniformly sampled states from the remainder.
        n_settings_sample:  if set, randomly sub-select this many settings
                            from the full-raw file using ``settings_seed``.
                            Output rows span the chosen settings in ascending
                            original order.
        settings_seed:      seed used to pick the subset of settings.
        n_extreme:          deprecated alias for high-only sampling.
    """
    data = load_full_raw(path)
    cfg = data["config"]
    phi = data["phi"]
    full_n_settings, states_per_setting = phi.shape[:2]
    if n_sample > states_per_setting:
        raise ValueError(
            f"{path}: n_sample={n_sample} exceeds states_per_setting="
            f"{states_per_setting}"
        )

    if n_settings_sample is None:
        chosen_settings = np.arange(full_n_settings, dtype=np.int32)
    else:
        if n_settings_sample <= 0:
            raise ValueError("n_settings_sample must be positive")
        if n_settings_sample > full_n_settings:
            raise ValueError(
                f"{path}: n_settings_sample={n_settings_sample} exceeds "
                f"full_n_settings={full_n_settings}"
            )
        if n_settings_sample == full_n_settings:
            chosen_settings = np.arange(full_n_settings, dtype=np.int32)
        else:
            rng_settings = np.random.default_rng(int(settings_seed))
            chosen_settings = np.sort(
                rng_settings.choice(full_n_settings, size=n_settings_sample,
                                    replace=False)
            ).astype(np.int32)

    # Per-(setting, state) auxiliaries kept on disk for reproducibility and
    # for downstream code that needs to re-derive per-(state, t) queries.
    source_state_idx = np.empty((chosen_settings.size, n_sample), dtype=np.int32)
    sample_kind = np.empty((chosen_settings.size, n_sample), dtype=np.int8)
    T = int(data["config"]["T"])
    # Out arrays are now *per current-time query* instead of (state, T, m).
    # The first axis is (n_settings * n_sample * T) ordered
    #   ((setting, state, t), (setting, state, t), ...).
    n_query = int(chosen_settings.size) * int(n_sample) * T
    query_setting_idx = np.empty(n_query, dtype=np.int32)
    query_source_state = np.empty(n_query, dtype=np.int32)
    query_t_index = np.empty(n_query, dtype=np.int16)
    phi_sampled = np.empty((n_query, phi.shape[-1]), dtype=np.float32)
    V_sampled = np.empty((n_query,), dtype=np.float32)
    # Per-(setting, state) raw state rows and per-setting constants are kept
    # as well so callers that want to recover raw state coordinates can do so.
    states_per_query = np.empty(
        (n_query, data["states"].shape[-1]), dtype=data["states"].dtype
    )
    valuations_sampled = np.empty(
        (chosen_settings.size,) + data["valuations"].shape[1:],
        dtype=np.float32,
    )
    matrix0_sampled = np.empty(
        (chosen_settings.size,) + data["matrix0"].shape[1:],
        dtype=np.float32,
    )
    matrix1_sampled = np.empty(
        (chosen_settings.size,) + data["matrix1"].shape[1:],
        dtype=np.float32,
    )

    # Config seed + CLI seed makes the result deterministic per source config.
    base_seed = int(cfg.get("seed", 0)) + int(seed)
    if n_extreme is not None:
        if n_extreme_high is not None or n_extreme_low is not None:
            raise ValueError(
                "n_extreme cannot be combined with n_extreme_high/"
                "n_extreme_low"
            )
        n_extreme_high = int(n_extreme)
        n_extreme_low = 0
    if (n_extreme_high is None and n_extreme_low is None
            and n_random is None):
        n_extreme_high = n_sample // 4
        n_extreme_low = n_sample // 4
        n_random = n_sample - n_extreme_high - n_extreme_low
    elif n_extreme_high is None or n_extreme_low is None:
        raise ValueError(
            "n_extreme_high and n_extreme_low must be supplied together"
        )
    elif n_random is None:
        n_random = n_sample - int(n_extreme_high) - int(n_extreme_low)

    n_extreme_high = int(n_extreme_high)
    n_extreme_low = int(n_extreme_low)
    n_random = int(n_random)
    if min(n_extreme_high, n_extreme_low, n_random) < 0:
        raise ValueError(
            "n_extreme_high/n_extreme_low/n_random must be non-negative"
        )
    if n_extreme_high + n_extreme_low + n_random != n_sample:
        raise ValueError(
            "n_extreme_high + n_extreme_low + n_random must equal n_sample"
        )

    # The first axis is (n_settings, n_sample, T) ordered
    #   setting-major, state-major, time-minor — so query index
    #   out_si * (n_sample * T) + s * T + t.  Each (state, t) pair becomes one
    #   independent current-round policy query.
    for out_si, src_si in enumerate(chosen_settings.tolist()):
        rng = np.random.default_rng(base_seed + src_si)
        idx, kind = sample_state_indices(
            phi[src_si], n_sample, rng, return_kind=True,
            n_extreme_high=n_extreme_high,
            n_extreme_low=n_extreme_low,
            n_random=n_random,
        )
        source_state_idx[out_si] = idx
        sample_kind[out_si] = kind
        base = out_si * (n_sample * T)
        s_offset = np.arange(n_sample) * T
        for s_local in range(n_sample):
            row_start = base + s_local * T
            row_end = row_start + T
            query_setting_idx[row_start:row_end] = src_si
            query_source_state[row_start:row_end] = idx[s_local]
            query_t_index[row_start:row_end] = np.arange(T, dtype=np.int16)
            phi_sampled[row_start:row_end] = phi[src_si, idx[s_local]]
            V_sampled[row_start:row_end] = data["V"][src_si, idx[s_local]]
            states_per_query[row_start:row_end] = data["states"][idx[s_local]]
        valuations_sampled[out_si] = data["valuations"][src_si]
        matrix0_sampled[out_si] = data["matrix0"][src_si]
        matrix1_sampled[out_si] = data["matrix1"][src_si]

    if not np.all((sample_kind == 2).sum(axis=1) == n_extreme_high):
        raise RuntimeError("High-phi sample count verification failed")
    if not np.all((sample_kind == 1).sum(axis=1) == n_extreme_low):
        raise RuntimeError("Low-phi sample count verification failed")
    if not np.all((sample_kind == 0).sum(axis=1) == n_random):
        raise RuntimeError("Random sample count verification failed")
    if any(np.unique(row).size != n_sample for row in source_state_idx):
        raise RuntimeError("Duplicate source state indices detected")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        # Raw state grid is retained once.  source_state_idx identifies which
        # rows were selected for each setting (per-setting shape, kept for
        # back-compat with v2 readers).
        states=data["states"],
        source_state_idx=source_state_idx,
        sample_kind=sample_kind,
        setting_idx=chosen_settings,
        # Per-(setting, group, K) constants are kept as well so callers that
        # need to recover per-setting raw valuations / matrices can do so.
        valuations=valuations_sampled,
        matrix0=matrix0_sampled,
        matrix1=matrix1_sampled,
        # Per-(state, t) flat query arrays: each row is one independent
        # current-round policy query (state, t/T) -> phi_t.  No sample
        # internal T axis remains.
        states_per_query=states_per_query,
        query_setting_idx=query_setting_idx,
        query_source_state=query_source_state,
        query_t_index=query_t_index,
        phi=phi_sampled,
        V=V_sampled,
        group_sizes=np.asarray(cfg["group_sizes"], dtype=np.int16),
        T=np.int16(cfg["T"]),
        delta=np.float32(cfg["delta"]),
        epsilon=np.float32(cfg["epsilon"]),
        n_sample=np.int16(n_sample),
        n_extreme=np.int16(n_extreme_high + n_extreme_low),
        n_extreme_high=np.int16(n_extreme_high),
        n_extreme_low=np.int16(n_extreme_low),
        n_random=np.int16(n_random),
        sample_kind_labels=np.asarray(["random", "low_phi", "high_phi"]),
        phi_rank_score="mean_over_time_of_group_0_phi",
        n_settings_sample=np.int32(chosen_settings.size),
        settings_seed=np.int64(settings_seed),
        sampler_seed=np.int64(seed),
        config=json.dumps(cfg),
        labels=np.asarray(["phi", "V"]),
        payment_source="online_simulation_or_oracle_on_demand",
        source_schema=FULL_SCHEMA,
        data_schema=SAMPLED_SCHEMA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dir", default="data_cache/1to9_full_raw")
    parser.add_argument("--out_dir", default="data_cache/1to9_sampled_raw")
    parser.add_argument("--tag", default=None, help="Sample only one config tag")
    parser.add_argument("--n_sample", type=int, default=160,
                        help="Number of states per setting to keep (default: "
                             "160 = 40 high + 40 low + 80 random)")
    parser.add_argument("--n_extreme_high", type=int, default=None,
                        help="Highest-phi states per setting (default: "
                             "n_sample // 4; 40 when n_sample=160)")
    parser.add_argument("--n_extreme_low", type=int, default=None,
                        help="Lowest-phi states per setting (default: "
                             "n_sample // 4; 40 when n_sample=160)")
    parser.add_argument("--n_random", type=int, default=None,
                        help="Uniform states from the remainder (default: "
                             "n_sample - high - low; 80 when n_sample=160)")
    parser.add_argument("--n_extreme", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--n_settings_sample", type=int, default=640,
                        help="Randomly keep this many settings per full-raw "
                             "file (default: 640)")
    parser.add_argument("--settings_seed", type=int, default=12345,
                        help="Seed for the settings sub-sampling step.")
    parser.add_argument("--seed", type=int, default=999,
                        help="Per-setting RNG seed (combined with config seed).")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if args.n_sample <= 0:
        raise ValueError("--n_sample must be positive")
    if args.n_extreme is not None:
        if args.n_extreme_high is not None or args.n_extreme_low is not None:
            raise ValueError(
                "--n_extreme cannot be combined with --n_extreme_high/"
                "--n_extreme_low"
            )
        args.n_extreme_high = args.n_extreme
        args.n_extreme_low = 0
    if (args.n_extreme_high is None and args.n_extreme_low is None
            and args.n_random is None):
        args.n_extreme_high = args.n_sample // 4
        args.n_extreme_low = args.n_sample // 4
        args.n_random = (
            args.n_sample - args.n_extreme_high - args.n_extreme_low
        )
    elif args.n_extreme_high is None or args.n_extreme_low is None:
        raise ValueError(
            "--n_extreme_high and --n_extreme_low must be given together"
        )
    elif args.n_random is None:
        args.n_random = (
            args.n_sample - args.n_extreme_high - args.n_extreme_low
        )
    if min(args.n_extreme_high, args.n_extreme_low, args.n_random) < 0:
        raise ValueError(
            "--n_extreme_high/--n_extreme_low/--n_random must be non-negative"
        )
    if (args.n_extreme_high + args.n_extreme_low + args.n_random
            != args.n_sample):
        raise ValueError(
            f"--n_extreme_high ({args.n_extreme_high}) + "
            f"--n_extreme_low ({args.n_extreme_low}) + "
            f"--n_random ({args.n_random}) must equal "
            f"--n_sample ({args.n_sample})"
        )

    full_dir = Path(args.full_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(full_dir.glob("*.npz"))
    if args.tag is not None:
        files = [path for path in files if path.stem == args.tag]
    if not files:
        raise FileNotFoundError(
            f"No matching full-raw NPZ files found in {full_dir}"
        )

    for path in files:
        out_path = out_dir / path.name
        if args.skip_existing and out_path.exists():
            print(f"[SKIP] {out_path}")
            continue
        print(
            f"Sampling {path.name}: n_sample={args.n_sample} "
            f"n_extreme_high={args.n_extreme_high} "
            f"n_extreme_low={args.n_extreme_low} "
            f"n_random={args.n_random} "
            f"n_settings_sample={args.n_settings_sample}")
        sample_full_raw(
            str(path), str(out_path), args.n_sample, args.seed,
            n_extreme_high=args.n_extreme_high,
            n_extreme_low=args.n_extreme_low,
            n_random=args.n_random,
            n_settings_sample=args.n_settings_sample,
            settings_seed=args.settings_seed,
        )
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  saved {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
