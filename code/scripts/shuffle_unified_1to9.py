"""Merge sampled-raw files by (T, m, K) bucket, shuffle once, and save splits."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, PROJECT_DIR)

from code.scripts.train_unified_1to9 import (
    load_sampled_raw,
    split_setting_indices,
    build_split_for_settings,
    pad_m_dim,
    M_MAX,
)


def _shuffle_arrays(arrs: List[np.ndarray], rng: np.random.Generator) -> List[np.ndarray]:
    n = arrs[0].shape[0]
    perm = rng.permutation(n)
    return [a[perm] for a in arrs]


def build_unified_shuffled(data_dir: str, out_dir: str,
                            split_seed: int = 123456, shuffle_seed: int = 1234567,
                            skip_existing: bool = False) -> None:
    """Merge all sampled-raw files by (T, m, K) and save globally shuffled splits."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Bucket key: (T, m, K) -> split -> list of padded numpy tuples
    buckets: Dict[Tuple[int, int, int], Dict[str, List[Tuple]]] = {}

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
    print(f"Found {len(files)} .npz files in {data_dir}")

    for fname in files:
        path = os.path.join(data_dir, fname)
        tag = fname[:-4]
        print(f"\nLoading {fname} (tag={tag}) ...")
        file_data = load_sampled_raw(path)
        cfg = file_data["config"]
        m = len(cfg["group_sizes"])
        K = int(cfg["num_states"])
        T = file_data["T"]
        n_settings = int(file_data["n_settings"])

        split_seed_for_cfg = int(split_seed) + int(cfg.get("seed", 0))
        train_settings, val_settings, test_settings = split_setting_indices(
            n_settings, split_seed_for_cfg
        )
        split_settings = {
            "train": train_settings,
            "val": val_settings,
            "test": test_settings,
        }

        key = (T, m, K)
        if key not in buckets:
            buckets[key] = {"train": [], "val": [], "test": []}

        for split_name, setting_rows in split_settings.items():
            split_tup = build_split_for_settings(file_data, setting_rows)
            s, t, v, tr, scale, phi, V = split_tup
            if s.shape[0] == 0:
                print(f"  [WARN] {tag}: empty {split_name} split")
                continue
            s_pad, t_p, v_pad, tr_pad, scale_pad, phi_pad, mask = pad_m_dim(
                s, t, v, tr, scale, phi, m
            )
            buckets[key][split_name].append(
                (s_pad, t_p, v_pad, tr_pad, scale_pad, phi_pad, V, mask)
            )

        print(
            f"  tag={tag}: settings train/val/test="
            f"{len(train_settings)}/{len(val_settings)}/{len(test_settings)}; "
            f"m={m} K={K} T={T}"
        )

    # Stream 0 is the global bucket-shuffle stream; training reserves the
    # remaining child streams for subsample/chunk/per-epoch operations.
    bucket_shuffle_stream = np.random.SeedSequence(shuffle_seed).spawn(5)[0]
    rng = np.random.default_rng(bucket_shuffle_stream)

    print(f"\nSaving unified shuffled data to {out_dir} ...")
    for key in sorted(buckets):
        T, m, K = key
        for split_name in ["train", "val", "test"]:
            parts = buckets[key][split_name]
            if not parts:
                print(f"  [WARN] bucket {key} {split_name}: empty; skipped")
                continue

            concat = []
            for arrs in zip(*parts):
                concat.append(np.concatenate(arrs, axis=0))
            s, t, v, tr, scale, phi, V, mask = concat

            if split_name == "train":
                concat = _shuffle_arrays([s, t, v, tr, scale, phi, V, mask], rng)
                s, t, v, tr, scale, phi, V, mask = concat

            save_path = out_path / f"{split_name}_T{T}_m{m}_K{K}.npz"
            if skip_existing and save_path.exists():
                print(f"  [SKIP] {save_path.name}")
                continue
            np.savez(
                save_path,
                s=s.astype(np.float32),
                t=t.astype(np.float32),
                v=v.astype(np.float32),
                tr=tr.astype(np.float32),
                scale=scale.astype(np.float32),
                phi=phi.astype(np.float32),
                V=V.astype(np.float32),
                mask=mask.astype(bool),
            )
            print(
                f"  saved {save_path.name}: {s.shape[0]} rows, "
                f"shape s={s.shape}, T={T}, m={m}, K={K}"
            )

    print("Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data_cache/1to9_sampled_raw")
    p.add_argument("--out_dir", type=str, default="data_cache/1to9_unified_shuffled")
    p.add_argument("--split_seed", type=int, default=123456,
                   help="Dedicated seed for the 8:1:1 setting-level split")
    p.add_argument("--shuffle_seed", type=int, default=1234567,
                   help="Root seed whose first child stream shuffles buckets")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip generating unified files that already exist")
    args = p.parse_args()
    build_unified_shuffled(
        args.data_dir, args.out_dir,
        split_seed=args.split_seed,
        shuffle_seed=args.shuffle_seed,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
