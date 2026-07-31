"""Export sampled-raw NPZ files to CSV summaries and previews."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, PROJECT_DIR)


def _summary_rows(tag: str, data: dict, n_settings: int, n_sample: int) -> list[dict]:
    """One row per (setting, source_state_idx) with compact aggregates."""
    phi = data["phi"]            # (n_settings, n_sample, T, m)
    V = data["V"]                # (n_settings, n_sample, T)
    sample_kind = data["sample_kind"]
    source_idx = data["source_state_idx"]
    T = int(data["T"])
    m = phi.shape[-1]

    phi_max_t = phi.max(axis=2)               # (n_settings, n_sample, m)
    phi_mean_t = phi.mean(axis=2)             # (n_settings, n_sample, m)
    V_max = V.max(axis=2)                     # (n_settings, n_sample)
    V_mean = V.mean(axis=2)

    rows = []
    for si in range(n_settings):
        for k in range(n_sample):
            row = {
                "tag": tag,
                "setting_idx": si,
                "source_state_idx": int(source_idx[si, k]),
                # 2=high phi, 1=low phi, 0=random.
                "sample_kind": int(sample_kind[si, k]),
                "n_phi_groups": m,
                "n_timesteps": T,
                "phi_max_per_group": json.dumps([float(x) for x in phi_max_t[si, k].tolist()]),
                "phi_mean_per_group": json.dumps([float(x) for x in phi_mean_t[si, k].tolist()]),
                "V_max": float(V_max[si, k]),
                "V_mean": float(V_mean[si, k]),
            }
            rows.append(row)
    return rows


def _preview_rows(tag: str, data: dict, states: np.ndarray,
                  n_settings: int, n_sample: int,
                  preview_settings: int) -> list[dict]:
    """One row per (setting, source_state_idx, t) with raw state vector."""
    phi = data["phi"]            # (n_settings, n_sample, T, m)
    V = data["V"]                # (n_settings, n_sample, T)
    sample_kind = data["sample_kind"]
    source_idx = data["source_state_idx"]
    m = phi.shape[-1]
    T = int(data["T"])

    rows = []
    n_to_emit = min(n_settings, preview_settings)
    for si in range(n_to_emit):
        for k in range(n_sample):
            src = int(source_idx[si, k])
            state_vec = states[src].tolist()  # length m*K (each group has K counts)
            state_str = json.dumps([int(x) for x in state_vec])
            for t in range(T):
                row = {
                    "tag": tag,
                    "setting_idx": si,
                    "source_state_idx": src,
                    "sample_kind": int(sample_kind[si, k]),
                    "t": t,
                    "state_vector": state_str,
                    "V": float(V[si, k, t]),
                }
                row["phi_per_group"] = json.dumps([float(x) for x in phi[si, k, t].tolist()])
                rows.append(row)
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_one(npz_path: Path, out_dir: Path, preview_settings: int) -> None:
    z = np.load(npz_path, allow_pickle=True)
    if "data_schema" not in z.files or str(z["data_schema"]) not in (
        "sampled_raw_v2",
        "sampled_raw",
    ):
        raise ValueError(f"{npz_path}: not a sampled_raw file")
    n_settings = int(z["phi"].shape[0])
    n_sample = int(z["n_sample"])
    tag = npz_path.stem
    states = z["states"]
    # Restrict to the subset of fields actually consumed by the writers so
    # we don't accidentally pickle scalar 0-d arrays from ``z.files``.
    data = {k: z[k] for k in
            ("phi", "V", "sample_kind", "source_state_idx", "T")}

    print(f"[csv] {tag}: n_settings={n_settings} n_sample={n_sample}")

    summary = _summary_rows(tag, data, n_settings, n_sample)
    _write_csv(
        out_dir / f"{tag}_summary.csv",
        list(summary[0].keys()),
        summary,
    )

    preview = _preview_rows(tag, data, states, n_settings, n_sample, preview_settings)
    _write_csv(
        out_dir / f"{tag}_preview.csv",
        list(preview[0].keys()),
        preview,
    )

    print(f"    summary rows = {len(summary)}  preview rows = {len(preview)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="data_cache/1to9_sampled_raw")
    p.add_argument("--out_dir", default="results/csv/sampled_raw")
    p.add_argument("--preview_settings", type=int, default=32,
                   help="Maximum number of settings included in *_preview.csv")
    p.add_argument("--tag", default=None, help="Process only one config tag")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    files = sorted(in_dir.glob("*.npz"))
    if args.tag is not None:
        files = [p for p in files if p.stem == args.tag]
    if not files:
        raise FileNotFoundError(f"No sampled-raw NPZ found under {in_dir}")

    for path in files:
        process_one(path, out_dir, args.preview_settings)
    print(f"\nWrote CSVs under {out_dir}")


if __name__ == "__main__":
    main()