"""Train a single unified model across multiple experiment settings."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, PROJECT_DIR)

from model.backbone_flex import FlexibleFairPivotNet
from train.losses import fair_pivot_loss
from train.train_loop_flex import (
    normalize_query_samples,
    normalize_raw_samples,
)

M_MAX = 4
K_MAX = 4
TRANS_DIM = 2 * K_MAX * K_MAX
EXPECTED_SAMPLED_SCHEMA = "sampled_raw"


def pad_m_dim(s, t, v, tr, scale, phi, m_actual):
    """Pad the group axis to M_MAX."""
    N, _, K = s.shape
    pad = M_MAX - m_actual
    if pad < 0:
        raise ValueError(f"m_actual={m_actual} exceeds M_MAX={M_MAX}")

    s_pad = np.pad(s, ((0, 0), (0, pad), (0, 0)))
    v_pad = np.pad(v, ((0, 0), (0, pad), (0, 0)))
    tr_pad = np.pad(tr, ((0, 0), (0, pad), (0, 0)))
    scale_pad = np.pad(scale, ((0, 0), (0, pad), (0, 0)))
    phi_pad = np.pad(phi, ((0, 0), (0, pad)))

    mask = np.zeros((N, M_MAX), dtype=bool)
    mask[:, :m_actual] = True

    return s_pad, t, v_pad, tr_pad, scale_pad, phi_pad, mask


def load_sampled_raw(path: str) -> Dict:
    """Load and validate one sampled-raw NPZ file."""
    z = np.load(path, allow_pickle=True)
    required = {
        "states", "source_state_idx", "sample_kind", "setting_idx",
        "valuations", "matrix0", "matrix1", "phi", "V",
        "query_setting_idx", "query_source_state", "query_t_index",
        "group_sizes", "T", "delta", "epsilon",
        "n_sample", "sampler_seed", "config", "source_schema", "data_schema",
    }
    missing = sorted(required.difference(z.files))
    if missing:
        raise ValueError(f"{path}: missing sampled-raw fields {missing}")
    schema = str(z["data_schema"])
    if schema != EXPECTED_SAMPLED_SCHEMA:
        raise ValueError(
            f"{path}: expected {EXPECTED_SAMPLED_SCHEMA}, got {schema!r}"
        )

    cfg = json.loads(str(z["config"]))
    n_query = int(z["phi"].shape[0])
    if z["V"].shape != (n_query,):
        raise ValueError(
            f"{path}: V must have shape ({n_query},), got {z['V'].shape}"
        )
    if z["query_setting_idx"].shape != (n_query,):
        raise ValueError(
            f"{path}: query_setting_idx must have shape ({n_query},), "
            f"got {z['query_setting_idx'].shape}"
        )
    n_settings = int(np.unique(np.asarray(z["query_setting_idx"])).size)
    n_sample = int(z["n_sample"])
    setting_idx = np.asarray(z["setting_idx"])
    if setting_idx.shape != (n_settings,):
        raise ValueError(
            f"{path}: setting_idx must have shape ({n_settings},), "
            f"got {setting_idx.shape}"
        )
    if np.unique(setting_idx).size != n_settings:
        raise ValueError(f"{path}: setting_idx contains duplicate source settings")

    source_state_idx = np.asarray(z["source_state_idx"])  # (n_settings, n_sample)
    if source_state_idx.shape != (n_settings, n_sample):
        raise ValueError(
            f"{path}: source_state_idx must have shape "
            f"({n_settings}, {n_sample}), got {source_state_idx.shape}"
        )

    return {
        "config": cfg,
        "states": z["states"],
        "source_state_idx": source_state_idx,
        "sample_kind": np.asarray(z["sample_kind"]),
        "setting_idx": setting_idx,
        "n_settings": n_settings,
        "valuations": np.asarray(z["valuations"]),
        "matrix0": np.asarray(z["matrix0"]),
        "matrix1": np.asarray(z["matrix1"]),
        "phi": np.asarray(z["phi"]),                       # (N_query, m)
        "V": np.asarray(z["V"]),                            # (N_query,)
        "query_setting_idx": np.asarray(z["query_setting_idx"]),
        "query_source_state": np.asarray(z["query_source_state"]),
        "query_t_index": np.asarray(z["query_t_index"]),
        "states_per_query": (
            np.asarray(z["states_per_query"])
            if "states_per_query" in z.files else None
        ),
        "group_sizes": np.asarray(cfg["group_sizes"]),
        "T": int(cfg["T"]),
        "delta": float(cfg["delta"]),
        "epsilon": float(cfg["epsilon"]),
        "n_sample": n_sample,
        "sampler_seed": int(z["sampler_seed"]),
    }


def build_split_for_settings(
    file_data: Dict,
    settings: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Materialise query rows for requested settings and normalise."""
    if settings.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return tuple(empty for _ in range(7))

    cfg = file_data["config"]
    m = len(cfg["group_sizes"])
    K = int(cfg["num_states"])
    T = file_data["T"]
    delta = file_data["delta"]
    epsilon = file_data["epsilon"]
    states_grid = file_data["states"]
    valuations = file_data["valuations"]
    matrix0 = file_data["matrix0"]
    matrix1 = file_data["matrix1"]

    query_setting_idx = np.asarray(file_data["query_setting_idx"])
    query_source_state = np.asarray(file_data["query_source_state"])
    phi_all = np.asarray(file_data["phi"]).astype(np.float32, copy=False)
    V_all = np.asarray(file_data["V"]).astype(np.float32, copy=False)

    keep = np.isin(query_setting_idx, settings)
    if not keep.any():
        empty = np.zeros((0,), dtype=np.float32)
        return tuple(empty for _ in range(7))
    phi_all = phi_all[keep]
    V_all = V_all[keep]
    sel_source = query_source_state[keep]
    sel_settings = query_setting_idx[keep]
    if "states_per_query" in file_data:
        states_all = np.asarray(file_data["states_per_query"])[keep]
    else:
        states_all = states_grid[sel_source]
    val_all = valuations[sel_settings]
    mat0_all = matrix0[sel_settings]
    mat1_all = matrix1[sel_settings]

    t_indices = np.asarray(file_data["query_t_index"])[keep]
    s_feat, t_feat, v_feat, tr_feat, scale_feat = normalize_query_samples(
        states_all.astype(np.float64, copy=False),
        val_all,
        mat0_all,
        mat1_all,
        np.asarray(cfg["group_sizes"]),
        delta,
        epsilon,
        t_indices,
    )
    phi_flat = phi_all
    V_flat = V_all
    return (s_feat, t_feat, v_feat, tr_feat, scale_feat, phi_flat, V_flat)


def split_setting_indices(n_settings: int, seed: int) -> Tuple[np.ndarray, ...]:
    """Split settings into train/val/test at 8:1:1 without leakage."""
    if n_settings < 3:
        raise ValueError(
            f"At least 3 settings are required for train/val/test, got {n_settings}"
        )
    n_val = max(1, n_settings // 10)
    n_test = max(1, n_settings // 10)
    n_train = n_settings - n_val - n_test
    order = np.random.default_rng(int(seed)).permutation(n_settings)
    train_settings = np.sort(order[:n_train])
    val_settings = np.sort(order[n_train:n_train + n_val])
    test_settings = np.sort(order[n_train + n_val:])
    return train_settings, val_settings, test_settings


def tensor_dataset_from_split(
    split_tup: Tuple[np.ndarray, ...],
    m: int,
    tag: str,
    split_name: str,
) -> Optional[TensorDataset]:
    """Pad a materialised split; return None if empty."""
    s, t, v, tr, scale, phi, V = split_tup
    if s.shape[0] == 0:
        print(f"  [WARN] {tag}: empty {split_name} split")
        return None
    (s_pad, t_p, v_pad, tr_pad, scale_pad,
     phi_pad, mask) = pad_m_dim(s, t, v, tr, scale, phi, m)
    return TensorDataset(
        torch.from_numpy(s_pad).float(),
        torch.from_numpy(t_p).float(),
        torch.from_numpy(v_pad).float(),
        torch.from_numpy(tr_pad).float(),
        torch.from_numpy(scale_pad).float(),
        torch.from_numpy(phi_pad).float(),
        torch.from_numpy(V).float(),
        torch.from_numpy(mask).bool(),
    )


def _shuffle_tensors(tensors: List[torch.Tensor], perm: np.ndarray) -> List[torch.Tensor]:
    """Shuffle tensors along the first axis."""
    perm_t = torch.from_numpy(perm).long()
    return [t[perm_t] for t in tensors]


def load_unified_shuffled(data_dir: str):
    """Load pre-shuffled unified splits from NPZ files."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
    print(f"Found {len(files)} .npz files in {data_dir} (unified shuffled layout)")

    train_datasets: List = []
    val_datasets: List = []
    test_datasets: List = []
    bucket_info: List[Tuple[Tuple[int, int, int], int, int, int]] = []

    splits = {"train": [], "val": [], "test": []}
    for fname in files:
        if not (fname.startswith("train_") or fname.startswith("val_") or fname.startswith("test_")):
            continue
        prefix, rest = fname.split("_", 1)
        rest = rest[:-4]
        parts = rest.split("_")
        if len(parts) != 3 or not all(p[0] in "TmK" for p in parts):
            print(f"  [WARN] skipping {fname}: unexpected filename format")
            continue
        T = int(parts[0][1:])
        m = int(parts[1][1:])
        K = int(parts[2][1:])
        key = (T, m, K)

        data = np.load(os.path.join(data_dir, fname))
        arrays = {name: np.asarray(data[name]) for name in
                  ("s", "t", "v", "tr", "scale", "phi", "V", "mask")}
        # Unified-shuffled cache is in per-query layout; each row is one (state, t) query.
        if arrays["s"].ndim != 3:
            raise ValueError(
                f"{fname}: expected point-wise s shape (N,M_MAX,K), "
                f"got {arrays['s'].shape}"
            )
        s = torch.from_numpy(arrays["s"]).float()
        t = torch.from_numpy(arrays["t"]).float()
        v = torch.from_numpy(arrays["v"]).float()
        tr = torch.from_numpy(arrays["tr"]).float()
        scale = torch.from_numpy(arrays["scale"]).float()
        phi = torch.from_numpy(arrays["phi"]).float()
        V = torch.from_numpy(arrays["V"]).float()
        mask = torch.from_numpy(arrays["mask"]).bool()
        ds = TensorDataset(s, t, v, tr, scale, phi, V, mask)

        if prefix == "train":
            splits["train"].append((key, ds))
        elif prefix == "val":
            splits["val"].append((key, ds))
        elif prefix == "test":
            splits["test"].append((key, ds))

    # Group by key and sort so all splits share the same bucket order.
    train_by_key: Dict[Tuple[int, int, int], TensorDataset] = {k: ds for k, ds in splits["train"]}
    val_by_key: Dict[Tuple[int, int, int], TensorDataset] = {k: ds for k, ds in splits["val"]}
    test_by_key: Dict[Tuple[int, int, int], TensorDataset] = {k: ds for k, ds in splits["test"]}

    for key in sorted(train_by_key):
        if key not in val_by_key or key not in test_by_key:
            print(f"  [WARN] bucket {key}: missing one of train/val/test; skipped")
            continue
        train_datasets.append(train_by_key[key])
        val_datasets.append(val_by_key[key])
        test_datasets.append(test_by_key[key])
        n_train = len(train_by_key[key])
        n_val = len(val_by_key[key])
        n_test = len(test_by_key[key])
        bucket_info.append((key, n_train, n_val, n_test))
        print(f"  bucket {key}: rows train/val/test={n_train}/{n_val}/{n_test}")

    return train_datasets, val_datasets, test_datasets, bucket_info


def build_datasets(data_dir: str, split_seed: int = 123456,
                   shuffle_seed: int = 1234567):
    """Load sampled-raw files and emit per-(T,m,K) bucket datasets."""
    # Auto-detect unified-shuffled layout and skip per-file split/shuffle if data
    # has already been prepared.
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
    is_unified = any(f.startswith(("train_T", "val_T", "test_T")) for f in files)
    if is_unified:
        return load_unified_shuffled(data_dir)

    # Bucket key: (T, m, K) -> Dict[str, List[TensorDataset]]
    buckets: Dict[Tuple[int, int, int], Dict[str, List[TensorDataset]]] = {}

    print(f"Found {len(files)} .npz files in {data_dir}")

    for fname in files:
        path = os.path.join(data_dir, fname)
        tag = fname[:-4]
        print(f"\nLoading {fname} (tag={tag}) ...")
        file_data = load_sampled_raw(path)
        cfg = file_data["config"]
        m = len(cfg["group_sizes"])
        K = int(cfg["num_states"])
        T = cfg["T"]
        n_settings = int(file_data["n_settings"])
        n_sample = file_data["n_sample"]

        # Config seeds are stable across machines; combining one with the CLI
        # seed gives a reproducible but config-specific setting permutation.
        split_seed_for_cfg = int(split_seed) + int(cfg.get("seed", 0))
        train_settings, val_settings, test_settings = split_setting_indices(
            n_settings, split_seed_for_cfg
        )
        split_settings = {
            "training": train_settings,
            "validation": val_settings,
            "test": test_settings,
        }

        key = (T, m, K)
        if key not in buckets:
            buckets[key] = {"training": [], "validation": [], "test": []}

        for split_name, setting_rows in split_settings.items():
            split_tup = build_split_for_settings(file_data, setting_rows)
            dataset = tensor_dataset_from_split(
                split_tup, m, tag, split_name
            )
            if dataset is not None:
                buckets[key][split_name].append(dataset)

        print(
            f"  tag={tag}: settings train/val/test="
            f"{len(train_settings)}/{len(val_settings)}/{len(test_settings)}; "
            f"m={m} K={K} T={T} eps={cfg.get('epsilon')} "
            f"n_sample={n_sample}"
        )

    # Merge each bucket's splits and globally shuffle the training set.
    train_datasets: List = []
    val_datasets: List = []
    test_datasets: List = []
    bucket_info: List[Tuple[Tuple[int, int, int], int, int, int]] = []
    # Stream 0 is reserved for the Step 2.5 / raw-layout bucket shuffle.
    bucket_shuffle_stream = np.random.SeedSequence(shuffle_seed).spawn(5)[0]
    rng = np.random.default_rng(bucket_shuffle_stream)

    for key in sorted(buckets):
        train_parts = buckets[key]["training"]
        val_parts = buckets[key]["validation"]
        test_parts = buckets[key]["test"]

        if not train_parts or not val_parts or not test_parts:
            print(f"  [WARN] bucket {key}: missing one of train/val/test; skipped")
            continue

        train_ds = ConcatDataset(train_parts)
        val_ds = ConcatDataset(val_parts)
        test_ds = ConcatDataset(test_parts)

        n_train = len(train_ds)
        n_val = len(val_ds)
        n_test = len(test_ds)

        # Global shuffle of training samples within the bucket.
        perm = rng.permutation(n_train)
        # ConcatDataset exposes .datasets with each tensor's data; reindexing
        # by raw indices is messy, so we materialize the merged tensors and
        # shuffle directly.
        merged = _concat_tensor_datasets(train_parts)
        merged = TensorDataset(*_shuffle_tensors(list(merged.tensors), perm))

        train_datasets.append(merged)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)
        bucket_info.append((key, n_train, n_val, n_test))
        print(
            f"  bucket {key}: merged rows train/val/test="
            f"{n_train}/{n_val}/{n_test}"
        )

    return train_datasets, val_datasets, test_datasets, bucket_info


def _concat_tensor_datasets(parts: List[TensorDataset]) -> TensorDataset:
    """Concatenate TensorDatasets by stacking each tensor."""
    if not parts:
        return TensorDataset()
    all_tensors = [[] for _ in range(len(parts[0].tensors))]
    for ds in parts:
        for i, t in enumerate(ds.tensors):
            all_tensors[i].append(t)
    stacked = [torch.cat(tlist, dim=0) for tlist in all_tensors]
    return TensorDataset(*stacked)


def balanced_subsample_by_epsilon(
    datasets_by_TK,
    budget: int,
    seed: int,
    target_epsilons=None,
):
    """Subsample equal training rows per epsilon value."""
    groups = {}
    for key, ds in datasets_by_TK.items():
        epsilon = float(key[2])
        if target_epsilons is not None and epsilon not in target_epsilons:
            continue
        groups.setdefault(epsilon, []).append((key, ds))

    if not groups:
        requested = sorted(target_epsilons) if target_epsilons is not None else []
        raise ValueError(f"No training data found for requested epsilons: {requested}")

    epsilons = sorted(groups)
    per_epsilon = budget // len(epsilons)
    remainder = budget % len(epsilons)
    if per_epsilon < 1:
        raise ValueError(
            f"subsample_train={budget} is smaller than the number of epsilon groups "
            f"({len(epsilons)})"
        )

    rng = np.random.default_rng(seed)
    sampled = {}
    print(f"Balanced epsilon sampling: {len(epsilons)} groups, "
          f"target total={per_epsilon * len(epsilons) + remainder}")

    for eps_i, epsilon in enumerate(epsilons):
        members = groups[epsilon]
        total = sum(len(ds) for _, ds in members)
        target = per_epsilon + (1 if eps_i < remainder else 0)
        target = min(target, total)

        raw = [target * len(ds) / total for _, ds in members]
        alloc = [int(x) for x in raw]
        left = target - sum(alloc)
        order = np.argsort([x - int(x) for x in raw])[::-1]
        for j in order[:left]:
            alloc[int(j)] += 1

        selected_total = 0
        for (key, ds), n_select in zip(members, alloc):
            if n_select <= 0:
                continue
            idx = rng.choice(len(ds), size=n_select, replace=False)
            tensors = [tensor.numpy()[idx] for tensor in ds.tensors]
            sampled[key] = TensorDataset(*[torch.from_numpy(tensor) for tensor in tensors])
            selected_total += n_select
        print(f"  epsilon={epsilon:g}: {selected_total} samples from {total}")

    return sampled


def evaluate(model, datasets: List, device):
    model.eval()
    total_metrics = {"loss": 0.0, "phi_mae": 0.0, "n": 0}
    with torch.no_grad():
        for ds in datasets:
            ld = DataLoader(ds, batch_size=512, shuffle=False)
            for s, t, v, tr, scale, phi, V, mask in ld:
                s, t, v, tr, scale, phi, V, mask = [
                    x.to(device) for x in (s, t, v, tr, scale, phi, V, mask)
                ]
                pred = model(s, t, v, tr, mask, scale)
                losses = fair_pivot_loss(pred, {"phi": phi}, mask=mask)

                mask_f = mask.float()
                phi_mae = (torch.abs(pred["phi"] - phi) * mask_f).sum() / (mask_f.sum() + 1e-9)

                bs = s.shape[0]
                total_metrics["loss"] += losses["total"].item() * bs
                total_metrics["phi_mae"] += phi_mae.item() * bs
                total_metrics["n"] += bs
    if total_metrics["n"] == 0:
        raise ValueError("Cannot evaluate an empty dataset collection")
    for k in ["loss", "phi_mae"]:
        total_metrics[k] /= total_metrics["n"]
    return total_metrics


def _reshuffle_bucket_datasets(
    bucket_datasets: List[TensorDataset],
    seed_sequence: np.random.SeedSequence,
) -> List[TensorDataset]:
    """Return TensorDatasets with the first axis shuffled."""
    rng = np.random.default_rng(seed_sequence)
    shuffled = []
    for ds in bucket_datasets:
        n = len(ds)
        perm = rng.permutation(n)
        shuffled.append(TensorDataset(*_shuffle_tensors(list(ds.tensors), perm)))
    return shuffled


def _chunk_dataset(ds: TensorDataset, chunk_size: int) -> List[TensorDataset]:
    """Split a TensorDataset into contiguous chunks."""
    n = len(ds)
    if n <= chunk_size:
        return [ds]
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append(TensorDataset(*[t[start:end] for t in ds.tensors]))
    return chunks


def chunk_train_datasets(
    train_datasets: List[TensorDataset], chunk_size: int,
    seed_sequence: np.random.SeedSequence,
) -> List[TensorDataset]:
    """Split train buckets into chunks and shuffle them."""
    rng = np.random.default_rng(seed_sequence)
    all_chunks = []
    for ds in train_datasets:
        all_chunks.extend(_chunk_dataset(ds, chunk_size))
    rng.shuffle(all_chunks)
    return all_chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data_cache/1to9_sampled_raw")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--output", type=str, default="results/ckpt/unified_1to9.pth")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--split_seed", type=int, default=123456,
                   help="Dedicated seed for the 8:1:1 setting-level split")
    p.add_argument("--shuffle_seed", type=int, default=1234567,
                   help="Root seed for independently spawned shuffle-family streams")
    p.add_argument("--init_seed", type=int, default=42,
                   help="Dedicated seed for torch/np model initialization")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--subsample_train", type=int, default=None,
                   help="Randomly subsample N training samples in total for quick experiments")
    p.add_argument("--balance_epsilon", action="store_true",
                   help="Sample an equal number of training rows for each epsilon")
    p.add_argument("--balance_epsilons", type=str, default=None,
                   help="Comma-separated epsilon values to balance, e.g. 1,5,9; "
                        "default: all epsilon values present")
    p.add_argument("--dry_run", action="store_true",
                   help="Run only a single epoch and skip saving to quickly test "
                        "memory and throughput")
    p.add_argument("--dry_run_batches", type=int, default=None,
                   help="If set with --dry_run, stop after this many training batches")
    p.add_argument("--chunk_size", type=int, default=8192,
                   help="Split each train bucket into chunks of ~N rows before round-robin "
                        "interleaving. Use 0 to disable chunking. Default: 8192")
    p.add_argument("--early_stop_patience", type=int, default=15,
                   help="Stop training if val loss has not improved for this many epochs. "
                        "Set to 0 to disable early stopping.")
    p.add_argument("--early_stop_min_delta", type=float, default=1e-5,
                   help="Minimum drop in val loss to count as an improvement.")
    args = p.parse_args()

    if args.device:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] --device cuda but CUDA unavailable (CPU-only PyTorch?). Falling back to CPU.")
            device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    torch.manual_seed(args.init_seed)
    np.random.seed(args.init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.init_seed)
    # Tighten determinism for any cuDNN op (conv paths inside the RNN backbone).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Building datasets from {args.data_dir} ...")
    train_datasets, val_datasets, test_datasets, bucket_info = build_datasets(
        args.data_dir,
        split_seed=args.split_seed,
        shuffle_seed=args.shuffle_seed,
    )
    if not train_datasets or not val_datasets or not test_datasets:
        raise ValueError(
            "The 8:1:1 split must contain train, validation, and test data"
        )

    print(f"\nBuckets ({len(bucket_info)} total):")
    for key, n_train, n_val, n_test in bucket_info:
        print(f"  {key}: train={n_train} val={n_val} test={n_test}")

    # Fixed stream assignment prevents optional branches from shifting other
    # random streams: 0=bucket shuffle, 1=subsample, 2=chunk,
    # 3=epoch sample reshuffle, 4=epoch bucket order.
    shuffle_streams = np.random.SeedSequence(args.shuffle_seed).spawn(5)
    subsample_rng = np.random.default_rng(shuffle_streams[1])
    chunk_stream = shuffle_streams[2]
    epoch_reshuffle_root = shuffle_streams[3]
    bucket_order_root = shuffle_streams[4]
    epoch_reshuffle_streams = epoch_reshuffle_root.spawn(
        1 if args.dry_run else args.epochs
    )
    bucket_order_streams = bucket_order_root.spawn(
        1 if args.dry_run else args.epochs
    )

    if args.subsample_train is not None and args.subsample_train > 0:
        # Subsample proportionally from each metadata bucket.  All model rows
        # are already independent (state, time) samples with common (m, K).
        total_train = sum(len(ds) for ds in train_datasets)
        budget = min(args.subsample_train, total_train)
        weights = [len(ds) / total_train for ds in train_datasets]
        allocs = [int(budget * w) for w in weights]
        remainder = budget - sum(allocs)
        for _ in range(remainder):
            allocs[np.argmax([w - a / budget for w, a in zip(weights, allocs)])] += 1

        subsampled = []
        for ds, alloc in zip(train_datasets, allocs):
            if alloc <= 0:
                subsampled.append(ds)
                continue
            idx = subsample_rng.choice(len(ds), size=alloc, replace=False)
            tensors = [t.numpy()[idx] for t in ds.tensors]
            subsampled.append(TensorDataset(*[torch.from_numpy(t) for t in tensors]))
        train_datasets = subsampled
        print(f"Subsampled training set: {budget} rows across {len(train_datasets)} buckets")

    if args.chunk_size > 0:
        # Split large metadata buckets into smaller point-wise sample chunks so
        # round-robin interleaving remains uniform across configurations.
        train_datasets = chunk_train_datasets(
            train_datasets, args.chunk_size, chunk_stream
        )
        print(
            f"Chunked training set: {len(train_datasets)} chunks "
            f"(~{args.chunk_size} rows each)"
        )

    if args.balance_epsilon:
        # Not implemented with the new bucket-based design because epsilon is a
        # scalar feature, not a bucket key.  Use --dry_run for quick tests.
        raise NotImplementedError(
            "--balance_epsilon is not yet supported with the new bucket-based "
            "splitting. Use --dry_run to test quickly instead."
        )

    model = FlexibleFairPivotNet(hidden_dim=128, m_max=M_MAX, K_max=K_MAX).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    def get_state_dict():
        return model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    epochs = 1 if args.dry_run else args.epochs
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        train_losses = []

        # Each epoch receives independent child streams for sample reshuffling
        # and bucket ordering.
        epoch_train_datasets = _reshuffle_bucket_datasets(
            train_datasets, epoch_reshuffle_streams[ep]
        )
        train_loaders = {
            i: DataLoader(
                ds,
                batch_size=min(args.batch_size, len(ds)),
                shuffle=False,  # already shuffled globally above
                drop_last=False,
            )
            for i, ds in enumerate(epoch_train_datasets)
        }

        # Round-robin over buckets.  Order is shuffled each epoch so different
        # buckets lead the update distribution.
        bucket_order = list(range(len(train_loaders)))
        np.random.default_rng(bucket_order_streams[ep]).shuffle(bucket_order)
        iterators = {i: iter(train_loaders[i]) for i in bucket_order}
        active = set(bucket_order)
        batch_count = 0
        while active:
            for i in list(active):
                try:
                    s, t, v, tr, scale, phi, V, mask = next(iterators[i])
                except StopIteration:
                    active.remove(i)
                    continue
                s, t, v, tr, scale, phi, V, mask = [
                    x.to(device) for x in (s, t, v, tr, scale, phi, V, mask)
                ]
                pred = model(s, t, v, tr, mask, scale)
                losses = fair_pivot_loss(pred, {"phi": phi}, mask=mask)
                optim.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                train_losses.append(losses["total"].item())
                batch_count += 1
                if args.dry_run and args.dry_run_batches is not None and batch_count >= args.dry_run_batches:
                    print(f"  [DRY RUN] stopping after {batch_count} batches")
                    active.clear()
                    break

        if not args.dry_run:
            sched.step()

        val_metrics = evaluate(model, val_datasets, device)
        improved = val_metrics["loss"] < best_val - args.early_stop_min_delta
        if not args.dry_run and improved:
            best_val = val_metrics["loss"]
            best_epoch = ep
            epochs_since_best = 0
            torch.save({
                "state_dict": get_state_dict(),
                "args": vars(args),
            }, args.output)
        else:
            if not args.dry_run:
                epochs_since_best += 1

        if ep % args.log_every == 0 or ep == epochs - 1:
            print(f" e{ep:3d} | tr loss={np.mean(train_losses):.6f} "
                  f"| val loss={val_metrics['loss']:.6f} "
                  f"phi_mae={val_metrics['phi_mae']:.6f} "
                  f"| best={best_val:.6f}@e{best_epoch} "
                  f"| wait={epochs_since_best}/{args.early_stop_patience} "
                  f"| {time.time()-t0:.1f}s", flush=True)

        if (not args.dry_run
                and args.early_stop_patience > 0
                and epochs_since_best >= args.early_stop_patience):
            print(f"\n[EARLY STOP] no val improvement for "
                  f"{args.early_stop_patience} epochs; best was e{best_epoch} "
                  f"with val_loss={best_val:.6f}", flush=True)
            break

    if args.dry_run:
        print("\n[DRY RUN] completed. No model was saved.")
        return

    ckpt = torch.load(args.output, map_location=device)
    state_dict = ckpt["state_dict"]
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    test_metrics = evaluate(model, test_datasets, device)
    metrics_path = f"{args.output}.metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split_ratio": [8, 1, 1],
                "split_unit": "setting",
                "best_validation_loss": best_val,
                "test": test_metrics,
                "buckets": [
                    {"T": key[0], "m": key[1], "K": key[2],
                     "n_train": n_train, "n_val": n_val, "n_test": n_test}
                    for key, n_train, n_val, n_test in bucket_info
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved best model to {args.output}")
    print(
        f"Test | loss={test_metrics['loss']:.2f} "
        f"phi_mae={test_metrics['phi_mae']:.4f} "
        f"rows={test_metrics['n']}"
    )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
