"""Parameterized multi-setting training loop for the learned model."""
from __future__ import annotations
import argparse, os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))          # code/
sys.path.insert(0, os.path.join(os.path.dirname(_here), ".."))  # project root

from model.backbone_flex import FlexibleFairPivotNet
from train.losses import fair_pivot_loss


# Fixed global reference scales from the original 1-9 experiments.
GROUP_SIZE_REF = 70.0
VALUATION_REF = 400.0
SCALE_DIM = 2  # [group_size / GROUP_SIZE_REF, group_v_max / VALUATION_REF]


def generate_one_setting(oracle):
    """Convert one oracle output to relative features, group scales and labels."""
    S = oracle["states"].shape[0]
    T = int(oracle["T"])
    m = oracle["group_sizes"].size
    K = oracle["num_states"]
    Ng = oracle["group_sizes"]                         # (m,)
    delta = oracle["delta"]
    epsilon = oracle["epsilon"]
    valuations = oracle["valuations"]                  # (m, K)
    matrix0_list = oracle["matrix0"]                   # list of m (K, K)
    matrix1_list = oracle["matrix1"]                   # list of m (K, K)
    # Backward compatibility: old oracle saved a single ndarray
    if isinstance(matrix0_list, np.ndarray):
        matrix0_list = [matrix0_list] * m
    if isinstance(matrix1_list, np.ndarray):
        matrix1_list = [matrix1_list] * m
    group_v_max = np.max(valuations, axis=1)             # (m,)
    if np.any(Ng <= 0) or np.any(group_v_max <= 0):
        raise ValueError("Every group must have positive size and valuations")
    if np.any(Ng > GROUP_SIZE_REF):
        raise ValueError(
            f"group size {int(np.max(Ng))} exceeds GROUP_SIZE_REF={GROUP_SIZE_REF:g}")
    if np.any(group_v_max > VALUATION_REF):
        raise ValueError(
            f"valuation {float(np.max(group_v_max))} exceeds VALUATION_REF={VALUATION_REF:g}")

    # Relative state counts per group population.
    s_raw = oracle["states"].astype(np.float64).copy()
    for k in range(m):
        for s in range(K):
            s_raw[:, k * K + s] /= Ng[k]
    s_norm = s_raw.reshape(S, 1, m, K).repeat(T, axis=1)  # (S, T, m, K)

    t_feat = np.zeros((S, T, 3), dtype=np.float32)
    for tidx in range(T):
        t_feat[:, tidx, 0] = tidx / T
        t_feat[:, tidx, 1] = delta
        t_feat[:, tidx, 2] = epsilon / 10.0

    v_relative = valuations / group_v_max[:, None]
    v_feat = np.broadcast_to(
        v_relative.reshape(1, 1, m, K), (S, T, m, K)
    ).copy()

    # Per-group absolute scales preserve both market shape and magnitude:
    #   state_count = relative_state * group_size
    #   valuation   = relative_valuation * group_v_max
    group_scale = np.stack(
        [Ng.astype(np.float64) / GROUP_SIZE_REF,
         group_v_max / VALUATION_REF],
        axis=-1,
    )                                                        # (m, 2)
    scale_feat = np.broadcast_to(
        group_scale.reshape(1, 1, m, SCALE_DIM), (S, T, m, SCALE_DIM)
    ).copy()

    # Per-group (2*K*K,) concatenated → (S, T, m, 2*K*K)
    tr_list = [np.concatenate([matrix0_list[k].flatten(), matrix1_list[k].flatten()])
               for k in range(m)]                       # m vectors of length 2*K*K
    tr_feat = np.zeros((S, T, m, 2 * K * K), dtype=np.float32)
    for k in range(m):
        tr_feat[:, :, k, :] = tr_list[k]

    # Payment is deliberately omitted; callers use fair_pivot_bi(compute_payment=True).
    phi   = oracle["phi"].astype(np.float32)       # (S, T, m)
    V_lab = oracle["V"].astype(np.float32)         # (S, T)

    return s_norm.astype(np.float32), t_feat, v_feat.astype(np.float32), \
           tr_feat.astype(np.float32), scale_feat.astype(np.float32), \
           phi, V_lab


def sample_state_indices(phi, n_sample, rng, return_kind=False,
                          n_extreme_high=None, n_extreme_low=None,
                          n_random=None, *, n_extreme=None):
    """Select high-phi, low-phi, and random raw-state indices.

    States are ranked by group 0's mean phi over time.  The default split is
    one quarter high, one quarter low, and the remainder random.  The
    deprecated ``n_extreme`` keyword is treated as high-phi count only.
    """
    phi = np.asarray(phi)
    if phi.ndim != 3:
        raise ValueError(f"phi must have shape (S, T, m), got {phi.shape}")
    S = phi.shape[0]
    if n_sample is None:
        idx = np.arange(S, dtype=np.int64)
        if return_kind:
            return idx, np.full(S, -1, dtype=np.int8)
        return idx
    if n_sample <= 0:
        raise ValueError(f"n_sample must be positive, got {n_sample}")
    if n_sample > S:
        raise ValueError(
            f"n_sample ({n_sample}) cannot exceed the number of states ({S})"
        )

    if n_extreme is not None:
        if n_extreme_high is not None or n_extreme_low is not None:
            raise ValueError(
                "n_extreme cannot be combined with n_extreme_high/"
                "n_extreme_low"
            )
        n_extreme_high = n_extreme
        n_extreme_low = 0

    if (n_extreme_high is None and n_extreme_low is None
            and n_random is None):
        n_extreme_high = n_sample // 4
        n_extreme_low = n_sample // 4
        n_random = n_sample - n_extreme_high - n_extreme_low
    else:
        if n_extreme_high is None or n_extreme_low is None:
            raise ValueError(
                "n_extreme_high and n_extreme_low must be supplied together"
            )
        if n_random is None:
            n_random = n_sample - int(n_extreme_high) - int(n_extreme_low)

    n_extreme_high = int(n_extreme_high)
    n_extreme_low = int(n_extreme_low)
    n_random = int(n_random)
    if min(n_extreme_high, n_extreme_low, n_random) < 0:
        raise ValueError(
            "n_extreme_high/n_extreme_low/n_random must be non-negative, "
            f"got {n_extreme_high}/{n_extreme_low}/{n_random}"
        )
    if n_extreme_high + n_extreme_low + n_random != n_sample:
        raise ValueError(
            f"n_extreme_high ({n_extreme_high}) + "
            f"n_extreme_low ({n_extreme_low}) + n_random ({n_random}) "
            f"must equal n_sample ({n_sample})"
        )
    if n_extreme_high + n_extreme_low > S:
        raise ValueError(
            "n_extreme_high + n_extreme_low cannot exceed the number of "
            f"states ({S})"
        )

    all_idx = np.arange(S, dtype=np.int64)
    # A common scalar axis is essential: for m=2, taking max/min over groups
    # would rank the same extreme states at both ends because phi sums to one.
    phi_score = phi[:, :, 0].mean(axis=1)
    high_idx = np.argsort(-phi_score, kind="stable")[:n_extreme_high]
    remaining = np.setdiff1d(all_idx, high_idx, assume_unique=True)
    low_order = np.argsort(phi_score[remaining], kind="stable")
    low_idx = remaining[low_order[:n_extreme_low]]
    remaining = np.setdiff1d(remaining, low_idx, assume_unique=True)
    random_idx = rng.choice(remaining, size=n_random, replace=False)

    idx_parts = []
    kind_parts = []
    if n_extreme_high:
        idx_parts.append(high_idx)
        kind_parts.append(np.full(n_extreme_high, 2, dtype=np.int8))
    if n_extreme_low:
        idx_parts.append(low_idx)
        kind_parts.append(np.full(n_extreme_low, 1, dtype=np.int8))
    if n_random:
        idx_parts.append(random_idx)
        kind_parts.append(np.zeros(n_random, dtype=np.int8))

    idx = np.concatenate(idx_parts).astype(np.int64, copy=False)
    kind = np.concatenate(kind_parts)
    order = rng.permutation(n_sample)
    idx = idx[order]
    kind = kind[order]
    if return_kind:
        return idx, kind
    return idx


def sample_states(state_feat, time_feat, val_feat, trans_feat, scale_feat,
                  phi, V_lab, n_sample, rng):
    """Backward-compatible feature sampler."""
    idx = sample_state_indices(phi, n_sample, rng)
    return (state_feat[idx], time_feat[idx], val_feat[idx], trans_feat[idx],
            scale_feat[idx], phi[idx], V_lab[idx])


def normalize_raw_samples(states, valuations, matrix0, matrix1,
                          group_sizes, T, delta, epsilon):
    """Normalise raw Oracle rows and flatten ``(state, time)`` into samples.

    Returns ``state, time, val, trans, scale`` with leading dimension ``N*T``.
    """
    states = np.asarray(states)
    valuations = np.asarray(valuations)
    matrix0 = np.asarray(matrix0)
    matrix1 = np.asarray(matrix1)
    Ng = np.asarray(group_sizes, dtype=np.float64)
    N = states.shape[0]
    m = Ng.size
    K = valuations.shape[-1]

    if states.shape != (N, m * K):
        raise ValueError(f"states must have shape {(N, m * K)}, got {states.shape}")
    expected_market = (N, m, K)
    if valuations.shape != expected_market:
        raise ValueError(
            f"valuations must have shape {expected_market}, got {valuations.shape}")
    expected_matrix = (N, m, K, K)
    if matrix0.shape != expected_matrix or matrix1.shape != expected_matrix:
        raise ValueError(
            f"matrix0/1 must have shape {expected_matrix}, got "
            f"{matrix0.shape} and {matrix1.shape}")

    group_v_max = valuations.max(axis=-1)  # (N, m)
    if np.any(Ng <= 0) or np.any(group_v_max <= 0):
        raise ValueError("Every group must have positive size and valuations")
    if np.any(Ng > GROUP_SIZE_REF):
        raise ValueError(
            f"group size {int(Ng.max())} exceeds GROUP_SIZE_REF={GROUP_SIZE_REF:g}")
    if np.any(group_v_max > VALUATION_REF):
        raise ValueError(
            f"valuation {float(group_v_max.max())} exceeds "
            f"VALUATION_REF={VALUATION_REF:g}")

    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    state_per_group = states.reshape(N, m, K).astype(np.float64)
    state_relative = state_per_group / Ng.reshape(1, m, 1)
    state_feat = np.repeat(state_relative, T, axis=0)              # (N*T, m, K)

    time_per_t = np.stack([
        np.arange(T, dtype=np.float64) / T,
        np.full(T, delta, dtype=np.float64),
        np.full(T, epsilon / 10.0, dtype=np.float64),
    ], axis=-1)
    time_feat = np.tile(time_per_t, (N, 1))                        # (N*T, 3)

    val_relative = valuations / group_v_max[:, :, None]
    val_feat = np.repeat(val_relative, T, axis=0)                  # (N*T, m, K)

    group_size_scale = np.broadcast_to(
        Ng.reshape(1, m), (N, m)
    ) / GROUP_SIZE_REF
    scale_per_row = np.stack(
        [group_size_scale, group_v_max / VALUATION_REF], axis=-1
    )
    scale_feat = np.repeat(scale_per_row, T, axis=0)               # (N*T, m, 2)

    trans_per_row = np.concatenate([
        matrix0.reshape(N, m, K * K),
        matrix1.reshape(N, m, K * K),
    ], axis=-1)
    trans_feat = np.repeat(trans_per_row, T, axis=0)               # (N*T, m, 2*K*K)

    return tuple(x.astype(np.float32) for x in (
        state_feat, time_feat, val_feat, trans_feat, scale_feat
    ))


def normalize_query_samples(states, valuations, matrix0, matrix1,
                           group_sizes, delta, epsilon, t_indices):
    """Normalise a batch of per-(state, t) current-time queries."""
    states = np.asarray(states)
    valuations = np.asarray(valuations)
    matrix0 = np.asarray(matrix0)
    matrix1 = np.asarray(matrix1)
    t_indices = np.asarray(t_indices, dtype=np.int64)
    Ng = np.asarray(group_sizes, dtype=np.float64)
    N = states.shape[0]
    m = Ng.size
    K = valuations.shape[-1]

    if states.shape != (N, m * K):
        raise ValueError(f"states must have shape {(N, m * K)}, got {states.shape}")
    if valuations.shape != (N, m, K):
        raise ValueError(
            f"valuations must have shape {(N, m, K)}, got {valuations.shape}")
    if matrix0.shape != (N, m, K, K) or matrix1.shape != (N, m, K, K):
        raise ValueError(
            f"matrix0/1 must have shape {(N, m, K, K)}, got "
            f"{matrix0.shape} and {matrix1.shape}")
    if t_indices.shape != (N,):
        raise ValueError(
            f"t_indices must have shape ({N},), got {t_indices.shape}")

    T = int(t_indices.max()) + 1 if N > 0 else 1

    group_v_max = valuations.max(axis=-1)  # (N, m)
    if np.any(Ng <= 0) or np.any(group_v_max <= 0):
        raise ValueError("Every group must have positive size and valuations")
    if np.any(Ng > GROUP_SIZE_REF):
        raise ValueError(
            f"group size {int(Ng.max())} exceeds GROUP_SIZE_REF={GROUP_SIZE_REF:g}")
    if np.any(group_v_max > VALUATION_REF):
        raise ValueError(
            f"valuation {float(group_v_max.max())} exceeds "
            f"VALUATION_REF={VALUATION_REF:g}")

    state_per_group = states.reshape(N, m, K).astype(np.float64)
    state_relative = state_per_group / Ng.reshape(1, m, 1)
    state_feat = state_relative                              # (N, m, K)

    time_feat = np.stack([
        t_indices.astype(np.float64) / T,
        np.full(N, delta, dtype=np.float64),
        np.full(N, epsilon / 10.0, dtype=np.float64),
    ], axis=-1)                                              # (N, 3)

    val_relative = valuations / group_v_max[:, :, None]
    val_feat = val_relative                                   # (N, m, K)

    group_size_scale = np.broadcast_to(
        Ng.reshape(1, m), (N, m)
    ) / GROUP_SIZE_REF
    scale_per_row = np.stack(
        [group_size_scale, group_v_max / VALUATION_REF], axis=-1
    )
    scale_feat = scale_per_row                               # (N, m, 2)

    trans_per_row = np.concatenate([
        matrix0.reshape(N, m, K * K),
        matrix1.reshape(N, m, K * K),
    ], axis=-1)
    trans_feat = trans_per_row                               # (N, m, 2*K*K)

    return tuple(x.astype(np.float32) for x in (
        state_feat, time_feat, val_feat, trans_feat, scale_feat
    ))


# Top-level worker for multiprocessing pickling
def _build_one_setting(args):
    import sys, os
    # Worker processes do not inherit sys.path; add project roots.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))          # code/
    sys.path.insert(0, os.path.join(os.path.dirname(_here), ".."))  # project root
    from code.oracle.fair_pivot_bi import fair_pivot_bi
    si, ms, n_settings, eps, group_sizes, num_states, T, delta, n_sample_states, rng_sample = args

    if si % 256 == 0:
        print(f"    [{si}/{n_settings}]  eps={eps}", flush=True)

    # Per-group matrices: group 0 uses mats[0,1], group 1 uses mats[2,3], etc.
    # For m > 2, we cycle through the 4 available matrices.
    mat0s = [ms["mats"][(2 * k) % 4] for k in range(len(group_sizes))]
    mat1s = [ms["mats"][(2 * k + 1) % 4] for k in range(len(group_sizes))]
    v_per_group = []
    for k in range(len(group_sizes)):
        if k == 0:
            v_per_group.append(np.array(ms["v_low"][:num_states], dtype=np.float64))
        elif k == 1:
            v_per_group.append(np.array(ms["v_high"][:num_states], dtype=np.float64))
        else:
            # For m >= 3, alternate low/high pools
            pool = ms["v_low"] if k % 2 == 0 else ms["v_high"]
            v_per_group.append(np.array(pool[:num_states], dtype=np.float64))

    oracle = fair_pivot_bi(
        group_sizes=group_sizes,
        num_states=num_states,
        T=T,
        delta=delta,
        epsilon=eps,
        valuations=np.vstack(v_per_group),
        matrix0=mat0s, matrix1=mat1s,  # per-group independent matrices
        compute_payment=False,
    )
    s, t, v, tr, scale, phi, V_lab = generate_one_setting(oracle)

    if n_sample_states is not None:
        s, t, v, tr, scale, phi, V_lab = sample_states(
            s, t, v, tr, scale, phi, V_lab, n_sample_states, rng_sample)

    # generate_one_setting keeps (state, time) separate until state sampling.
    # The model contract is point-wise, so fold time into the sample axis.
    n_rows, n_times, actual_m, actual_K = s.shape
    s = s.reshape(n_rows * n_times, actual_m, actual_K)
    t = t.reshape(n_rows * n_times, 3)
    v = v.reshape(n_rows * n_times, actual_m, actual_K)
    tr = tr.reshape(n_rows * n_times, actual_m, tr.shape[-1])
    scale = scale.reshape(n_rows * n_times, actual_m, SCALE_DIM)
    phi = phi.reshape(n_rows * n_times, actual_m)
    V_lab = V_lab.reshape(n_rows * n_times)

    return s, t, v, tr, scale, phi, V_lab


def build_dataset(config: dict, seed: int = 12345678, n_jobs: int = 1) -> tuple:
    """Generate and concatenate oracle data for one experiment configuration."""
    group_sizes = config["group_sizes"]
    num_states = config["num_states"]
    T = config["T"]
    delta = config["delta"]
    epsilons = config["epsilons"]
    n_settings = config["n_settings"]
    n_sample_states = config.get("n_sample_states", 128)

    rng_market = np.random.default_rng(seed)
    rng_sample = np.random.default_rng(seed + 999)

    market_settings = []
    for _ in range(n_settings):
        sm = int(rng_market.integers(0, 2**32 - 1, dtype=np.uint32))
        mr = np.random.default_rng(sm)
        mats = mr.random((4, num_states, num_states))
        for i in range(4):
            mats[i] /= mats[i].sum(axis=1, keepdims=True)
        sv = int(rng_market.integers(0, 2**32 - 1, dtype=np.uint32))
        vr = np.random.default_rng(sv)
        # Valuation pools use the same ranges as the original experiments.
        v_pool_low = sorted(vr.choice(range(360, 401), num_states, replace=False).astype(np.float64))
        v_pool_high = sorted(vr.choice(range(240, 281), num_states, replace=False).astype(np.float64))
        market_settings.append({
            "mats": [mats[i] for i in range(4)],
            "v_low": v_pool_low,
            "v_high": v_pool_high,
        })

    all_s, all_t, all_v, all_tr, all_scale = [], [], [], [], []
    all_phi, all_V = [], []

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, n_jobs)

    for eps in epsilons:
        print(f"  Oracle generation for epsilon={eps} ...")
        sys.stdout.flush()

        # Prepare seed per worker for sampling (must be independent per setting).
        base_args = (n_settings, eps, group_sizes, num_states, T, delta, n_sample_states)
        worker_args = [(si, ms, *base_args, np.random.default_rng(seed + 999 + si)) for si, ms in enumerate(market_settings)]

        if n_jobs == 1:
            results = [_build_one_setting(a) for a in worker_args]
        else:
            from multiprocessing import Pool
            with Pool(processes=n_jobs) as pool:
                results = pool.map(_build_one_setting, worker_args)

        for s, t, v, tr, scale, phi, V_lab in results:
            all_s.append(s); all_t.append(t); all_v.append(v); all_tr.append(tr)
            all_scale.append(scale)
            all_phi.append(phi); all_V.append(V_lab)

    return (np.concatenate(all_s, axis=0), np.concatenate(all_t, axis=0),
            np.concatenate(all_v, axis=0), np.concatenate(all_tr, axis=0),
            np.concatenate(all_scale, axis=0), np.concatenate(all_phi, axis=0),
            np.concatenate(all_V, axis=0))


# Back-compat wrapper for the old CLI (epsilon-only)
def build_1fairness_dataset(epsilons, n_settings, n_sample_states, seed=12345678):
    """Legacy wrapper calling build_dataset with default group_sizes=[30,30]."""
    return build_dataset(
        {
            "group_sizes": [30, 30],
            "num_states": 2,
            "T": 5,
            "delta": 0.6,
            "epsilons": epsilons,
            "n_settings": n_settings,
            "n_sample_states": n_sample_states,
        }, seed=seed
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_json", type=str, default=None,
                   help="Path to JSON config for experiment")
    p.add_argument("--epsilons", type=str, default="1,5,9",
                   help="Comma-separated epsilon values (used if --config_json not provided)")
    p.add_argument("--settings", type=int, default=1024)
    p.add_argument("--sample_states", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--m_max", type=int, default=4)
    p.add_argument("--K_max", type=int, default=4)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=12345678)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    if args.config_json is not None:
        with open(args.config_json, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        epsilons = [int(x) for x in args.epsilons.split(",")]
        config = {
            "group_sizes": [30, 30],
            "num_states": 2,
            "T": 5,
            "delta": 0.6,
            "epsilons": epsilons,
            "n_settings": args.settings,
            "n_sample_states": args.sample_states,
        }

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    epsilons = config.get("epsilons", [])

    if args.output is None:
        exp_tag = config.get("exp_tag", "default")
        args.output = os.path.join(
            os.path.dirname(__file__), "..", "results", "ckpt",
            f"flex_{exp_tag}_e{'-'.join(map(str, epsilons))}_s{config['n_settings']}.pth"
        )

    print(f"=== Model Training ===")
    print(f"config={json.dumps(config, indent=2)}")
    print(f"epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"device={device}  hidden_dim={args.hidden_dim}  m_max={args.m_max}  K_max={args.K_max}")
    sys.stdout.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    s, t, v, tr, scale, phi, V_lab = build_dataset(config, seed=args.seed)
    N = s.shape[0]
    print(f"Dataset: {N} independent (state, time) samples | state={s.shape} phi={phi.shape}")
    print(f"  oracle+convert time: {time.time()-t0:.0f}s")
    sys.stdout.flush()

    perm = np.random.default_rng(args.seed + 1).permutation(N)
    n_train = int(N * (1 - args.val_split))
    tr_idx = perm[:n_train]
    vl_idx = perm[n_train:]
    print(f"Train: {n_train}  Val: {N - n_train}")

    def to_t(arr, idx):
        return torch.from_numpy(arr[idx]).float().to(device)

    tr_ds = TensorDataset(
        to_t(s, tr_idx), to_t(t, tr_idx), to_t(v, tr_idx), to_t(tr, tr_idx),
        to_t(scale, tr_idx), to_t(phi, tr_idx), to_t(V_lab, tr_idx))
    vl_ds = TensorDataset(
        to_t(s, vl_idx), to_t(t, vl_idx), to_t(v, vl_idx), to_t(tr, vl_idx),
        to_t(scale, vl_idx), to_t(phi, vl_idx), to_t(V_lab, vl_idx))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    vl_ld = DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False)

    model = FlexibleFairPivotNet(
        hidden_dim=args.hidden_dim, m_max=args.m_max, K_max=args.K_max).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    sys.stdout.flush()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for ep in range(args.epochs):
        te = time.time()
        model.train()
        tr_stats = {"total": [], "l_phi": []}
        for ss, tt, vv, trr, sc, pp, VV in tr_ld:
            pred = model(ss, tt, vv, trr, scale_feat=sc)
            L = fair_pivot_loss(pred, {"phi": pp})
            optim.zero_grad()
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            for k in tr_stats:
                tr_stats[k].append(L[k].item())
        sched.step()

        model.eval()
        vl_stats = {"total": [], "l_phi": []}
        with torch.no_grad():
            for ss, tt, vv, trr, sc, pp, VV in vl_ld:
                pred = model(ss, tt, vv, trr, scale_feat=sc)
                L = fair_pivot_loss(pred, {"phi": pp})
                for k in vl_stats:
                    vl_stats[k].append(L[k].item())

        vt = np.mean(vl_stats["total"])
        if vt < best_val:
            best_val = vt
            torch.save(model.state_dict(), args.output)

        if ep % args.log_every == 0 or ep == args.epochs - 1:
            t_phi = np.mean(tr_stats['l_phi'])
            v_phi = np.mean(vl_stats['l_phi'])
            print(f" e{ep:3d} | tr phi={t_phi:.4f}"
                  f" | vl phi={v_phi:.4f}"
                  f" | best={best_val:.4f} | {time.time()-te:.1f}s")
            sys.stdout.flush()

    print(f"\n=== DONE: best={best_val:.6f} saved to {args.output} ===")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
