"""End-to-end evaluation of the learned model on multiple experiments."""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "code_dp"))

from model.backbone_flex import FlexibleFairPivotNet
from train.train_loop_flex import GROUP_SIZE_REF, VALUATION_REF, normalize_raw_samples

from auction_env.config import AuctionConfig
from auction_env.data_generator import generate_data
from pvt_fair import FairnessRealAuction


_MODEL = None
_DEVICE = None

def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _load_model(ckpt_path: str, **model_kwargs) -> FlexibleFairPivotNet:
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    _DEVICE = _get_device()
    model = FlexibleFairPivotNet(**model_kwargs)
    ckpt = torch.load(ckpt_path, map_location=_DEVICE)
    # Backwards-compatible: train_loop.py saves {state_dict, args, ...};
    # train_loop_flex.py saves raw state_dict.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.to(_DEVICE)
    model.eval()
    _MODEL = model
    print(f"  FlexNet loaded from {ckpt_path} on {_DEVICE}")
    return model


def predict_p0(
    model: FlexibleFairPivotNet,
    group_sizes: np.ndarray,   # (m,)
    valuations: np.ndarray,    # (m, K)
    matrix0: np.ndarray,        # (K, K)
    matrix1: np.ndarray,        # (K, K)
    T: int,
    delta: float,
    epsilon: float,
) -> dict:
    """
    Enumerate all calc states, run FlexNet inference, return
    {(t, g0s1, g1s1, ...): p0} dict matching FairnessRealAuction.real_rnn_p0 format.
    """
    m = len(group_sizes)
    K = valuations.shape[1]
    group_v_max = np.max(valuations, axis=1)
    if np.any(group_sizes > GROUP_SIZE_REF):
        raise ValueError(
            f"group size exceeds training reference GROUP_SIZE_REF={GROUP_SIZE_REF:g}"
        )
    if np.any(group_v_max > VALUATION_REF):
        raise ValueError(
            f"valuation exceeds training reference VALUATION_REF={VALUATION_REF:g}"
        )

    from oracle.fair_pivot_bi import enumerate_market_states_K
    states = enumerate_market_states_K(group_sizes.tolist(), K)  # (S, m*K)
    S = states.shape[0]

    # Reuse the training-time normalisation so online eval matches training.
    val_b = np.broadcast_to(valuations, (S, m, K))
    mat0_b = np.broadcast_to(matrix0, (S, m, K, K))
    mat1_b = np.broadcast_to(matrix1, (S, m, K, K))
    state_feat, time_feat, val_feat, trans_feat, scale_feat = normalize_raw_samples(
        states, val_b, mat0_b, mat1_b,
        np.asarray(group_sizes), T, delta, epsilon,
    )

    sf = torch.from_numpy(state_feat).float().to(_DEVICE)
    tf = torch.from_numpy(time_feat).float().to(_DEVICE)
    vf = torch.from_numpy(val_feat).float().to(_DEVICE)
    trf = torch.from_numpy(trans_feat).float().to(_DEVICE)
    scf = torch.from_numpy(scale_feat).float().to(_DEVICE)

    with torch.no_grad():
        out = model(sf, tf, vf, trf, scale_feat=scf)
        # Model output is point-wise (S*T, m); table shape is restored outside.
        p0_tensor = out["phi"][:, 0].cpu().numpy().reshape(S, T)

    # Build dict: key = (t, g0s1, g1s1, ...)
    result = {}
    for s_idx in range(S):
        state_key = tuple(int(states[s_idx, k * K + 1]) for k in range(m))
        for t in range(T):
            result[(t, ) + state_key] = float(p0_tensor[s_idx, t])
    return result



_WORKER_GROUPS = None
_WORKER_AGENTS = None
_WORKER_OTHER = None
_WORKER_CONFIG = None
_WORKER_AUC_GS = None
_WORKER_CGS = None
_WORKER_PRECOMPUTED_P0 = None


# _worker_init / _WORKER_* removed: each task carries its own groups/agents/p0
# (so re-generated groups are visible to the worker for variance injection).


def _worker_init_single(args, auc_gs, cgs):
    """Single-process variant: just set globals for _run_one_auction."""
    global _WORKER_GROUPS, _WORKER_AGENTS, _WORKER_OTHER, _WORKER_CONFIG
    global _WORKER_AUC_GS, _WORKER_CGS, _WORKER_PRECOMPUTED_P0
    auction_seed, run_data = args
    groups, agents, other_data, config, flex_p0 = run_data
    _WORKER_GROUPS = groups
    _WORKER_AGENTS = agents
    _WORKER_OTHER = other_data
    _WORKER_CONFIG = config
    _WORKER_AUC_GS = auc_gs
    _WORKER_CGS = cgs
    _WORKER_PRECOMPUTED_P0 = flex_p0


def _run_one_auction(args) -> dict:
    auction_seed, run_data = args
    groups, agents, other_data, config, flex_p0 = run_data
    rng = np.random.default_rng(auction_seed)
    auction = FairnessRealAuction(
        groups, agents, other_data, config, rng,
        precomputed_rnn_p0=flex_p0,
    )
    res = auction.run()
    return _extract_results(auction, res, config.T)


def _worker_run_with_init(args_and_sizes):
    """Pool worker entry: set globals then call _run_one_auction."""
    args, auc_gs, cgs = args_and_sizes
    _worker_init_single(args, auc_gs, cgs)
    return _run_one_auction(args)


def _extract_results(auction, res, T):
    out = {"pvt_w": [0.0]*T, "pvt_p": [0.0]*T, "pvt_g0": [0.0]*T, "pvt_g1": [0.0]*T,
           "dp_w": [0.0]*T,  "dp_p": [0.0]*T,  "dp_p0": [0.0]*T,
           "dp_g0": [0.0]*T, "dp_g1": [0.0]*T,
           "dp_ow0": [0.0]*T, "dp_ow1": [0.0]*T,
           "dp_cow0": [0.0]*T, "dp_cow1": [0.0]*T,
           "rnn_w": [0.0]*T, "rnn_p": [0.0]*T, "rnn_p0": [0.0]*T,
           "rnn_g0": [0.0]*T, "rnn_g1": [0.0]*T,
           "rnn_ow0": [0.0]*T, "rnn_ow1": [0.0]*T,
           "rnn_cow0": [0.0]*T, "rnn_cow1": [0.0]*T,
           "dp_time": 0.0, "rnn_time": 0.0, "pvt_time": 0.0}
    for t in range(T):
        out["pvt_w"][t]  = float(res["pvt_real_social_welfare"][t])
        out["pvt_p"][t]  = float(res["pvt_real_payment"][t])
        out["dp_w"][t]   = float(res["dp_real_social_welfare"][t])
        out["dp_p"][t]   = float(res["dp_real_payment"][t])
        out["rnn_w"][t]  = float(res["rnn_real_social_welfare"][t])
        out["rnn_p"][t]  = float(res["rnn_real_payment"][t])
        out["dp_p0"][t]  = float(res["statistic_dp_p0"][t])
        out["rnn_p0"][t] = float(res["statistic_rnn_p0"][t])
        out["pvt_g0"][t] = float(res["pvt_real_social_welfare_group0"][t]) / max(_WORKER_AUC_GS[0], 1)
        out["pvt_g1"][t] = float(res["pvt_real_social_welfare_group1"][t]) / max(_WORKER_AUC_GS[1], 1)
        out["dp_g0"][t]  = float(res["dp_real_social_welfare_group0"][t])  / max(_WORKER_AUC_GS[0], 1)
        out["dp_g1"][t]  = float(res["dp_real_social_welfare_group1"][t])  / max(_WORKER_AUC_GS[1], 1)
        out["rnn_g0"][t] = float(res["rnn_real_social_welfare_group0"][t]) / max(_WORKER_AUC_GS[0], 1)
        out["rnn_g1"][t] = float(res["rnn_real_social_welfare_group1"][t]) / max(_WORKER_AUC_GS[1], 1)
        out["dp_ow0"][t]  = float(res["dp_only_expected_social_welfare_group0"][t])  / max(_WORKER_CGS[0], 1)
        out["dp_ow1"][t]  = float(res["dp_only_expected_social_welfare_group1"][t])  / max(_WORKER_CGS[1], 1)
        out["rnn_ow0"][t] = float(res["rnn_only_expected_social_welfare_group0"][t]) / max(_WORKER_CGS[0], 1)
        out["rnn_ow1"][t] = float(res["rnn_only_expected_social_welfare_group1"][t]) / max(_WORKER_CGS[1], 1)
        out["dp_cow0"][t]  = float(res["dp_calc_only_expected_social_welfare_group0"][t])  / max(_WORKER_CGS[0], 1)
        out["dp_cow1"][t]  = float(res["dp_calc_only_expected_social_welfare_group1"][t])  / max(_WORKER_CGS[1], 1)
        out["rnn_cow0"][t] = float(res["rnn_calc_only_expected_social_welfare_group0"][t]) / max(_WORKER_CGS[0], 1)
        out["rnn_cow1"][t] = float(res["rnn_calc_only_expected_social_welfare_group1"][t]) / max(_WORKER_CGS[1], 1)
    out["dp_time"]  = float(res.get("dp_time_cost", 0))
    out["rnn_time"] = float(res.get("rnn_time_cost", 0))
    out["pvt_time"] = float(res.get("pvt_time_cost", 0))
    return out



def run_flex_experiment(
    exp_config: dict,
    model_ckpt: str,
    csv_dir: str,
    csv_name: str,
    num_runs: int = 500,
):
    """Run one experiment evaluation and write CSVs.

    If the legacy CSV already exists, the function returns early.
    """
    legacy_path = os.path.join(csv_dir, csv_name)
    if os.path.exists(legacy_path):
        print(f"[skip-existing] {legacy_path} already exists; skipping simulation.")
        return
    print(f"\n{'='*60}")
    print(f"=== FlexNet eval: {csv_name} ===")
    print(f"{'='*60}")

    model = _load_model(model_ckpt,
                         hidden_dim=128, m_max=4, K_max=4)

    config = AuctionConfig()
    config.EPSILON = float(exp_config["epsilon"])
    config.T = int(exp_config["T"])
    config.DELTA = float(exp_config["delta"])

    # === 3 independent RNGs ===
    # 1. training-data sampling seed: cfg["seed"] (used in training script)
    # 2. neural-network seed: torch.manual_seed() (used in training script)
    # 3. simulation seed: config.SEED = 314159 (AuctionConfig default)
    #    -> used to generate market + auction seed pool
    seed_pool_rng = np.random.default_rng(config.SEED)
    seed_of_value    = int(seed_pool_rng.integers(0, 2**32 - 1, dtype=np.uint32))
    seed_of_matrix   = int(seed_pool_rng.integers(0, 2**32 - 1, dtype=np.uint32))
    seed_of_grouping = int(seed_pool_rng.integers(0, 2**32 - 1, dtype=np.uint32))

    groups, agents, other_data, config = generate_data(
        config=config, epsilon=config.EPSILON,
        seed_of_value=seed_of_value,
        seed_of_matrix=seed_of_matrix,
        seed_of_grouping=seed_of_grouping,
        specified_group_sizes=exp_config["group_sizes"],
    )
    config.EPSILON_VALUES = [config.EPSILON]
    print(f"  m={len(groups)} K={groups[0].valuations.shape[0]} T={config.T} delta={config.DELTA} epsilon={config.EPSILON}")

    NUM_RUNS = num_runs
    g_size_arr = np.array(exp_config["group_sizes"], dtype=np.int64)
    seed_pool_rng2 = np.random.default_rng(config.SEED)
    run_seed_pool = []
    for _ in range(NUM_RUNS):
        s_v = int(seed_pool_rng2.integers(0, 2**32 - 1, dtype=np.uint32))
        s_m = int(seed_pool_rng2.integers(0, 2**32 - 1, dtype=np.uint32))
        s_g = int(seed_pool_rng2.integers(0, 2**32 - 1, dtype=np.uint32))
        run_seed_pool.append((s_v, s_m, s_g))
    auction_seeds = [int(s) for s in
                     np.random.SeedSequence(config.SEED + 1).generate_state(NUM_RUNS)]

    # Pre-generate groups for each run; batch FlexNet inference across all runs.
    from oracle.fair_pivot_bi import enumerate_market_states_K
    t0_flex = time.perf_counter()
    R = NUM_RUNS
    S = enumerate_market_states_K(exp_config["group_sizes"], 2).shape[0]
    T_ = int(exp_config["T"])

    per_run_vals = np.zeros((R, len(exp_config["group_sizes"]), 2), dtype=np.float32)
    per_run_m0  = np.zeros((R, 2, 2), dtype=np.float32)
    per_run_m1  = np.zeros((R, 2, 2), dtype=np.float32)
    states_list = []
    run_data_list = []
    for run_id in range(R):
        s_v, s_m, s_g = run_seed_pool[run_id]
        cfg_run = AuctionConfig()
        cfg_run.EPSILON = float(exp_config["epsilon"])
        cfg_run.T = int(exp_config["T"])
        cfg_run.DELTA = float(exp_config["delta"])
        cfg_run.SEED_OF_VALUE = np.uint32(s_v)
        cfg_run.SEED_OF_MATRIX = np.uint32(s_m)
        cfg_run.SEED_OF_GROUPING = np.uint32(s_g)
        cfg_run.EPSILON_VALUES = [cfg_run.EPSILON]
        g, a, od, _ = generate_data(
            config=cfg_run, epsilon=cfg_run.EPSILON,
            seed_of_value=s_v, seed_of_matrix=s_m, seed_of_grouping=s_g,
            specified_group_sizes=exp_config["group_sizes"],
        )
        vals = np.vstack([gi.valuations for gi in g])
        per_run_vals[run_id] = vals.astype(np.float32)
        per_run_m0[run_id]  = g[0].matrix0.astype(np.float32)
        per_run_m1[run_id]  = g[0].matrix1.astype(np.float32)
        states = enumerate_market_states_K(exp_config["group_sizes"], 2)
        states_list.append(states)
        run_data_list.append((g, a, od, cfg_run, None))  # placeholder

    # Stack: (R, S, m*K) states
    states_all = np.stack(states_list)  # (R, S, m*K)
    val_b   = np.broadcast_to(per_run_vals[:, None, :, :], (R, S, *per_run_vals.shape[1:]))  # (R, S, m, K)
    # Add a dummy m axis to matrices (same matrix used for all groups)
    m_groups = per_run_vals.shape[1]
    mat0_b = np.broadcast_to(per_run_m0[:, None, None, :, :], (R, S, m_groups, *per_run_m0.shape[1:]))  # (R, S, m, K, K)
    mat1_b = np.broadcast_to(per_run_m1[:, None, None, :, :], (R, S, m_groups, *per_run_m1.shape[1:]))  # (R, S, m, K, K)

    # Normalisation expands each (run, state) row into independent time rows,
    # ordered as run -> state -> time.  The model receives no T axis.
    m_groups = per_run_vals.shape[1]
    state_feat, time_feat, val_feat, trans_feat, scale_feat = normalize_raw_samples(
        states_all.reshape(R*S, -1),
        val_b.reshape(R*S, m_groups, per_run_vals.shape[2]),
        mat0_b.reshape(R*S, m_groups, per_run_m0.shape[1], per_run_m0.shape[2]),
        mat1_b.reshape(R*S, m_groups, per_run_m1.shape[1], per_run_m1.shape[2]),
        np.asarray(exp_config["group_sizes"], dtype=np.int64),
        T_,
        float(exp_config["delta"]),
        float(exp_config["epsilon"]),
    )
    sf_all = torch.from_numpy(state_feat).float()
    tf_all = torch.from_numpy(time_feat).float()
    vf_all = torch.from_numpy(val_feat).float()
    trf_all = torch.from_numpy(trans_feat).float()
    scf_all = torch.from_numpy(scale_feat).float()

    n_queries = R * S * T_
    p0_flat = np.zeros(n_queries, dtype=np.float32)
    chunk_size = 4096
    with torch.no_grad():
        for start in range(0, n_queries, chunk_size):
            end = min(start + chunk_size, n_queries)
            sl = slice(start, end)
            out = model(
                sf_all[sl].to(_DEVICE),
                tf_all[sl].to(_DEVICE),
                vf_all[sl].to(_DEVICE),
                trf_all[sl].to(_DEVICE),
                scale_feat=scf_all[sl].to(_DEVICE),
            )
            p0_flat[sl] = out["phi"][:, 0].cpu().numpy()
    p0_all = p0_flat.reshape(R, S, T_)
    del sf_all, tf_all, vf_all, trf_all, scf_all

    m_groups = len(exp_config["group_sizes"])
    for run_id in range(R):
        flex_p0_i = {}
        for s_idx in range(S):
            state_key = tuple(int(states_list[run_id][s_idx, k * 2 + 1]) for k in range(m_groups))
            for t in range(T_):
                flex_p0_i[(t,) + state_key] = float(p0_all[run_id, s_idx, t])
        g, a, od, cfg_run, _ = run_data_list[run_id]
        run_data_list[run_id] = (g, a, od, cfg_run, flex_p0_i)
    flex_time = time.perf_counter() - t0_flex
    print(f"  FlexNet batched p0 prediction: {R} runs in {flex_time:.2f}s "
          f"({R*S*T_} independent state-time queries)")

    auc_gs = list(exp_config["group_sizes"])
    cgs = list(exp_config.get("calc_group_sizes", auc_gs))
    print(f"  auc_group_sizes={auc_gs} calc_group_sizes={cgs}")

    T = config.T
    # per-run accumulators for 1-9 style detailed CSV
    pvt_w_runs  = np.zeros((NUM_RUNS, T))
    pvt_p_runs  = np.zeros((NUM_RUNS, T))
    pvt_g0_runs = np.zeros((NUM_RUNS, T)); pvt_g1_runs = np.zeros((NUM_RUNS, T))
    dp_w_runs   = np.zeros((NUM_RUNS, T))
    dp_p_runs   = np.zeros((NUM_RUNS, T))
    dp_p0_runs  = np.zeros((NUM_RUNS, T))
    dp_g0_runs  = np.zeros((NUM_RUNS, T));  dp_g1_runs  = np.zeros((NUM_RUNS, T))
    dp_ow0_runs = np.zeros((NUM_RUNS, T));  dp_ow1_runs = np.zeros((NUM_RUNS, T))
    dp_cow0_runs = np.zeros((NUM_RUNS, T)); dp_cow1_runs = np.zeros((NUM_RUNS, T))
    rnn_w_runs  = np.zeros((NUM_RUNS, T))
    rnn_p_runs  = np.zeros((NUM_RUNS, T))
    rnn_p0_runs = np.zeros((NUM_RUNS, T))
    rnn_g0_runs = np.zeros((NUM_RUNS, T)); rnn_g1_runs = np.zeros((NUM_RUNS, T))
    rnn_ow0_runs = np.zeros((NUM_RUNS, T)); rnn_ow1_runs = np.zeros((NUM_RUNS, T))
    rnn_cow0_runs = np.zeros((NUM_RUNS, T)); rnn_cow1_runs = np.zeros((NUM_RUNS, T))
    dp_time_runs  = np.zeros(NUM_RUNS)
    rnn_time_runs = np.zeros(NUM_RUNS)
    pvt_time_runs = np.zeros(NUM_RUNS)
    flex_offline_runs = np.zeros(NUM_RUNS)   # offline p0-table prep amortized per run
    # Track which run_ids have *all* T rows landed in detail CSV (for resume).
    detailed_done = np.zeros(NUM_RUNS, dtype=bool)
    # Track whether the detailed CSV has already received its header (first write).
    detailed_header_written = False

    # run_ids and pre-populate the per-run arrays so that the final summary /
    # ci95 / legacy outputs match a fully uninterrupted experiment.
    os.makedirs(csv_dir, exist_ok=True)
    csv_stem = os.path.splitext(csv_name)[0]
    out_detail  = os.path.join(csv_dir, f"{csv_stem}_detailed.csv")
    out_summary = os.path.join(csv_dir, f"{csv_stem}_summary.csv")
    out_ci      = os.path.join(csv_dir, f"{csv_stem}_ci95.csv")
    out_legacy  = os.path.join(csv_dir, csv_name)

    if os.path.exists(out_detail):
        try:
            _prev = pd.read_csv(out_detail)
            if "run_id" in _prev.columns:
                # Each run contributes T rows with the same run_id.
                # We treat a run as "done" if it has the full T rows.
                _counts = _prev.groupby("run_id").size()
                _prev_done_ids = sorted(int(r) for r in _counts.index
                                        if int(_counts.loc[r]) >= T)
                if _prev_done_ids:
                    print(f"  [resume] {out_detail} exists with "
                          f"{len(_prev_done_ids)}/{NUM_RUNS} finished runs "
                          f"(fully-populated). Restoring per-run accumulators.")
                    _DETAIL_NUM_COLS = [
                        "pvt_real_welfare", "pvt_real_payment",
                        "pvt_real_welfare_group0", "pvt_real_welfare_group1",
                        "dp_real_welfare", "dp_real_payment",
                        "dp_real_welfare_group0", "dp_real_welfare_group1",
                        "dp_p0", "rnn_p0",
                        "rnn_real_welfare", "rnn_real_payment",
                        "rnn_real_welfare_group0", "rnn_real_welfare_group1",
                        "dp_only_expected_welfare0", "dp_only_expected_welfare1",
                        "rnn_only_expected_welfare0", "rnn_only_expected_welfare1",
                        "dp_calc_only_expected_welfare_group0",
                        "dp_calc_only_expected_welfare_group1",
                        "rnn_calc_only_expected_welfare_group0",
                        "rnn_calc_only_expected_welfare_group1",
                    ]
                    _TIME_COLS = ["dp_time", "rnn_time", "pvt_time"]
                    # The detailed CSV is (run_id, period)-major. Pivot to (run, T) for arrays.
                    for _rid in _prev_done_ids:
                        rows = _prev[_prev["run_id"] == _rid].sort_values(by="period")
                        if len(rows) != T:
                            continue
                        _map = {
                            "pvt_w_runs":  "pvt_real_welfare",
                            "pvt_p_runs":  "pvt_real_payment",
                            "pvt_g0_runs": "pvt_real_welfare_group0",
                            "pvt_g1_runs": "pvt_real_welfare_group1",
                            "dp_w_runs":   "dp_real_welfare",
                            "dp_p_runs":   "dp_real_payment",
                            "dp_g0_runs":  "dp_real_welfare_group0",
                            "dp_g1_runs":  "dp_real_welfare_group1",
                            "dp_p0_runs":  "dp_p0",
                            "rnn_p0_runs": "rnn_p0",
                            "rnn_w_runs":  "rnn_real_welfare",
                            "rnn_p_runs":  "rnn_real_payment",
                            "rnn_g0_runs": "rnn_real_welfare_group0",
                            "rnn_g1_runs": "rnn_real_welfare_group1",
                            "dp_ow0_runs": "dp_only_expected_welfare0",
                            "dp_ow1_runs": "dp_only_expected_welfare1",
                            "rnn_ow0_runs": "rnn_only_expected_welfare0",
                            "rnn_ow1_runs": "rnn_only_expected_welfare1",
                            "dp_cow0_runs": "dp_calc_only_expected_welfare_group0",
                            "dp_cow1_runs": "dp_calc_only_expected_welfare_group1",
                            "rnn_cow0_runs": "rnn_calc_only_expected_welfare_group0",
                            "rnn_cow1_runs": "rnn_calc_only_expected_welfare_group1",
                        }
                        for arr_name, col_name in _map.items():
                            vals = rows[col_name].to_numpy()
                            if vals.shape[0] == T:
                                globals()[arr_name][_rid] = vals
                        # Time columns: a single value per run. Use period-0 row.
                        if "dp_time" in rows.columns:
                            dp_time_runs[_rid] = float(rows["dp_time"].iloc[0])
                        if "rnn_time" in rows.columns:
                            rnn_time_runs[_rid] = float(rows["rnn_time"].iloc[0])
                        if "pvt_time" in rows.columns:
                            pvt_time_runs[_rid] = float(rows["pvt_time"].iloc[0])
                        detailed_done[_rid] = True
                    if _prev_done_ids:
                        # If the detailed CSV already has rows, we are continuing
                        # an append-only stream, so future writes must NOT write
                        # the header again.
                        detailed_header_written = True
            del _prev
        except Exception as _exc:
            print(f"  [resume] could not read existing {out_detail}: {_exc}; "
                  "starting fresh.")
            detailed_done[:] = False
            detailed_header_written = False
            # Truncate so we start clean if the existing file is malformed.
            try:
                os.remove(out_detail)
            except OSError:
                pass

    pending_ids = [i for i in range(NUM_RUNS) if not detailed_done[i]]
    if not pending_ids and NUM_RUNS > 0:
        # All runs already done: skip the simulation entirely.
        # We still need to rebuild summary/ci95/legacy from the detail CSV (see end).
        results_list = []
    else:
        # Default 8; override via env var EVAL_N_WORKERS.  For large-group
        # configs (e.g. b_n100/140 group=[50,50] and c_part partitions) each
        # auction enumerates (N+1)^2 calc states, so parallel workers can OOM
        # the box.  Cap to 8 workers when group_size >= 50 to keep memory in
        # check (Windows 30 GB / 18 cores baseline); small-group configs keep
        # the user-provided cap.
        _max_workers = int(os.environ.get("EVAL_N_WORKERS", "8"))
        _large_group_cap = 8 if max(auc_gs) >= 50 else _max_workers
        n_workers = max(
            1,
            min(len(pending_ids), _large_group_cap, (os.cpu_count() or 4)),
        )
        print(f"  Using {n_workers} CPU workers "
              f"(NUM_RUNS={NUM_RUNS}, pending={len(pending_ids)}, "
              f"max_gs={max(auc_gs)})")

        # Build per-run tasks: (run_id, auction_seed, run_data_tuple).
        # Carry run_id explicitly so we know where to write back after imap_unordered.
        # Each worker task is (auction_seed, run_data) — same shape consumed by
        # ``_worker_init_single`` and ``_run_one_auction``.
        all_tasks = [
            (i, auction_seeds[i], (auction_seeds[i], run_data_list[i]))
            for i in pending_ids
        ]

        def _write_detail_rows(run_id, r):
            """Append T rows for a single run_id to out_detail (header once)."""
            nonlocal detailed_header_written
            eps_local = exp_config["epsilon"]
            per_call_local = flex_time / max(R * S * T_, 1)
            rnn_t_local = T_ * per_call_local
            rnn_offline_local = flex_time / R
            local_rows = []
            for t in range(T):
                # For new runs we use the freshly computed values; for resumed rows
                # the arrays already hold the values from the prior write.
                pvt_f_local = abs(pvt_g0_runs[run_id, t] - pvt_g1_runs[run_id, t])
                dp_f_local  = abs(dp_g0_runs[run_id, t]  - dp_g1_runs[run_id, t])
                rnn_f_local = abs(rnn_g0_runs[run_id, t] - rnn_g1_runs[run_id, t])
                dp_only_f_local  = abs(dp_ow0_runs[run_id, t]  - dp_ow1_runs[run_id, t])
                rnn_only_f_local = abs(rnn_ow0_runs[run_id, t] - rnn_ow1_runs[run_id, t])
                local_rows.append({
                    "epsilon": eps_local,
                    "run_id": run_id,
                    "period": t,
                    "pvt_real_fairness":       float(pvt_f_local),
                    "dp_real_fairness":        float(dp_f_local),
                    "rnn_real_fairness":       float(rnn_f_local),
                    "dp_only_expected_fairness":  float(dp_only_f_local),
                    "rnn_only_expected_fairness": float(rnn_only_f_local),
                    "dp_calc_only_expected_fairness":
                        abs(dp_cow0_runs[run_id, t] - dp_cow1_runs[run_id, t]),
                    "rnn_calc_only_expected_fairness":
                        abs(rnn_cow0_runs[run_id, t] - rnn_cow1_runs[run_id, t]),
                    "pvt_real_welfare":        float(pvt_w_runs[run_id, t]),
                    "pvt_real_payment":        float(pvt_p_runs[run_id, t]),
                    "pvt_real_welfare_group0": float(pvt_g0_runs[run_id, t]),
                    "pvt_real_welfare_group1": float(pvt_g1_runs[run_id, t]),
                    "dp_p0":                   float(dp_p0_runs[run_id, t]),
                    "dp_real_welfare":         float(dp_w_runs[run_id, t]),
                    "dp_real_payment":         float(dp_p_runs[run_id, t]),
                    "dp_real_welfare_group0":  float(dp_g0_runs[run_id, t]),
                    "dp_real_welfare_group1":  float(dp_g1_runs[run_id, t]),
                    "dp_only_expected_welfare0":  float(dp_ow0_runs[run_id, t]),
                    "dp_only_expected_welfare1":  float(dp_ow1_runs[run_id, t]),
                    "dp_calc_only_expected_welfare_group0": float(dp_cow0_runs[run_id, t]),
                    "dp_calc_only_expected_welfare_group1": float(dp_cow1_runs[run_id, t]),
                    "rnn_p0":                  float(rnn_p0_runs[run_id, t]),
                    "rnn_real_welfare":        float(rnn_w_runs[run_id, t]),
                    "rnn_real_payment":        float(rnn_p_runs[run_id, t]),
                    "rnn_real_welfare_group0": float(rnn_g0_runs[run_id, t]),
                    "rnn_real_welfare_group1": float(rnn_g1_runs[run_id, t]),
                    "rnn_only_expected_welfare0": float(rnn_ow0_runs[run_id, t]),
                    "rnn_only_expected_welfare1": float(rnn_ow1_runs[run_id, t]),
                    "rnn_calc_only_expected_welfare_group0": float(rnn_cow0_runs[run_id, t]),
                    "rnn_calc_only_expected_welfare_group1": float(rnn_cow1_runs[run_id, t]),
                    "group0_size": int(auc_gs[0]),
                    "group1_size": int(auc_gs[1]) if len(auc_gs) > 1 else 0,
                    "auction_seed": int(auction_seeds[run_id]),
                    "dp_time":  float(dp_time_runs[run_id]),
                    "rnn_time": float(rnn_t_local),
                    "pvt_time": float(pvt_time_runs[run_id]),
                })
            df_local = pd.DataFrame(local_rows)
            write_header = not detailed_header_written
            df_local.to_csv(out_detail, mode="a", header=write_header, index=False)
            detailed_header_written = True
            return local_rows

        def _accumulate_and_persist(run_id, r):
            """Write one run's results into the per-run arrays and append to CSV."""
            pvt_w_runs[run_id]  = r["pvt_w"]
            pvt_p_runs[run_id]  = r["pvt_p"]
            dp_w_runs[run_id]   = r["dp_w"]
            dp_p_runs[run_id]   = r["dp_p"]
            rnn_w_runs[run_id]  = r["rnn_w"]
            rnn_p_runs[run_id]  = r["rnn_p"]
            dp_p0_runs[run_id]  = r["dp_p0"]
            rnn_p0_runs[run_id] = r["rnn_p0"]
            pvt_g0_runs[run_id] = r["pvt_g0"]; pvt_g1_runs[run_id] = r["pvt_g1"]
            dp_g0_runs[run_id]  = r["dp_g0"];  dp_g1_runs[run_id]  = r["dp_g1"]
            rnn_g0_runs[run_id] = r["rnn_g0"]; rnn_g1_runs[run_id] = r["rnn_g1"]
            dp_ow0_runs[run_id] = r["dp_ow0"]; dp_ow1_runs[run_id] = r["dp_ow1"]
            rnn_ow0_runs[run_id] = r["rnn_ow0"]; rnn_ow1_runs[run_id] = r["rnn_ow1"]
            dp_cow0_runs[run_id] = r["dp_cow0"]; dp_cow1_runs[run_id] = r["dp_cow1"]
            rnn_cow0_runs[run_id] = r["rnn_cow0"]; rnn_cow1_runs[run_id] = r["rnn_cow1"]
            dp_time_runs[run_id]  = r["dp_time"]
            # rnn_time semantics: real online auction cost = per-call forward × T (steps/run).
            # Per-call time = batched forward total / (R*S*T) where S=states, T=timesteps.
            # The batched `flex_time` is used to PRE-COMPUTE the full p0 table (for calc-only
            # expected fairness, which is an offline metric, not a real-time auction cost).
            # We store that as a SEPARATE column `flex_offline_time` so time_bar_chart can
            # show the honest per-call online cost only.
            per_call = flex_time / max(R * S * T_, 1)
            rnn_time_runs[run_id] = T_ * per_call  # one auction calls the net T times
            pvt_time_runs[run_id] = r.get("pvt_time", 0.0)
            flex_offline_runs[run_id] = flex_time / R  # offline prep amortized per run (informational)
            detailed_done[run_id] = True
            _write_detail_rows(run_id, r)

        t0_auc = time.perf_counter()
        if n_workers == 1:
            for (run_id, _seed, task) in all_tasks:
                _worker_init_single(task, auc_gs, cgs)
                r = _run_one_auction(task)
                _accumulate_and_persist(run_id, r)
        else:
            ctx = mp.get_context("spawn")
            def _packed(task_with_id):
                rid, seed, task = task_with_id
                return (task, auc_gs, cgs)
            packed = [_packed(t) for t in all_tasks]
            run_id_by_pos = {pos: all_tasks[pos][0] for pos in range(len(all_tasks))}
            with ctx.Pool(processes=n_workers) as pool:
                for pos, r in enumerate(pool.imap_unordered(
                        _worker_run_with_init, packed, chunksize=1)):
                    run_id = run_id_by_pos[pos]
                    _accumulate_and_persist(run_id, r)

        print(f"  {NUM_RUNS} runs in {time.perf_counter() - t0_auc:.1f}s "
              f"(pending={len(pending_ids)})")
    # ---- detailed CSV has already been written incrementally ----------
    # Each completed run appended T rows to ``out_detail`` as soon as its
    # auction finished. After the pool loop, the CSV already contains every
    # (run, period) row we need. We only sanity-check the row count here.
    eps = exp_config["epsilon"]
    os.makedirs(csv_dir, exist_ok=True)
    csv_stem = os.path.splitext(csv_name)[0]
    out_detail = os.path.join(csv_dir, f"{csv_stem}_detailed.csv")
    if os.path.exists(out_detail):
        df_disk = pd.read_csv(out_detail)
        expected_rows = NUM_RUNS * T
        if len(df_disk) != expected_rows:
            print(f"  [warn] {out_detail} has {len(df_disk)} rows, expected "
                  f"{expected_rows}; check for interrupted writes.")
        print(f"  -> {out_detail}  ({len(df_disk)} rows, incrementally appended)")
    else:
        # No pending runs and no prior CSV: very small NUM_RUNS (e.g. dry-run)
        # path. Write the (now empty) detail CSV with header only.
        empty = pd.DataFrame(columns=[
            "epsilon", "run_id", "period",
            "dp_calc_only_expected_fairness", "rnn_calc_only_expected_fairness",
            "pvt_real_welfare", "pvt_real_payment",
            "pvt_real_welfare_group0", "pvt_real_welfare_group1",
            "dp_p0", "dp_real_welfare", "dp_real_payment",
            "dp_real_welfare_group0", "dp_real_welfare_group1",
            "dp_only_expected_welfare0", "dp_only_expected_welfare1",
            "dp_calc_only_expected_welfare_group0",
            "dp_calc_only_expected_welfare_group1",
            "rnn_p0", "rnn_real_welfare", "rnn_real_payment",
            "rnn_real_welfare_group0", "rnn_real_welfare_group1",
            "rnn_only_expected_welfare0", "rnn_only_expected_welfare1",
            "rnn_calc_only_expected_welfare_group0",
            "rnn_calc_only_expected_welfare_group1",
            "group0_size", "group1_size", "auction_seed",
            "dp_time", "rnn_time", "pvt_time",
        ])
        empty.to_csv(out_detail, index=False)
        print(f"  -> {out_detail}  (0 rows, header only)")

    rows_summary = []
    for t in range(T):
        # Fairness aligned with 1.csv convention: first mean across runs,
        # then |g0 - g1| (per_capita).  This makes the aggregate numbers
        # directly comparable to 1fairness_parameter/1/1.csv.
        pvt_f = abs(pvt_g0_runs[:, t].mean() - pvt_g1_runs[:, t].mean())
        dp_f  = abs(dp_g0_runs[:, t].mean()  - dp_g1_runs[:, t].mean())
        rnn_f = abs(rnn_g0_runs[:, t].mean() - rnn_g1_runs[:, t].mean())
        rows_summary.append({
            "epsilon": eps,
            "period": t,
            "pvt_real_welfare_mean": float(pvt_w_runs[:, t].mean()),
            "pvt_real_welfare_std":  float(pvt_w_runs[:, t].std()),
            "dp_real_welfare_mean":  float(dp_w_runs[:, t].mean()),
            "dp_real_welfare_std":   float(dp_w_runs[:, t].std()),
            "rnn_real_welfare_mean": float(rnn_w_runs[:, t].mean()),
            "rnn_real_welfare_std":  float(rnn_w_runs[:, t].std()),
            "pvt_real_fairness_mean": float(pvt_f.mean()),
            "dp_real_fairness_mean":  float(dp_f.mean()),
            "rnn_real_fairness_mean": float(rnn_f.mean()),
            "dp_time_sum":  float(dp_time_runs.sum()),
            "rnn_time_sum": float(rnn_time_runs.sum()),
        })
    df_summary = pd.DataFrame(rows_summary)
    df_summary.to_csv(out_summary, index=False)
    print(f"  -> {out_summary}  ({len(df_summary)} rows)")

    from scipy import stats as _stats
    rows_ci = []
    # 1-9 baseline formula (run_experiments.py:412):
    #     fair = abs( sum_run g0 - sum_run g1 )
    # i.e. gap-of-sums, not mean-of-|gaps|.  CI propagates from each group
    # separately:  SE_diff = sqrt(SE_g0^2 + SE_g1^2) where SE_* = sem of the
    # per-run *_g0 / *_g1 arrays.
    def _gap_of_sums_ci(g0, g1):
        """Return (mean_gap, ci_low, ci_high) using the 1-9 baseline formula.

        mean_gap = |mean(g0) - mean(g1)|    (== abs(Σg0 - Σg1) / N)
        SE_diff  = sqrt( SE(g0)^2 + SE(g1)^2 )
        CI       = mean_gap ± t_{N-1, 0.975} * SE_diff
        """
        g0 = np.asarray(g0, dtype=np.float64)
        g1 = np.asarray(g1, dtype=np.float64)
        if g0.size < 2:
            m = float(abs(g0.mean() - g1.mean()))
            return m, m, m
        m0, m1 = float(g0.mean()), float(g1.mean())
        gap = abs(m0 - m1)
        se0 = float(_stats.sem(g0)) if g0.std() >= 1e-12 else 0.0
        se1 = float(_stats.sem(g1)) if g1.std() >= 1e-12 else 0.0
        se_diff = float(np.sqrt(se0 * se0 + se1 * se1))
        if se_diff < 1e-12:
            return gap, gap, gap
        lo, hi = _stats.t.interval(0.95, g0.size - 1, loc=gap, scale=se_diff)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return gap, gap, gap
        return gap, float(lo), float(hi)
    def _ci(arr):
        arr = np.atleast_1d(np.asarray(arr, dtype=np.float64))
        m = float(arr.mean())
        if arr.size < 2 or float(arr.std()) < 1e-12:
            return m, m, m
        lo, hi = _stats.t.interval(0.95, arr.size-1, loc=m, scale=_stats.sem(arr))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = m, m
        return m, float(lo), float(hi)
    for t in range(T):
        pvt_f_m, pvt_f_l, pvt_f_h   = _gap_of_sums_ci(pvt_g0_runs[:, t],  pvt_g1_runs[:, t])
        dp_f_m,  dp_f_l,  dp_f_h    = _gap_of_sums_ci(dp_g0_runs[:, t],   dp_g1_runs[:, t])
        rnn_f_m, rnn_f_l, rnn_f_h   = _gap_of_sums_ci(rnn_g0_runs[:, t],  rnn_g1_runs[:, t])
        dp_only_f_m, dp_only_f_l, dp_only_f_h   = _gap_of_sums_ci(dp_ow0_runs[:, t],  dp_ow1_runs[:, t])
        rnn_only_f_m, rnn_only_f_l, rnn_only_f_h = _gap_of_sums_ci(rnn_ow0_runs[:, t], rnn_ow1_runs[:, t])
        dp_calc_only_f_m, dp_calc_only_f_l, dp_calc_only_f_h   = _gap_of_sums_ci(dp_cow0_runs[:, t],  dp_cow1_runs[:, t])
        rnn_calc_only_f_m, rnn_calc_only_f_l, rnn_calc_only_f_h = _gap_of_sums_ci(rnn_cow0_runs[:, t], rnn_cow1_runs[:, t])
        pvt_w_m, pvt_w_l, pvt_w_h = _ci(pvt_w_runs[:, t])
        dp_w_m,  dp_w_l,  dp_w_h  = _ci(dp_w_runs[:, t])
        rnn_w_m, rnn_w_l, rnn_w_h = _ci(rnn_w_runs[:, t])
        pvt_p_m, pvt_p_l, pvt_p_h = _ci(pvt_p_runs[:, t])
        dp_p_m,  dp_p_l,  dp_p_h  = _ci(dp_p_runs[:, t])
        rnn_p_m, rnn_p_l, rnn_p_h = _ci(rnn_p_runs[:, t])
        dp_p0_m, dp_p0_l, dp_p0_h = _ci(dp_p0_runs[:, t])
        rnn_p0_m, rnn_p0_l, rnn_p0_h = _ci(rnn_p0_runs[:, t])
        rows_ci.append({
            "epsilon": eps, "period": t,
            "dp_time_sum": float(dp_time_runs.sum()),
            "rnn_time_sum": float(rnn_time_runs.sum()),
            "pvt_time_sum": float(pvt_time_runs.sum()),
            "pvt_real_fairness": pvt_f_m, "pvt_real_fairness_ci95_low": pvt_f_l, "pvt_real_fairness_ci95_high": pvt_f_h,
            "dp_real_fairness": dp_f_m,   "dp_real_fairness_ci95_low": dp_f_l,   "dp_real_fairness_ci95_high": dp_f_h,
            "rnn_real_fairness": rnn_f_m, "rnn_real_fairness_ci95_low": rnn_f_l, "rnn_real_fairness_ci95_high": rnn_f_h,
            "dp_only_expected_fairness": dp_only_f_m, "dp_only_expected_fairness_ci95_low": dp_only_f_l, "dp_only_expected_fairness_ci95_high": dp_only_f_h,
            "rnn_only_expected_fairness": rnn_only_f_m, "rnn_only_expected_fairness_ci95_low": rnn_only_f_l, "rnn_only_expected_fairness_ci95_high": rnn_only_f_h,
            "dp_calc_only_expected_fairness": dp_calc_only_f_m, "dp_calc_only_expected_fairness_ci95_low": dp_calc_only_f_l, "dp_calc_only_expected_fairness_ci95_high": dp_calc_only_f_h,
            "rnn_calc_only_expected_fairness": rnn_calc_only_f_m, "rnn_calc_only_expected_fairness_ci95_low": rnn_calc_only_f_l, "rnn_calc_only_expected_fairness_ci95_high": rnn_calc_only_f_h,
            "pvt_welfare": pvt_w_m, "pvt_welfare_ci95_low": pvt_w_l, "pvt_welfare_ci95_high": pvt_w_h,
            "dp_welfare":  dp_w_m,  "dp_welfare_ci95_low":  dp_w_l,  "dp_welfare_ci95_high":  dp_w_h,
            "rnn_welfare": rnn_w_m, "rnn_welfare_ci95_low": rnn_w_l, "rnn_welfare_ci95_high": rnn_w_h,
            "pvt_payment": pvt_p_m, "pvt_payment_ci95_low": pvt_p_l, "pvt_payment_ci95_high": pvt_p_h,
            "dp_payment":  dp_p_m,  "dp_payment_ci95_low":  dp_p_l,  "dp_payment_ci95_high":  dp_p_h,
            "rnn_payment": rnn_p_m, "rnn_payment_ci95_low": rnn_p_l, "rnn_payment_ci95_high": rnn_p_h,
            "dp_p0":  dp_p0_m,  "dp_p0_ci95_low":  dp_p0_l,  "dp_p0_ci95_high":  dp_p0_h,
            "rnn_p0": rnn_p0_m, "rnn_p0_ci95_low": rnn_p0_l, "rnn_p0_ci95_high": rnn_p0_h,
            # PVT doesn't have its own "only expected" arrays (oracle, not a
            # learned policy) so the two *_only_expected columns share pvt_f
            # values and the same CI as real_fairness.
            "pvt_real_only_expected_fairness": pvt_f_m,
            "pvt_real_only_expected_fairness_ci95_low": pvt_f_l,
            "pvt_real_only_expected_fairness_ci95_high": pvt_f_h,
            "pvt_only_expected_fairness": pvt_f_m,
            "pvt_only_expected_fairness_ci95_low": pvt_f_l,
            "pvt_only_expected_fairness_ci95_high": pvt_f_h,
            "dp_real_only_expected_fairness": dp_only_f_m,
            "dp_real_only_expected_fairness_ci95_low": dp_only_f_l,
            "dp_real_only_expected_fairness_ci95_high": dp_only_f_h,
            "rnn_real_only_expected_fairness": rnn_only_f_m,
            "rnn_real_only_expected_fairness_ci95_low": rnn_only_f_l,
            "rnn_real_only_expected_fairness_ci95_high": rnn_only_f_h,
            # Time semantics:
            #   dp_time / pvt_time = SUM over all runs (per-run wall-clock cost).
            #   rnn_time = SUM over all runs of T * per-call forward time
            #              (this is the *real online auction* cost: each of T steps
            #               per run calls the network once). Batched forward prep
            #               time used for the full p0 table is reported separately
            #               as `flex_offline_time` (informational, offline metric).
            "dp_time":  float(dp_time_runs.sum()),
            "rnn_time": float(rnn_time_runs.sum()),
            "pvt_time": float(pvt_time_runs.sum()),
            "flex_offline_time": float(flex_offline_runs.sum()),
        })
    df_ci = pd.DataFrame(rows_ci)
    df_ci.to_csv(out_ci, index=False)
    print(f"  -> {out_ci}  ({len(df_ci)} rows)")

    records = []
    for t in range(T):
        # Fairness: mean-then-abs (aligned with 1.csv run_experiments.py:411-417).
        pvt_f = float(abs(pvt_g0_runs[:, t].mean() - pvt_g1_runs[:, t].mean()))
        dp_f  = float(abs(dp_g0_runs[:, t].mean()  - dp_g1_runs[:, t].mean()))
        rnn_f = float(abs(rnn_g0_runs[:, t].mean() - rnn_g1_runs[:, t].mean()))
        dp_only_f  = float(abs(dp_ow0_runs[:, t].mean()  - dp_ow1_runs[:, t].mean()))
        rnn_only_f = float(abs(rnn_ow0_runs[:, t].mean() - rnn_ow1_runs[:, t].mean()))
        dp_calc_only_f  = float(abs(dp_cow0_runs[:, t].mean()  - dp_cow1_runs[:, t].mean()))
        rnn_calc_only_f = float(abs(rnn_cow0_runs[:, t].mean() - rnn_cow1_runs[:, t].mean()))
        records.append({
            'period': t, 'epsilon': exp_config["epsilon"],
            'pvt_welfare': float(pvt_w_runs[:, t].mean()),
            'pvt_payment': float(pvt_p_runs[:, t].mean()),
            # PVT only has one fairness signal (no expected-only decomposition).
            'pvt_real_fairness': pvt_f,
            'pvt_real_only_expected_fairness': pvt_f,
            'pvt_only_expected_fairness': pvt_f,
            'dp_p0': float(dp_p0_runs[:, t].mean()),
            'dp_welfare': float(dp_w_runs[:, t].mean()),
            'dp_payment': float(dp_p_runs[:, t].mean()),
            'dp_real_fairness': dp_f,
            'dp_real_only_expected_fairness': dp_only_f,
            'dp_only_expected_fairness': dp_calc_only_f,
            'rnn_p0': float(rnn_p0_runs[:, t].mean()),
            'rnn_welfare': float(rnn_w_runs[:, t].mean()),
            'rnn_payment': float(rnn_p_runs[:, t].mean()),
            'rnn_real_fairness': rnn_f,
            'rnn_real_only_expected_fairness': rnn_only_f,
            'rnn_only_expected_fairness': rnn_calc_only_f,
            'dp_time':  float(dp_time_runs.sum()),
            'rnn_time': float(rnn_time_runs.sum()),
            'pvt_time': float(pvt_time_runs.sum()),
            'flex_offline_time': float(flex_offline_runs.sum()),
            'pvt_real_social_welfare_group0': float(pvt_g0_runs[:, t].mean()),
            'dp_real_social_welfare_group0':  float(dp_g0_runs[:, t].mean()),
            'rnn_real_social_welfare_group0': float(rnn_g0_runs[:, t].mean()),
            'pvt_real_social_welfare_group1': float(pvt_g1_runs[:, t].mean()),
            'dp_real_social_welfare_group1':  float(dp_g1_runs[:, t].mean()),
            'rnn_real_social_welfare_group1': float(rnn_g1_runs[:, t].mean()),
        })
    df_legacy = pd.DataFrame(records).sort_values('period', ascending=True).reset_index(drop=True)
    df_legacy.to_csv(out_legacy, index=False)
    print(f"  -> {out_legacy}  ({len(df_legacy)} rows, legacy 5-row period mean)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="FlexNet end-to-end eval on all experiments")
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(CODE_DIR, "results", "ckpt", "flex_1fairness_full.pth"),
                    help="Path to trained FlexNet weights")
    ap.add_argument("--csv_dir", type=str,
                    default=os.path.join(PROJECT_DIR, "results", "run", "all_experiments"))
    ap.add_argument("--num_runs", type=int, default=500)
    ap.add_argument("--epsilon", type=float, default=5.0,
                    help="Epsilon for the fairness parameter exp")
    ap.add_argument("--csv_name", type=str, default=None)
    args = ap.parse_args()

    exp = {
        "group_sizes": [30, 30],
        "num_states": 2,
        "T": 5,
        "delta": 0.6,
        "epsilon": args.epsilon,
        "seed": 12345678,
    }
    csv_name = args.csv_name or f"exp1_eps{int(args.epsilon)}.csv"
    run_flex_experiment(exp, args.ckpt, args.csv_dir, csv_name, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
