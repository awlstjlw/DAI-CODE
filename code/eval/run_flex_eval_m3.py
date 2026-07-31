"""End-to-end evaluation of the learned FlexNet model injecting predicted p0 into pvt_fair_m3."""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)                     # code/
PROJECT_DIR = os.path.dirname(CODE_DIR)               # project root
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "code_dp"))

from model.backbone_flex import FlexibleFairPivotNet
from train.train_loop_flex import GROUP_SIZE_REF, VALUATION_REF, normalize_raw_samples

# m=3 variant: use the dedicated three-group config/data generator copied from
# original experiment 3. The generic auction_env generator is intentionally m=2.
from auction_env.config_m3 import AuctionConfigM3 as AuctionConfig
from auction_env.data_generator_m3 import generate_data_m3 as generate_data
from pvt_fair_m3 import FairnessRealAuction

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
    """Return {(t, state...): p0} for every calc state and period."""
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
        # Point-wise output (S*T, m); restore the external policy-table shape.
        phi_arr = out["phi"].cpu().numpy().reshape(S, T, m)

    # Build dict: key = (t, g0s1, g1s1, g2s1, ...), value = (p0, p1, p2, ...)
    result = {}
    for s_idx in range(S):
        state_key = tuple(int(states[s_idx, k * K + 1]) for k in range(m))
        for t in range(T):
            result[(t, ) + state_key] = tuple(float(phi_arr[s_idx, t, g]) for g in range(m))
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
    """Set per-run globals for _run_one_auction."""
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
    """m=3 variant: dynamically handle all groups, and store multiple p-values."""
    m = len(_WORKER_AUC_GS)
    out = {"pvt_w": [0.0]*T, "pvt_p": [0.0]*T,
           "dp_w": [0.0]*T, "dp_p": [0.0]*T,
           "rnn_w": [0.0]*T, "rnn_p": [0.0]*T,
           # For m=3 dp_p0/rnn_p0 are tuples (p0,p1,p2); store as a list of lists.
           "dp_p0":  [None]*T,
           "rnn_p0": [None]*T,
           # Per-group welfare (per_capita).  We dynamically allocate m slots.
           **{f"pvt_g{i}": [0.0]*T for i in range(m)},
           **{f"dp_g{i}":  [0.0]*T for i in range(m)},
           **{f"rnn_g{i}": [0.0]*T for i in range(m)},
           **{f"dp_ow{i}":  [0.0]*T for i in range(m)},
           **{f"rnn_ow{i}": [0.0]*T for i in range(m)},
           **{f"dp_cow{i}":  [0.0]*T for i in range(m)},
           **{f"rnn_cow{i}": [0.0]*T for i in range(m)},
           "dp_time": 0.0, "rnn_time": 0.0, "pvt_time": 0.0}
    for t in range(T):
        out["pvt_w"][t]  = float(res["pvt_real_social_welfare"][t])
        out["pvt_p"][t]  = float(res["pvt_real_payment"][t])
        out["dp_w"][t]   = float(res["dp_real_social_welfare"][t])
        out["dp_p"][t]   = float(res["dp_real_payment"][t])
        out["rnn_w"][t]  = float(res["rnn_real_social_welfare"][t])
        out["rnn_p"][t]  = float(res["rnn_real_payment"][t])
        # statistic_*_p0 may be scalar (m=2) or tuple (m=3).  Convert to tuple.
        dp_p0_t  = res["statistic_dp_p0"][t]
        rnn_p0_t = res["statistic_rnn_p0"][t]
        if isinstance(dp_p0_t, (tuple, list, np.ndarray)):
            out["dp_p0"][t]  = tuple(float(x) for x in dp_p0_t)
            out["rnn_p0"][t] = tuple(float(x) for x in rnn_p0_t)
        else:
            # Backwards-compat: scalar p0 (e.g. legacy 1-9 m=2 outputs).
            out["dp_p0"][t]  = (float(dp_p0_t),)
            out["rnn_p0"][t] = (float(rnn_p0_t),)
        for g in range(m):
            out[f"pvt_g{g}"][t]    = float(res[f"pvt_real_social_welfare_group{g}"][t]) / max(_WORKER_AUC_GS[g], 1)
            out[f"dp_g{g}"][t]     = float(res[f"dp_real_social_welfare_group{g}"][t])  / max(_WORKER_AUC_GS[g], 1)
            out[f"rnn_g{g}"][t]    = float(res[f"rnn_real_social_welfare_group{g}"][t]) / max(_WORKER_AUC_GS[g], 1)
            out[f"dp_ow{g}"][t]    = float(res[f"dp_only_expected_social_welfare_group{g}"][t]) / max(_WORKER_CGS[g], 1)
            out[f"rnn_ow{g}"][t]   = float(res[f"rnn_only_expected_social_welfare_group{g}"][t]) / max(_WORKER_CGS[g], 1)
            out[f"dp_cow{g}"][t]   = float(res[f"dp_calc_only_expected_social_welfare_group{g}"][t]) / max(_WORKER_CGS[g], 1)
            out[f"rnn_cow{g}"][t]  = float(res[f"rnn_calc_only_expected_social_welfare_group{g}"][t]) / max(_WORKER_CGS[g], 1)
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
    """Run one experiment and write 1-9 style detailed/summary/ci95/legacy CSVs."""
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

    # Build per-run seed pool (value / matrix / grouping), each run gets fresh market
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

    states_all = np.stack(states_list)  # (R, S, m*K)
    val_b   = np.broadcast_to(per_run_vals[:, None, :, :], (R, S, *per_run_vals.shape[1:]))  # (R, S, m, K)
    # Add a dummy m axis to matrices (same matrix used for all groups)
    m_groups = per_run_vals.shape[1]
    mat0_b = np.broadcast_to(per_run_m0[:, None, None, :, :], (R, S, m_groups, *per_run_m0.shape[1:]))  # (R, S, m, K, K)
    mat1_b = np.broadcast_to(per_run_m1[:, None, None, :, :], (R, S, m_groups, *per_run_m1.shape[1:]))  # (R, S, m, K, K)

    # Expand (run, state) into independent time-conditioned model queries.
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
    phi_flat = np.zeros((n_queries, m_groups), dtype=np.float32)
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
            phi_flat[sl] = out["phi"].cpu().numpy()
    p0_all = phi_flat.reshape(R, S, T_, m_groups)
    del sf_all, tf_all, vf_all, trf_all, scf_all

    for run_id in range(R):
        flex_p0_i = {}
        for s_idx in range(S):
            state_key = tuple(int(states_list[run_id][s_idx, k * 2 + 1]) for k in range(m_groups))
            for t in range(T_):
                flex_p0_i[(t,) + state_key] = tuple(float(x) for x in p0_all[run_id, s_idx, t])
        g, a, od, cfg_run, _ = run_data_list[run_id]
        run_data_list[run_id] = (g, a, od, cfg_run, flex_p0_i)
    flex_time = time.perf_counter() - t0_flex
    print(f"  FlexNet batched p0 prediction: {R} runs in {flex_time:.2f}s "
          f"({R*S*T_} independent state-time queries)")

    auc_gs = list(exp_config["group_sizes"])
    cgs = list(exp_config.get("calc_group_sizes", auc_gs))
    print(f"  auc_group_sizes={auc_gs} calc_group_sizes={cgs}")

    T = config.T
    m_groups = len(exp_config["group_sizes"])
    pvt_w_runs  = np.zeros((NUM_RUNS, T))
    pvt_p_runs  = np.zeros((NUM_RUNS, T))
    pvt_g_runs  = np.zeros((NUM_RUNS, T, m_groups))
    dp_w_runs   = np.zeros((NUM_RUNS, T))
    dp_p_runs   = np.zeros((NUM_RUNS, T))
    dp_p0_runs  = np.zeros((NUM_RUNS, T))
    dp_g_runs   = np.zeros((NUM_RUNS, T, m_groups))
    dp_ow_runs  = np.zeros((NUM_RUNS, T, m_groups))
    dp_cow_runs = np.zeros((NUM_RUNS, T, m_groups))
    rnn_w_runs  = np.zeros((NUM_RUNS, T))
    rnn_p_runs  = np.zeros((NUM_RUNS, T))
    rnn_p0_runs = np.zeros((NUM_RUNS, T))
    rnn_g_runs  = np.zeros((NUM_RUNS, T, m_groups))
    rnn_ow_runs = np.zeros((NUM_RUNS, T, m_groups))
    rnn_cow_runs = np.zeros((NUM_RUNS, T, m_groups))
    dp_time_runs  = np.zeros(NUM_RUNS)
    rnn_time_runs = np.zeros(NUM_RUNS)
    pvt_time_runs = np.zeros(NUM_RUNS)
    flex_offline_runs = np.zeros(NUM_RUNS)   # offline p0-table prep amortized per run
    # Track which run_ids have *all* T rows landed in detail CSV (for resume).
    detailed_done = np.zeros(NUM_RUNS, dtype=bool)
    detailed_header_written = False

    # ---- Resume support -------------------------------------------------
    # If a previous run wrote part of the detailed CSV, reload the completed
    # run_ids and pre-populate the per-run arrays so that the final summary /
    # ci95 / legacy outputs match a fully uninterrupted experiment.
    os.makedirs(csv_dir, exist_ok=True)
    csv_stem = os.path.splitext(csv_name)[0]
    out_detail  = os.path.join(csv_dir, f"{csv_stem}_detailed.csv")
    out_summary = os.path.join(csv_dir, f"{csv_stem}_summary.csv")
    out_ci      = os.path.join(csv_dir, f"{csv_stem}_ci95.csv")
    out_legacy  = os.path.join(csv_dir, csv_name)

    def _fair_at_m3(arr, run_id, t):
        return float(arr[run_id, t, :].max() - arr[run_id, t, :].min())

    def _fair_period_m3(arr, t):
        means = arr[:, t, :].mean(axis=0)
        return float(means.max() - means.min())

    if os.path.exists(out_detail):
        try:
            _prev = pd.read_csv(out_detail)
            if "run_id" in _prev.columns:
                _counts = _prev.groupby("run_id").size()
                _prev_done_ids = sorted(int(r) for r in _counts.index
                                        if int(_counts.loc[r]) >= T)
                if _prev_done_ids:
                    print(f"  [resume] {out_detail} exists with "
                          f"{len(_prev_done_ids)}/{NUM_RUNS} finished runs "
                          f"(fully-populated). Restoring per-run accumulators.")
                    for _rid in _prev_done_ids:
                        rows = _prev[_prev["run_id"] == _rid].sort_values(by="period")
                        if len(rows) != T:
                            continue
                        pvt_w_runs[_rid]   = rows["pvt_real_welfare"].to_numpy()
                        pvt_p_runs[_rid]   = rows["pvt_real_payment"].to_numpy()
                        dp_w_runs[_rid]    = rows["dp_real_welfare"].to_numpy()
                        dp_p_runs[_rid]    = rows["dp_real_payment"].to_numpy()
                        dp_p0_runs[_rid]   = rows["dp_p0"].to_numpy()
                        rnn_w_runs[_rid]   = rows["rnn_real_welfare"].to_numpy()
                        rnn_p_runs[_rid]   = rows["rnn_real_payment"].to_numpy()
                        rnn_p0_runs[_rid]  = rows["rnn_p0"].to_numpy()
                        for g in range(m_groups):
                            pvt_g_runs[_rid,  :, g] = rows[f"pvt_real_welfare_group{g}"].to_numpy()
                            dp_g_runs[_rid,   :, g] = rows[f"dp_real_welfare_group{g}"].to_numpy()
                            rnn_g_runs[_rid,  :, g] = rows[f"rnn_real_welfare_group{g}"].to_numpy()
                            dp_ow_runs[_rid,  :, g] = rows[f"dp_only_expected_welfare{g}"].to_numpy()
                            rnn_ow_runs[_rid, :, g] = rows[f"rnn_only_expected_welfare{g}"].to_numpy()
                            dp_cow_runs[_rid, :, g] = rows[f"dp_calc_only_expected_welfare_group{g}"].to_numpy()
                            rnn_cow_runs[_rid,:, g] = rows[f"rnn_calc_only_expected_welfare_group{g}"].to_numpy()
                        if "dp_time" in rows.columns:
                            dp_time_runs[_rid] = float(rows["dp_time"].iloc[0])
                        if "rnn_time" in rows.columns:
                            rnn_time_runs[_rid] = float(rows["rnn_time"].iloc[0])
                        if "pvt_time" in rows.columns:
                            pvt_time_runs[_rid] = float(rows["pvt_time"].iloc[0])
                        detailed_done[_rid] = True
                    if _prev_done_ids:
                        detailed_header_written = True
            del _prev
        except Exception as _exc:
            print(f"  [resume] could not read existing {out_detail}: {_exc}; "
                  "starting fresh.")
            detailed_done[:] = False
            detailed_header_written = False
            try:
                os.remove(out_detail)
            except OSError:
                pass

    pending_ids = [i for i in range(NUM_RUNS) if not detailed_done[i]]

    if not pending_ids and NUM_RUNS > 0:
        # All runs already done: skip the simulation entirely.
        results_list = []
    else:
        # Cap parallel workers to keep the machine responsive.
        # Default 8; override via env var EVAL_N_WORKERS.
        _max_workers = int(os.environ.get("EVAL_N_WORKERS", "8"))
        n_workers = max(1, min(len(pending_ids), _max_workers, (os.cpu_count() or 4)))
        print(f"  Using {n_workers} CPU workers "
              f"(NUM_RUNS={NUM_RUNS}, pending={len(pending_ids)})")

        # Carry original run_id alongside each task for ordered writeback.
        all_tasks = [(i, auction_seeds[i], run_data_list[i]) for i in pending_ids]

        def _write_detail_rows(run_id, r):
            """Append T rows for a single run_id to out_detail (header once)."""
            nonlocal detailed_header_written
            eps_local = exp_config["epsilon"]
            per_call_local = flex_time / max(R * S * T_, 1)
            rnn_t_local = T_ * per_call_local
            local_rows = []
            for t in range(T):
                row = {
                    "epsilon": eps_local,
                    "run_id": run_id,
                    "period": t,
                    "dp_calc_only_expected_fairness": _fair_at_m3(dp_cow_runs, run_id, t),
                    "rnn_calc_only_expected_fairness": _fair_at_m3(rnn_cow_runs, run_id, t),
                    "pvt_real_welfare":        float(pvt_w_runs[run_id, t]),
                    "pvt_real_payment":        float(pvt_p_runs[run_id, t]),
                    "dp_p0":                   float(dp_p0_runs[run_id, t]),
                    "dp_real_welfare":         float(dp_w_runs[run_id, t]),
                    "dp_real_payment":         float(dp_p_runs[run_id, t]),
                    "rnn_p0":                  float(rnn_p0_runs[run_id, t]),
                    "rnn_real_welfare":        float(rnn_w_runs[run_id, t]),
                    "rnn_real_payment":        float(rnn_p_runs[run_id, t]),
                    "auction_seed": int(auction_seeds[run_id]),
                    "dp_time":  float(dp_time_runs[run_id]),
                    "rnn_time": float(rnn_t_local),
                    "pvt_time": float(pvt_time_runs[run_id]),
                }
                for g in range(m_groups):
                    row[f"pvt_real_welfare_group{g}"]            = float(pvt_g_runs[run_id, t, g])
                    row[f"dp_real_welfare_group{g}"]             = float(dp_g_runs[run_id, t, g])
                    row[f"dp_only_expected_welfare{g}"]          = float(dp_ow_runs[run_id, t, g])
                    row[f"dp_calc_only_expected_welfare_group{g}"] = float(dp_cow_runs[run_id, t, g])
                    row[f"rnn_real_welfare_group{g}"]            = float(rnn_g_runs[run_id, t, g])
                    row[f"rnn_only_expected_welfare{g}"]         = float(rnn_ow_runs[run_id, t, g])
                    row[f"rnn_calc_only_expected_welfare_group{g}"] = float(rnn_cow_runs[run_id, t, g])
                    row[f"group{g}_size"] = int(auc_gs[g])
                local_rows.append(row)
            df_local = pd.DataFrame(local_rows)
            write_header = not detailed_header_written
            df_local.to_csv(out_detail, mode="a", header=write_header, index=False)
            detailed_header_written = True

        def _accumulate_and_persist(run_id, r):
            """Write one run's results into the per-run arrays and append to CSV."""
            pvt_w_runs[run_id]  = r["pvt_w"]
            pvt_p_runs[run_id]  = r["pvt_p"]
            dp_w_runs[run_id]   = r["dp_w"]
            dp_p_runs[run_id]   = r["dp_p"]
            rnn_w_runs[run_id]  = r["rnn_w"]
            rnn_p_runs[run_id]  = r["rnn_p"]
            dp_p0_run  = r["dp_p0"]
            rnn_p0_run = r["rnn_p0"]
            if isinstance(dp_p0_run[0], (tuple, list, np.ndarray)):
                dp_p0_runs[run_id]  = float(dp_p0_run[0][0])
                rnn_p0_runs[run_id] = float(rnn_p0_run[0][0])
            else:
                dp_p0_runs[run_id]  = float(dp_p0_run[0])
                rnn_p0_runs[run_id] = float(rnn_p0_run[0])
            for g in range(m_groups):
                pvt_g_runs[run_id,  :, g]  = r[f"pvt_g{g}"]
                dp_g_runs[run_id,   :, g]  = r[f"dp_g{g}"]
                rnn_g_runs[run_id,  :, g]  = r[f"rnn_g{g}"]
                dp_ow_runs[run_id,  :, g]  = r[f"dp_ow{g}"]
                rnn_ow_runs[run_id, :, g]  = r[f"rnn_ow{g}"]
                dp_cow_runs[run_id, :, g]  = r[f"dp_cow{g}"]
                rnn_cow_runs[run_id,:, g]  = r[f"rnn_cow{g}"]
            dp_time_runs[run_id]  = r["dp_time"]
            per_call = flex_time / max(R * S * T_, 1)
            rnn_time_runs[run_id] = T_ * per_call
            pvt_time_runs[run_id] = r.get("pvt_time", 0.0)
            flex_offline_runs[run_id] = flex_time / R
            detailed_done[run_id] = True
            _write_detail_rows(run_id, r)

        t0_auc = time.perf_counter()
        if n_workers == 1:
            for (run_id, _seed, task) in all_tasks:
                _worker_init_single((_seed, task), auc_gs, cgs)
                r = _run_one_auction((_seed, task))
                _accumulate_and_persist(run_id, r)
        else:
            ctx = mp.get_context("spawn")
            def _packed(task_with_id):
                _rid, _seed, task = task_with_id
                # task is the 5-tuple (g, a, od, cfg_run, flex_p0); _run_one_auction
                # and _worker_init_single expect args=(_seed, run_data), so wrap.
                return ((_seed, task), auc_gs, cgs)
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
    # auction finished. Sanity-check the row count on disk here.
    if os.path.exists(out_detail):
        df_disk = pd.read_csv(out_detail)
        expected_rows = NUM_RUNS * T
        if len(df_disk) != expected_rows:
            print(f"  [warn] {out_detail} has {len(df_disk)} rows, expected "
                  f"{expected_rows}; check for interrupted writes.")
        print(f"  -> {out_detail}  ({len(df_disk)} rows, incrementally appended)")
    else:
        empty_cols = ["epsilon", "run_id", "period",
                      "dp_calc_only_expected_fairness",
                      "rnn_calc_only_expected_fairness",
                      "pvt_real_welfare", "pvt_real_payment",
                      "dp_p0", "dp_real_welfare", "dp_real_payment",
                      "rnn_p0", "rnn_real_welfare", "rnn_real_payment",
                      "auction_seed", "dp_time", "rnn_time", "pvt_time"]
        for g in range(m_groups):
            empty_cols += [f"pvt_real_welfare_group{g}",
                           f"dp_real_welfare_group{g}",
                           f"dp_only_expected_welfare{g}",
                           f"dp_calc_only_expected_welfare_group{g}",
                           f"rnn_real_welfare_group{g}",
                           f"rnn_only_expected_welfare{g}",
                           f"rnn_calc_only_expected_welfare_group{g}",
                           f"group{g}_size"]
        pd.DataFrame(columns=empty_cols).to_csv(out_detail, index=False)
        print(f"  -> {out_detail}  (header only)")
    eps = exp_config["epsilon"]
    csv_stem = os.path.splitext(csv_name)[0]

    # ---- write 1-9 detailed_experiment_results_summary.csv ---------
    # m-agnostic: per-period fairness = max - min over m group-means
    # (recovers |mean(g0) - mean(g1)| for m=2).
    def _fair_period(arr, t):
        means = arr[:, t, :].mean(axis=0)        # (m,)
        return float(means.max() - means.min())

    rows_summary = []
    for t in range(T):
        pvt_f = _fair_period(pvt_g_runs, t)
        dp_f  = _fair_period(dp_g_runs,  t)
        rnn_f = _fair_period(rnn_g_runs, t)
        rows_summary.append({
            "epsilon": eps,
            "period": t,
            "pvt_real_welfare_mean": float(pvt_w_runs[:, t].mean()),
            "pvt_real_welfare_std":  float(pvt_w_runs[:, t].std()),
            "dp_real_welfare_mean":  float(dp_w_runs[:, t].mean()),
            "dp_real_welfare_std":   float(dp_w_runs[:, t].std()),
            "rnn_real_welfare_mean": float(rnn_w_runs[:, t].mean()),
            "rnn_real_welfare_std":  float(rnn_w_runs[:, t].std()),
            "pvt_real_fairness_mean": pvt_f,
            "dp_real_fairness_mean":  dp_f,
            "rnn_real_fairness_mean": rnn_f,
            "dp_time_sum":  float(dp_time_runs.sum()),
            "rnn_time_sum": float(rnn_time_runs.sum()),
        })
    df_summary = pd.DataFrame(rows_summary)
    df_summary.to_csv(out_summary, index=False)
    print(f"  -> {out_summary}  ({len(df_summary)} rows)")

    # ---- write 1-9 summary_with_ci95.csv ---------------------------
    from scipy import stats as _stats
    rows_ci = []
    # Per-period fairness (m-agnostic): max-min over m group-means.
    def _fair_period_ci(arr, t):
        means = arr[:, t, :].mean(axis=0)
        return float(means.max() - means.min())
    for t in range(T):
        # Fairness: mean-then-..., aligned with 1.csv (and m=3 spread).
        pvt_f           = _fair_period_ci(pvt_g_runs,  t)
        dp_f            = _fair_period_ci(dp_g_runs,   t)
        rnn_f           = _fair_period_ci(rnn_g_runs,  t)
        dp_only_f       = _fair_period_ci(dp_ow_runs,  t)
        rnn_only_f      = _fair_period_ci(rnn_ow_runs, t)
        dp_calc_only_f  = _fair_period_ci(dp_cow_runs, t)
        rnn_calc_only_f = _fair_period_ci(rnn_cow_runs,t)
        def _ci(arr):
            arr = np.atleast_1d(np.asarray(arr, dtype=np.float64))
            m = float(arr.mean())
            if arr.size < 2 or float(arr.std()) < 1e-12:
                return m, m, m
            lo, hi = _stats.t.interval(0.95, arr.size-1, loc=m, scale=_stats.sem(arr))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = m, m
            return m, float(lo), float(hi)
        pvt_w_m, pvt_w_l, pvt_w_h = _ci(pvt_w_runs[:, t])
        dp_w_m,  dp_w_l,  dp_w_h  = _ci(dp_w_runs[:, t])
        rnn_w_m, rnn_w_l, rnn_w_h = _ci(rnn_w_runs[:, t])
        pvt_p_m, pvt_p_l, pvt_p_h = _ci(pvt_p_runs[:, t])
        dp_p_m,  dp_p_l,  dp_p_h  = _ci(dp_p_runs[:, t])
        rnn_p_m, rnn_p_l, rnn_p_h = _ci(rnn_p_runs[:, t])
        dp_p0_m, dp_p0_l, dp_p0_h = _ci(dp_p0_runs[:, t])
        rnn_p0_m, rnn_p0_l, rnn_p0_h = _ci(rnn_p0_runs[:, t])
        pvt_f_m, pvt_f_l, pvt_f_h = _ci(pvt_f)
        dp_f_m,  dp_f_l,  dp_f_h  = _ci(dp_f)
        rnn_f_m, rnn_f_l, rnn_f_h = _ci(rnn_f)
        dp_only_f_m, dp_only_f_l, dp_only_f_h = _ci(dp_only_f)
        rnn_only_f_m, rnn_only_f_l, rnn_only_f_h = _ci(rnn_only_f)
        dp_calc_only_f_m, dp_calc_only_f_l, dp_calc_only_f_h = _ci(dp_calc_only_f)
        rnn_calc_only_f_m, rnn_calc_only_f_l, rnn_calc_only_f_h = _ci(rnn_calc_only_f)
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
            "pvt_real_only_expected_fairness": pvt_f_m,
            "pvt_only_expected_fairness": pvt_f_m,
            "dp_real_only_expected_fairness": dp_only_f_m,
            "rnn_real_only_expected_fairness": rnn_only_f_m,
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

    # ---- write legacy 5-row period-mean table ---------------------
    # m-agnostic fairness: max - min over m group-means
    # (recovers |mean(g0) - mean(g1)| for m=2).
    def _fair_legacy(arr, t):
        means = arr[:, t, :].mean(axis=0)
        return float(means.max() - means.min())
    records = []
    for t in range(T):
        # Fairness: mean-then-abs (aligned with 1.csv run_experiments.py:411-417).
        pvt_f           = _fair_legacy(pvt_g_runs,  t)
        dp_f            = _fair_legacy(dp_g_runs,   t)
        rnn_f           = _fair_legacy(rnn_g_runs,  t)
        dp_only_f       = _fair_legacy(dp_ow_runs,  t)
        rnn_only_f      = _fair_legacy(rnn_ow_runs, t)
        dp_calc_only_f  = _fair_legacy(dp_cow_runs, t)
        rnn_calc_only_f = _fair_legacy(rnn_cow_runs,t)
        row = {
            'period': t, 'epsilon': exp_config["epsilon"],
            'pvt_welfare': float(pvt_w_runs[:, t].mean()),
            'pvt_payment': float(pvt_p_runs[:, t].mean()),
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
        }
        for g in range(m_groups):
            row[f'pvt_real_social_welfare_group{g}'] = float(pvt_g_runs[:, t, g].mean())
            row[f'dp_real_social_welfare_group{g}']  = float(dp_g_runs[:,  t, g].mean())
            row[f'rnn_real_social_welfare_group{g}'] = float(rnn_g_runs[:, t, g].mean())
        records.append(row)
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
