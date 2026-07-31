#!/usr/bin/env python3
"""Run the unified network on all 1-9 ablations and export per-group CSVs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# Match the original experiments 1--9 figure families. Experiment 6
# (number of states) is intentionally omitted. Every non-epsilon family starts
# with the shared baseline a_eps5, but that baseline is evaluated only once in
# group 1 and reused by the other plots.
BASELINE_TAG = "a_eps5"
BASELINE_GROUP = "1"
GROUPS = {
    "1": ["a_eps1", "a_eps5", "a_eps9"],
    "2": ["a_eps5", "b_n100", "b_n140"],
    "3": ["a_eps5", "c_m3"],
    "4": ["a_eps5", "c_part_20_40", "c_part_10_50"],
    "5": ["a_eps5", "d_val_truncnorm"],
    "7": ["a_eps5", "e_trans_persistent"],
    "8": ["a_eps5", "f_T9", "f_T14"],
    "9": ["f_d04", "a_eps5", "f_d08"],
}

def _pick_runner(cfg):
    """Dispatch to the right eval module by group count (m=2 vs m=3)."""
    n_groups = len(cfg.get("group_sizes", []))
    if n_groups >= 3:
        # m=3 config: dedicated runner (separate file, no shared mutable state).
        from code.eval.run_flex_eval_m3 import run_flex_experiment
        return run_flex_experiment
    # Default: m=2 (legacy and current 1-9 ablations).
    from code.eval.run_flex_eval import run_flex_experiment
    return run_flex_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="Path to trained model checkpoint")
    ap.add_argument("--out-dir", default="results/eval_all")
    ap.add_argument("--num-runs", type=int, default=500)
    ap.add_argument("--skip-groups", type=str, default="3",
                    help="Comma-separated group ids to skip (e.g. '3' or '3,4'). "
                         "Group 3 is m=3; default skips it. Pass empty string '' to run all.")
    args = ap.parse_args()
    if not args.ckpt:
        ap.error("--ckpt is required")

    skip_groups = {g.strip() for g in args.skip_groups.split(",") if g.strip()}
    if skip_groups:
        print(f"[run_all_experiments] skipping groups: {sorted(skip_groups)}")

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))

    with open(args.config_json, encoding="utf-8") as f:
        configs = {c["tag"]: c for c in json.load(f)["configs"]}
    baseline_dir = root / args.out_dir / BASELINE_GROUP / "csv"
    for group, tags in GROUPS.items():
        if group in skip_groups:
            print(f"[SKIP] group {group} (in --skip-groups)")
            continue
        selected = [configs[t] for t in tags if t in configs]
        if not selected:
            continue
        out = root / args.out_dir / group / "csv"
        out.mkdir(parents=True, exist_ok=True)

        # Reuse the one baseline simulation from group 1 in all seven other
        # ablation figures. This adds the baseline column without rerunning 500
        # auctions seven extra times.
        csv_paths = []
        for cfg in selected:
            csv_dir = baseline_dir if cfg["tag"] == BASELINE_TAG else out
            csv_paths.append(csv_dir / f"{cfg['tag']}.csv")
        for cfg, csv_path in zip(selected, csv_paths):
            if cfg["tag"] == BASELINE_TAG and group != BASELINE_GROUP:
                if not csv_path.is_file():
                    raise FileNotFoundError(
                        f"shared baseline CSV was not generated: {csv_path}"
                    )
                continue
            runner = _pick_runner(cfg)
            runner(cfg, args.ckpt, str(out), csv_path.name,
                   num_runs=args.num_runs)

if __name__ == "__main__":
    main()
