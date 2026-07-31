#!/usr/bin/env bash
# End-to-end pipeline (Linux / macOS)
set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$HERE"

CONFIG_JSON="code/configs/experiments_1to9.json"
RAW_DIR="data_cache/1to9_full_raw"
SAMPLED_DIR="data_cache/1to9_sampled_raw"
UNIFIED_DIR="data_cache/1to9_unified_shuffled"
N_JOBS_BUILD=1
N_SAMPLE_STATES=160
N_SETTINGS_SAMPLE=640
N_EXTREME_HIGH=40
N_EXTREME_LOW=40
N_RANDOM=80
SETTINGS_SEED=12345
SAMPLE_SEED=999

EPOCHS=50
BATCH_SIZE=256
LR=5e-4
WEIGHT_DECAY=1e-3
EARLY_STOP_PATIENCE=15
EARLY_STOP_MIN_DELTA=1e-5
TRAIN_OUTPUT="results/ckpt/unified_1to9.pth"
SHUFFLE_SEED=${SHUFFLE_SEED:-1234567}
SPLIT_SEED=${SPLIT_SEED:-123456}
INIT_SEED=${INIT_SEED:-42}
export PYTHONHASHSEED=0

EVAL_CKPT="$TRAIN_OUTPUT"
EVAL_NUM_RUNS=${EVAL_NUM_RUNS:-500}
EVAL_OUT_DIR="results/eval_all"
SKIP_GROUPS=${SKIP_GROUPS:-3}

CSV_OUT_DIR="results/csv/sampled_raw"
PREVIEW_SETTINGS=${PREVIEW_SETTINGS:-32}

SKIP_BUILD=0; SKIP_SAMPLE=0; SKIP_TRAIN=0; SKIP_EVAL=0; SKIP_CSV=0
FORCE_SAMPLE=0; ONLY_EVAL=0

for arg in "$@"; do
    case "$arg" in
        --skip-build)  SKIP_BUILD=1 ;;
        --skip-sample) SKIP_SAMPLE=1 ;;
        --force-sample) FORCE_SAMPLE=1 ;;
        --skip-train)  SKIP_TRAIN=1 ;;
        --skip-eval)   SKIP_EVAL=1 ;;
        --skip-csv)    SKIP_CSV=1 ;;
        --only-eval)   ONLY_EVAL=1; SKIP_BUILD=1; SKIP_SAMPLE=1; SKIP_TRAIN=1; SKIP_CSV=1 ;;
        --ckpt=*)      EVAL_CKPT="${arg#*=}" ;;
        --num-runs=*)  EVAL_NUM_RUNS="${arg#*=}" ;;
        --skip-groups=*) SKIP_GROUPS="${arg#*=}" ;;
        --epochs=*)    EPOCHS="${arg#*=}" ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "  --skip-build       Skip data generation"
            echo "  --skip-sample      Skip sampling"
            echo "  --force-sample     Force re-sampling even if data exists"
            echo "  --skip-train       Skip training"
            echo "  --skip-eval        Skip simulation evaluation"
            echo "  --skip-csv         Skip CSV export"
            echo "  --only-eval        Run evaluation only (needs --ckpt)"
            echo "  --ckpt=PATH        Override checkpoint path"
            echo "  --num-runs=N       Simulation runs per config (default 500)"
            echo "  --epochs=N         Training epochs (default 50)"
            echo "  --skip-groups=G    Comma-separated groups to skip (default 3)"
            exit 0
            ;;
        *) echo "[WARN] Unknown arg: $arg" ;;
    esac
done

mkdir -p "$RAW_DIR" "$SAMPLED_DIR" "$UNIFIED_DIR" "$(dirname "$TRAIN_OUTPUT")" \
         "$EVAL_OUT_DIR" "$CSV_OUT_DIR"

if [ $SKIP_SAMPLE -eq 0 ] && [ $FORCE_SAMPLE -eq 0 ] && [ -n "$(find "$SAMPLED_DIR" -maxdepth 1 -name '*.npz' -print -quit 2>/dev/null)" ]; then
    echo "[INFO] Existing sampled data detected; use --force-sample to regenerate."
    SKIP_SAMPLE=1
fi

ts() { date '+%Y-%m-%d %H:%M:%S'; }
step() {
    echo
    echo "==============================================================="
    echo "[$(ts)] STEP $1 — $2"
    echo "==============================================================="
}

if [ $SKIP_BUILD -eq 0 ]; then
    step 1 "Generate full-raw oracle data"
    python3 -m code.scripts.build_1to9_datasets \
        --config_json "$CONFIG_JSON" --out_dir "$RAW_DIR" \
        --n_jobs "$N_JOBS_BUILD" --skip_existing
fi

if [ $SKIP_SAMPLE -eq 0 ]; then
    step 2 "Sample states from full-raw"
    python3 -m code.scripts.sample_1to9_raw_datasets \
        --full_dir "$RAW_DIR" --out_dir "$SAMPLED_DIR" \
        --n_sample "$N_SAMPLE_STATES" \
        --n_extreme_high "$N_EXTREME_HIGH" --n_extreme_low "$N_EXTREME_LOW" \
        --n_random "$N_RANDOM" --n_settings_sample "$N_SETTINGS_SAMPLE" \
        --settings_seed "$SETTINGS_SEED" --seed "$SAMPLE_SEED" --skip_existing
fi

if [ $SKIP_SAMPLE -eq 0 ]; then
    step "2.5" "Merge and shuffle sampled-raw"
    python3 -m code.scripts.shuffle_unified_1to9 \
        --data_dir "$SAMPLED_DIR" --out_dir "$UNIFIED_DIR" \
        --split_seed "$SPLIT_SEED" --shuffle_seed "$SHUFFLE_SEED" --skip_existing
fi

if [ $SKIP_TRAIN -eq 0 ]; then
    step 3 "Train learned model"
    python3 -m code.scripts.train_unified_1to9 \
        --data_dir "$UNIFIED_DIR" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
        --lr "$LR" --weight_decay "$WEIGHT_DECAY" \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
        --output "$TRAIN_OUTPUT" \
        --split_seed "$SPLIT_SEED" --shuffle_seed "$SHUFFLE_SEED" --init_seed "$INIT_SEED"
fi

if [ $SKIP_EVAL -eq 0 ]; then
    step 4 "Run simulations and generate per-config CSVs"
    python3 -m code.scripts.run_all_experiments \
        --config-json "$CONFIG_JSON" --ckpt "$EVAL_CKPT" \
        --out-dir "$EVAL_OUT_DIR" --num-runs "$EVAL_NUM_RUNS" \
        --skip-groups "$SKIP_GROUPS"
fi

if [ $SKIP_CSV -eq 0 ]; then
    step 5 "Export sampled-raw to CSV"
    python3 -m code.scripts.sampled_raw_to_csv \
        --in_dir "$SAMPLED_DIR" --out_dir "$CSV_OUT_DIR" \
        --preview_settings "$PREVIEW_SETTINGS"
fi

echo
echo "[$(ts)] Done."
echo "  Data:    $RAW_DIR, $SAMPLED_DIR"
echo "  Model:   $TRAIN_OUTPUT"
echo "  CSV:     $CSV_OUT_DIR"
