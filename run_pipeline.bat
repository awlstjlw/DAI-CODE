@echo off
REM End-to-end pipeline (Windows)

setlocal EnableDelayedExpansion

set HERE=%~dp0
cd /d "%HERE%"

set CONFIG_JSON=code\configs\experiments_1to9.json
set RAW_DIR=data_cache\1to9_full_raw
set SAMPLED_DIR=data_cache\1to9_sampled_raw
set UNIFIED_DIR=data_cache\1to9_unified_shuffled
if "%N_JOBS_BUILD%"=="" set N_JOBS_BUILD=1
set N_SAMPLE_STATES=160
set N_SETTINGS_SAMPLE=640
set N_EXTREME_HIGH=40
set N_EXTREME_LOW=40
set N_RANDOM=80
set SETTINGS_SEED=12345
set SAMPLE_SEED=999

set EPOCHS=50
set BATCH_SIZE=256
set LR=5e-4
set WEIGHT_DECAY=1e-3
set EARLY_STOP_PATIENCE=15
set EARLY_STOP_MIN_DELTA=1e-5
set TRAIN_OUTPUT=results\ckpt\unified_1to9.pth
if "%SHUFFLE_SEED%"=="" set SHUFFLE_SEED=1234567
if "%SPLIT_SEED%"=="" set SPLIT_SEED=123456
if "%INIT_SEED%"=="" set INIT_SEED=42
set PYTHONHASHSEED=0

set EVAL_CKPT=%TRAIN_OUTPUT%
set EVAL_NUM_RUNS=500
set EVAL_OUT_DIR=results\eval_all
set SKIP_GROUPS=3

set CSV_OUT_DIR=results\csv\sampled_raw
set PREVIEW_SETTINGS=32

set SKIP_BUILD=0
set SKIP_SAMPLE=0
set FORCE_SAMPLE=0
set SKIP_TRAIN=0
set SKIP_EVAL=0
set SKIP_CSV=0
set ONLY_EVAL=0

:parse_args
if "%~1"=="" goto after_parse
if /i "%~1"=="--skip-build"  (set SKIP_BUILD=1 & shift & goto parse_args)
if /i "%~1"=="--skip-sample" (set SKIP_SAMPLE=1 & shift & goto parse_args)
if /i "%~1"=="--force-sample" (set FORCE_SAMPLE=1 & shift & goto parse_args)
if /i "%~1"=="--skip-train"  (set SKIP_TRAIN=1 & shift & goto parse_args)
if /i "%~1"=="--skip-eval"   (set SKIP_EVAL=1 & shift & goto parse_args)
if /i "%~1"=="--skip-csv"    (set SKIP_CSV=1 & shift & goto parse_args)
if /i "%~1"=="--only-eval"   (
    set ONLY_EVAL=1
    set SKIP_BUILD=1
    set SKIP_SAMPLE=1
    set SKIP_TRAIN=1
    set SKIP_CSV=1
    shift
    goto parse_args
)
if /i "%~1"=="--ckpt" (
    if "%~2"=="" goto arg_error
    set "EVAL_CKPT=%~2"
    shift & shift
    goto parse_args
)
if /i "%~1"=="--num-runs" (
    if "%~2"=="" goto arg_error
    set "EVAL_NUM_RUNS=%~2"
    shift & shift
    goto parse_args
)
if /i "%~1"=="--epochs" (
    if "%~2"=="" goto arg_error
    set "EPOCHS=%~2"
    shift & shift
    goto parse_args
)
if /i "%~1"=="--skip-groups" (
    if "%~2"=="" goto arg_error
    set "SKIP_GROUPS=%~2"
    shift & shift
    goto parse_args
)
set "_ARG=%~1"
if /i "!_ARG:~0,7!=="--ckpt=" (set "EVAL_CKPT=!_ARG:~7!" & shift & goto parse_args)
if /i "!_ARG:~0,11!=="--num-runs=" (set "EVAL_NUM_RUNS=!_ARG:~11!" & shift & goto parse_args)
if /i "!_ARG:~0,9!=="--epochs=" (set "EPOCHS=!_ARG:~9!" & shift & goto parse_args)
if /i "!_ARG:~0,14!=="--skip-groups=" (set "SKIP_GROUPS=!_ARG:~14!" & shift & goto parse_args)
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
echo [WARN] Unknown arg: %~1
shift
goto parse_args

:after_parse

if not exist "%RAW_DIR%" mkdir "%RAW_DIR%"
if not exist "%SAMPLED_DIR%" mkdir "%SAMPLED_DIR%"
if not exist "%UNIFIED_DIR%" mkdir "%UNIFIED_DIR%"
if !SKIP_SAMPLE!==0 if !FORCE_SAMPLE!==0 (
    dir /b "%SAMPLED_DIR%\*.npz" >nul 2>nul && (
        echo [INFO] Existing sampled data detected; use --force-sample to regenerate.
        set SKIP_SAMPLE=1
    )
)
for %%P in ("%TRAIN_OUTPUT%") do if not exist "%%~dpP" mkdir "%%~dpP"
if not exist "%EVAL_OUT_DIR%" mkdir "%EVAL_OUT_DIR%"
if not exist "%CSV_OUT_DIR%" mkdir "%CSV_OUT_DIR%"

if !SKIP_BUILD!==0 (
    python -m code.scripts.build_1to9_datasets --config_json "%CONFIG_JSON%" --out_dir "%RAW_DIR%" --n_jobs %N_JOBS_BUILD% --skip_existing
    if errorlevel 1 goto :error
)

if !SKIP_SAMPLE!==0 goto :skip_sample
    python -m code.scripts.sample_1to9_raw_datasets --full_dir "%RAW_DIR%" --out_dir "%SAMPLED_DIR%" --n_sample %N_SAMPLE_STATES% --n_extreme_high %N_EXTREME_HIGH% --n_extreme_low %N_EXTREME_LOW% --n_random %N_RANDOM% --n_settings_sample %N_SETTINGS_SAMPLE% --settings_seed %SETTINGS_SEED% --seed %SAMPLE_SEED% --skip_existing
    if errorlevel 1 goto :error

    python -m code.scripts.shuffle_unified_1to9 --data_dir "%SAMPLED_DIR%" --out_dir "%UNIFIED_DIR%" --split_seed %SPLIT_SEED% --shuffle_seed %SHUFFLE_SEED% --skip_existing
    if errorlevel 1 goto :error
:skip_sample

if !SKIP_TRAIN!==0 (
    python -m code.scripts.train_unified_1to9 --data_dir "%UNIFIED_DIR%" --epochs %EPOCHS% --batch_size %BATCH_SIZE% --lr %LR% --weight_decay %WEIGHT_DECAY% --early_stop_patience %EARLY_STOP_PATIENCE% --early_stop_min_delta %EARLY_STOP_MIN_DELTA% --output "%TRAIN_OUTPUT%" --split_seed %SPLIT_SEED% --shuffle_seed %SHUFFLE_SEED% --init_seed %INIT_SEED%
    if errorlevel 1 goto :error
)

if !SKIP_EVAL!==0 (
    python -m code.scripts.run_all_experiments --config-json "%CONFIG_JSON%" --ckpt "%EVAL_CKPT%" --out-dir "%EVAL_OUT_DIR%" --num-runs %EVAL_NUM_RUNS% --skip-groups "%SKIP_GROUPS%"
    if errorlevel 1 goto :error
)

if !SKIP_CSV!==0 (
    python -m code.scripts.sampled_raw_to_csv --in_dir "%SAMPLED_DIR%" --out_dir "%CSV_OUT_DIR%" --preview_settings %PREVIEW_SETTINGS%
    if errorlevel 1 goto :error
)

exit /b 0

:show_help
echo Usage: run_pipeline.bat [options]
echo   --skip-build       Skip data generation
echo   --skip-sample      Skip sampling
echo   --force-sample     Force re-sampling even if data exists
echo   --skip-train       Skip training
echo   --skip-eval        Skip simulation evaluation
echo   --skip-csv         Skip CSV export
echo   --only-eval        Run evaluation only (needs --ckpt)
echo   --ckpt PATH        Override checkpoint path
echo   --num-runs N       Simulation runs per config (default 500)
echo   --epochs N         Training epochs (default 50)
echo   --skip-groups G    Comma-separated groups to skip (default 3)
exit /b 0

:arg_error
echo [ERROR] Missing value after %~1
exit /b 2

:error
echo.
echo !!! Error occurred (exit code %errorlevel%), terminating. !!!
exit /b %errorlevel%
