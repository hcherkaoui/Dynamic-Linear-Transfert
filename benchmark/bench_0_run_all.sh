#!/bin/bash

set -euo pipefail

################################################################################
# Globals
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"
DATASETS=("email" "boston")  # real dataset to use
EPS=(1.0 0.5 0.1)            # 'task difference' between target and source
ALPHA=0.05                   # transfert learning hyperparameter
JOBLIB_VERBOSE=100           # verbosity level for joblib
START_TIME=$(date)

echo "=============================================================================="
echo " DynTR Benchmark Runner"
echo "Started at: $START_TIME"
echo "=============================================================================="

################################################################################
# Experiment 1: Transfer gain Illustration (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 1] Illustrating the Transfer gain estimation on synthetic datasets..."

for eps in "${EPS[@]}"; do
    echo "------------------------------------------------------------------------------"
    echo "→ Epsilon: $eps"
    python -m benchmark.bench_1_delta_illustration_synthetic_data --eps "$eps" -n_t 1000 -n_s 2000 --joblib-verbose "$JOBLIB_VERBOSE"
done

echo "[Experiment 1] Merge figures from experiment 1"
mkdir -p _merge_figures
files=(_figures/1_*.pdf)
if [ -e "${files[0]}" ]; then
    pdftk "${files[@]}" cat output "_merge_figures/_merge_figures_exp_1.pdf"
fi

echo "[✓] Finished Experiment 1"
echo

################################################################################
# Experiment 2: Sigma influence (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 2] Influence of sigma_S on synthetic datasets..."

for eps in "${EPS[@]}"; do
    echo "------------------------------------------------------------------------------"
    echo "→ Epsilon: $eps"
    python -m benchmark.bench_2_sigma_influence_synthetic_data --eps "$eps" --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"
done

echo "[Experiment 2] Merge figures from experiment 2"
files=(_figures/2_*.pdf)
if [ -e "${files[0]}" ]; then
    pdftk "${files[@]}" cat output "_merge_figures/_merge_figures_exp_2.pdf"
fi

echo "[✓] Finished Experiment 2"
echo

################################################################################
# Experiment 3: Epsilon influence (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 3] Influence of epsilon on synthetic datasets..."

    python -m benchmark.bench_3_epsilon_influence_synthetic_data --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"

echo "[Experiment 3] Merge figures from experiment 3"
files=(_figures/3_*.pdf)
if [ -e "${files[0]}" ]; then
    pdftk "${files[@]}" cat output "_merge_figures/_merge_figures_exp_3.pdf"
fi

echo "[✓] Finished Experiment 3"
echo

################################################################################
# Experiment 4: Performance Comparison (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 4] Comparing collaboration on synthetic datasets..."

for eps in "${EPS[@]}"; do
    echo "------------------------------------------------------------------------------"
    echo "→ Epsilon: $eps"
    python -m benchmark.bench_4_performance_synthetic_data --eps "$eps" -n_t 1000 -n_s 2000 --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"
done

echo "[Experiment 4] Merge figures from experiment 4"
files=(_figures/4_*.pdf)
if [ -e "${files[0]}" ]; then
    pdftk "${files[@]}" cat output "_merge_figures/_merge_figures_exp_4.pdf"
fi

echo "[✓] Finished Experiment 4"
echo

################################################################################
# Experiment 5: Performance Comparison (Real Data)
echo "=============================================================================="
echo "[Experiment 5] Comparing collaboration on real datasets..."

for dataset in "${DATASETS[@]}"; do
    echo "------------------------------------------------------------------------------"
    echo "→ Dataset: $dataset"
    python -m benchmark.bench_5_performance_real_data --dataset-name "$dataset" -n_t 1000 -n_s 2000 --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"
done

echo "[Experiment 5] Merge figures from experiment 5"
files=(_figures/5_*.pdf)
if [ -e "${files[0]}" ]; then
    pdftk "${files[@]}" cat output "_merge_figures/_merge_figures_exp_5.pdf"
fi

echo "[✓] Finished Experiment 5"
echo

################################################################################
# Experiment 6 additional: Lambda influence (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 6] Influence of epsilon on synthetic datasets..."

python -m benchmark.bench_6_add_lbda_influence_synthetic_data --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"

echo "[Experiment 6] Merge figures from experiment 6"
files=(_figures/6_*.pdf)
if [ -e "${files[0]}" ]; then
    pdftk "${files[@]}" cat output "_merge_figures/_merge_figures_exp_6.pdf"
fi

echo "[✓] Finished Experiment 6"
echo

################################################################################
# Experiment 7 additional: Introduction illustration (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 7] Introduction illustration on synthetic datasets..."

python -m benchmark.bench_7_intro_illustration --eps 0.5 -n_t 1500 -n_s 4000 --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"

echo "[Experiment 7] Merge figures from experiment 7"
files=(_figures/7_*.pdf)
if [ -e "${files[0]}" ]; then
    pdftk "${files[@]}" cat output "_merge_figures/_merge_figures_exp_7.pdf"
fi

echo "[✓] Finished Experiment 7"
echo

################################################################################
# Summary
END_TIME=$(date)
echo "=============================================================================="
echo "All experiments completed."
echo "Started at: $START_TIME"
echo "Finished at: $END_TIME"
