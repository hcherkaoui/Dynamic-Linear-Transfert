#!/usr/bin/env bash

#SBATCH --job-name=dyntr_bench
#SBATCH --output=dyntr_bench_%j.out
#SBATCH --error=dyntr_bench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH -A jcx@v100
#SBATCH -p gpu_p13

set -euo pipefail

# Avoid nested threading inside joblib workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Load Jean Zay PyTorch module inside the job
module purge
module load pytorch-gpu/py3/2.2.0

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
cd "$root_dir"

# Check dependencies
python -c "import torch; print('torch OK:', torch.__version__)" \
  || { echo "[error] torch not found after module load"; exit 1; }
python -c "import torchvision; print('torchvision OK:', torchvision.__version__)" \
  || echo "[warn] torchvision not found in this module environment."
python -c "import dyntr, joblib, matplotlib, numpy, numba, sklearn, pandas; print('bench deps OK')"

####################################################################################################
# Main
DATASETS=("email" "boston")  # real dataset to use
EPS=(1.0 0.5 0.1)            # 'task difference' between target and source
ALPHA=0.05                   # transfert learning hyperparameter
JOBLIB_VERBOSE=100           # verbosity level for joblib
START_TIME=$(date)

echo "=============================================================================="
echo "DynTR Benchmark Runner"
echo "Started at: $START_TIME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "CPUS: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "=============================================================================="

# Experiment 1: Transfer gain Illustration (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 1] Illustrating the Transfer gain estimation on synthetic datasets..."

for eps in "${EPS[@]}"; do
  echo "------------------------------------------------------------------------------"
  echo "→ Epsilon: $eps"
  srun --ntasks=1 python -m benchmark.bench_1_delta_illustration_synthetic_data --eps "$eps" -n_t 1000 -n_s 2000 --joblib-verbose "$JOBLIB_VERBOSE"
done

echo "[✓] Finished Experiment 1"
echo

# Experiment 2: Sigma influence (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 2] Influence of sigma_S on synthetic datasets..."

for eps in "${EPS[@]}"; do
  echo "------------------------------------------------------------------------------"
  echo "→ Epsilon: $eps"
  srun --ntasks=1 python -m benchmark.bench_2_sigma_influence_synthetic_data --eps "$eps" --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"
done

echo "[✓] Finished Experiment 2"
echo

# Experiment 3: Epsilon influence (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 3] Influence of epsilon on synthetic datasets..."

srun --ntasks=1 python -m benchmark.bench_3_epsilon_influence_synthetic_data --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"

echo "[✓] Finished Experiment 3"
echo

# Experiment 4: Performance Comparison (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 4] Comparing collaboration on synthetic datasets..."

for eps in "${EPS[@]}"; do
  echo "------------------------------------------------------------------------------"
  echo "→ Epsilon: $eps"
  srun --ntasks=1 python -m benchmark.bench_4_performance_synthetic_data --eps "$eps" -n_t 1000 -n_s 2000 --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"
done

echo "[✓] Finished Experiment 4"
echo

# Experiment 5: Performance Comparison (Real Data)
echo "=============================================================================="
echo "[Experiment 5] Comparing collaboration on real datasets..."

for dataset in "${DATASETS[@]}"; do
  echo "------------------------------------------------------------------------------"
  echo "→ Dataset: $dataset"
  srun --ntasks=1 python -m benchmark.bench_5_performance_real_data --dataset-name "$dataset" -n_t 1000 -n_s 2000 --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"
done

echo "[✓] Finished Experiment 5"
echo

# Experiment 6 additional: Lambda influence (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 6] Lambda influence on synthetic datasets..."

srun --ntasks=1 python -m benchmark.bench_6_add_lbda_influence_synthetic_data --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"

echo "[✓] Finished Experiment 6"
echo

# Experiment 7 additional: Introduction illustration (Synthetic Data)
echo "=============================================================================="
echo "[Experiment 7] Introduction illustration on synthetic datasets..."

srun --ntasks=1 python -m benchmark.bench_7_intro_illustration --eps 0.5 -n_t 1500 -n_s 4000 --alpha $ALPHA --joblib-verbose "$JOBLIB_VERBOSE"

echo "[✓] Finished Experiment 7"
echo

# Summary
END_TIME=$(date)
echo "=============================================================================="
echo "All experiments completed."
echo "Started at: $START_TIME"
echo "Finished at: $END_TIME"
