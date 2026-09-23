#!/bin/bash

#SBATCH --job-name=SuStaIn_SZ_main
#SBATCH --partition=normal
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL

echo "===== SuStaIn SZ MAIN RUN gestartet ====="
echo "Hostname: $(hostname)"
echo "Startzeit: $(date)"

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
conda activate "${SUSTAIN_CONDA_ENV:-sustain_env}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

: "${SUSTAIN_PROJECT_DIR:?Set SUSTAIN_PROJECT_DIR to the repository root before submission}"
cd "$SUSTAIN_PROJECT_DIR"

python -u HPC_Scripts/sustain_main_SZ.py

echo "===== JOB ENDE ====="
echo "Endzeit: $(date)"
