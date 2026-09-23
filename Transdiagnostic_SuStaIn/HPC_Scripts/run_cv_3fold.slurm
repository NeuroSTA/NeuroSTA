#!/bin/bash

#SBATCH --job-name=sustain_cv3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=120G
#SBATCH --time=10-00:00:00
#SBATCH --partition=normal
#SBATCH --output=sustain_cv3_%j.out
#SBATCH --error=sustain_cv3_%j.err
#SBATCH --mail-type=END,FAIL

echo "===== JOB STARTED ====="
echo "Hostname: $(hostname)"
echo "Startzeit: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
conda activate "${SUSTAIN_CONDA_ENV:-sustain_env}"

# tqdm-Ausgabe ins Log deaktivieren
export TQDM_DISABLE=1

# BLAS/MKL/OpenMP Threads begrenzen
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

: "${SUSTAIN_PROJECT_DIR:?Set SUSTAIN_PROJECT_DIR to the repository root before submission}"
cd "$SUSTAIN_PROJECT_DIR"

echo "Starte Python-Script: sustain_cv_3fold.py"
python -u HPC_Scripts/sustain_cv_3fold.py

echo "=========================================="
echo "===== JOB FINISHED ====="
echo "Endzeit: $(date)"
