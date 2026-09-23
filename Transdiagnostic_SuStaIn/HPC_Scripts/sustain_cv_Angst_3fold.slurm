#!/bin/bash

#SBATCH --job-name=sustain_cv_Angst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10-00:00:00
#SBATCH --partition=normal
#SBATCH --output=sustain_cv_Angst_%j.out
#SBATCH --error=sustain_cv_Angst_%j.err
#SBATCH --mail-type=END,FAIL

echo "===== SuStaIn ANGST CV JOB STARTED ====="
echo "Hostname: $(hostname)"
echo "Startzeit: $(date)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
conda activate "${SUSTAIN_CONDA_ENV:-sustain_env}"

# PySuStaIn nutzt eigene Prozesse -> BLAS-Threads begrenzen
# PySuStaIn nutzt eigene Prozesse -> BLAS-Threads begrenzen
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

: "${SUSTAIN_PROJECT_DIR:?Set SUSTAIN_PROJECT_DIR to the repository root before submission}"
cd "$SUSTAIN_PROJECT_DIR"

python -u HPC_Scripts/sustain_cv_Angst_3fold.py

echo "===== SuStaIn ANGST CV JOB FINISHED ====="
echo "Endzeit: $(date)"
