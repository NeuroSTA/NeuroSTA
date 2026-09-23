#!/bin/bash
#SBATCH --job-name=sustain_main_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=120G
#SBATCH --time=10-00:00:00
#SBATCH --partition=normal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL

echo "===== JOB STARTED ====="
echo "Hostname: $(hostname)"
echo "Startzeit: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# ---------------------------------------------------------
# Module / Conda-Umgebung
# ---------------------------------------------------------
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
conda activate "${SUSTAIN_CONDA_ENV:-sustain_env}"

# ---------------------------------------------------------
# tqdm & Threads konfigurieren
# ---------------------------------------------------------
# tqdm-Fortschrittsbalken deaktivieren (schreibt sonst massiv ins Log)
export TQDM_DISABLE=1

# Verhindere Oversubscription durch BLAS/MKL/OpenMP
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ---------------------------------------------------------
# Ausführen
# ---------------------------------------------------------
: "${SUSTAIN_PROJECT_DIR:?Set SUSTAIN_PROJECT_DIR to the repository root before submission}"
cd "$SUSTAIN_PROJECT_DIR"

echo "Starte Python-Script: run_sustain_main.py"
python -u HPC_Scripts/run_sustain_main.py

echo "=========================================="
echo "===== JOB FINISHED ====="
echo "Endzeit: $(date)"
