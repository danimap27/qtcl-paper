#!/bin/bash
#SBATCH --job-name=qtcl_seed
#SBATCH --output=logs/quantics_%A_%a.out
#SBATCH --error=logs/quantics_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danitanke02@gmail.com

# ── QTCL QUANTICS 2026 — seeds en paralelo (array job) ────────────────────────
# Lanza 3 jobs simultáneos, uno por seed. Más rápido que el job único.
# Después de que los 3 terminen, agregar con:
#   python hercules/aggregate_results.py
#
# Uso:
#   sbatch hercules/submit_quantics_array.slurm
#   # esperar a que termine: squeue -u $USER
#   python hercules/aggregate_results.py

SEED=$SLURM_ARRAY_TASK_ID

echo "========================================================"
echo "QTCL QUANTICS — Seed $SEED  (Array job $SLURM_ARRAY_JOB_ID)"
echo "Nodo: $SLURM_NODELIST | CPUs: $SLURM_CPUS_PER_TASK"
echo "Inicio: $(date)"
echo "========================================================"

module purge
module load Miniconda3
source activate qtcl

mkdir -p logs figures results data

python -u code/qtcl_quantics_experiment.py --mode main --seed $SEED

echo "Seed $SEED completado: $(date)"
