#!/bin/bash
#SBATCH --job-name=qtcl_v6
#SBATCH --output=logs/qtcl_%j.out
#SBATCH --error=logs/qtcl_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmarper2@upo.es

# ── QTCL v6 — experimento completo (3 seeds × 5 métodos × 5 tareas) ──────────
# Lanzar con: sbatch hercules/submit_qtcl.slurm
# Monitorear: squeue -u $USER
# Ver log:    tail -f logs/qtcl_<JOB_ID>.out

echo "============================================"
echo "QTCL v6 — Split-MNIST Continual Learning"
echo "Nodo: $SLURM_NODELIST | CPUs: $SLURM_CPUS_PER_TASK"
echo "Inicio: $(date)"
echo "============================================"

module purge
module load python/3.10
module load miniconda/23.5
source activate qtcl-env

mkdir -p logs figures results data

# Ejecutar experimento completo
python -u code/qtcl_v6_experiment.py

echo ""
echo "============================================"
echo "Completado: $(date)"
echo "Resultados en: figures/ y results*.json"
echo "============================================"
