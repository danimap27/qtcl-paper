#!/bin/bash
#SBATCH --job-name=qtcl_quantics
#SBATCH --output=logs/quantics_%j.out
#SBATCH --error=logs/quantics_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --partition=standard
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danitanke02@gmail.com

# ── QTCL QUANTICS 2026 — experimento completo ─────────────────────────────────
# Ejecuta: experimento principal (3 seeds × 5 métodos) + ablación (λ + ρ)
# Genera:  results/quantics_main.json, results/quantics_ablation.json, figures/
#
# Uso:
#   sbatch hercules/submit_quantics_main.slurm
#
# Monitorear:
#   squeue -u $USER
#   tail -f logs/quantics_<JOB_ID>.out

echo "========================================================"
echo "QTCL QUANTICS 2026 — Experimento completo"
echo "Nodo: $SLURM_NODELIST | CPUs: $SLURM_CPUS_PER_TASK"
echo "Inicio: $(date)"
echo "========================================================"

module purge
module load Miniconda3
source activate qtcl

mkdir -p logs figures results data

echo ""
echo "── Fase 1: Experimento principal (Table 1 del paper) ──"
python -u code/qtcl_quantics_experiment.py --mode main

echo ""
echo "── Fase 2: Ablación lambda ──"
python -u code/qtcl_quantics_experiment.py --mode ablation_lambda

echo ""
echo "── Fase 3: Ablación rehearsal ──"
python -u code/qtcl_quantics_experiment.py --mode ablation_rehearsal

echo ""
echo "── Fase 4: Generar todas las figuras ──"
python -u code/qtcl_quantics_experiment.py --mode figures

echo ""
echo "========================================================"
echo "Completado: $(date)"
echo "Resultados: results/quantics_main.json"
echo "            results/quantics_ablation.json"
echo "Figuras:    figures/"
echo "========================================================"
