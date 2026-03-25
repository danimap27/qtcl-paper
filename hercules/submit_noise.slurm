#!/bin/bash
#SBATCH --job-name=qtcl_noise
#SBATCH --output=logs/noise_%j.out
#SBATCH --error=logs/noise_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=standard
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmarper2@upo.es

# ── QTCL Noise Experiment — IBM FakeBrisbane ───────────────────────────────
# Ejecuta QTCL limpio vs QTCL con ruido real de IBM FakeBrisbane.
# Más lento que el experimento limpio: parameter-shift + N_SHOTS por circuito.
#
# Uso:
#   sbatch hercules/submit_noise.slurm
#
# Monitorizar:
#   squeue -u $USER
#   tail -f logs/noise_<JOB_ID>.out

echo "========================================================"
echo "QTCL Noise Experiment — IBM FakeBrisbane"
echo "Nodo: $SLURM_NODELIST | CPUs: $SLURM_CPUS_PER_TASK"
echo "Inicio: $(date)"
echo "========================================================"

module purge
module load Miniconda3
source activate qtcl

# Instalar dependencias extra si no están
pip install -q pennylane-qiskit qiskit-aer qiskit-ibm-runtime 2>/dev/null

mkdir -p logs figures results data

echo ""
echo "── Experimento ruido IBM FakeBrisbane ──"
python -u code/qtcl_noise_experiment.py

echo ""
echo "========================================================"
echo "Completado: $(date)"
echo "Resultados: results/noise_results.json"
echo "Figuras:    figures/noise_comparison.pdf"
echo "            figures/noise_degradation.pdf"
echo "========================================================"
