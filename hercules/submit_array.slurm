#!/bin/bash
#SBATCH --job-name=qtcl_seed
#SBATCH --output=logs/qtcl_%A_%a.out
#SBATCH --error=logs/qtcl_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --array=0-2          # 3 seeds en paralelo
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmarper2@upo.es

# ── Job Array: 1 seed por nodo, todos en paralelo ─────────────────────────────
# Lanzar con: sbatch hercules/submit_array.slurm
# Monitorear: squeue -u $USER
# Ver logs:   tail -f logs/qtcl_<ARRAY_ID>_<SEED>.out

SEED=$SLURM_ARRAY_TASK_ID

echo "============================================"
echo "QTCL v6 — Seed $SEED / Job $SLURM_ARRAY_JOB_ID"
echo "Nodo: $SLURM_NODELIST | CPUs: $SLURM_CPUS_PER_TASK"
echo "Inicio: $(date)"
echo "============================================"

module purge
module load python/3.10
module load miniconda/23.5
source activate qtcl-env

mkdir -p logs figures data results

# Patch N_SEEDS=1 y arrancar desde el seed correcto via monkey-patch
python -u - <<PYEOF
import sys, json, os
import numpy as np

# Añadir code/ al path
sys.path.insert(0, 'code')

# Importar módulo y parchear constantes antes de ejecutar lógica
import qtcl_v6_experiment as mod

SEED = $SEED
mod.N_SEEDS = 1  # solo 1 seed en este job

# Reemplazar el loop de seeds para arrancar desde SEED
original_main = mod.main

def patched_main():
    import pandas as pd
    T = len(mod.TASKS)
    methods = ["Classical Naive", "Classical EWC",
               "Quantum Naive",   "Quantum EWC",  "QTCL"]

    all_accs    = {m: [] for m in methods}
    all_metrics = {m: [] for m in methods}

    print(f"\n[Seed {SEED}] Cargando Split-MNIST...")
    tasks_tr, tasks_te = mod.load_split_mnist(seed=SEED*37+5)

    for m in methods:
        print(f"  [{m}]")
        acc = mod.run_method(m, tasks_tr, tasks_te, SEED)
        all_accs[m].append(acc)
        all_metrics[m].append(mod.cl_metrics(acc))

    # Métricas
    summary = {}
    for m in methods:
        aa  = float(all_metrics[m][0]["AA"])
        bwt = float(all_metrics[m][0]["BWT"])
        f   = float(all_metrics[m][0]["F"])
        summary[m] = {"seed": SEED, "AA": aa, "BWT": bwt, "F": f}
        print(f"  {m:20s}  AA={aa:.4f}  BWT={bwt:+.4f}  F={f:.4f}")

    # Guardar resultados individuales
    out_path = f"results/seed_{SEED}_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {out_path}")

    return summary

patched_main()
PYEOF

echo "Seed $SEED completado: $(date)"
