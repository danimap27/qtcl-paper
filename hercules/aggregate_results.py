#!/usr/bin/env python3
"""
Agrega los resultados de los 3 jobs del array en un único JSON.
Ejecutar después de que los 3 seeds hayan terminado:
    python hercules/aggregate_results.py
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
METHODS = ["Classical Naive", "Classical EWC",
           "Quantum Naive", "Quantum EWC", "QTCL"]
N_SEEDS = 3


def aggregate():
    # Cargar resultados por seed
    per_seed = []
    for seed in range(N_SEEDS):
        path = RESULTS_DIR / f"seed_{seed}_main.json"
        if not path.exists():
            print(f"[WARN] Falta {path} — seed {seed} no completado")
            continue
        with open(path) as f:
            per_seed.append(json.load(f))

    if not per_seed:
        print("No hay resultados que agregar.")
        return

    print(f"Seeds cargados: {len(per_seed)}")
    print()

    # Agregar métricas por método
    summary = {}
    header = f"{'Método':22s}  {'AA':>10s}  {'BWT':>10s}  {'FWT':>10s}  {'F':>10s}"
    print(header)
    print("-" * len(header))

    for m in METHODS:
        aa_vals  = [s[m]["AA"]  for s in per_seed if m in s]
        bwt_vals = [s[m]["BWT"] for s in per_seed if m in s]
        fwt_vals = [s[m]["FWT"] for s in per_seed if m in s]
        f_vals   = [s[m]["F"]   for s in per_seed if m in s]

        summary[m] = {
            "AA_mean":  float(np.mean(aa_vals)),
            "AA_std":   float(np.std(aa_vals)),
            "BWT_mean": float(np.mean(bwt_vals)),
            "BWT_std":  float(np.std(bwt_vals)),
            "FWT_mean": float(np.mean(fwt_vals)),
            "FWT_std":  float(np.std(fwt_vals)),
            "F_mean":   float(np.mean(f_vals)),
            "F_std":    float(np.std(f_vals)),
            "n_seeds":  len(aa_vals),
        }

        aa  = f"{np.mean(aa_vals):.3f}±{np.std(aa_vals):.3f}"
        bwt = f"{np.mean(bwt_vals):+.3f}±{np.std(bwt_vals):.3f}"
        fwt = f"{np.mean(fwt_vals):+.3f}±{np.std(fwt_vals):.3f}"
        f   = f"{np.mean(f_vals):.3f}±{np.std(f_vals):.3f}"
        print(f"{m:22s}  {aa:>10s}  {bwt:>10s}  {fwt:>10s}  {f:>10s}")

    # Guardar resultado final
    out_path = RESULTS_DIR / "quantics_main.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResultados guardados en {out_path}")

    # Agregar ablación si existe
    abl_seeds = []
    for seed in range(N_SEEDS):
        path = RESULTS_DIR / f"seed_{seed}_ablation.json"
        if path.exists():
            with open(path) as f:
                abl_seeds.append(json.load(f))

    if abl_seeds:
        abl_summary = {}
        for key in abl_seeds[0]:          # "lambda" y "rehearsal"
            abl_summary[key] = {}
            for val in abl_seeds[0][key]:
                aa_list = [s[key][val]["AA"] for s in abl_seeds if key in s and val in s[key]]
                abl_summary[key][val] = {
                    "AA_mean": float(np.mean(aa_list)),
                    "AA_std":  float(np.std(aa_list)),
                }
        abl_out = RESULTS_DIR / "quantics_ablation.json"
        with open(abl_out, "w") as f:
            json.dump(abl_summary, f, indent=2)
        print(f"Ablación guardada en {abl_out}")


if __name__ == "__main__":
    aggregate()
