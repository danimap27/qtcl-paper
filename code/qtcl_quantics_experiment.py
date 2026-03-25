"""
QTCL QUANTICS 2026 — Experimentation script for Hercules HPC
=================================================================
Reproduces Table 1 and the ablation figures from the paper:
  "Quantum Continual Learning via Fisher EWC and Experience Replay
   in Variational Quantum Circuits"

Execution modes (--mode):
  main               — Main experiment: 5 methods × N_SEEDS seeds
  main --seed K      — A single seed K (for array jobs)
  ablation_lambda    — Sweeps λ_Q ∈ {50,100,200,500,1000,2000}
  ablation_rehearsal — Sweeps ρ ∈ {0,0.1,0.15,0.2,0.25,0.3,0.4}
  figures            — Generates figures from results saved in results/

Outputs:
  results/seed_{K}_main.json       — results for a single seed (array mode)
  results/quantics_main.json       — aggregated results (full main mode)
  results/quantics_ablation.json   — ablation
  figures/*.pdf, figures/*.png

Typical usage on Hercules:
  # Single job:
  sbatch hercules/submit_quantics_main.slurm

  # Seeds in parallel (faster):
  sbatch hercules/submit_quantics_array.slurm
  # After the 3 finish:
  python hercules/aggregate_results.py
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Import v6 experiment core ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from qtcl_v6_experiment import (
    # Configuration
    N_QUBITS, N_SHARED_LAYERS, N_TASK_LAYERS, ENC_HIDDEN,
    N_TRAIN_PER_TASK, N_TEST_PER_TASK, N_EPOCHS, BATCH_SIZE, LR,
    LAMBDA_EWC_Q, LAMBDA_EWC_C, REHEARSAL_RATIO, N_SEEDS, TASKS, DEVICE,
    COLORS, FIGURES_DIR,
    # Functions
    load_split_mnist, run_method, cl_metrics,
    # Figures
    fig_cl_metrics_ci, fig_acc_matrix, fig_acc_evolution,
    fig_forgetting, fig_radar, fig_summary_table,
    fig_architecture, fig_circuit_diagram, fig_mnist_tasks,
    fig_quantum_vs_classical,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

METHODS = ["Classical Naive", "Classical EWC",
           "Quantum Naive", "Quantum EWC", "QTCL"]

# Ablation values (match those in the paper)
LAMBDA_SWEEP    = [50, 100, 200, 500, 1000, 2000]
REHEARSAL_SWEEP = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header(title: str):
    sep = "=" * 60
    print(f"\n{sep}")
    print(title)
    print(sep)


def _save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Mode: main experiment ─────────────────────────────────────────────────────

def run_main(seed: int = None):
    """Main experiment: 5 methods, N_SEEDS seeds (or one if seed is not None)."""
    seeds = [seed] if seed is not None else list(range(N_SEEDS))
    _print_header(
        f"QTCL QUANTICS — Main experiment\n"
        f"Seeds: {seeds} | Methods: {len(METHODS)} | Tasks: {len(TASKS)}\n"
        f"λ_Q={LAMBDA_EWC_Q} λ_C={LAMBDA_EWC_C} ρ={REHEARSAL_RATIO} "
        f"epochs={N_EPOCHS} lr={LR}"
    )

    all_accs    = {m: [] for m in METHODS}
    all_metrics = {m: [] for m in METHODS}

    for s in seeds:
        print(f"\n── Seed {s} ──────────────────────────────")
        tasks_tr, tasks_te = load_split_mnist(seed=s * 37 + 5)

        for m in METHODS:
            print(f"  [{m}]")
            acc = run_method(m, tasks_tr, tasks_te, seed=s)
            all_accs[m].append(acc)
            mets = cl_metrics(acc)
            all_metrics[m].append(mets)
            print(f"    AA={mets['AA']:.4f}  BWT={mets['BWT']:+.4f}  "
                  f"FWT={mets['FWT']:+.4f}  F={mets['F']:.4f}")

    # Print summary table
    print(f"\n{'Method':22s}  {'AA':>10s}  {'BWT':>10s}  {'FWT':>10s}  {'F':>10s}")
    print("-" * 70)
    for m in METHODS:
        aa  = [x["AA"]  for x in all_metrics[m]]
        bwt = [x["BWT"] for x in all_metrics[m]]
        fwt = [x["FWT"] for x in all_metrics[m]]
        f   = [x["F"]   for x in all_metrics[m]]
        print(f"{m:22s}  "
              f"{np.mean(aa):.3f}±{np.std(aa):.3f}  "
              f"{np.mean(bwt):+.3f}±{np.std(bwt):.3f}  "
              f"{np.mean(fwt):+.3f}±{np.std(fwt):.3f}  "
              f"{np.mean(f):.3f}±{np.std(f):.3f}")

    # Save results
    if seed is not None:
        # Array mode: save result for the individual seed
        out = {
            m: {
                "seed":  seed,
                "AA":    float(all_metrics[m][0]["AA"]),
                "BWT":   float(all_metrics[m][0]["BWT"]),
                "FWT":   float(all_metrics[m][0]["FWT"]),
                "F":     float(all_metrics[m][0]["F"]),
                "acc_matrix": all_accs[m][0].tolist(),
            }
            for m in METHODS
        }
        _save_json(out, RESULTS_DIR / f"seed_{seed}_main.json")
    else:
        # Full mode: save aggregated summary
        out = {
            m: {
                "AA_mean":  float(np.mean([x["AA"]  for x in all_metrics[m]])),
                "AA_std":   float(np.std( [x["AA"]  for x in all_metrics[m]])),
                "BWT_mean": float(np.mean([x["BWT"] for x in all_metrics[m]])),
                "BWT_std":  float(np.std( [x["BWT"] for x in all_metrics[m]])),
                "FWT_mean": float(np.mean([x["FWT"] for x in all_metrics[m]])),
                "FWT_std":  float(np.std( [x["FWT"] for x in all_metrics[m]])),
                "F_mean":   float(np.mean([x["F"]   for x in all_metrics[m]])),
                "F_std":    float(np.std( [x["F"]   for x in all_metrics[m]])),
                "n_seeds":  len(seeds),
            }
            for m in METHODS
        }
        _save_json(out, RESULTS_DIR / "quantics_main.json")

        # Generate figures for the main experiment
        print("\n── Figures ──────────────────────────────")
        tasks_demo, _ = load_split_mnist(seed=42)
        fig_mnist_tasks(tasks_demo)
        fig_architecture()
        try:
            fig_circuit_diagram()
        except Exception as e:
            print(f"  → vqc_circuit (skipped: {e})")

        # Figures that need per-seed data
        acc_dict_mean = {
            m: np.mean(all_accs[m], axis=0) for m in METHODS
        }
        fig_acc_matrix(
            {m: acc_dict_mean[m] for m in ["Classical EWC", "QTCL"]},
            T=len(TASKS)
        )
        fig_acc_evolution(acc_dict_mean, T=len(TASKS))
        fig_forgetting(acc_dict_mean, T=len(TASKS))
        fig_cl_metrics_ci(all_metrics)
        fig_quantum_vs_classical(all_metrics)
        metrics_mean = {
            m: {k: float(np.mean([x[k] for x in all_metrics[m]]))
                for k in ["AA", "BWT", "FWT", "F"]}
            for m in METHODS
        }
        fig_radar(metrics_mean)
        fig_summary_table(metrics_mean)
        print("\nFigures saved in figures/")

    return all_accs, all_metrics


# ── Mode: lambda ablation ─────────────────────────────────────────────────────

def run_ablation_lambda(n_seeds: int = 2):
    """Sweeps λ_Q with full QTCL method, 2 seeds."""
    _print_header(
        f"Ablation λ_Q — values: {LAMBDA_SWEEP}\n"
        f"Method: QTCL | Seeds: {n_seeds}"
    )

    import qtcl_v6_experiment as mod

    results = {}
    for lam in LAMBDA_SWEEP:
        print(f"\n  λ_Q = {lam}")
        mod.LAMBDA_EWC_Q = lam
        aa_vals = []
        for s in range(n_seeds):
            tasks_tr, tasks_te = load_split_mnist(seed=s * 37 + 5)
            acc = run_method("QTCL", tasks_tr, tasks_te, seed=s)
            aa_vals.append(cl_metrics(acc)["AA"])
        m = float(np.mean(aa_vals))
        std = float(np.std(aa_vals))
        results[str(lam)] = {"AA": m, "AA_std": std}
        print(f"    AA = {m:.4f} ± {std:.4f}")

    # Restore original value
    mod.LAMBDA_EWC_Q = LAMBDA_EWC_Q

    out_path = RESULTS_DIR / "ablation_lambda.json"
    _save_json({"lambda": results}, out_path)

    # Figure
    _fig_ablation(
        x_vals=LAMBDA_SWEEP,
        y_means=[results[str(v)]["AA"]     for v in LAMBDA_SWEEP],
        y_stds= [results[str(v)]["AA_std"] for v in LAMBDA_SWEEP],
        xlabel=r"EWC strength $\lambda_Q$",
        ylabel="Average Accuracy",
        title=r"Ablation: EWC strength $\lambda_Q$ (QTCL on Split-MNIST)",
        fname="ablation_lambda",
        xscale="log",
        vline=LAMBDA_EWC_Q,
    )
    return results


def run_ablation_rehearsal(n_seeds: int = 2):
    """Sweeps ρ with full QTCL method, 2 seeds."""
    _print_header(
        f"Ablation ρ — values: {REHEARSAL_SWEEP}\n"
        f"Method: QTCL | Seeds: {n_seeds}"
    )

    import qtcl_v6_experiment as mod

    results = {}
    for rho in REHEARSAL_SWEEP:
        print(f"\n  ρ = {rho}")
        mod.REHEARSAL_RATIO = rho
        aa_vals = []
        for s in range(n_seeds):
            tasks_tr, tasks_te = load_split_mnist(seed=s * 37 + 5)
            acc = run_method("QTCL", tasks_tr, tasks_te, seed=s)
            aa_vals.append(cl_metrics(acc)["AA"])
        m = float(np.mean(aa_vals))
        std = float(np.std(aa_vals))
        results[str(rho)] = {"AA": m, "AA_std": std}
        print(f"    AA = {m:.4f} ± {std:.4f}")

    # Restore original value
    mod.REHEARSAL_RATIO = REHEARSAL_RATIO

    out_path = RESULTS_DIR / "ablation_rehearsal.json"
    _save_json({"rehearsal": results}, out_path)

    # Figure
    _fig_ablation(
        x_vals=REHEARSAL_SWEEP,
        y_means=[results[str(v)]["AA"]     for v in REHEARSAL_SWEEP],
        y_stds= [results[str(v)]["AA_std"] for v in REHEARSAL_SWEEP],
        xlabel=r"Rehearsal ratio $\rho$",
        ylabel="Average Accuracy",
        title=r"Ablation: rehearsal ratio $\rho$ (QTCL on Split-MNIST)",
        fname="ablation_rehearsal",
        vline=REHEARSAL_RATIO,
    )
    return results


def _fig_ablation(x_vals, y_means, y_stds, xlabel, ylabel, title,
                  fname, xscale=None, vline=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x_vals, y_means, yerr=[1.96 * s for s in y_stds],
                marker="o", color="#0D47A1", lw=2, ms=7, capsize=5,
                label="QTCL (mean ± 95% CI)")
    if vline is not None:
        ax.axvline(vline, color="#FF6F00", ls="--", lw=1.5,
                   label=f"paper default ({vline})")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=11)
    if xscale:
        ax.set_xscale(xscale)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in x_vals], rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{fname}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


# ── Mode: generate figures from JSON ─────────────────────────────────────────

def run_figures():
    """Generates all figures from saved JSON files."""
    _print_header("Generating figures from saved results")

    main_path = RESULTS_DIR / "quantics_main.json"
    if not main_path.exists():
        print(f"[ERROR] {main_path} not found. Run --mode main first.")
        sys.exit(1)

    data = _load_json(main_path)
    print(f"  Methods in results: {list(data.keys())}")

    # Figures that do not need raw data
    tasks_demo, _ = load_split_mnist(seed=42)
    fig_mnist_tasks(tasks_demo)
    fig_architecture()
    try:
        fig_circuit_diagram()
    except Exception as e:
        print(f"  → vqc_circuit (skipped: {e})")

    # Metrics figure from saved means/std
    metrics_mean = {
        m: {
            "AA":  data[m]["AA_mean"],
            "BWT": data[m]["BWT_mean"],
            "FWT": data[m]["FWT_mean"],
            "F":   data[m]["F_mean"],
        }
        for m in METHODS if m in data
    }
    fig_radar(metrics_mean)
    fig_summary_table(metrics_mean)

    # Ablation figures if they exist
    for fname, key in [("ablation_lambda", "lambda"),
                       ("ablation_rehearsal", "rehearsal")]:
        path = RESULTS_DIR / f"{fname}.json"
        if path.exists():
            abl = _load_json(path)[key]
            x_vals = [float(k) for k in abl]
            y_means = [abl[k]["AA"]     for k in abl]
            y_stds  = [abl[k]["AA_std"] for k in abl]
            xlabel = (r"EWC strength $\lambda_Q$"
                      if key == "lambda" else r"Rehearsal ratio $\rho$")
            vline = LAMBDA_EWC_Q if key == "lambda" else REHEARSAL_RATIO
            _fig_ablation(x_vals, y_means, y_stds,
                          xlabel=xlabel, ylabel="Average Accuracy",
                          title=f"Ablation: {key}",
                          fname=fname, vline=vline,
                          xscale="log" if key == "lambda" else None)

    print("\nFigures saved in figures/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="QTCL QUANTICS 2026 — Experiments for Hercules HPC"
    )
    p.add_argument(
        "--mode",
        choices=["main", "ablation_lambda", "ablation_rehearsal", "figures"],
        required=True,
        help="Execution mode",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Specific seed (only for --mode main, used in array jobs)",
    )
    p.add_argument(
        "--abl_seeds",
        type=int,
        default=2,
        help="Number of seeds for ablation (default: 2)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "main":
        run_main(seed=args.seed)

    elif args.mode == "ablation_lambda":
        run_ablation_lambda(n_seeds=args.abl_seeds)

    elif args.mode == "ablation_rehearsal":
        run_ablation_rehearsal(n_seeds=args.abl_seeds)

    elif args.mode == "figures":
        run_figures()
