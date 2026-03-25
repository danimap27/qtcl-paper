"""
QTCL Noise Experiment — Real IBM QPU noise (FakeBrisbane)
=============================================================
Runs clean QTCL vs QTCL with real IBM hardware noise and compares metrics.

Noise source: FakeBrisbane (qiskit_ibm_runtime.fake_provider)
  - Real calibration data from the ibm_brisbane processor (Eagle r3, 127 qubits)
  - Includes: depolarizing error, thermal relaxation (T1/T2), readout error
  - Does not require IBM token or internet connection

Result: clean vs noisy table + comparison figure for the paper.

Usage on Hercules:
  sbatch hercules/submit_noise.slurm

Outputs:
  results/noise_results.json
  figures/noise_comparison.pdf / .png
  figures/noise_degradation.pdf / .png
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from qtcl_v6_experiment import (
    N_QUBITS, N_SHARED_LAYERS, N_TASK_LAYERS, ENC_HIDDEN,
    N_TRAIN_PER_TASK, N_TEST_PER_TASK, N_EPOCHS, BATCH_SIZE, LR,
    LAMBDA_EWC_Q, REHEARSAL_RATIO, TASKS, DEVICE, COLORS,
    load_split_mnist, cl_metrics, EWC, ClassicalModel,
    make_loader, train_task, eval_model, FIGURES_DIR,
)

import pennylane as qml

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IBM_BACKEND  = "FakeBrisbane"   # calibration backend
N_SEEDS      = 2                # seeds for the noise experiment
N_SHOTS      = 1024             # shots per circuit (required with noise)


# ─── Noise model from IBM FakeBrisbane ────────────────────────────────────────

def load_ibm_noise_model():
    """
    Loads the FakeBrisbane noise model.
    Contains: gate errors, T1/T2 thermal relaxation, readout errors
    from real hardware calibration data of ibm_brisbane.
    """
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    from qiskit_aer.noise import NoiseModel

    print(f"  Loading noise model from {IBM_BACKEND}...")
    backend    = FakeBrisbane()
    noise_model = NoiseModel.from_backend(backend)

    gates   = list(noise_model.noise_instructions)
    n_errs  = len(noise_model._local_quantum_errors) + len(noise_model._default_quantum_errors)
    print(f"  Gates with noise: {gates}")
    print(f"  Total quantum errors registered: {n_errs}")

    # Save noise model info for the paper
    info = {
        "backend":        IBM_BACKEND,
        "basis_gates":    list(backend.operation_names),
        "n_qubits_hw":    backend.num_qubits,
        "noisy_gates":    gates,
        "n_error_entries": n_errs,
    }
    with open(RESULTS_DIR / "noise_model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Info saved: results/noise_model_info.json")

    return noise_model


# ─── VQC with noise ────────────────────────────────────────────────────────────

def _make_noisy_qnode(noise_model, n_qubits=N_QUBITS,
                      n_shared=N_SHARED_LAYERS, n_task=N_TASK_LAYERS):
    """
    QNode with qiskit.aer + real IBM noise model.
    Uses parameter-shift (adjoint not compatible with noise model).
    shots=N_SHOTS for realistic stochastic simulation.
    """
    dev = qml.device(
        "qiskit.aer",
        wires=n_qubits,
        noise_model=noise_model,
        shots=N_SHOTS,
    )

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, shared_w, task_w):
        for i in range(n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)
        qml.StronglyEntanglingLayers(shared_w, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(task_w,   wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class NoisyQuantumModel(nn.Module):
    """QuantumModel with noisy circuit (qiskit.aer + IBM noise model)."""

    def __init__(self, noise_model, input_dim=784):
        super().__init__()
        self.n_qubits  = N_QUBITS
        self.encoder   = nn.Sequential(
            nn.Linear(input_dim, ENC_HIDDEN), nn.ReLU(),
            nn.LayerNorm(ENC_HIDDEN),
            nn.Linear(ENC_HIDDEN, N_QUBITS), nn.Tanh()
        )
        self.shared_w  = nn.Parameter(
            torch.randn(N_SHARED_LAYERS, N_QUBITS, 3) * 0.1
        )
        self.task_w    = nn.ParameterDict()
        self._add_task(0)
        self.current_task = 0
        self.post_q    = nn.Linear(N_QUBITS, 2)
        self.circuit   = _make_noisy_qnode(noise_model)

    def _add_task(self, task_id):
        key = f"t{task_id}"
        if key not in self.task_w:
            self.task_w[key] = nn.Parameter(
                torch.randn(N_TASK_LAYERS, N_QUBITS, 3) * 0.1)

    def set_task(self, task_id):
        self._add_task(task_id)
        self.current_task = task_id

    def forward(self, x):
        z  = self.encoder(x)
        tw = self.task_w[f"t{self.current_task}"]
        outs = []
        for i in range(z.shape[0]):
            outs.append(torch.stack(self.circuit(z[i], self.shared_w, tw)))
        return self.post_q(torch.stack(outs).float())


# ─── Generic QTCL runner ─────────────────────────────────────────────────────

def run_qtcl(model, tasks_tr, tasks_te, seed):
    """Runs QTCL (EWC + rehearsal) with the given model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    T        = len(tasks_tr)
    acc      = np.zeros((T, T))
    ewc_obj  = EWC(model, LAMBDA_EWC_Q)
    reh_X, reh_y = [], []

    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    task {i+1}/{T}...", end=" ", flush=True)
        model.set_task(i)

        if reh_X:
            Xr = torch.cat([X_tr] + reh_X)
            yr = torch.cat([y_tr] + reh_y)
        else:
            Xr, yr = X_tr, y_tr

        train_task(model, Xr, yr, N_EPOCHS, ewc=ewc_obj)
        ewc_obj.register(X_tr, y_tr, i)

        n_keep = max(4, int(REHEARSAL_RATIO * len(X_tr)))
        idx    = np.random.choice(len(X_tr), n_keep, replace=False)
        reh_X.append(X_tr[idx])
        reh_y.append(y_tr[idx])

        for j in range(T):
            model.set_task(j)
            acc[i, j] = eval_model(model, *tasks_te[j])

        print(f"AA={acc[i, :i+1].mean():.3f}", flush=True)

    return acc


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_noise_experiment():
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"QTCL Noise Experiment — {IBM_BACKEND}")
    print(f"Seeds: {N_SEEDS} | Shots: {N_SHOTS} | Tasks: {len(TASKS)}")
    print(sep)

    # Load noise model once
    noise_model = load_ibm_noise_model()

    clean_metrics_list = []
    noisy_metrics_list = []

    for s in range(N_SEEDS):
        print(f"\n── Seed {s} ──────────────────────────────")
        tasks_tr, tasks_te = load_split_mnist(seed=s * 37 + 5)

        # ── Clean (lightning.qubit, adjoint) ──
        print("  [Clean — lightning.qubit]")
        from qtcl_v6_experiment import QuantumModel
        clean_model = QuantumModel().to(DEVICE)
        clean_acc   = run_qtcl(clean_model, tasks_tr, tasks_te, seed=s)
        cm          = cl_metrics(clean_acc)
        clean_metrics_list.append(cm)
        print(f"  Clean → AA={cm['AA']:.4f}  F={cm['F']:.4f}")

        # ── Noisy (qiskit.aer + IBM noise model) ──
        print(f"  [Noisy — qiskit.aer + {IBM_BACKEND}]")
        noisy_model = NoisyQuantumModel(noise_model).to(DEVICE)
        noisy_acc   = run_qtcl(noisy_model, tasks_tr, tasks_te, seed=s)
        nm          = cl_metrics(noisy_acc)
        noisy_metrics_list.append(nm)
        print(f"  Noisy → AA={nm['AA']:.4f}  F={nm['F']:.4f}")

    # ─── Aggregated results ───────────────────────────────────────────────────
    def _agg(lst, key):
        vals = [x[key] for x in lst]
        return float(np.mean(vals)), float(np.std(vals))

    results = {
        "backend":   IBM_BACKEND,
        "n_shots":   N_SHOTS,
        "n_seeds":   N_SEEDS,
        "clean": {k: {"mean": _agg(clean_metrics_list, k)[0],
                      "std":  _agg(clean_metrics_list, k)[1]}
                  for k in ["AA", "BWT", "FWT", "F"]},
        "noisy": {k: {"mean": _agg(noisy_metrics_list, k)[0],
                      "std":  _agg(noisy_metrics_list, k)[1]}
                  for k in ["AA", "BWT", "FWT", "F"]},
    }

    # Absolute degradation
    for k in ["AA", "F"]:
        delta = results["noisy"][k]["mean"] - results["clean"][k]["mean"]
        results[f"delta_{k}"] = round(delta, 4)

    with open(RESULTS_DIR / "noise_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ─── Console table ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"QTCL Noise Results — {IBM_BACKEND} ({N_SHOTS} shots/circuit)")
    print(f"{'Condition':20s}  {'AA':>12s}  {'BWT':>12s}  {'FWT':>12s}  {'F':>12s}")
    print("-" * 75)
    for cond, lst in [("Clean (ideal)", clean_metrics_list),
                      (f"Noisy ({IBM_BACKEND})", noisy_metrics_list)]:
        row = {}
        for k in ["AA", "BWT", "FWT", "F"]:
            m, s = _agg(lst, k)
            row[k] = f"{m:.3f}±{s:.3f}"
        print(f"{cond:20s}  {row['AA']:>12s}  {row['BWT']:>12s}  "
              f"{row['FWT']:>12s}  {row['F']:>12s}")
    for k in ["AA", "F"]:
        print(f"  Δ{k} (noisy−clean): {results[f'delta_{k}']:+.4f}")
    print(sep)

    # ─── Figures ──────────────────────────────────────────────────────────────
    _fig_noise_comparison(results)
    _fig_noise_degradation_bar(results)

    print(f"\nResults: results/noise_results.json")
    print(f"Figures: figures/noise_comparison.pdf")
    print(f"         figures/noise_degradation.pdf")

    return results


# ─── Figures ─────────────────────────────────────────────────────────────────

def _fig_noise_comparison(results):
    """Grouped bars: clean vs noisy for AA, BWT, FWT, F."""
    metrics = ["AA", "BWT", "FWT", "F"]
    labels  = ["Average\nAccuracy", "Backward\nTransfer", "Forward\nTransfer", "Forgetting"]
    x = np.arange(len(metrics))
    w = 0.35

    clean_m = [results["clean"][k]["mean"] for k in metrics]
    clean_s = [results["clean"][k]["std"]  for k in metrics]
    noisy_m = [results["noisy"][k]["mean"] for k in metrics]
    noisy_s = [results["noisy"][k]["std"]  for k in metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, clean_m, w, yerr=clean_s, capsize=5,
                color="#0D47A1", alpha=0.85, label="QTCL clean (lightning.qubit)")
    b2 = ax.bar(x + w/2, noisy_m, w, yerr=noisy_s, capsize=5,
                color="#FF6F00", alpha=0.85, label=f"QTCL noisy ({IBM_BACKEND})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"QTCL: ideal vs. hardware noise ({IBM_BACKEND}, {results['n_shots']} shots)",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylim(-0.2, 1.05)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"noise_comparison.{ext}", bbox_inches="tight")
    plt.close()
    print("  → noise_comparison")


def _fig_noise_degradation_bar(results):
    """Absolute degradation Δ(noisy − clean) for AA and F."""
    metrics = ["AA", "F"]
    labels  = ["Average Accuracy", "Forgetting"]
    deltas  = [results[f"delta_{k}"] for k in metrics]
    colors  = ["#C62828" if d < 0 else "#2E7D32" for d in deltas]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, deltas, color=colors, alpha=0.85, edgecolor="white", lw=1.2)
    ax.axhline(0, color="black", lw=1.0)
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (0.005 if val >= 0 else -0.015),
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("Δ (noisy − clean)", fontsize=11)
    ax.set_title(
        f"Noise degradation: QTCL on {IBM_BACKEND}",
        fontsize=11
    )
    ax.set_ylim(min(deltas) - 0.05, max(deltas) + 0.05)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"noise_degradation.{ext}", bbox_inches="tight")
    plt.close()
    print("  → noise_degradation")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_noise_experiment()
