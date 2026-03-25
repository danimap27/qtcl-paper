"""
QTCL Noise Experiment — Real IBM QPU noise (FakeBrisbane)
=============================================================
Runs clean QTCL vs QTCL with real IBM hardware noise and compares metrics.

Noise source: FakeBrisbane (qiskit_ibm_runtime.fake_provider)
  - Real gate error rates (sx, cx) extracted from ibm_brisbane calibration data
  - Applied as depolarizing channels in PennyLane default.mixed device
  - Does not require IBM token or internet connection

Why default.mixed instead of qiskit.aer + parameter-shift:
  - qiskit.aer + parameter-shift + shots is O(params × shots) per batch → very slow
  - default.mixed uses exact density matrix simulation → backprop compatible
  - Same real IBM error rates, ~10x faster

Usage on Hercules:
  sbatch hercules/submit_noise.slurm

Outputs:
  results/noise_results.json
  results/noise_model_info.json   — IBM error rates used
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from qtcl_v6_experiment import (
    N_QUBITS, N_SHARED_LAYERS, N_TASK_LAYERS, ENC_HIDDEN,
    N_TRAIN_PER_TASK, N_TEST_PER_TASK, N_EPOCHS, BATCH_SIZE, LR,
    LAMBDA_EWC_Q, REHEARSAL_RATIO, TASKS, DEVICE,
    load_split_mnist, cl_metrics, EWC, QuantumModel,
    train_task, eval_model, FIGURES_DIR,
)

import pennylane as qml

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IBM_BACKEND   = "FakeBrisbane"
N_SEEDS       = 2
# Reduced for noise experiment (robustness analysis, not full benchmark)
NOISE_EPOCHS  = 15   # vs 25 in main experiment
NOISE_SAMPLES = 100  # train samples per task (vs 150)


# ─── Extract real IBM error rates ────────────────────────────────────────────

def extract_ibm_error_rates():
    """
    Extracts average 1Q (sx) and 2Q (cx) gate error rates from FakeBrisbane.
    Falls back to typical IBM values if extraction fails.
    Returns (p1q, p2q, info_dict).
    """
    try:
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        backend = FakeBrisbane()
        props   = backend.properties()
        n_q     = min(N_QUBITS, backend.num_qubits)

        # 1Q: sx gate error per qubit
        sx_errors = []
        for q in range(n_q):
            try:
                sx_errors.append(props.gate_error("sx", q))
            except Exception:
                pass

        # 2Q: cx gate error for adjacent pairs
        cx_errors = []
        for q in range(n_q - 1):
            try:
                cx_errors.append(props.gate_error("cx", [q, q + 1]))
            except Exception:
                pass

        p1q = float(np.mean(sx_errors)) if sx_errors else 0.001
        p2q = float(np.mean(cx_errors)) if cx_errors else 0.010

        info = {
            "backend":       IBM_BACKEND,
            "source":        "FakeBrisbane calibration data (real IBM hardware)",
            "n_qubits_used": N_QUBITS,
            "p1q_sx_mean":   round(p1q, 6),
            "p2q_cx_mean":   round(p2q, 6),
            "sx_per_qubit":  [round(e, 6) for e in sx_errors],
            "cx_per_pair":   [round(e, 6) for e in cx_errors],
            "noise_model":   "DepolarizingChannel (default.mixed)",
        }
        print(f"  IBM error rates extracted from {IBM_BACKEND}:")
        print(f"    p1q (sx avg): {p1q:.4e}")
        print(f"    p2q (cx avg): {p2q:.4e}")

    except Exception as e:
        print(f"  FakeBrisbane extraction failed ({e}), using typical IBM values")
        p1q = 0.001   # typical IBM 1Q error ~0.1%
        p2q = 0.010   # typical IBM 2Q error ~1%
        info = {
            "backend":     IBM_BACKEND,
            "source":      "fallback — typical IBM values",
            "p1q_sx_mean": p1q,
            "p2q_cx_mean": p2q,
            "noise_model": "DepolarizingChannel (default.mixed)",
        }

    with open(RESULTS_DIR / "noise_model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: results/noise_model_info.json")

    return p1q, p2q, info


# ─── Noisy QNode (default.mixed + depolarizing channels) ─────────────────────

def _make_noisy_qnode(p1q: float, p2q: float,
                      n_qubits=N_QUBITS,
                      n_shared=N_SHARED_LAYERS,
                      n_task=N_TASK_LAYERS):
    """
    QNode using PennyLane default.mixed (density matrix simulator).
    Applies DepolarizingChannel with real IBM error rates after each layer.
    Uses backprop → same speed as clean simulation.

    Noise model per layer:
      - After encoding RY gates: DepolarizingChannel(p1q) on each qubit
      - After StronglyEntanglingLayers: DepolarizingChannel(p1q) on each qubit
        + DepolarizingChannel(p2q) on adjacent pairs (2Q gate approximation)
    """
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, shared_w, task_w):
        # Encoding
        for i in range(n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)
            qml.DepolarizingChannel(p1q, wires=i)

        # Shared entangling layers
        qml.StronglyEntanglingLayers(shared_w, wires=range(n_qubits))
        for i in range(n_qubits):
            qml.DepolarizingChannel(p1q, wires=i)
        for i in range(n_qubits - 1):
            qml.DepolarizingChannel(p2q, wires=i)

        # Task-specific layers
        qml.StronglyEntanglingLayers(task_w, wires=range(n_qubits))
        for i in range(n_qubits):
            qml.DepolarizingChannel(p1q, wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class NoisyQuantumModel(nn.Module):
    """
    QuantumModel with depolarizing noise (default.mixed + IBM error rates).
    Identical architecture to QuantumModel; only the circuit device changes.
    """

    def __init__(self, p1q: float, p2q: float, input_dim: int = 784):
        super().__init__()
        self.n_qubits = N_QUBITS
        self.encoder  = nn.Sequential(
            nn.Linear(input_dim, ENC_HIDDEN), nn.ReLU(),
            nn.LayerNorm(ENC_HIDDEN),
            nn.Linear(ENC_HIDDEN, N_QUBITS), nn.Tanh()
        )
        self.shared_w = nn.Parameter(
            torch.randn(N_SHARED_LAYERS, N_QUBITS, 3) * 0.1
        )
        self.task_w   = nn.ParameterDict()
        self._add_task(0)
        self.current_task = 0
        self.post_q   = nn.Linear(N_QUBITS, 2)
        self.circuit  = _make_noisy_qnode(p1q, p2q)

    def _add_task(self, task_id: int):
        key = f"t{task_id}"
        if key not in self.task_w:
            self.task_w[key] = nn.Parameter(
                torch.randn(N_TASK_LAYERS, N_QUBITS, 3) * 0.1)

    def set_task(self, task_id: int):
        self._add_task(task_id)
        self.current_task = task_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    T       = len(tasks_tr)
    acc     = np.zeros((T, T))
    ewc_obj = EWC(model, LAMBDA_EWC_Q)
    reh_X, reh_y = [], []

    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    task {i+1}/{T}...", end=" ", flush=True)
        model.set_task(i)

        Xr = torch.cat([X_tr] + reh_X) if reh_X else X_tr
        yr = torch.cat([y_tr] + reh_y) if reh_y else y_tr

        train_task(model, Xr, yr, NOISE_EPOCHS, ewc=ewc_obj)
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
    print(f"QTCL Noise Experiment — {IBM_BACKEND} (default.mixed)")
    print(f"Seeds: {N_SEEDS} | Tasks: {len(TASKS)} | Epochs: {NOISE_EPOCHS} | Samples/task: {NOISE_SAMPLES}")
    print(sep)

    p1q, p2q, noise_info = extract_ibm_error_rates()

    clean_metrics_list = []
    noisy_metrics_list = []

    for s in range(N_SEEDS):
        print(f"\n── Seed {s} ──────────────────────────────")
        tasks_tr, tasks_te = load_split_mnist(seed=s * 37 + 5)
        # Trim to NOISE_SAMPLES per task to speed up noise experiment
        tasks_tr = [(X[:NOISE_SAMPLES], y[:NOISE_SAMPLES]) for X, y in tasks_tr]

        # Clean (lightning.qubit, adjoint)
        print("  [Clean — lightning.qubit]")
        clean_model = QuantumModel().to(DEVICE)
        clean_acc   = run_qtcl(clean_model, tasks_tr, tasks_te, seed=s)
        cm          = cl_metrics(clean_acc)
        clean_metrics_list.append(cm)
        print(f"  Clean → AA={cm['AA']:.4f}  F={cm['F']:.4f}")

        # Noisy (default.mixed + IBM depolarizing)
        print(f"  [Noisy — default.mixed + {IBM_BACKEND} error rates]")
        noisy_model = NoisyQuantumModel(p1q, p2q).to(DEVICE)
        noisy_acc   = run_qtcl(noisy_model, tasks_tr, tasks_te, seed=s)
        nm          = cl_metrics(noisy_acc)
        noisy_metrics_list.append(nm)
        print(f"  Noisy → AA={nm['AA']:.4f}  F={nm['F']:.4f}")

    # Aggregated results
    def _agg(lst, key):
        vals = [x[key] for x in lst]
        return float(np.mean(vals)), float(np.std(vals))

    results = {
        "backend":      IBM_BACKEND,
        "noise_model":  "DepolarizingChannel (default.mixed)",
        "p1q":          p1q,
        "p2q":          p2q,
        "n_seeds":      N_SEEDS,
        "clean": {k: {"mean": _agg(clean_metrics_list, k)[0],
                      "std":  _agg(clean_metrics_list, k)[1]}
                  for k in ["AA", "BWT", "FWT", "F"]},
        "noisy": {k: {"mean": _agg(noisy_metrics_list, k)[0],
                      "std":  _agg(noisy_metrics_list, k)[1]}
                  for k in ["AA", "BWT", "FWT", "F"]},
    }
    for k in ["AA", "F"]:
        results[f"delta_{k}"] = round(
            results["noisy"][k]["mean"] - results["clean"][k]["mean"], 4)

    with open(RESULTS_DIR / "noise_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Console table
    print(f"\n{sep}")
    print(f"QTCL Noise Results — {IBM_BACKEND} (p1q={p1q:.1e}, p2q={p2q:.1e})")
    print(f"{'Condition':28s}  {'AA':>12s}  {'BWT':>12s}  {'FWT':>12s}  {'F':>12s}")
    print("-" * 80)
    for cond, lst in [("Clean (ideal)", clean_metrics_list),
                      (f"Noisy ({IBM_BACKEND})", noisy_metrics_list)]:
        row = {k: f"{_agg(lst, k)[0]:.3f}±{_agg(lst, k)[1]:.3f}"
               for k in ["AA", "BWT", "FWT", "F"]}
        print(f"{cond:28s}  {row['AA']:>12s}  {row['BWT']:>12s}  "
              f"{row['FWT']:>12s}  {row['F']:>12s}")
    for k in ["AA", "F"]:
        print(f"  Δ{k} (noisy−clean): {results[f'delta_{k}']:+.4f}")
    print(sep)

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
    x, w    = np.arange(len(metrics)), 0.35

    clean_m = [results["clean"][k]["mean"] for k in metrics]
    clean_s = [results["clean"][k]["std"]  for k in metrics]
    noisy_m = [results["noisy"][k]["mean"] for k in metrics]
    noisy_s = [results["noisy"][k]["std"]  for k in metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, clean_m, w, yerr=clean_s, capsize=5,
           color="#0D47A1", alpha=0.85, label="QTCL clean (lightning.qubit)")
    ax.bar(x + w/2, noisy_m, w, yerr=noisy_s, capsize=5,
           color="#FF6F00", alpha=0.85,
           label=f"QTCL noisy ({IBM_BACKEND}, p1q={results['p1q']:.1e})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"QTCL: ideal vs. hardware noise ({IBM_BACKEND}, depolarizing)",
        fontsize=11)
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
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.005 if val >= 0 else -0.015),
                f"{val:+.3f}", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("Δ (noisy − clean)", fontsize=11)
    ax.set_title(f"Noise degradation: QTCL on {IBM_BACKEND}", fontsize=11)
    ax.set_ylim(min(deltas) - 0.05, max(deltas) + 0.05)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"noise_degradation.{ext}", bbox_inches="tight")
    plt.close()
    print("  → noise_degradation")


if __name__ == "__main__":
    run_noise_experiment()
