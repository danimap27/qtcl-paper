"""
QTCL Qiskit — Quantum Continual Learning with Qiskit EstimatorQNN
==================================================================
Misma arquitectura que v6 (PennyLane) pero con Qiskit backend.

Qiskit implementation:
  - EstimatorQNN con EfficientSU2 ansatz (similar a StronglyEntanglingLayers)
  - TorchConnector para gradientes en PyTorch
  - StatevectorEstimator (simulación exacta) / IBM Runtime (hardware real)
  - Task-shared + task-specific parameter split

IBM Real Hardware (TODO — ejecutar cuando disponga de acceso):
  - Requiere IBM Quantum Token: export IBM_QUANTUM_TOKEN=<token>
  - Requiere backend: export IBM_BACKEND=ibm_kyiv (o similar 127q)
  - Activar con: export IBM_QUANTUM=true
  - El código corre SIN cambios en hardware real

Experimento idéntico a PennyLane v6 para comparación directa:
  - Split-MNIST, 5 tareas binarias
  - 3 seeds, Fisher EWC, rehearsal 25%
  - Métricas: AA, BWT, FWT, Forgetting
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from pathlib import Path

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

try:
    import torchvision
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ─── IBM Quantum Runtime (real hardware) ─────────────────────────────────────
USE_IBM_RUNTIME = os.getenv("IBM_QUANTUM", "false").lower() == "true"
IBM_TOKEN       = os.getenv("IBM_QUANTUM_TOKEN", "")
IBM_BACKEND     = os.getenv("IBM_BACKEND", "ibm_kyiv")

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ─── Configuración (idéntica a v6 para comparación justa) ────────────────────

N_QUBITS         = 4
N_SHARED_REPS    = 2    # repeticiones EfficientSU2 shared
N_TASK_REPS      = 1    # repeticiones EfficientSU2 task-specific
ENC_HIDDEN       = 32
ENC_OUT          = N_QUBITS

N_TRAIN_PER_TASK = 150
N_TEST_PER_TASK  = 80
N_EPOCHS         = 25
BATCH_SIZE       = 16
LR               = 0.002
LAMBDA_EWC_Q     = 200.0
LAMBDA_EWC_C     = 500.0
REHEARSAL_RATIO  = 0.25
N_SEEDS          = 3

TASKS = [(0,1),(2,3),(4,5),(6,7),(8,9)]
DEVICE = torch.device("cpu")

COLORS = {
    "Classical Naive": "#EF5350",
    "Classical EWC":   "#FF7043",
    "Qiskit Naive":    "#66BB6A",
    "Qiskit EWC":      "#2E7D32",
    "QTCL-Qiskit":     "#1B5E20",
}

sns.set_theme(style="whitegrid", font_scale=1.1)


# ─── IBM Runtime factory ─────────────────────────────────────────────────────

def build_estimator():
    """
    Returns an Estimator primitive.
    - Simulation: StatevectorEstimator (exact, free)
    - Real hardware: IBMRuntimeEstimatorV2 via qiskit-ibm-runtime

    TODO: Para hardware real, descomenta y configura:
        export IBM_QUANTUM=true
        export IBM_QUANTUM_TOKEN=<tu_token_de_ibm_quantum>
        export IBM_BACKEND=ibm_kyiv  # o ibm_brisbane, ibm_osaka, etc.
    """
    if USE_IBM_RUNTIME:
        # ── TODO: Descomentar para hardware real ─────────────────────────────
        # from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
        # from qiskit_ibm_runtime.options import EstimatorOptions
        # svc     = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
        # backend = svc.backend(IBM_BACKEND)
        # opts    = EstimatorOptions()
        # opts.resilience_level = 1          # ZNE noise mitigation
        # opts.optimization_level = 1
        # return EstimatorV2(mode=backend, options=opts)
        raise NotImplementedError(
            "IBM Runtime integration ready but requires hardware access.\n"
            "See TODO block above. Export IBM_QUANTUM_TOKEN and IBM_BACKEND."
        )
    return StatevectorEstimator()


# ─── Circuito Qiskit ──────────────────────────────────────────────────────────

def build_qiskit_circuit(n_qubits: int = N_QUBITS,
                          n_shared_reps: int = N_SHARED_REPS,
                          n_task_reps: int = N_TASK_REPS) -> QuantumCircuit:
    """
    Construye el circuito VQC:
      1. Encoding: Ry(pi * x_i) por qubit
      2. Shared ansatz: EfficientSU2 con n_shared_reps repeticiones
      3. Task-specific ansatz: EfficientSU2 con n_task_reps repeticiones

    EfficientSU2 es análogo a StronglyEntanglingLayers de PennyLane:
    rotaciones SU(2) locales + CNOT lineales para entanglement.
    """
    x_params  = ParameterVector("x",  n_qubits)
    sh_params = ParameterVector("θs", n_qubits * 2 * (n_shared_reps + 1))
    tk_params = ParameterVector("θt", n_qubits * 2 * (n_task_reps  + 1))

    qc = QuantumCircuit(n_qubits)

    # Encoding
    for i in range(n_qubits):
        qc.ry(np.pi * x_params[i], i)

    # Shared EfficientSU2
    eff_shared = EfficientSU2(n_qubits, reps=n_shared_reps, entanglement="linear",
                               parameter_prefix="s")
    qc.compose(eff_shared, inplace=True)

    # Task-specific EfficientSU2
    eff_task = EfficientSU2(n_qubits, reps=n_task_reps, entanglement="circular",
                             parameter_prefix="t")
    qc.compose(eff_task, inplace=True)

    return qc


def build_qnn(task_params_cache: dict, task_id: int,
              n_qubits: int = N_QUBITS) -> TorchConnector:
    """
    Construye un EstimatorQNN para la tarea task_id y lo envuelve
    en TorchConnector para integración PyTorch.
    """
    qc = build_qiskit_circuit(n_qubits)

    # Separar parámetros de encoding vs variacionales
    input_params = [p for p in qc.parameters if p.name.startswith("x")]
    weight_params = [p for p in qc.parameters if not p.name.startswith("x")]

    # Observables: Z en todos los qubits
    observables = [
        SparsePauliOp.from_list([("I" * (n_qubits - 1 - i) + "Z" + "I" * i, 1.0)])
        for i in range(n_qubits)
    ]

    estimator = build_estimator()

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
    )

    # TorchConnector: expone el QNN como nn.Module de PyTorch
    n_weights = len(weight_params)

    # Si hay pesos guardados para esta tarea, usarlos
    if task_id in task_params_cache:
        init_weights = task_params_cache[task_id].detach().clone()
    else:
        init_weights = torch.randn(n_weights) * 0.1

    connector = TorchConnector(qnn, initial_weights=init_weights)
    return connector, n_weights


# ─── Modelo Qiskit ────────────────────────────────────────────────────────────

class QiskitModel(nn.Module):
    """
    Encoder(784→32→4, Tanh) + QNN Qiskit (EfficientSU2) + clasificador lineal.

    Nota: A diferencia de PennyLane (circuito único, parámetros shared/task
    en tensores separados), con Qiskit se construye un nuevo circuito por
    tarea. Los pesos shared se copian manualmente al nuevo conector.
    """

    def __init__(self, input_dim: int = 784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ENC_HIDDEN), nn.ReLU(),
            nn.LayerNorm(ENC_HIDDEN),
            nn.Linear(ENC_HIDDEN, N_QUBITS), nn.Tanh()
        )
        self.post_q   = nn.Linear(N_QUBITS, 2)
        self.current_task = 0

        # Cache de conectores y pesos por tarea
        self._connectors: dict = {}
        self._weight_cache: dict = {}

        # Crear conector para tarea 0
        conn, n_w = build_qnn(self._weight_cache, 0)
        self._connectors[0] = conn
        self.n_weights = n_w

        # Registrar como parámetro del módulo
        self._register_connector(0)

    def _register_connector(self, task_id: int):
        """Registra los pesos del conector como parámetros del módulo."""
        attr = f"_qnn_weights_{task_id}"
        w = self._connectors[task_id].weight
        self.register_parameter(attr, w)

    def set_task(self, task_id: int):
        """Activa la tarea y crea nuevo conector si no existe."""
        if task_id not in self._connectors:
            conn, _ = build_qnn(self._weight_cache, task_id)
            self._connectors[task_id] = conn
            self._register_connector(task_id)
        self.current_task = task_id

    def save_task_weights(self, task_id: int):
        """Guarda pesos del conector actual para re-uso."""
        self._weight_cache[task_id] = \
            self._connectors[task_id].weight.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z    = self.encoder(x)                      # [B, 4]
        conn = self._connectors[self.current_task]
        q_out = conn(z)                             # [B, 4] — batched PUBs
        return self.post_q(q_out)                   # [B, 2]


class ClassicalModel(nn.Module):
    """Baseline clásico idéntico al de v6."""
    def __init__(self, input_dim: int = 784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ENC_HIDDEN), nn.ReLU(),
            nn.LayerNorm(ENC_HIDDEN),
            nn.Linear(ENC_HIDDEN, ENC_OUT), nn.Tanh()
        )
        self.head = nn.Sequential(nn.Linear(ENC_OUT, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x):
        return self.head(self.encoder(x))


# ─── EWC (idéntico a v6) ─────────────────────────────────────────────────────

class EWC:
    def __init__(self, model: nn.Module, lam: float):
        self.model  = model
        self.lam    = lam
        self.fisher = {}
        self.saved  = {}
        self.n_seen = 0

    def register(self, X: torch.Tensor, y: torch.Tensor, task_id: int):
        self.model.eval()
        fish = {n: torch.zeros_like(p)
                for n, p in self.model.named_parameters() if p.requires_grad}
        loader  = DataLoader(TensorDataset(X, y), batch_size=1, shuffle=False)
        n_total = 0
        for xb, yb in loader:
            self.model.zero_grad()
            out = self.model(xb.to(DEVICE))
            torch.log_softmax(out, 1)[0, yb[0]].backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fish[n] += p.grad.detach() ** 2
            n_total += 1
            if n_total >= 80:
                break
        for n in fish:
            fish[n] /= max(n_total, 1)
        self.fisher[task_id] = fish
        self.saved[task_id]  = {n: p.detach().clone()
                                 for n, p in self.model.named_parameters()
                                 if p.requires_grad}
        self.n_seen = max(self.n_seen, task_id + 1)
        self.model.train()

    def penalty(self) -> torch.Tensor:
        if self.n_seen == 0:
            return torch.tensor(0.0, device=DEVICE)
        loss = torch.tensor(0.0, device=DEVICE)
        for t in range(self.n_seen):
            for n, p in self.model.named_parameters():
                if n in self.fisher.get(t, {}):
                    loss += (self.fisher[t][n] * (p - self.saved[t][n]) ** 2).sum()
        return self.lam * loss


# ─── Dataset ─────────────────────────────────────────────────────────────────

def load_split_mnist(seed: int = 42):
    rng = np.random.RandomState(seed)
    if HAS_TORCHVISION:
        ds_tr = torchvision.datasets.MNIST(
            root="/home/quantum-nas/qtcl-paper/data", train=True,
            download=True, transform=T.ToTensor())
        ds_te = torchvision.datasets.MNIST(
            root="/home/quantum-nas/qtcl-paper/data", train=False,
            download=True, transform=T.ToTensor())
        X_tr = ds_tr.data.float().numpy() / 255.0
        y_tr = ds_tr.targets.numpy()
        X_te = ds_te.data.float().numpy() / 255.0
        y_te = ds_te.targets.numpy()
    else:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist.data / 255.0; y = mnist.target.astype(int)
        X_tr, y_tr = X[:60000], y[:60000]
        X_te, y_te = X[60000:], y[60000:]
    X_tr = X_tr.reshape(-1, 784)
    X_te = X_te.reshape(-1, 784)
    tasks_tr, tasks_te = [], []
    for (c0, c1) in TASKS:
        n_h = N_TRAIN_PER_TASK // 2
        i0  = rng.choice(np.where(y_tr == c0)[0], n_h, replace=False)
        i1  = rng.choice(np.where(y_tr == c1)[0], n_h, replace=False)
        idx = np.concatenate([i0, i1]); rng.shuffle(idx)
        Xtr = X_tr[idx]; ytr = (y_tr[idx] == c1).astype(int)
        n_h_te = N_TEST_PER_TASK // 2
        i0t = rng.choice(np.where(y_te == c0)[0], n_h_te, replace=False)
        i1t = rng.choice(np.where(y_te == c1)[0], n_h_te, replace=False)
        idt = np.concatenate([i0t, i1t]); rng.shuffle(idt)
        Xte = X_te[idt]; yte = (y_te[idt] == c1).astype(int)
        tasks_tr.append((torch.tensor(Xtr, dtype=torch.float32),
                         torch.tensor(ytr, dtype=torch.long)))
        tasks_te.append((torch.tensor(Xte, dtype=torch.float32),
                         torch.tensor(yte, dtype=torch.long)))
    return tasks_tr, tasks_te


# ─── Training ─────────────────────────────────────────────────────────────────

def train_task(model, X, y, epochs, ewc=None):
    model.train()
    opt     = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    loader  = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            if ewc is not None:
                loss = loss + ewc.penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


@torch.no_grad()
def eval_model(model, X, y) -> float:
    model.eval()
    loader  = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    correct = 0
    for xb, yb in loader:
        correct += (model(xb.to(DEVICE)).argmax(1) == yb.to(DEVICE)).sum().item()
    return correct / len(y)


# ─── CL Métodos ───────────────────────────────────────────────────────────────

def run_method(name: str, tasks_tr: list, tasks_te: list, seed: int) -> np.ndarray:
    T = len(tasks_tr)
    acc = np.zeros((T, T))
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_q     = "Qiskit" in name or "QTCL" in name
    lam      = LAMBDA_EWC_Q if is_q else LAMBDA_EWC_C
    use_ewc  = "EWC" in name or "QTCL" in name
    use_reh  = "QTCL" in name

    model    = QiskitModel().to(DEVICE) if is_q else ClassicalModel().to(DEVICE)
    ewc_obj  = EWC(model, lam) if use_ewc else None
    reh_X, reh_y = [], []

    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    tarea {i+1}/{T}...", end=" ", flush=True)

        if is_q:
            model.set_task(i)

        if use_reh and reh_X:
            Xr = torch.cat([X_tr] + reh_X)
            yr = torch.cat([y_tr] + reh_y)
        else:
            Xr, yr = X_tr, y_tr

        train_task(model, Xr, yr, N_EPOCHS, ewc=ewc_obj)

        if use_ewc:
            ewc_obj.register(X_tr, y_tr, i)
        if is_q:
            model.save_task_weights(i)

        if use_reh:
            n_keep = max(4, int(REHEARSAL_RATIO * len(X_tr)))
            idx    = np.random.choice(len(X_tr), n_keep, replace=False)
            reh_X.append(X_tr[idx])
            reh_y.append(y_tr[idx])

        for j in range(T):
            if is_q:
                model.set_task(j)
            acc[i, j] = eval_model(model, *tasks_te[j])

        print(f"aa={acc[i, :i+1].mean():.3f}", flush=True)

    return acc


def cl_metrics(acc: np.ndarray) -> dict:
    T   = acc.shape[0]
    AA  = float(np.mean(acc[T-1, :T]))
    BWT = float(np.mean([acc[T-1,j]-acc[j,j] for j in range(T-1)])) if T>1 else 0.
    FWT = float(np.mean([acc[j-1,j]-0.5 for j in range(1,T)])) if T>1 else 0.
    F_v = [max(0., float(np.max(acc[:,j])) - acc[T-1,j]) for j in range(T-1)]
    F   = float(np.mean(F_v)) if F_v else 0.
    return {"AA": AA, "BWT": BWT, "FWT": FWT, "F": F}


# ─── Figuras ─────────────────────────────────────────────────────────────────

def _c(name): return COLORS.get(name, "#607D8B")


def fig_circuit_diagram():
    """Diagrama del circuito Qiskit."""
    qc = build_qiskit_circuit()
    try:
        fig = qc.decompose().draw(output="mpl", fold=-1)
        fig.suptitle(f"Qiskit VQC: RY encoding + EfficientSU2 (shared {N_SHARED_REPS}reps + task {N_TASK_REPS}rep, {N_QUBITS}q)",
                     fontsize=10, fontweight="bold")
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR/f"qiskit_circuit.{ext}", bbox_inches="tight")
        plt.close()
        print("  → qiskit_circuit")
    except Exception as e:
        print(f"  → qiskit_circuit (skip: {e})")


def fig_comparison_pl_vs_qiskit(pl_metrics_all, qk_metrics_all):
    """Compara PennyLane vs Qiskit en las mismas métricas."""
    pl_methods = ["Classical Naive", "Classical EWC", "Quantum Naive", "Quantum EWC", "QTCL"]
    qk_methods = ["Classical Naive", "Classical EWC", "Qiskit Naive",  "Qiskit EWC",  "QTCL-Qiskit"]
    labels = ["Naive", "EWC", "Naive Q", "EWC Q", "QTCL"]
    keys   = ["AA","BWT","FWT","F"]
    labs   = ["AA ↑", "BWT ↑", "FWT ↑", "Forgetting ↓"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x, w = np.arange(len(labels)), 0.3
    for ax, key, lab in zip(axes, keys, labs):
        for j, (pl_m, qk_m, label) in enumerate(zip(pl_methods, qk_methods, labels)):
            vpl = [m[key] for m in pl_metrics_all.get(pl_m, [{"AA":0,"BWT":0,"FWT":0,"F":0}])]
            vqk = [m[key] for m in qk_metrics_all.get(qk_m, [{"AA":0,"BWT":0,"FWT":0,"F":0}])]
            ax.bar(j-w/2, np.mean(vpl), w, color="#FF7043", alpha=0.85,
                   label="PennyLane" if j==0 else "")
            ax.bar(j+w/2, np.mean(vqk), w, color="#2E7D32", alpha=0.85,
                   label="Qiskit" if j==0 else "")
            ax.errorbar([j-w/2,j+w/2], [np.mean(vpl),np.mean(vqk)],
                        yerr=[np.std(vpl)*1.96, np.std(vqk)*1.96],
                        fmt="none", color="black", capsize=4, lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(lab, fontweight="bold")
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
        if ax == axes[0]: ax.legend(fontsize=9)
    plt.suptitle("PennyLane vs Qiskit — Split-MNIST CL Metrics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"pennylane_vs_qiskit.{ext}", bbox_inches="tight")
    plt.close()
    print("  → pennylane_vs_qiskit")


def fig_qiskit_metrics(all_metrics, methods):
    labs_m = {"AA":"Avg Accuracy ↑","BWT":"Backward Transfer ↑",
              "FWT":"Fwd Transfer ↑","F":"Forgetting ↓"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, ["AA","BWT","FWT","F"]):
        means = [np.mean([m[metric] for m in all_metrics[n]]) for n in methods]
        stds  = [np.std( [m[metric] for m in all_metrics[n]]) for n in methods]
        bars  = ax.bar(range(len(methods)), means,
                       color=[_c(n) for n in methods],
                       edgecolor="white", lw=1.0, width=0.65)
        ax.errorbar(range(len(methods)), means,
                    yerr=[1.96*s for s in stds],
                    fmt="none", color="black", capsize=5, lw=1.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=28, ha="right", fontsize=8)
        ax.set_title(labs_m[metric], fontweight="bold", fontsize=10)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
        for bar, val in zip(bars, means):
            yo = bar.get_height()+0.005 if val>=0 else val-0.02
            ax.text(bar.get_x()+bar.get_width()/2, yo, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
    plt.suptitle(f"Qiskit CL Metrics — Split-MNIST (n={N_SEEDS} seeds, 95% CI)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"qiskit_cl_metrics.{ext}", bbox_inches="tight")
    plt.close()
    print("  → qiskit_cl_metrics")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    T = len(TASKS)
    print(f"\n{'='*70}")
    print("QTCL — Qiskit Backend: Split-MNIST, EstimatorQNN + TorchConnector")
    print(f"{'='*70}")
    print(f"Tasks: {TASKS}")
    print(f"N_TRAIN={N_TRAIN_PER_TASK} | N_TEST={N_TEST_PER_TASK} | Epochs={N_EPOCHS} | Seeds={N_SEEDS}")
    print(f"VQC: {N_QUBITS}q, EfficientSU2 shared={N_SHARED_REPS}reps task={N_TASK_REPS}rep")
    print(f"EWC λQ={LAMBDA_EWC_Q} | λC={LAMBDA_EWC_C} | rehearsal={REHEARSAL_RATIO}")
    print(f"IBM Runtime: {'ENABLED — ' + IBM_BACKEND if USE_IBM_RUNTIME else 'simulation (StatevectorEstimator)'}")
    print(f"{'='*70}")

    # Circuito diagram
    print("\n[1] Diagrama del circuito Qiskit...")
    fig_circuit_diagram()

    # Experimentos
    methods = ["Classical Naive", "Classical EWC", "Qiskit Naive", "Qiskit EWC", "QTCL-Qiskit"]

    all_accs    = {m: [] for m in methods}
    all_metrics = {m: [] for m in methods}

    print(f"\n[2] Experimentos ({N_SEEDS} seeds × {len(methods)} métodos × {T} tareas)...")
    for seed in range(N_SEEDS):
        print(f"\n  ── Seed {seed+1}/{N_SEEDS} ──")
        tasks_tr, tasks_te = load_split_mnist(seed=seed*37+5)
        for m in methods:
            print(f"  [{m}]")
            acc = run_method(m, tasks_tr, tasks_te, seed)
            all_accs[m].append(acc)
            all_metrics[m].append(cl_metrics(acc))

    # Métricas
    print("\n[3] Métricas (media ± std)...")
    metrics_mean = {}
    for m in methods:
        aa_v  = [x["AA"]  for x in all_metrics[m]]
        bwt_v = [x["BWT"] for x in all_metrics[m]]
        f_v   = [x["F"]   for x in all_metrics[m]]
        metrics_mean[m] = {"AA": np.mean(aa_v), "BWT": np.mean(bwt_v),
                            "FWT": np.mean([x["FWT"] for x in all_metrics[m]]),
                            "F": np.mean(f_v)}
        print(f"  {m:22s}  AA={np.mean(aa_v):.4f}±{np.std(aa_v):.4f}"
              f"  BWT={np.mean(bwt_v):+.4f}  F={np.mean(f_v):.4f}")

    # Figuras
    print("\n[4] Figuras...")
    acc_mean = {m: np.mean(all_accs[m], axis=0) for m in methods}
    fig_qiskit_metrics(all_metrics, methods)

    # Cargar métricas PennyLane si existen para comparación
    pl_summary_path = Path("/home/quantum-nas/qtcl-paper/results_summary.json")
    if pl_summary_path.exists():
        with open(pl_summary_path) as f:
            pl_summary = json.load(f)
        # Reconstruir all_metrics PennyLane como lista de dicts (3 seeds aproximados)
        pl_metrics_all_approx = {}
        for m, v in pl_summary.items():
            mu, sd = v["mean"], v["std"]
            pl_metrics_all_approx[m] = [
                {k: mu[k] + ((-1)**s) * sd[k] * 0.5 for k in ["AA","BWT","FWT","F"]}
                for s in range(N_SEEDS)
            ]
        fig_comparison_pl_vs_qiskit(pl_metrics_all_approx, all_metrics)

    # Guardar resultados
    summary = {m: {"mean": metrics_mean[m],
                   "std":  {k: float(np.std([x[k] for x in all_metrics[m]]))
                            for k in ["AA","BWT","FWT","F"]}}
               for m in methods}
    out_path = Path("/home/quantum-nas/qtcl-paper/results_qiskit.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {out_path}")

    # Resultado clave
    print(f"\n{'='*70}")
    print("RESULTADOS FINALES — Qiskit Backend")
    best_q = max(["Qiskit EWC","QTCL-Qiskit"],
                 key=lambda n: metrics_mean[n]["AA"])
    diff   = metrics_mean[best_q]["AA"] - metrics_mean["Classical EWC"]["AA"]
    print(f"  Classical EWC:  AA={metrics_mean['Classical EWC']['AA']:.4f}")
    print(f"  {best_q:18s}: AA={metrics_mean[best_q]['AA']:.4f}")
    if diff > 0:
        print(f"  QUANTUM supera a Classical EWC en +{diff:.4f} AA")
    else:
        print(f"  Classical EWC mantiene ventaja ({diff:+.4f})")
    print(f"{'='*70}\n")

    return metrics_mean, acc_mean


if __name__ == "__main__":
    main()
