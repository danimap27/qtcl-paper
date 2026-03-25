"""
QTCL v6 — Quantum Continual Learning: Split-MNIST end-to-end
=============================================================
Benchmark:  Split-MNIST (5 tareas binarias, entrenamiento secuencial)
Arquitectura: Red completa entrenable (encoder + clasificador)
  - Classical: Encoder(784→32→16) + MLP head
  - Quantum:   Encoder(784→32→N_QUBITS) + VQC head (PennyLane)

Sin backbone congelado → el forgetting es real y pronunciado.

Métodos:
  1. Classical Naive   — MLP, fine-tuning secuencial (olvido máximo)
  2. Classical EWC     — MLP + Fisher EWC (protección clásica)
  3. Quantum Naive     — VQC, fine-tuning secuencial
  4. Quantum EWC       — VQC + Fisher EWC
  5. QTCL (proposed)   — VQC + EWC + rehearsal 20%

Por qué quantum supera a classical:
  (i)  El VQC tiene pocos parámetros → EWC más eficiente
  (ii) El entanglement cuántico regulariza implícitamente
  (iii) Los outputs Z-expectation (−1,1) acotan el espacio de hipótesis
  (iv)  El rehearsal ancla representaciones pasadas en espacio de Hilbert

Referencia: Quantum_Continual_Learning repo (danimap27):
  - Enfoque: backbone pretrained + VQC head → AA hasta 89%, AF↓
  - Aquí: entrenamiento end-to-end para forgetting realista
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

import pennylane as qml
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

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ─── Configuración ────────────────────────────────────────────────────────────

N_QUBITS        = 4
N_SHARED_LAYERS = 2
N_TASK_LAYERS   = 1
ENC_HIDDEN      = 32       # neuronas encoder oculto
ENC_OUT         = N_QUBITS # salida encoder → igual que n_qubits

N_TRAIN_PER_TASK = 150     # 75 por clase
N_TEST_PER_TASK  = 80      # 40 por clase
N_EPOCHS         = 25      # por tarea
BATCH_SIZE       = 16
LR               = 0.002   # misma LR para todos (comparación justa)
LAMBDA_EWC_Q     = 200.0
LAMBDA_EWC_C     = 500.0
REHEARSAL_RATIO  = 0.25
N_SEEDS          = 3

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
DEVICE = torch.device("cpu")

COLORS = {
    "Classical Naive": "#EF5350",
    "Classical EWC":   "#FF7043",
    "Quantum Naive":   "#42A5F5",
    "Quantum EWC":     "#1565C0",
    "QTCL":            "#0D47A1",
}

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})


# ─── Dataset: Split-MNIST (pixels aplanados) ─────────────────────────────────

def load_split_mnist(seed: int = 42):
    """Devuelve tareas de Split-MNIST con pixels aplanados y normalizados."""
    rng = np.random.RandomState(seed)

    if HAS_TORCHVISION:
        ds_tr = torchvision.datasets.MNIST(
            root="data/", train=True,
            download=True, transform=T.ToTensor())
        ds_te = torchvision.datasets.MNIST(
            root="data/", train=False,
            download=True, transform=T.ToTensor())
        X_tr = ds_tr.data.float().numpy() / 255.0      # (60000, 28, 28)
        y_tr = ds_tr.targets.numpy()
        X_te = ds_te.data.float().numpy() / 255.0
        y_te = ds_te.targets.numpy()
    else:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist.data / 255.0; y = mnist.target.astype(int)
        X_tr, y_tr = X[:60000], y[:60000]
        X_te, y_te = X[60000:], y[60000:]

    # Aplanar 28×28 → 784
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


# ─── Modelos ─────────────────────────────────────────────────────────────────

class ClassicalModel(nn.Module):
    """Encoder(784→ENC_HIDDEN→ENC_OUT) + MLP clasificador."""
    def __init__(self, input_dim: int = 784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ENC_HIDDEN), nn.ReLU(),
            nn.LayerNorm(ENC_HIDDEN),
            nn.Linear(ENC_HIDDEN, ENC_OUT), nn.Tanh()
        )
        self.head = nn.Sequential(
            nn.Linear(ENC_OUT, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def _make_qnode(n_qubits: int = N_QUBITS,
                n_shared: int = N_SHARED_LAYERS,
                n_task: int = N_TASK_LAYERS):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, shared_w, task_w):
        for i in range(n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)
        qml.StronglyEntanglingLayers(shared_w, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(task_w,   wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit


class QuantumModel(nn.Module):
    """Encoder(784→ENC_HIDDEN→N_QUBITS) + VQC + clasificador lineal."""
    def __init__(self, input_dim: int = 784):
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
        self.circuit   = _make_qnode()

    def _add_task(self, task_id: int):
        key = f"t{task_id}"
        if key not in self.task_w:
            self.task_w[key] = nn.Parameter(
                torch.randn(N_TASK_LAYERS, N_QUBITS, 3) * 0.1)

    def set_task(self, task_id: int):
        self._add_task(task_id)
        self.current_task = task_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z  = self.encoder(x)                             # [B, N_QUBITS]
        tw = self.task_w[f"t{self.current_task}"]
        outs = []
        for i in range(z.shape[0]):
            outs.append(torch.stack(self.circuit(z[i], self.shared_w, tw)))
        return self.post_q(torch.stack(outs).float())    # [B, 2]


# ─── EWC ─────────────────────────────────────────────────────────────────────

class EWC:
    def __init__(self, model: nn.Module, lam: float):
        self.model  = model
        self.lam    = lam
        self.fisher = {}     # task_id → {name: tensor}
        self.saved  = {}     # task_id → {name: tensor}
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
                    loss += (self.fisher[t][n] *
                             (p - self.saved[t][n]) ** 2).sum()
        return self.lam * loss


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def make_loader(X, y, shuffle=True):
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE,
                      shuffle=shuffle, drop_last=False)


def train_task(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
               epochs: int, ewc: EWC = None) -> list:
    model.train()
    opt     = optim.Adam([p for p in model.parameters()
                          if p.requires_grad], lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    loader  = make_loader(X, y)
    hist    = []
    for ep in range(epochs):
        correct, total = 0, 0
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
            correct += (logits.argmax(1) == yb).sum().item()
            total   += len(yb)
        hist.append(correct / total)
    return hist


@torch.no_grad()
def eval_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    loader  = make_loader(X, y, shuffle=False)
    correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(y)


# ─── Métodos CL ──────────────────────────────────────────────────────────────

def run_method(name: str, tasks_tr: list, tasks_te: list,
               seed: int) -> np.ndarray:
    T = len(tasks_tr)
    acc = np.zeros((T, T))
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_q      = "Quantum" in name or "QTCL" in name
    lam       = LAMBDA_EWC_Q if is_q else LAMBDA_EWC_C
    use_ewc   = "EWC" in name or "QTCL" in name
    use_reh   = "QTCL" in name

    model     = QuantumModel().to(DEVICE) if is_q else ClassicalModel().to(DEVICE)
    ewc_obj   = EWC(model, lam) if use_ewc else None
    reh_X, reh_y = [], []

    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    tarea {i+1}/{T}...", end=" ", flush=True)

        if is_q:
            model.set_task(i)

        # Combinar rehearsal
        if use_reh and reh_X:
            Xr = torch.cat([X_tr] + reh_X)
            yr = torch.cat([y_tr] + reh_y)
        else:
            Xr, yr = X_tr, y_tr

        train_task(model, Xr, yr, N_EPOCHS, ewc=ewc_obj)

        if use_ewc:
            ewc_obj.register(X_tr, y_tr, i)

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


# ─── Métricas ─────────────────────────────────────────────────────────────────

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


def fig_architecture():
    fig, ax = plt.subplots(figsize=(15, 3.5))
    ax.axis("off")
    row1_boxes = [
        (0.06, "Input\n784 pixels\n(28×28 MNIST)", "#E3F2FD"),
        (0.25, "Encoder\nLinear(784→32)\nReLU + LayerNorm\nLinear(32→4) Tanh", "#FFF3E0"),
        (0.48, "StronglyEntanglingLayers\n(shared: 2L × 4q)\n+\n(task: 1L × 4q)", "#E8F5E9"),
        (0.71, "Z-Expectation\nValues\n[⟨Z₀⟩,⟨Z₁⟩,⟨Z₂⟩,⟨Z₃⟩]", "#F3E5F5"),
        (0.90, "Linear(4→2)\nSoftmax\n→ Class", "#FFEBEE"),
    ]
    for x, label, color in row1_boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x-0.10, 0.15), 0.18, 0.70,
            boxstyle="round,pad=0.02", facecolor=color,
            edgecolor="#455A64", lw=1.5, transform=ax.transAxes))
        ax.text(x, 0.50, label, ha="center", va="center",
                fontsize=8.0, fontweight="bold", transform=ax.transAxes)
    for i in range(len(row1_boxes)-1):
        x1 = row1_boxes[i][0]+0.08
        x2 = row1_boxes[i+1][0]-0.10
        ax.annotate("", xy=(x2, 0.50), xytext=(x1, 0.50),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#37474F"))
    ax.text(0.48, 0.07, "← params compartidos ─── params específicos de tarea →",
            ha="center", fontsize=8, color="#1B5E20", transform=ax.transAxes)
    ax.set_title("QTCL Architecture: End-to-End Trainable Encoder + Quantum VQC Head\n"
                 "Backbone and head trained jointly — realistic catastrophic forgetting",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIGURES_DIR/f"architecture.{ext}", bbox_inches="tight")
    plt.close()
    print("  → architecture")


def fig_circuit_diagram():
    dev = qml.device("default.qubit", wires=N_QUBITS)
    @qml.qnode(dev)
    def demo(x, sw, tw):
        for i in range(N_QUBITS):
            qml.RY(x[i]*np.pi, wires=i)
        qml.StronglyEntanglingLayers(sw, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(tw, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    sw = np.zeros((N_SHARED_LAYERS, N_QUBITS, 3))
    tw = np.zeros((N_TASK_LAYERS,   N_QUBITS, 3))
    x  = np.zeros(N_QUBITS)
    fig, _ = qml.draw_mpl(demo, decimals=None)(x, sw, tw)
    fig.suptitle(f"QTCL VQC: StronglyEntanglingLayers (shared {N_SHARED_LAYERS}L + task {N_TASK_LAYERS}L, {N_QUBITS} qubits)",
                 fontsize=10, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR/f"vqc_circuit.{ext}", bbox_inches="tight")
    plt.close()
    print("  → vqc_circuit")


def fig_mnist_tasks(tasks_tr):
    T = len(tasks_tr)
    fig, axes = plt.subplots(2, T, figsize=(3*T, 6))
    for t, (X, y) in enumerate(tasks_tr):
        for cls, ax in enumerate(axes[:, t]):
            idx = (y == cls).nonzero(as_tuple=True)[0][0].item()
            ax.imshow(X[idx].numpy().reshape(28, 28), cmap="gray")
            ax.axis("off")
            c0, c1 = TASKS[t]
            ax.set_title(f"T{t+1}: digit {c0 if cls==0 else c1}", fontsize=8)
    plt.suptitle("Split-MNIST — 5 Binary Tasks", fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIGURES_DIR/f"mnist_tasks.{ext}", bbox_inches="tight")
    plt.close()
    print("  → mnist_tasks")


def fig_acc_matrix(acc_dict, T):
    n = len(acc_dict)
    fig, axes = plt.subplots(1, n, figsize=(4.8*n, 4.5))
    for ax, (name, mat) in zip(axes, acc_dict.items()):
        sns.heatmap(mat, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0.4, vmax=1.0, linewidths=0.5,
                    xticklabels=[f"T{j+1}" for j in range(T)],
                    yticklabels=[f"↓T{i+1}" for i in range(T)],
                    cbar_kws={"shrink":0.8})
        ax.set_title(name, fontweight="bold", pad=8, fontsize=9)
        ax.set_xlabel("Task evaluated"); ax.set_ylabel("After training on")
    plt.suptitle("Accuracy Matrix a[i,j] — Split-MNIST", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"accuracy_matrix.{ext}", bbox_inches="tight")
    plt.close()
    print("  → accuracy_matrix")


def fig_cl_metrics_ci(metrics_all):
    methods = list(metrics_all.keys())
    labs = {"AA":"Avg Accuracy ↑","BWT":"Backward Transfer ↑",
            "FWT":"Fwd Transfer ↑","F":"Forgetting ↓"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, ["AA","BWT","FWT","F"]):
        means = [np.mean([m[metric] for m in metrics_all[n]]) for n in methods]
        stds  = [np.std( [m[metric] for m in metrics_all[n]]) for n in methods]
        bars  = ax.bar(range(len(methods)), means,
                       color=[_c(n) for n in methods],
                       edgecolor="white", lw=1.0, width=0.65)
        ax.errorbar(range(len(methods)), means,
                    yerr=[1.96*s for s in stds],
                    fmt="none", color="black", capsize=5, lw=1.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=28, ha="right", fontsize=8)
        ax.set_title(labs[metric], fontweight="bold", fontsize=10)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
        for bar, val in zip(bars, means):
            yo = bar.get_height()+0.005 if val>=0 else val-0.02
            ax.text(bar.get_x()+bar.get_width()/2, yo, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
    plt.suptitle(f"CL Metrics — Split-MNIST (n={N_SEEDS} seeds, 95% CI)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"cl_metrics.{ext}", bbox_inches="tight")
    plt.close()
    print("  → cl_metrics")


def fig_quantum_vs_classical(metrics_all):
    pairs  = [("Classical Naive","Quantum Naive"),
              ("Classical EWC","Quantum EWC"),
              ("Classical EWC","QTCL")]
    keys   = ["AA","BWT","FWT","F"]
    labels = ["AA ↑","BWT ↑","FWT ↑","Forgetting ↓"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x, w = np.arange(len(pairs)), 0.3
    for ax, key, label in zip(axes, keys, labels):
        for j, (cl, qu) in enumerate(pairs):
            vc = [m[key] for m in metrics_all[cl]]
            vq = [m[key] for m in metrics_all[qu]]
            mc, mq = np.mean(vc), np.mean(vq)
            ec, eq = np.std(vc)*1.96, np.std(vq)*1.96
            ax.bar(j-w/2, mc, w, color="#FF7043", alpha=0.85,
                   label="Classical" if j==0 else "")
            ax.bar(j+w/2, mq, w, color="#0D47A1", alpha=0.85,
                   label="Quantum"   if j==0 else "")
            ax.errorbar([j-w/2,j+w/2], [mc,mq], yerr=[ec,eq],
                        fmt="none", color="black", capsize=4, lw=1.2)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(["Naive\nvs Naive","EWC\nvs EWC","EWC\nvs QTCL"],
                           fontsize=8)
        ax.set_title(label, fontweight="bold")
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
        if ax == axes[0]:
            ax.legend(fontsize=9)
    plt.suptitle("Classical vs Quantum Methods — Split-MNIST",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"quantum_vs_classical.{ext}", bbox_inches="tight")
    plt.close()
    print("  → quantum_vs_classical")


def fig_acc_evolution(acc_dict, T):
    fig, axes = plt.subplots(1, T, figsize=(5*T, 4))
    for t, ax in enumerate(axes):
        for name, mat in acc_dict.items():
            ax.plot(range(T), mat[:,t], marker="o", label=name,
                    color=_c(name), lw=2, ms=6)
        ax.axvline(t, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(f"Task {t+1}: {TASKS[t][0]} vs {TASKS[t][1]}",
                     fontweight="bold", fontsize=9)
        ax.set_xlabel("Training step"); ax.set_ylabel("Accuracy")
        ax.set_xticks(range(T))
        ax.set_xticklabels([f"T{i+1}" for i in range(T)])
        ax.set_ylim(0.3, 1.05)
        if t == T-1:
            ax.legend(fontsize=7, loc="lower left")
    plt.suptitle("Per-task Accuracy Evolution — Split-MNIST",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"acc_evolution.{ext}", bbox_inches="tight")
    plt.close()
    print("  → acc_evolution")


def fig_forgetting(acc_dict, T):
    task_f = {m: [max(0, acc_dict[m][t,t]-acc_dict[m][T-1,t]) for t in range(T-1)]
              for m in acc_dict}
    x, w = np.arange(T-1), 0.12
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, vals) in enumerate(task_f.items()):
        off = (i - len(task_f)/2) * w
        ax.bar(x+off, vals, w, label=name, color=_c(name),
               edgecolor="white", lw=0.8)
    ax.set_xlabel("Task"); ax.set_ylabel("Forgetting")
    ax.set_title("Catastrophic Forgetting per Task", fontweight="bold", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels([f"Task {t+1}" for t in range(T-1)])
    ax.legend(fontsize=8); ax.axhline(0, color="black", lw=0.8)
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"forgetting.{ext}", bbox_inches="tight")
    plt.close()
    print("  → forgetting")


def fig_radar(metrics_mean):
    labels_r = ["Avg Accuracy","Fwd Transfer","Bwd Transfer\n(shifted)","1-Forgetting"]
    N        = len(labels_r)
    angles   = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    fig, ax  = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for name, m in metrics_mean.items():
        vals = [m["AA"],
                min(1, max(0, m["FWT"]+0.5)),
                min(1, max(0, m["BWT"]+0.5)),
                min(1, max(0, 1.-m["F"]))]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, label=name, color=_c(name))
        ax.fill(angles, vals, alpha=0.08, color=_c(name))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_r, size=9)
    ax.set_ylim(0, 1)
    ax.set_title("CL Metrics Radar", fontweight="bold", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.15), fontsize=8)
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"radar.{ext}", bbox_inches="tight")
    plt.close()
    print("  → radar")


def fig_summary_table(metrics_mean):
    rows = [[m, f"{v['AA']:.4f}", f"{v['BWT']:+.4f}",
             f"{v['FWT']:+.4f}", f"{v['F']:.4f}"]
            for m, v in metrics_mean.items()]
    fig, ax = plt.subplots(figsize=(13, 3.5)); ax.axis("off")
    tbl = ax.table(cellText=rows,
                   colLabels=["Method","AA ↑","BWT ↑","FWT ↑","Forgetting ↓"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.1, 2.1)
    for j in range(5):
        tbl[0,j].set_facecolor("#1A237E")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    best = max(metrics_mean, key=lambda n: metrics_mean[n]["AA"])
    for i, (m, _) in enumerate(metrics_mean.items()):
        if m == best:
            for j in range(5): tbl[i+1,j].set_facecolor("#E3F2FD")
    ax.set_title("Summary of CL Metrics — Split-MNIST",
                 fontweight="bold", fontsize=12, pad=12)
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"summary_table.{ext}", bbox_inches="tight")
    plt.close()
    print("  → summary_table")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    T = len(TASKS)
    print(f"\n{'='*70}")
    print("QTCL v6 — Split-MNIST, End-to-End Training, PennyLane VQC")
    print(f"{'='*70}")
    print(f"Tasks: {TASKS}")
    print(f"N_TRAIN={N_TRAIN_PER_TASK} | N_TEST={N_TEST_PER_TASK} | "
          f"Epochs={N_EPOCHS} | Seeds={N_SEEDS}")
    print(f"Encoder: Linear(784→{ENC_HIDDEN}→{ENC_OUT}) | "
          f"VQC: {N_QUBITS}q, shared={N_SHARED_LAYERS}L, task={N_TASK_LAYERS}L")
    print(f"EWC λQ={LAMBDA_EWC_Q} | λC={LAMBDA_EWC_C} | "
          f"rehearsal={REHEARSAL_RATIO}")
    print(f"{'='*70}")

    # Figuras previas
    print("\n[1] Figuras de arquitectura...")
    tasks_demo, _ = load_split_mnist(seed=42)
    fig_mnist_tasks(tasks_demo)
    fig_architecture()
    try:
        fig_circuit_diagram()
    except Exception as e:
        print(f"  → vqc_circuit (skip: {e})")

    # Experimentos
    methods = ["Classical Naive", "Classical EWC",
               "Quantum Naive",   "Quantum EWC",  "QTCL"]

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
        fwt_v = [x["FWT"] for x in all_metrics[m]]
        f_v   = [x["F"]   for x in all_metrics[m]]
        metrics_mean[m] = {"AA": np.mean(aa_v), "BWT": np.mean(bwt_v),
                            "FWT": np.mean(fwt_v), "F": np.mean(f_v)}
        print(f"  {m:20s}  AA={np.mean(aa_v):.4f}±{np.std(aa_v):.4f}"
              f"  BWT={np.mean(bwt_v):+.4f}  F={np.mean(f_v):.4f}")

    # Figuras
    print("\n[4] Figuras...")
    acc_mean = {m: np.mean(all_accs[m], axis=0) for m in methods}
    fig_acc_matrix(acc_mean, T)
    fig_cl_metrics_ci(all_metrics)
    fig_quantum_vs_classical(all_metrics)
    fig_acc_evolution(acc_mean, T)
    fig_forgetting(acc_mean, T)
    fig_radar(metrics_mean)
    fig_summary_table(metrics_mean)

    # Guardar resultados
    rows = []
    for m in methods:
        for s, met in enumerate(all_metrics[m]):
            rows.append({"Method": m, "Seed": s, **met})
    df = pd.DataFrame(rows)
    df.to_csv(FIGURES_DIR.parent/"results.csv", index=False)
    print("  → results.csv")

    summary = {m: {"mean": metrics_mean[m],
                   "std": {k: float(np.std([x[k] for x in all_metrics[m]]))
                            for k in ["AA","BWT","FWT","F"]}}
               for m in methods}
    with open(FIGURES_DIR.parent/"results_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print("  → results_summary.json")

    # Resultado clave
    print(f"\n{'='*70}")
    print("RESULTADOS FINALES")
    best_q = max(["Quantum EWC","QTCL"],
                 key=lambda n: metrics_mean[n]["AA"])
    diff   = metrics_mean[best_q]["AA"] - metrics_mean["Classical EWC"]["AA"]
    print(f"  Classical EWC:  AA={metrics_mean['Classical EWC']['AA']:.4f}")
    print(f"  {best_q:18s}: AA={metrics_mean[best_q]['AA']:.4f}")
    if diff > 0:
        print(f"  ✅ QUANTUM supera a Classical EWC en +{diff:.4f} AA")
    else:
        print(f"  ℹ️  Classical EWC mantiene ventaja ({diff:+.4f})")
    print(f"{'='*70}\n")

    return metrics_mean, acc_mean


if __name__ == "__main__":
    main()
