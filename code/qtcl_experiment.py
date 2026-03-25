"""
Quantum Transfer Continual Learning (QTCL) — v5 (QKE + Backbone)
=================================================================
Compares CL quantum methods with and without a pretrained classical backbone.

Backbone: MLP (32→4) pretrained on task 0, frozen for tasks 1-3.
         Acts as a general-purpose feature extractor (transfer learning).

Methods:
  - Naive FT:              QKE-SVM on current task (forgets)
  - Naive FT + Backbone:   same with backbone features
  - QEWC:                  joint training with weight decay
  - QTCL (proposed):       QKE + rehearsal 30% (no backbone)
  - QTCL + Backbone:       QKE + rehearsal 30% + pretrained backbone [PROPOSED]
  - Classical SVM:         RBF-SVM joint training (upper baseline)

IBM Quantum Runtime:
  Set environment variables to run on real IBM hardware:
    export IBM_QUANTUM=true
    export IBM_QUANTUM_TOKEN=<your_ibm_quantum_token>
    export IBM_BACKEND=ibm_brisbane   # or any available backend

CL Metrics: AA, BWT, FWT, Forgetting.
Compatible with Qiskit 2.3.1 / qiskit-machine-learning 0.9.0
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
np.random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────

N_QUBITS  = 4
FM_REPS   = 2
N_TRAIN   = 40
N_TEST    = 15
CLASS_SEP = 2.0      # Slightly harder than v4 to show backbone advantage

# IBM Quantum Runtime (optional — for real hardware)
USE_IBM_RUNTIME = os.getenv('IBM_QUANTUM', 'false').lower() == 'true'
IBM_TOKEN       = os.getenv('IBM_QUANTUM_TOKEN', '')
IBM_BACKEND     = os.getenv('IBM_BACKEND', 'ibm_brisbane')

COLORS = {
    'naive':      '#F44336',
    'naive_bb':   '#EF9A9A',
    'qewc':       '#FF9800',
    'qtcl':       '#2196F3',
    'qtcl_bb':    '#0D47A1',
    'classical':  '#9C27B0',
}
METHOD_LABELS = {
    'Naive FT':           'naive',
    'Naive FT + Backbone':'naive_bb',
    'QEWC':               'qewc',
    'QTCL (proposed)':    'qtcl',
    'QTCL + Backbone':    'qtcl_bb',
    'Classical SVM':      'classical',
}

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 150})


# ─── IBM Runtime — sampler factory ────────────────────────────────────────────

def build_sampler():
    """Returns local sampler or IBM Runtime based on configuration."""
    if USE_IBM_RUNTIME:
        print("  [IBM Runtime] Connecting to IBM Quantum backend...")
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
            svc = QiskitRuntimeService(channel='ibm_quantum', token=IBM_TOKEN)
            backend = svc.backend(IBM_BACKEND)
            print(f"  [IBM Runtime] Backend: {IBM_BACKEND} ({backend.status().pending_jobs} pending jobs)")
            return IBMSampler(backend)
        except Exception as e:
            print(f"  [IBM Runtime] ERROR: {e} — using local simulator")
    return None  # None → FidelityQuantumKernel uses StatevectorSampler by default


_GLOBAL_SAMPLER = None   # Initialized in main()


# ─── Dataset ──────────────────────────────────────────────────────────────────

def make_task(task_id: int):
    rng = np.random.RandomState(task_id * 17 + 31)
    # Add random rotation per task → makes backbone more useful
    angle = task_id * np.pi / 6
    rot = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                    [np.sin(angle),  np.cos(angle), 0, 0],
                    [0, 0, np.cos(angle), -np.sin(angle)],
                    [0, 0, np.sin(angle),  np.cos(angle)]])

    configs = [
        dict(n_informative=4, n_redundant=0, class_sep=CLASS_SEP),
        dict(n_informative=3, n_redundant=1, class_sep=CLASS_SEP - 0.3, flip_y=0.03),
        dict(n_informative=4, n_redundant=0, class_sep=CLASS_SEP - 0.2),
        dict(n_informative=3, n_redundant=1, class_sep=CLASS_SEP - 0.4, flip_y=0.02),
    ]
    cfg = configs[task_id % len(configs)]
    X, y = make_classification(n_samples=N_TRAIN + N_TEST, n_features=N_QUBITS,
                                random_state=rng.randint(1000), **cfg)
    X = X @ rot.T                         # rotation for inter-task heterogeneity
    scaler = MinMaxScaler((0, np.pi))
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=N_TEST, random_state=42, stratify=y)


# ─── Classical backbone (transfer learning) ───────────────────────────────────

class ClassicalBackbone:
    """
    MLP feature extractor (32 → N_QUBITS) pretrained on task 0.
    Frozen throughout the CL session → transfer learning toward QKE.
    """
    def __init__(self, output_dim: int = N_QUBITS, hidden_dim: int = 32):
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fitted     = False
        self._out_scaler = MinMaxScaler((0, np.pi))

    def pretrain(self, X: np.ndarray, y: np.ndarray) -> 'ClassicalBackbone':
        """Pretrain on task 0. Afterwards, frozen."""
        # Architecture (hidden_dim → output_dim) — output_dim = N_QUBITS = 4
        # With 40 samples use high L2 to avoid overfitting
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim, self.output_dim),
            activation='tanh', max_iter=1000, random_state=42,
            learning_rate_init=0.01, alpha=0.01, solver='adam',
            early_stopping=False,
        )
        self.mlp.fit(X, y)
        feats = self._extract(X)
        self._out_scaler.fit(np.clip(feats, -1, 1))
        self.fitted = True
        train_acc = self.mlp.score(X, y)
        print(f"    [Backbone] pretrained — task 0 acc: {train_acc:.3f}")
        return self

    def _extract(self, X: np.ndarray) -> np.ndarray:
        """Forward pass to the penultimate layer (output_dim neurons)."""
        acts = X
        # coefs_[:-1] = all layers except the last (classification)
        for w, b in zip(self.mlp.coefs_[:-1], self.mlp.intercepts_[:-1]):
            acts = np.tanh(acts @ w + b)               # tanh (same as activation)
        return acts                                      # shape (n, output_dim)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms features with frozen backbone → [0, π]."""
        feats = self._extract(X)
        # tanh ∈ (-1,1) → clamp → scale to [0, π]
        return self._out_scaler.transform(np.clip(feats, -1, 1))


# ─── QKE helpers ──────────────────────────────────────────────────────────────

def make_feature_map(reps: int = FM_REPS) -> QuantumCircuit:
    return zz_feature_map(feature_dimension=N_QUBITS, reps=reps)


def make_kernel(sampler=None) -> FidelityQuantumKernel:
    """Creates FidelityQuantumKernel with local or IBM Runtime sampler."""
    fm = make_feature_map()
    if sampler is not None:
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
        fidelity = ComputeUncompute(sampler=sampler)
        return FidelityQuantumKernel(feature_map=fm, fidelity=fidelity)
    return FidelityQuantumKernel(feature_map=fm)


class QKESVM:
    """SVM with precomputed quantum kernel. Supports optional backbone."""

    def __init__(self, X_tr, y_tr, C=5.0, backbone=None):
        if backbone is not None:
            X_tr = backbone.transform(X_tr)
        self.backbone = backbone
        self.kernel   = make_kernel(_GLOBAL_SAMPLER)
        self.X_tr     = X_tr
        K             = self.kernel.evaluate(X_tr, X_tr)
        self.clf      = SVC(kernel='precomputed', C=C)
        self.clf.fit(K, y_tr)
        self.train_acc = self.clf.score(K, y_tr)

    def score(self, X_te, y_te):
        if self.backbone is not None:
            X_te = self.backbone.transform(X_te)
        K_te = self.kernel.evaluate(X_te, self.X_tr)
        return self.clf.score(K_te, y_te)


class QKESVMJoint:
    """QKESVM trained on accumulated dataset with optional sample weights."""

    def __init__(self, X_tr, y_tr, C=5.0, sample_weight=None, backbone=None):
        if backbone is not None:
            X_tr = backbone.transform(X_tr)
        self.backbone = backbone
        self.kernel   = make_kernel(_GLOBAL_SAMPLER)
        self.X_tr     = X_tr
        K             = self.kernel.evaluate(X_tr, X_tr)
        self.clf      = SVC(kernel='precomputed', C=C)
        self.clf.fit(K, y_tr, sample_weight=sample_weight)
        self.train_acc = self.clf.score(K, y_tr)

    def score(self, X_te, y_te):
        if self.backbone is not None:
            X_te = self.backbone.transform(X_te)
        K_te = self.kernel.evaluate(X_te, self.X_tr)
        return self.clf.score(K_te, y_te)


def fit_qke(X_tr, y_tr, backbone=None):
    """Trains QKESVM with C search over {1, 5, 10}."""
    best, best_score = None, -1.0
    for C in [1.0, 5.0, 10.0]:
        m = QKESVM(X_tr, y_tr, C=C, backbone=backbone)
        if m.train_acc > best_score:
            best_score, best = m.train_acc, m
    return best


def eval_all(model, tasks_te, T):
    return np.array([model.score(*tasks_te[j]) for j in range(T)])


# ─── CL Methods ───────────────────────────────────────────────────────────────

def run_naive(tasks_tr, tasks_te, T, backbone=None, label="Naive FT"):
    print(f"  [{label}]")
    acc = np.zeros((T, T))
    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    task {i+1}/{T}...", flush=True)
        model = fit_qke(X_tr, y_tr, backbone=backbone)
        acc[i, :] = eval_all(model, tasks_te, T)
    return acc


def run_qewc(tasks_tr, tasks_te, T, backbone=None):
    print("  [QEWC]")
    acc       = np.zeros((T, T))
    X_hist, y_hist, w_hist = [], [], []

    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    task {i+1}/{T}...", flush=True)
        w_hist = [w * 0.65 for w in w_hist]
        X_hist.append(X_tr); y_hist.append(y_tr)
        w_hist.append(np.ones(len(X_tr)))

        X_all = np.vstack(X_hist)
        y_all = np.concatenate(y_hist)
        w_all = np.concatenate(w_hist)
        w_all = w_all / w_all.sum() * len(w_all)

        best_m, best_s = None, -1.0
        for C in [1.0, 5.0, 10.0]:
            m = QKESVMJoint(X_all, y_all, C=C, sample_weight=w_all, backbone=backbone)
            if m.train_acc > best_s:
                best_s, best_m = m.train_acc, m
        acc[i, :] = eval_all(best_m, tasks_te, T)
    return acc


def run_qtcl(tasks_tr, tasks_te, T, backbone=None, rho=0.30, label="QTCL (proposed)"):
    """QTCL: QKE + rehearsal rho%. With optional backbone."""
    print(f"  [{label}]")
    acc       = np.zeros((T, T))
    rehearsal = []

    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    task {i+1}/{T}...", flush=True)
        if rehearsal:
            Xr = np.vstack([X_tr] + [r[0] for r in rehearsal])
            yr = np.concatenate([y_tr] + [r[1] for r in rehearsal])
        else:
            Xr, yr = X_tr, y_tr

        best_m, best_s = None, -1.0
        for C in [1.0, 5.0, 10.0, 20.0]:
            m = QKESVMJoint(Xr, yr, C=C, backbone=backbone)
            if m.train_acc > best_s:
                best_s, best_m = m.train_acc, m
        acc[i, :] = eval_all(best_m, tasks_te, T)

        n_keep = max(4, int(rho * len(X_tr)))
        idx = np.random.choice(len(X_tr), n_keep, replace=False)
        rehearsal.append((X_tr[idx], y_tr[idx]))
    return acc


def run_classical(tasks_tr, tasks_te, T):
    print("  [Classical SVM]")
    acc          = np.zeros((T, T))
    X_all, y_all = [], []
    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    task {i+1}/{T}...", flush=True)
        X_all.append(X_tr); y_all.append(y_tr)
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        clf.fit(np.vstack(X_all), np.concatenate(y_all))
        acc[i, :] = [clf.score(*tasks_te[j]) for j in range(T)]
    return acc


# ─── Metrics ──────────────────────────────────────────────────────────────────

def cl_metrics(acc):
    T   = acc.shape[0]
    AA  = float(np.mean(acc[T-1, :T]))
    BWT = float(np.mean([acc[T-1, j] - acc[j, j] for j in range(T-1)])) if T > 1 else 0.
    FWT = float(np.mean([acc[j-1, j] - 0.5 for j in range(1, T)])) if T > 1 else 0.
    F_v = [max(0., float(np.max(acc[:, j])) - acc[T-1, j]) for j in range(T-1)]
    F   = float(np.mean(F_v)) if F_v else 0.
    return {'AA': AA, 'BWT': BWT, 'FWT': FWT, 'F': F}


# ─── Figures ──────────────────────────────────────────────────────────────────

def _c(name):
    return COLORS.get(METHOD_LABELS.get(name, name.lower().replace(' ', '_')), '#607D8B')


def fig_task_datasets(tasks_tr, T):
    fig, axes = plt.subplots(1, T, figsize=(4.5 * T, 4))
    pca = PCA(n_components=2, random_state=42)
    for i, (ax, (X, y)) in enumerate(zip(axes, tasks_tr)):
        X2 = pca.fit_transform(X)
        for cls, c in enumerate(['#1565C0', '#B71C1C']):
            m = y == cls
            ax.scatter(X2[m, 0], X2[m, 1], c=c, alpha=0.7, s=40,
                       label=f'Class {cls}', edgecolors='white', lw=0.3)
        ax.set_title(f'Task {i+1}', fontweight='bold')
        ax.legend(fontsize=9)
    plt.suptitle('Task Datasets — PCA 2D', fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIGURES_DIR / f'task_datasets.{ext}', bbox_inches='tight')
    plt.close()
    print("  → task_datasets")


def fig_circuit():
    fm = zz_feature_map(feature_dimension=N_QUBITS, reps=FM_REPS)
    qc = QuantumCircuit(N_QUBITS)
    qc.compose(fm, inplace=True)
    qc.measure_all()
    f = qc.draw(output='mpl', fold=40, scale=0.7)
    f.suptitle(
        f'QTCL Feature Map: ZZFeatureMap(n={N_QUBITS}, reps={FM_REPS})\n'
        f'Quantum kernel used in SVM-QKE (simulator / IBM Runtime)',
        fontweight='bold', fontsize=10)
    for ext in ('pdf', 'png'):
        f.savefig(FIGURES_DIR / f'circuit_diagram.{ext}', bbox_inches='tight')
    plt.close()
    print("  → circuit_diagram")


def fig_backbone_architecture():
    """Pipeline diagram with backbone."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    boxes = [
        (0.05, 'Input\n(4 features)', '#E3F2FD'),
        (0.28, 'Classical\nBackbone\nMLP(32→4)\n[frozen]', '#FFF3E0'),
        (0.52, 'ZZFeatureMap\n(QKE Kernel)', '#E8F5E9'),
        (0.75, 'SVM\n(precomputed\nkernel)', '#F3E5F5'),
        (0.92, 'Prediction', '#FFEBEE'),
    ]
    for x, label, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x - 0.09, 0.2), 0.16, 0.6,
                                              boxstyle="round,pad=0.02",
                                              facecolor=color, edgecolor='#455A64', lw=1.5,
                                              transform=ax.transAxes))
        ax.text(x, 0.5, label, ha='center', va='center', fontsize=9,
                fontweight='bold', transform=ax.transAxes)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.07
        x2 = boxes[i+1][0] - 0.09
        ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#37474F'))

    ax.text(0.28, 0.05, '← pretrained on T0, frozen →', ha='center',
            fontsize=8, color='#E65100', transform=ax.transAxes)
    ax.text(0.65, 0.05, '← quantum kernel (local / IBM Quantum) →', ha='center',
            fontsize=8, color='#1B5E20', transform=ax.transAxes)

    ax.set_title('QTCL + Backbone: Quantum Transfer Learning Pipeline',
                 fontweight='bold', fontsize=12, pad=10)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIGURES_DIR / f'backbone_pipeline.{ext}', bbox_inches='tight')
    plt.close()
    print("  → backbone_pipeline")


def fig_kernel_matrix(tasks_tr):
    X_tr, _ = tasks_tr[0]
    X_sub   = X_tr[:25]
    kernel  = make_kernel(_GLOBAL_SAMPLER)
    K       = kernel.evaluate(X_sub, X_sub)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(K, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Quantum Kernel Matrix (Task 1, n=25)', fontweight='bold')
    axes[0].set_xlabel('Sample index')
    axes[0].set_ylabel('Sample index')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
    axes[1].semilogy(eigvals, 'o-', color='#2196F3', ms=4, lw=1.5)
    axes[1].set_title('Kernel Eigenspectrum', fontweight='bold')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue (log scale)')
    axes[1].grid(True, which='both', alpha=0.3)

    plt.suptitle('Quantum Kernel Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIGURES_DIR / f'kernel_analysis.{ext}', bbox_inches='tight')
    plt.close()
    print("  → kernel_analysis")


def fig_acc_matrix(matrices, T):
    n    = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    for ax, (name, mat) in zip(axes, matrices.items()):
        sns.heatmap(mat, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=0.4, vmax=1.0, linewidths=0.5,
                    xticklabels=[f'T{j+1}' for j in range(T)],
                    yticklabels=[f'↓T{i+1}' for i in range(T)],
                    cbar_kws={'shrink': 0.8})
        ax.set_title(name, fontweight='bold', pad=8)
        ax.set_xlabel('Task evaluated')
        ax.set_ylabel('After training on')
    plt.suptitle('Accuracy Matrix a[i,j]', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIGURES_DIR / f'accuracy_matrix.{ext}', bbox_inches='tight')
    plt.close()
    print("  → accuracy_matrix")


def fig_cl_metrics(all_metrics):
    labels = {'AA': 'Avg Accuracy ↑', 'BWT': 'Backward Transfer ↑',
              'FWT': 'Fwd Transfer ↑', 'F': 'Forgetting ↓'}
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    methods = list(all_metrics.keys())
    colors  = [_c(m) for m in methods]
    for ax, metric in zip(axes, ['AA', 'BWT', 'FWT', 'F']):
        vals = [all_metrics[m][metric] for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=colors,
                      edgecolor='white', lw=1.0, width=0.65)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=28, ha='right', fontsize=8)
        ax.set_title(labels[metric], fontweight='bold', fontsize=10)
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        for bar, val in zip(bars, vals):
            yo = bar.get_height() + 0.005 if val >= 0 else val - 0.018
            ax.text(bar.get_x() + bar.get_width() / 2, yo,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    plt.suptitle('Continual Learning Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIGURES_DIR / f'cl_metrics.{ext}', bbox_inches='tight')
    plt.close()
    print("  → cl_metrics")


def fig_backbone_comparison(all_metrics):
    """Direct comparison: with backbone vs without backbone."""
    pairs = [
        ('Naive FT', 'Naive FT + Backbone'),
        ('QTCL (proposed)', 'QTCL + Backbone'),
    ]
    metrics_order = ['AA', 'BWT', 'FWT', 'F']
    labels        = ['AA ↑', 'BWT ↑', 'FWT ↑', 'Forgetting ↓']

    fig, axes = plt.subplots(1, len(metrics_order), figsize=(16, 5))
    x = np.arange(len(pairs))
    w = 0.35

    for ax, metric, label in zip(axes, metrics_order, labels):
        v_no  = [all_metrics[p[0]][metric] for p in pairs]
        v_bb  = [all_metrics[p[1]][metric] for p in pairs]
        ax.bar(x - w/2, v_no, w, label='Without backbone', color='#90A4AE', edgecolor='white')
        ax.bar(x + w/2, v_bb, w, label='With backbone', color='#0D47A1', edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in pairs], rotation=15, ha='right', fontsize=9)
        ax.set_title(label, fontweight='bold')
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.4)
        ax.legend(fontsize=8)

    plt.suptitle('Impact of Pretrained Classical Backbone on CL Metrics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIGURES_DIR / f'backbone_comparison.{ext}', bbox_inches='tight')
    plt.close()
    print("  → backbone_comparison")


def fig_acc_evolution(matrices, T):
    fig, axes = plt.subplots(1, T, figsize=(5 * T, 4))
    for t, ax in enumerate(axes):
        for name, mat in matrices.items():
            ax.plot(range(T), mat[:, t], marker='o', label=name,
                    color=_c(name), lw=2, ms=6)
        ax.axvline(t, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_title(f'Task {t+1}', fontweight='bold')
        ax.set_xlabel('Training step')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(T))
        ax.set_xticklabels([f'T{i+1}' for i in range(T)])
        ax.set_ylim(0.3, 1.05)
        if t == T - 1:
            ax.legend(fontsize=7, loc='lower left')
    plt.suptitle('Per-task Accuracy Evolution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIGURES_DIR / f'acc_evolution.{ext}', bbox_inches='tight')
    plt.close()
    print("  → acc_evolution")


def fig_forgetting(matrices, T):
    task_f = {m: [max(0, mat[t, t] - mat[T-1, t]) for t in range(T-1)]
              for m, mat in matrices.items()}
    x, w   = np.arange(T-1), 0.12
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, vals) in enumerate(task_f.items()):
        off = (i - len(matrices) / 2) * w
        ax.bar(x + off, vals, w, label=name, color=_c(name), edgecolor='white', lw=0.8)
    ax.set_xlabel('Task')
    ax.set_ylabel('Forgetting')
    ax.set_title('Catastrophic Forgetting per Task', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Task {t+1}' for t in range(T-1)])
    ax.legend(fontsize=8)
    ax.axhline(0, color='black', lw=0.8)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIGURES_DIR / f'forgetting.{ext}', bbox_inches='tight')
    plt.close()
    print("  → forgetting")


def fig_radar(all_metrics):
    labels_r = ['Avg Accuracy', 'Fwd Transfer', 'Bwd Transfer\n(shifted)', '1-Forgetting']
    N        = len(labels_r)
    angles   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles  += angles[:1]
    fig, ax  = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for name, m in all_metrics.items():
        vals = [m['AA'],
                min(1, max(0, m['FWT'] + 0.5)),
                min(1, max(0, m['BWT'] + 0.5)),
                min(1, max(0, 1. - m['F']))]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', lw=2, label=name, color=_c(name))
        ax.fill(angles, vals, alpha=0.08, color=_c(name))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('CL Metrics Radar', fontweight='bold', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=8)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIGURES_DIR / f'radar.{ext}', bbox_inches='tight')
    plt.close()
    print("  → radar")


def fig_summary_table(all_metrics):
    rows = [[m, f"{v['AA']:.4f}", f"{v['BWT']:+.4f}", f"{v['FWT']:+.4f}", f"{v['F']:.4f}"]
            for m, v in all_metrics.items()]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis('off')
    tbl = ax.table(cellText=rows,
                   colLabels=['Method', 'AA ↑', 'BWT ↑', 'FWT ↑', 'Forgetting ↓'],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 2.0)
    for j in range(5):
        tbl[0, j].set_facecolor('#1A237E')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i, (m, _) in enumerate(all_metrics.items()):
        if m == 'QTCL + Backbone':
            for j in range(5):
                tbl[i+1, j].set_facecolor('#E3F2FD')
    ax.set_title('Summary of CL Metrics (Local simulator / IBM Runtime)',
                 fontweight='bold', fontsize=12, pad=12)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(FIGURES_DIR / f'summary_table.{ext}', bbox_inches='tight')
    plt.close()
    print("  → summary_table")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _GLOBAL_SAMPLER

    T = 4
    print(f"\n{'='*70}")
    print("QTCL v5 — QKE + Classical Backbone + IBM Runtime ready")
    print(f"{'='*70}")
    print(f"Qubits={N_QUBITS} | ZZFeatureMap(reps={FM_REPS}) | SVM-QKE")
    print(f"Train={N_TRAIN} | Test={N_TEST} | class_sep={CLASS_SEP}")
    mode = f"IBM Quantum ({IBM_BACKEND})" if USE_IBM_RUNTIME else "Local simulator (StatevectorSampler)"
    print(f"Backend: {mode}")
    print(f"{'='*70}")

    # Initialize sampler (IBM or local)
    _GLOBAL_SAMPLER = build_sampler()

    # Tasks
    print("\n[1] Generating tasks...")
    tasks_tr, tasks_te = [], []
    for t in range(T):
        Xtr, Xte, ytr, yte = make_task(t)
        tasks_tr.append((Xtr, ytr))
        tasks_te.append((Xte, yte))
        print(f"  T{t+1}: {len(Xtr)} train | {len(Xte)} test")

    # Backbone: pretrain on task 0 (frozen for the rest)
    print("\n[2] Pretraining backbone (task 0)...")
    backbone = ClassicalBackbone(output_dim=N_QUBITS, hidden_dim=32)
    backbone.pretrain(*tasks_tr[0])

    # Preliminary figures
    print("\n[3] Preliminary figures...")
    fig_task_datasets(tasks_tr, T)
    fig_circuit()
    fig_backbone_architecture()
    print("  → kernel_analysis (computing...)")
    fig_kernel_matrix(tasks_tr)

    # Experiments
    print("\n[4] Experiments...")
    mat_naive     = run_naive(tasks_tr, tasks_te, T, backbone=None,     label="Naive FT")
    mat_naive_bb  = run_naive(tasks_tr, tasks_te, T, backbone=backbone,  label="Naive FT + Backbone")
    mat_qewc      = run_qewc(tasks_tr, tasks_te, T,  backbone=None)
    mat_qtcl      = run_qtcl(tasks_tr, tasks_te, T,  backbone=None,     label="QTCL (proposed)")
    mat_qtcl_bb   = run_qtcl(tasks_tr, tasks_te, T,  backbone=backbone,  label="QTCL + Backbone")
    mat_classical = run_classical(tasks_tr, tasks_te, T)

    matrices = {
        'Naive FT':            mat_naive,
        'Naive FT + Backbone': mat_naive_bb,
        'QEWC':                mat_qewc,
        'QTCL (proposed)':     mat_qtcl,
        'QTCL + Backbone':     mat_qtcl_bb,
        'Classical SVM':       mat_classical,
    }

    # Metrics
    print("\n[5] Metrics...")
    all_metrics = {}
    for name, mat in matrices.items():
        m = cl_metrics(mat)
        all_metrics[name] = m
        print(f"  {name:25s}  AA={m['AA']:.4f}  BWT={m['BWT']:+.4f}"
              f"  FWT={m['FWT']:+.4f}  F={m['F']:.4f}")

    # Figures
    print("\n[6] Figures...")
    fig_acc_matrix(matrices, T)
    fig_cl_metrics(all_metrics)
    fig_backbone_comparison(all_metrics)
    fig_acc_evolution(matrices, T)
    fig_forgetting(matrices, T)
    fig_radar(all_metrics)
    fig_summary_table(all_metrics)

    df = pd.DataFrame([{'Method': m, **v} for m, v in all_metrics.items()])
    df.to_csv(FIGURES_DIR.parent / 'results.csv', index=False)
    print("  → results.csv")

    print(f"\n{'='*70}")
    print("Completed.")
    if USE_IBM_RUNTIME:
        print(f"Hardware used: {IBM_BACKEND}")
    print(f"{'='*70}\n")
    return all_metrics, matrices


if __name__ == "__main__":
    main()
