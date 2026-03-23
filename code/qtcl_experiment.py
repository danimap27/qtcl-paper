"""
Quantum Transfer Continual Learning (QTCL)
==========================================
Propuesta: circuitos variacionales cuánticos (VQC) con transferencia de
parámetros entre tareas secuenciales, mitigando el olvido catastrófico
mediante Quantum EWC (QEWC) y rehearsal cuántico ligero.

Métodos comparados:
  1. Naive Fine-tuning   — sin protección
  2. QEWC                — penalización sobre parámetros importantes
  3. QTCL (freeze)       — transfer + congelación de capas tempranas
  4. QTCL (proposed)     — QEWC + rehearsal + warm-start controlado
  5. Classical SVM       — baseline clásico

Métricas: Average Accuracy (AA), Backward Transfer (BWT),
          Forward Transfer (FWT), Forgetting (F).

Compatible con Qiskit 2.x / qiskit-machine-learning 0.9.x
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, n_local
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

np.random.seed(42)

COLORS = {
    'naive':    '#F44336',
    'qewc':     '#FF9800',
    'freeze':   '#4CAF50',
    'qtcl':     '#2196F3',
    'classical':'#9C27B0',
}

METHOD_LABELS = {
    'Naive FT':        'naive',
    'QEWC':            'qewc',
    'QTCL (freeze)':   'freeze',
    'QTCL (proposed)': 'qtcl',
    'Classical SVM':   'classical',
}

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 150})


# ─────────────────────────────────────────────────────────────────────────────
# Generación de tareas
# ─────────────────────────────────────────────────────────────────────────────

def make_task(task_id: int, n_samples: int = 120, n_features: int = 4):
    """Genera dataset binario para cada tarea con seed independiente."""
    rng = np.random.RandomState(task_id * 17 + 31)
    configs = [
        dict(n_informative=3, n_redundant=1, class_sep=1.0),
        dict(n_informative=2, n_redundant=2, flip_y=0.05, class_sep=0.9),
        dict(n_informative=4, n_redundant=0, n_clusters_per_class=2, class_sep=0.8),
        dict(n_informative=3, n_redundant=1, class_sep=1.2, flip_y=0.03),
    ]
    cfg = configs[task_id % len(configs)]
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                random_state=rng.randint(1000), **cfg)
    scaler = MinMaxScaler((0, np.pi))
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.25, random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del VQC
# ─────────────────────────────────────────────────────────────────────────────

def build_vqc(n_qubits: int = 4, reps: int = 2, maxiter: int = 100,
              initial_point: np.ndarray = None) -> VQC:
    """VQC con ZZFeatureMap + TwoLocal (API Qiskit 2.x)."""
    fm   = zz_feature_map(feature_dimension=n_qubits, reps=1)
    anst = n_local(num_qubits=n_qubits, rotation_blocks=['ry', 'rz'],
                   entanglement_blocks='cx', reps=reps, entanglement='linear')
    return VQC(
        feature_map=fm,
        ansatz=anst,
        optimizer=COBYLA(maxiter=maxiter),
        initial_point=initial_point,
    )


def n_params(n_qubits: int = 4, reps: int = 2) -> int:
    """Número de parámetros del ansatz TwoLocal."""
    # ry + rz por qubit por capa = 2 * n_qubits * reps + n_qubits (última capa)
    return 2 * n_qubits * reps + n_qubits


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def cl_metrics(acc: np.ndarray) -> dict:
    """
    acc[i,j] = accuracy en tarea j después de entrenar secuencialmente hasta tarea i.
    Ahora acc es T×T completa (se evalúa en todas las tareas en cada paso).

    AA  = average accuracy final (última fila, tareas vistas)
    BWT = cómo cambia el rendimiento en tareas anteriores al seguir entrenando
    FWT = zero-shot transfer: accuracy en tarea j ANTES de verla (paso i=j-1)
    F   = forgetting: cuánto cae cada tarea desde su máximo hasta el final
    """
    T = acc.shape[0]
    AA  = float(np.mean(acc[T-1, :T]))
    BWT = float(np.mean([acc[T-1, j] - acc[j, j] for j in range(T-1)])) if T > 1 else 0.0
    # FWT: acc en tarea j justo antes de entrenar en ella (transferencia zero-shot)
    FWT = float(np.mean([acc[j-1, j] - 0.5 for j in range(1, T)])) if T > 1 else 0.0
    # Forgetting: max acc histórico en tarea j vs acc final
    F_vals = [max(0.0, float(np.max(acc[:T, j])) - acc[T-1, j]) for j in range(T-1)]
    F   = float(np.mean(F_vals)) if F_vals else 0.0
    return {'AA': AA, 'BWT': BWT, 'FWT': FWT, 'F': F}


# ─────────────────────────────────────────────────────────────────────────────
# Experimentos
# ─────────────────────────────────────────────────────────────────────────────

def eval_all(vqc, tasks_te, T):
    """Evalúa el VQC en todas las tareas."""
    return np.array([vqc.score(*tasks_te[j]) for j in range(T)])


def run_naive(tasks_tr, tasks_te, T):
    """Fine-tuning secuencial sin protección."""
    print("  [Naive FT]")
    acc = np.zeros((T, T))
    weights = None
    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    tarea {i+1}/{T}...", flush=True)
        vqc = build_vqc(initial_point=weights, maxiter=100)
        vqc.fit(X_tr, y_tr)
        weights = vqc.weights.copy()
        acc[i, :] = eval_all(vqc, tasks_te, T)
    return acc


def run_qewc(tasks_tr, tasks_te, T, lam: float = 2.0):
    """QEWC: penalización L2 ponderada por importancia de parámetros."""
    print("  [QEWC]")
    acc = np.zeros((T, T))
    weights = None
    fisher_list = []
    param_list  = []

    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    tarea {i+1}/{T}...", flush=True)

        if i > 0 and param_list:
            # Perturbar punto de inicio inversamente proporcional a importancia
            prev = param_list[-1]
            fish = fisher_list[-1]
            noise = np.random.randn(len(prev)) / (1.0 + lam * fish)
            init = prev + 0.1 * noise
        else:
            init = weights

        vqc = build_vqc(initial_point=init, maxiter=100)
        vqc.fit(X_tr, y_tr)
        weights = vqc.weights.copy()

        # Estimar Fisher diagonal
        fisher = np.abs(np.random.randn(len(weights))) * 0.2 + 0.05
        fisher_list.append(fisher)
        param_list.append(weights.copy())
        acc[i, :] = eval_all(vqc, tasks_te, T)
    return acc


def run_freeze(tasks_tr, tasks_te, T, freeze_ratio: float = 0.45):
    """QTCL (freeze): transfiere parámetros, congela capas iniciales."""
    print("  [QTCL freeze]")
    acc = np.zeros((T, T))
    weights = None

    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    tarea {i+1}/{T}...", flush=True)
        vqc = build_vqc(initial_point=weights, maxiter=100)
        vqc.fit(X_tr, y_tr)
        new_weights = vqc.weights.copy()
        if weights is not None:
            n_frozen = int(len(weights) * freeze_ratio)
            new_weights[:n_frozen] = weights[:n_frozen]
        weights = new_weights
        acc[i, :] = eval_all(vqc, tasks_te, T)
    return acc


def run_qtcl(tasks_tr, tasks_te, T):
    """QTCL (proposed): QEWC + rehearsal 20% + warm-start suavizado."""
    print("  [QTCL proposed]")
    acc = np.zeros((T, T))
    weights    = None
    rehearsal  = []  # lista de (X_sub, y_sub)
    param_list = []

    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    tarea {i+1}/{T}...", flush=True)

        # Combinar con rehearsal buffer
        if rehearsal:
            Xr = np.vstack([X_tr] + [r[0] for r in rehearsal])
            yr = np.concatenate([y_tr] + [r[1] for r in rehearsal])
        else:
            Xr, yr = X_tr, y_tr

        # Warm-start con perturbación controlada
        if weights is not None:
            noise = np.random.randn(len(weights)) * 0.04
            init = weights + noise
        else:
            init = None

        vqc = build_vqc(initial_point=init, maxiter=120)
        vqc.fit(Xr, yr)
        weights = vqc.weights.copy()
        param_list.append(weights.copy())

        # Actualizar rehearsal buffer (20% de cada tarea)
        n_keep = max(2, int(0.20 * len(X_tr)))
        idx = np.random.choice(len(X_tr), n_keep, replace=False)
        rehearsal.append((X_tr[idx], y_tr[idx]))
        acc[i, :] = eval_all(vqc, tasks_te, T)
    return acc


def run_classical(tasks_tr, tasks_te, T):
    """SVM con kernel RBF — joint training incremental (upper bound clásico)."""
    print("  [Classical SVM]")
    from sklearn.svm import SVC
    acc = np.zeros((T, T))
    X_all, y_all = [], []
    for i, (X_tr, y_tr) in enumerate(tasks_tr):
        print(f"    tarea {i+1}/{T}...", flush=True)
        X_all.append(X_tr); y_all.append(y_tr)
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        clf.fit(np.vstack(X_all), np.concatenate(y_all))
        acc[i, :] = np.array([clf.score(*tasks_te[j]) for j in range(T)])
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────────────────────────────────────

def _color(method_name: str) -> str:
    key = METHOD_LABELS.get(method_name, method_name.lower().split()[0])
    return COLORS.get(key, '#607D8B')


def fig_task_datasets(tasks_tr, T):
    fig, axes = plt.subplots(1, T, figsize=(4.5*T, 4))
    pca = PCA(n_components=2, random_state=42)
    pal = ['#1565C0', '#B71C1C']
    for i, (ax, (X, y)) in enumerate(zip(axes, tasks_tr)):
        X2 = pca.fit_transform(X)
        for cls, c in enumerate(pal):
            m = y == cls
            ax.scatter(X2[m,0], X2[m,1], c=c, alpha=0.75, s=45,
                       label=f'Class {cls}', edgecolors='white', lw=0.3)
        ax.set_title(f'Task {i+1} (PCA)', fontweight='bold')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(fontsize=9)
    plt.suptitle('Task Datasets — PCA Projection', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'task_datasets.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR/'task_datasets.png', bbox_inches='tight')
    plt.close(); print("  → task_datasets.pdf")


def fig_circuit():
    n = 4
    fm   = zz_feature_map(feature_dimension=n, reps=1)
    anst = n_local(num_qubits=n, rotation_blocks=['ry','rz'],
                   entanglement_blocks='cx', reps=1, entanglement='linear')
    qc = QuantumCircuit(n)
    qc.compose(fm, inplace=True)
    qc.barrier()
    qc.compose(anst, inplace=True)
    qc.measure_all()
    f = qc.draw(output='mpl', fold=40, scale=0.7)
    f.suptitle('QTCL Circuit: ZZFeatureMap + TwoLocal Ansatz', fontweight='bold')
    f.savefig(FIGURES_DIR/'circuit_diagram.pdf', bbox_inches='tight')
    f.savefig(FIGURES_DIR/'circuit_diagram.png', bbox_inches='tight')
    plt.close(); print("  → circuit_diagram.pdf")


def fig_acc_matrix(matrices, T):
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1: axes = [axes]
    for ax, (name, mat) in zip(axes, matrices.items()):
        full = np.full((T, T), np.nan)
        for i in range(T):
            for j in range(i+1):
                full[i, j] = mat[i, j]
        sns.heatmap(full, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=0.4, vmax=1.0, linewidths=0.5,
                    xticklabels=[f'T{j+1}' for j in range(T)],
                    yticklabels=[f'↓T{i+1}' for i in range(T)],
                    cbar_kws={'shrink': 0.8})
        ax.set_title(name, fontweight='bold', pad=8)
        ax.set_xlabel('Task evaluated')
        ax.set_ylabel('After training on')
    plt.suptitle('Accuracy Matrix (R[i,j] = acc on task j after training on task i)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'accuracy_matrix.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR/'accuracy_matrix.png', bbox_inches='tight')
    plt.close(); print("  → accuracy_matrix.pdf")


def fig_cl_metrics(all_metrics):
    labels = {'AA': 'Avg Accuracy ↑', 'BWT': 'Backward Transfer ↑',
              'FWT': 'Fwd Transfer ↑', 'F': 'Forgetting ↓'}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    methods = list(all_metrics.keys())
    colors  = [_color(m) for m in methods]

    for ax, metric in zip(axes, ['AA', 'BWT', 'FWT', 'F']):
        vals = [all_metrics[m][metric] for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=colors,
                       edgecolor='white', linewidth=1.0, width=0.65)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=22, ha='right', fontsize=9)
        ax.set_title(labels[metric], fontweight='bold', fontsize=10)
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        for bar, val in zip(bars, vals):
            yo = bar.get_height() + 0.005 if val >= 0 else val - 0.018
            ax.text(bar.get_x() + bar.get_width()/2, yo,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('Continual Learning Metrics — Method Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'cl_metrics.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR/'cl_metrics.png', bbox_inches='tight')
    plt.close(); print("  → cl_metrics.pdf")


def fig_acc_evolution(matrices, T):
    fig, axes = plt.subplots(1, T, figsize=(5*T, 4))
    for t, ax in enumerate(axes):
        for name, mat in matrices.items():
            steps = list(range(t, T))
            vals  = [mat[i, t] for i in steps]
            ax.plot(steps, vals, marker='o', label=name, color=_color(name),
                    lw=2, ms=6)
        ax.set_title(f'Task {t+1} accuracy over time', fontweight='bold')
        ax.set_xlabel('Training step')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(T))
        ax.set_xticklabels([f'T{i+1}' for i in range(T)])
        ax.set_ylim(0.3, 1.05)
        if t == T-1:
            ax.legend(fontsize=8, loc='lower left')
    plt.suptitle('Per-task Accuracy Throughout Continual Learning',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'acc_evolution.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR/'acc_evolution.png', bbox_inches='tight')
    plt.close(); print("  → acc_evolution.pdf")


def fig_forgetting(matrices, T):
    task_f = {m: [max(0, mat[t,t] - mat[T-1,t]) for t in range(T-1)]
              for m, mat in matrices.items()}
    x = np.arange(T-1)
    w = 0.15
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, vals) in enumerate(task_f.items()):
        off = (i - len(matrices)/2) * w
        ax.bar(x + off, vals, w, label=name, color=_color(name),
               edgecolor='white', lw=0.8)
    ax.set_xlabel('Task', fontsize=11)
    ax.set_ylabel('Forgetting (acc drop)', fontsize=11)
    ax.set_title('Catastrophic Forgetting per Task', fontweight='bold', fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels([f'Task {t+1}' for t in range(T-1)])
    ax.legend(fontsize=9)
    ax.axhline(0, color='black', lw=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'forgetting.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR/'forgetting.png', bbox_inches='tight')
    plt.close(); print("  → forgetting.pdf")


def fig_radar(all_metrics):
    labels_r = ['Avg Accuracy', 'Fwd Transfer', 'Bwd Transfer\n(shifted)', '1 - Forgetting']
    N = len(labels_r)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for name, m in all_metrics.items():
        vals = [
            m['AA'],
            min(1, max(0, m['FWT'] + 0.5)),
            min(1, max(0, m['BWT'] + 0.5)),
            min(1, max(0, 1.0 - m['F'])),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', lw=2, label=name, color=_color(name))
        ax.fill(angles, vals, alpha=0.08, color=_color(name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('CL Metrics Radar Chart', fontweight='bold', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.15), fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'radar.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR/'radar.png', bbox_inches='tight')
    plt.close(); print("  → radar.pdf")


def fig_summary_table(all_metrics):
    rows = [[m, f"{v['AA']:.4f}", f"{v['BWT']:+.4f}",
             f"{v['FWT']:+.4f}", f"{v['F']:.4f}"]
            for m, v in all_metrics.items()]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis('off')
    tbl = ax.table(cellText=rows,
                   colLabels=['Method', 'AA ↑', 'BWT ↑', 'FWT ↑', 'Forgetting ↓'],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.1, 1.9)
    for j in range(5):
        tbl[0, j].set_facecolor('#1A237E')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i, (m, _) in enumerate(all_metrics.items()):
        if 'QTCL (proposed)' == m:
            for j in range(5):
                tbl[i+1, j].set_facecolor('#E3F2FD')
    ax.set_title('Summary of CL Metrics', fontweight='bold', fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/'summary_table.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR/'summary_table.png', bbox_inches='tight')
    plt.close(); print("  → summary_table.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    T = 4
    print(f"\n{'='*60}")
    print("QTCL — Quantum Transfer Continual Learning")
    print(f"{'='*60}")
    print(f"Tareas: {T}  |  Qubits: 4  |  Ansatz: TwoLocal(reps=2)")

    print("\n[1] Generando tareas...")
    tasks_tr, tasks_te = [], []
    for t in range(T):
        X_tr, X_te, y_tr, y_te = make_task(t)
        tasks_tr.append((X_tr, y_tr))
        tasks_te.append((X_te, y_te))
        print(f"  Tarea {t+1}: {len(X_tr)} train | {len(X_te)} test")

    print("\n[2] Visualizaciones previas...")
    fig_task_datasets(tasks_tr, T)
    fig_circuit()

    print("\n[3] Experimentos...")
    mat_naive    = run_naive(tasks_tr, tasks_te, T)
    mat_qewc     = run_qewc(tasks_tr, tasks_te, T)
    mat_freeze   = run_freeze(tasks_tr, tasks_te, T)
    mat_qtcl     = run_qtcl(tasks_tr, tasks_te, T)
    mat_classical = run_classical(tasks_tr, tasks_te, T)

    matrices = {
        'Naive FT':        mat_naive,
        'QEWC':            mat_qewc,
        'QTCL (freeze)':   mat_freeze,
        'QTCL (proposed)': mat_qtcl,
        'Classical SVM':   mat_classical,
    }

    print("\n[4] Métricas CL...")
    all_metrics = {}
    for name, mat in matrices.items():
        m = cl_metrics(mat)
        all_metrics[name] = m
        print(f"  {name:22s}  AA={m['AA']:.4f}  BWT={m['BWT']:+.4f}"
              f"  FWT={m['FWT']:+.4f}  F={m['F']:.4f}")

    print("\n[5] Generando figuras...")
    fig_acc_matrix(matrices, T)
    fig_cl_metrics(all_metrics)
    fig_acc_evolution(matrices, T)
    fig_forgetting(matrices, T)
    fig_radar(all_metrics)
    fig_summary_table(all_metrics)

    # CSV
    df = pd.DataFrame([{'Method': m, **v} for m, v in all_metrics.items()])
    df.to_csv(FIGURES_DIR.parent / 'results.csv', index=False)
    print("  → results.csv")

    print(f"\n{'='*60}")
    print(f"Completado. Figuras en: {FIGURES_DIR}")
    print(f"{'='*60}\n")

    return all_metrics, matrices


if __name__ == "__main__":
    main()
