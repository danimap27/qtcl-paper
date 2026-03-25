"""
QTCL Ablation Study
===================
Studies the effect of:
  1. Lambda EWC (λ ∈ {50, 200, 500, 1000, 2000})
  2. Rehearsal ratio (ρ ∈ {0.0, 0.1, 0.2, 0.3, 0.4})
  3. Circuit depth (n_shared_layers ∈ {1, 2, 3})
  4. Qubit count (n_qubits ∈ {2, 4, 6}) — PennyLane only

Uses 2 seeds per configuration (balance between rigor and speed).
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from copy import deepcopy

import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

try:
    import torchvision, torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ─── Base config (same as v6) ─────────────────────────────────────────────────
N_QUBITS_DEFAULT  = 4
N_SHARED_DEFAULT  = 2
N_TASK_DEFAULT    = 1
ENC_HIDDEN        = 32
N_TRAIN_PER_TASK  = 150
N_TEST_PER_TASK   = 80
N_EPOCHS          = 25
BATCH_SIZE        = 16
LR                = 0.002
N_SEEDS_ABL       = 2   # 2 seeds for ablation (speed/rigor tradeoff)
TASKS = [(0,1),(2,3),(4,5),(6,7),(8,9)]
DEVICE = torch.device("cpu")

sns.set_theme(style="whitegrid", font_scale=1.1)


# ─── Dataset ──────────────────────────────────────────────────────────────────
def load_split_mnist(seed=42):
    rng = np.random.RandomState(seed)
    if HAS_TORCHVISION:
        ds_tr = torchvision.datasets.MNIST(
            root="data/", train=True,
            download=True, transform=T.ToTensor())
        ds_te = torchvision.datasets.MNIST(
            root="data/", train=False,
            download=True, transform=T.ToTensor())
        X_tr = ds_tr.data.float().numpy()/255.; y_tr = ds_tr.targets.numpy()
        X_te = ds_te.data.float().numpy()/255.; y_te = ds_te.targets.numpy()
    else:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784",version=1,as_frame=False)
        X,y = mnist.data/255., mnist.target.astype(int)
        X_tr,y_tr = X[:60000],y[:60000]; X_te,y_te = X[60000:],y[60000:]
    X_tr = X_tr.reshape(-1,784); X_te = X_te.reshape(-1,784)
    tasks_tr, tasks_te = [], []
    for (c0,c1) in TASKS:
        n_h = N_TRAIN_PER_TASK//2
        i0  = rng.choice(np.where(y_tr==c0)[0],n_h,replace=False)
        i1  = rng.choice(np.where(y_tr==c1)[0],n_h,replace=False)
        idx = np.concatenate([i0,i1]); rng.shuffle(idx)
        ytr = (y_tr[idx]==c1).astype(int)
        n_h_te = N_TEST_PER_TASK//2
        i0t=rng.choice(np.where(y_te==c0)[0],n_h_te,replace=False)
        i1t=rng.choice(np.where(y_te==c1)[0],n_h_te,replace=False)
        idt=np.concatenate([i0t,i1t]); rng.shuffle(idt)
        yte = (y_te[idt]==c1).astype(int)
        tasks_tr.append((torch.tensor(X_tr[idx],dtype=torch.float32),
                         torch.tensor(ytr,dtype=torch.long)))
        tasks_te.append((torch.tensor(X_te[idt],dtype=torch.float32),
                         torch.tensor(yte,dtype=torch.long)))
    return tasks_tr, tasks_te


# ─── Parametric model ─────────────────────────────────────────────────────────
def make_qnode(n_qubits, n_shared, n_task):
    dev = qml.device("lightning.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, shared_w, task_w):
        for i in range(n_qubits):
            qml.RY(inputs[i]*np.pi, wires=i)
        qml.StronglyEntanglingLayers(shared_w, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(task_w,   wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit


class QuantumModel(nn.Module):
    def __init__(self, n_qubits=N_QUBITS_DEFAULT, n_shared=N_SHARED_DEFAULT,
                 n_task=N_TASK_DEFAULT):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_shared = n_shared
        self.n_task   = n_task
        self.encoder  = nn.Sequential(
            nn.Linear(784, ENC_HIDDEN), nn.ReLU(), nn.LayerNorm(ENC_HIDDEN),
            nn.Linear(ENC_HIDDEN, n_qubits), nn.Tanh()
        )
        self.shared_w = nn.Parameter(torch.randn(n_shared, n_qubits, 3)*0.1)
        self.task_w   = nn.ParameterDict()
        self._add_task(0)
        self.current_task = 0
        self.post_q   = nn.Linear(n_qubits, 2)
        self.circuit  = make_qnode(n_qubits, n_shared, n_task)

    def _add_task(self, tid):
        k = f"t{tid}"
        if k not in self.task_w:
            self.task_w[k] = nn.Parameter(torch.randn(self.n_task, self.n_qubits, 3)*0.1)

    def set_task(self, tid):
        self._add_task(tid); self.current_task = tid

    def forward(self, x):
        z  = self.encoder(x)
        tw = self.task_w[f"t{self.current_task}"]
        outs = []
        for i in range(z.shape[0]):
            outs.append(torch.stack(self.circuit(z[i], self.shared_w, tw)))
        return self.post_q(torch.stack(outs).float())


class ClassicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, ENC_HIDDEN), nn.ReLU(), nn.LayerNorm(ENC_HIDDEN),
            nn.Linear(ENC_HIDDEN, N_QUBITS_DEFAULT), nn.Tanh()
        )
        self.head = nn.Sequential(nn.Linear(N_QUBITS_DEFAULT,8),nn.ReLU(),nn.Linear(8,2))
    def forward(self,x): return self.head(self.encoder(x))


class EWC:
    def __init__(self, model, lam):
        self.model=model; self.lam=lam
        self.fisher={}; self.saved={}; self.n_seen=0

    def register(self, X, y, tid):
        self.model.eval()
        fish = {n:torch.zeros_like(p) for n,p in self.model.named_parameters() if p.requires_grad}
        loader = DataLoader(TensorDataset(X,y), batch_size=1, shuffle=False)
        n_tot = 0
        for xb,yb in loader:
            self.model.zero_grad()
            out = self.model(xb)
            torch.log_softmax(out,1)[0,yb[0]].backward()
            for n,p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fish[n] += p.grad.detach()**2
            n_tot += 1
            if n_tot >= 60: break
        for n in fish: fish[n] /= max(n_tot,1)
        self.fisher[tid] = fish
        self.saved[tid]  = {n:p.detach().clone() for n,p in self.model.named_parameters() if p.requires_grad}
        self.n_seen = max(self.n_seen, tid+1)
        self.model.train()

    def penalty(self):
        if self.n_seen==0: return torch.tensor(0.0,device=DEVICE)
        loss = torch.tensor(0.0,device=DEVICE)
        for t in range(self.n_seen):
            for n,p in self.model.named_parameters():
                if n in self.fisher.get(t,{}):
                    loss += (self.fisher[t][n]*(p-self.saved[t][n])**2).sum()
        return self.lam * loss


def train_task(model, X, y, epochs, ewc=None):
    model.train()
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    fn  = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X,y), batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for xb,yb in loader:
            opt.zero_grad()
            loss = fn(model(xb),yb)
            if ewc: loss = loss + ewc.penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()


@torch.no_grad()
def eval_model(model, X, y):
    model.eval()
    loader = DataLoader(TensorDataset(X,y), batch_size=BATCH_SIZE, shuffle=False)
    correct = sum((model(xb).argmax(1)==yb).sum().item() for xb,yb in loader)
    return correct/len(y)


def run_qtcl(tasks_tr, tasks_te, seed, lam=200., rho=0.25,
             n_qubits=N_QUBITS_DEFAULT, n_shared=N_SHARED_DEFAULT,
             n_task=N_TASK_DEFAULT) -> dict:
    """Runs QTCL with configurable parameters. Returns cl_metrics."""
    torch.manual_seed(seed); np.random.seed(seed)
    T     = len(tasks_tr)
    acc   = np.zeros((T,T))
    model = QuantumModel(n_qubits, n_shared, n_task).to(DEVICE)
    ewc   = EWC(model, lam)
    reh_X, reh_y = [], []

    for i,(X_tr,y_tr) in enumerate(tasks_tr):
        model.set_task(i)
        if reh_X:
            Xr = torch.cat([X_tr]+reh_X); yr = torch.cat([y_tr]+reh_y)
        else:
            Xr,yr = X_tr,y_tr
        train_task(model,Xr,yr,N_EPOCHS,ewc=ewc)
        ewc.register(X_tr,y_tr,i)
        n_keep = max(4, int(rho*len(X_tr)))
        idx = np.random.choice(len(X_tr),n_keep,replace=False)
        reh_X.append(X_tr[idx]); reh_y.append(y_tr[idx])
        for j in range(T):
            model.set_task(j); acc[i,j] = eval_model(model,*tasks_te[j])

    T2 = acc.shape[0]
    AA  = float(np.mean(acc[T2-1,:T2]))
    F_v = [max(0.,float(np.max(acc[:,j]))-acc[T2-1,j]) for j in range(T2-1)]
    F   = float(np.mean(F_v))
    BWT = float(np.mean([acc[T2-1,j]-acc[j,j] for j in range(T2-1)]))
    return {"AA":AA,"F":F,"BWT":BWT}


def run_classical_ewc(tasks_tr, tasks_te, seed, lam=500., rho=0.) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    T   = len(tasks_tr)
    acc = np.zeros((T,T))
    model = ClassicalModel().to(DEVICE)
    ewc   = EWC(model, lam)
    reh_X, reh_y = [], []
    for i,(X_tr,y_tr) in enumerate(tasks_tr):
        if reh_X and rho>0:
            Xr=torch.cat([X_tr]+reh_X); yr=torch.cat([y_tr]+reh_y)
        else: Xr,yr=X_tr,y_tr
        train_task(model,Xr,yr,N_EPOCHS,ewc=ewc)
        ewc.register(X_tr,y_tr,i)
        if rho>0:
            n_keep=max(4,int(rho*len(X_tr))); idx=np.random.choice(len(X_tr),n_keep,replace=False)
            reh_X.append(X_tr[idx]); reh_y.append(y_tr[idx])
        for j in range(T): acc[i,j]=eval_model(model,*tasks_te[j])
    T2=acc.shape[0]
    AA = float(np.mean(acc[T2-1,:T2]))
    F  = float(np.mean([max(0.,float(np.max(acc[:,j]))-acc[T2-1,j]) for j in range(T2-1)]))
    BWT= float(np.mean([acc[T2-1,j]-acc[j,j] for j in range(T2-1)]))
    return {"AA":AA,"F":F,"BWT":BWT}


# ─── Ablation 1: Lambda EWC ───────────────────────────────────────────────────
def ablation_lambda():
    print("\n── Ablation 1: λ_EWC ──")
    lambdas = [50, 100, 200, 500, 1000, 2000]
    results = {"QTCL-Q":[], "Classical-EWC":[]}
    for lam in lambdas:
        print(f"  λ={lam}...")
        aa_q, aa_c = [], []
        for seed in range(N_SEEDS_ABL):
            tasks_tr, tasks_te = load_split_mnist(seed=seed*37+5)
            r_q = run_qtcl(tasks_tr, tasks_te, seed, lam=lam)
            r_c = run_classical_ewc(tasks_tr, tasks_te, seed, lam=lam, rho=0.25)
            aa_q.append(r_q["AA"]); aa_c.append(r_c["AA"])
        results["QTCL-Q"].append((lam, np.mean(aa_q), np.std(aa_q)))
        results["Classical-EWC"].append((lam, np.mean(aa_c), np.std(aa_c)))

    fig, ax = plt.subplots(figsize=(8,5))
    for name, color in [("QTCL-Q","#0D47A1"),("Classical-EWC","#FF7043")]:
        xs = [r[0] for r in results[name]]
        ys = [r[1] for r in results[name]]
        es = [r[2]*1.96 for r in results[name]]
        ax.errorbar(xs, ys, yerr=es, marker="o", lw=2, capsize=5,
                    label=name, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("λ_EWC", fontsize=11)
    ax.set_ylabel("Average Accuracy", fontsize=11)
    ax.set_title("Ablation: Effect of EWC Regularization Strength",
                 fontweight="bold", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"ablation_lambda.{ext}", bbox_inches="tight")
    plt.close()
    print("  → ablation_lambda")
    return results


# ─── Ablation 2: Rehearsal ratio ──────────────────────────────────────────────
def ablation_rehearsal():
    print("\n── Ablation 2: Rehearsal ratio ──")
    ratios = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    results = []
    for rho in ratios:
        print(f"  ρ={rho}...")
        aa_list = []
        for seed in range(N_SEEDS_ABL):
            tasks_tr, tasks_te = load_split_mnist(seed=seed*37+5)
            r = run_qtcl(tasks_tr, tasks_te, seed, rho=rho)
            aa_list.append(r["AA"])
        results.append((rho, np.mean(aa_list), np.std(aa_list)))

    fig, ax = plt.subplots(figsize=(8,5))
    xs=[r[0] for r in results]; ys=[r[1] for r in results]; es=[r[2]*1.96 for r in results]
    ax.errorbar(xs, ys, yerr=es, marker="o", lw=2, capsize=5, color="#0D47A1")
    ax.axvline(0.25, color="red", ls="--", lw=1.2, label="default ρ=0.25")
    ax.set_xlabel("Rehearsal ratio ρ", fontsize=11)
    ax.set_ylabel("Average Accuracy", fontsize=11)
    ax.set_title("Ablation: Effect of Rehearsal Buffer Size",
                 fontweight="bold", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"ablation_rehearsal.{ext}", bbox_inches="tight")
    plt.close()
    print("  → ablation_rehearsal")
    return results


# ─── Ablation 3: Circuit depth ────────────────────────────────────────────────
def ablation_depth():
    print("\n── Ablation 3: Circuit depth (n_shared_layers) ──")
    depths = [1, 2, 3, 4]
    results = []
    for d in depths:
        print(f"  n_shared={d}...")
        aa_list = []
        for seed in range(N_SEEDS_ABL):
            tasks_tr, tasks_te = load_split_mnist(seed=seed*37+5)
            r = run_qtcl(tasks_tr, tasks_te, seed, n_shared=d)
            aa_list.append(r["AA"])
        results.append((d, np.mean(aa_list), np.std(aa_list)))
        print(f"    AA={np.mean(aa_list):.4f}±{np.std(aa_list):.4f}")

    fig, ax = plt.subplots(figsize=(7,5))
    xs=[r[0] for r in results]; ys=[r[1] for r in results]; es=[r[2]*1.96 for r in results]
    ax.errorbar(xs, ys, yerr=es, marker="o", lw=2, capsize=5, color="#0D47A1")
    ax.axvline(2, color="red", ls="--", lw=1.2, label="default depth=2")
    ax.set_xlabel("Number of shared StronglyEntanglingLayers", fontsize=11)
    ax.set_ylabel("Average Accuracy", fontsize=11)
    ax.set_title("Ablation: Circuit Depth vs Accuracy\n(barren plateau effect)",
                 fontweight="bold", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"ablation_depth.{ext}", bbox_inches="tight")
    plt.close()
    print("  → ablation_depth")
    return results


# ─── Ablation 4: Qubit count ──────────────────────────────────────────────────
def ablation_qubits():
    print("\n── Ablation 4: Qubit count ──")
    qubit_counts = [2, 4, 6]
    results = []
    for nq in qubit_counts:
        print(f"  n_qubits={nq}...")
        aa_list = []
        for seed in range(N_SEEDS_ABL):
            tasks_tr, tasks_te = load_split_mnist(seed=seed*37+5)
            r = run_qtcl(tasks_tr, tasks_te, seed, n_qubits=nq)
            aa_list.append(r["AA"])
        results.append((nq, np.mean(aa_list), np.std(aa_list)))
        print(f"    AA={np.mean(aa_list):.4f}±{np.std(aa_list):.4f}")

    fig, ax = plt.subplots(figsize=(7,5))
    xs=[r[0] for r in results]; ys=[r[1] for r in results]; es=[r[2]*1.96 for r in results]
    ax.bar(xs, ys, width=0.8, color=["#90CAF9","#0D47A1","#1B5E20"],
           alpha=0.85, edgecolor="white")
    ax.errorbar(xs, ys, yerr=es, fmt="none", color="black", capsize=6, lw=1.5)
    ax.set_xlabel("Number of qubits", fontsize=11)
    ax.set_ylabel("Average Accuracy", fontsize=11)
    ax.set_title("Ablation: Qubit Count vs Accuracy",
                 fontweight="bold", fontsize=12)
    ax.set_xticks(xs); ax.grid(True, alpha=0.4, axis="y")
    plt.tight_layout()
    for ext in ("pdf","png"):
        plt.savefig(FIGURES_DIR/f"ablation_qubits.{ext}", bbox_inches="tight")
    plt.close()
    print("  → ablation_qubits")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print("QTCL Ablation Study — Split-MNIST")
    print(f"{'='*65}")
    print(f"Seeds per config: {N_SEEDS_ABL} | Epochs: {N_EPOCHS}/task")
    print(f"{'='*65}")

    results = {}
    results["lambda"]    = ablation_lambda()
    results["rehearsal"] = ablation_rehearsal()
    results["depth"]     = ablation_depth()
    results["qubits"]    = ablation_qubits()

    # Save results
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {m: [(float(x),float(y),float(s)) for x,y,s in lst]
                                for m,lst in v.items()}
        else:
            serializable[k] = [(float(x),float(y),float(s)) for x,y,s in v]

    with open("results/ablation.json","w") as f:
        json.dump(serializable, f, indent=2)
    print("\n→ results_ablation.json")

    # Summary
    print(f"\n{'='*65}")
    print("ABLATION SUMMARY")
    print(f"{'='*65}")
    best_lam = max(results["lambda"]["QTCL-Q"], key=lambda r: r[1])
    best_rho = max(results["rehearsal"], key=lambda r: r[1])
    best_dep = max(results["depth"], key=lambda r: r[1])
    best_q   = max(results["qubits"], key=lambda r: r[1])
    print(f"  Best λ_EWC:         {best_lam[0]}  (AA={best_lam[1]:.4f})")
    print(f"  Best rehearsal ρ:   {best_rho[0]}  (AA={best_rho[1]:.4f})")
    print(f"  Best circuit depth: {best_dep[0]}  (AA={best_dep[1]:.4f})")
    print(f"  Best qubit count:   {int(best_q[0])}  (AA={best_q[1]:.4f})")


if __name__ == "__main__":
    main()
