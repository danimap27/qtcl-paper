# Quantum Transfer Continual Learning (QTCL)

Paper y código de experimentación para el framework QTCL: aprendizaje continual con circuitos variacionales cuánticos usando transferencia de parámetros, Quantum EWC y rehearsal cuántico.

## Estructura

```
qtcl-paper/
├── code/
│   └── qtcl_experiment.py   # Experimento completo (5 métodos, 4 tareas)
├── figures/                 # Generadas automáticamente por el experimento
│   ├── task_datasets.pdf
│   ├── circuit_diagram.pdf
│   ├── accuracy_matrix.pdf
│   ├── cl_metrics.pdf
│   ├── acc_evolution.pdf
│   ├── forgetting.pdf
│   ├── radar.pdf
│   └── summary_table.pdf
├── paper/
│   ├── main.tex             # Paper LaTeX (5 páginas, twocolumn)
│   ├── main.pdf             # PDF compilado
│   └── references.bib
└── results.csv              # Métricas finales en CSV
```

## Requisitos

```bash
pip install qiskit qiskit-machine-learning qiskit-aer \
            matplotlib numpy scipy scikit-learn pandas seaborn pylatexenc
```

Para compilar el PDF (solo si modificas el .tex):
```bash
sudo apt-get install texlive-latex-extra texlive-fonts-extra texlive-science latexmk
```

## Ejecutar los experimentos

```bash
python3 code/qtcl_experiment.py
```

Tiempo estimado: **~8 minutos** en CPU (Intel i5). Las figuras se guardan en `figures/`.

## Resultados

| Method | AA ↑ | BWT ↑ | FWT ↑ | Forgetting ↓ |
|--------|------|-------|-------|--------------|
| Naive FT | 0.508 | -0.078 | -0.033 | 0.078 |
| QEWC | 0.492 | -0.200 | -0.022 | 0.200 |
| QTCL (freeze) | 0.525 | -0.122 | -0.011 | 0.122 |
| **QTCL (proposed)** | **0.592** | **+0.011** | -0.022 | **0.033** |
| Classical SVM | 0.883 | -0.044 | +0.278 | 0.044 |

QTCL (proposed) es el único método cuántico con BWT positivo (sin olvido neto) y reduce el forgetting un 57% respecto al fine-tuning naive.

## Compilar el paper

```bash
cd paper/
latexmk -pdf -interaction=nonstopmode main.tex
```

## Entorno probado

- Python 3.12
- Qiskit 2.3.1
- qiskit-machine-learning 0.9.0
- qiskit-aer (simulador estacionario, sin ruido de shot)
- NumPy seed: 42
