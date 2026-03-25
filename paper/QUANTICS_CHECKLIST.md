# QUANTICS 2026 — Submission Checklist

**Congreso:** QUANTICS 2026, Porto, 16-18 Julio 2026
**Serie:** Springer CCIS (Communications in Computer and Information Science)
**Deadline:** 16 Abril 2026
**Límite:** 8 páginas máximo
**Archivo:** `quantics_main.tex`

---

## Qué se incluyó vs. qué se omitió

### Incluido

| Sección | Estado | Notas |
|---------|--------|-------|
| Abstract (≤150 palabras) | OK | Contiene claim AA=0.849 y +21pp |
| Introduction (~0.5 pág) | OK | Motivación + contribuciones numeradas |
| Background: MAML, EWC, QFIM | OK | 1 párrafo cada uno |
| Method: arquitectura VQC | OK | Encoder MLP + $R_Y$ encoding + parámetros compartidos/específicos |
| Method: Fisher EWC híbrido | OK | Ecuación FIM, diferenciación adjunta |
| Algorithm (pseudocódigo) | OK | Versión compacta de 5 líneas |
| Experiments: setup | OK | Split-MNIST, 5 métodos, métricas AA/F/BWT/FWT |
| Tabla de resultados principales | OK | Solo PennyLane (Qiskit pendiente) |
| Figura cl_metrics.pdf | OK | Barras de error 95% CI |
| Figura accuracy_matrix.pdf | OK | Comparativa Classical EWC vs QTCL |
| Ablation: λ (lambda) | OK | Figura ablation_lambda.pdf |
| Ablation: rehearsal ρ | OK | Figura ablation_rehearsal.pdf |
| Conclusions + future work | OK | Claim central + extensiones |
| Referencias (10 clave) | OK | Ver lista abajo |

### Omitido (respecto al paper original)

| Sección omitida | Razón |
|-----------------|-------|
| Resultados Qiskit | Pendientes; añadir en versión final si hay tiempo |
| Sección IBM Real Hardware (§5 original) | TODO pendiente, no hay resultados |
| Análisis teórico extenso (barren plateaus detallado) | Espacio |
| Subsección PennyLane vs. Qiskit | Sin resultados Qiskit completos |
| Ablation: circuit depth ($L_s$) | Omitida para ganar espacio |
| Ablation: qubit count ($n_q$) | Omitida para ganar espacio |
| Sección Discussion (§7 original) | Resumida en párrafos de resultados |
| Sección Related Work (§6 original) | Integrada brevemente en Introduction |
| Apéndices (reproducibilidad, detalles circuito) | No caben en 8 páginas |
| Nombres de autores e institución | Double-blind |

---

## Instrucciones para formato INSTICC / Springer CCIS (llncs)

### Paso 1: Cambiar la clase de documento

```latex
% Reemplazar:
\documentclass[a4paper,12pt]{article}
\usepackage[margin=2.5cm]{geometry}

% Por:
\documentclass{llncs}
```

Descargar `llncs.cls` de: https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines

### Paso 2: Ajustar autor e institución (camera-ready, post-aceptación)

```latex
\author{Daniel Martínez Pérez}
\institute{Quantum Machine Learning Group, Universidad Pablo de Olavide,
           Sevilla, Spain\\
           \email{dmarper2@upo.es}}
```

Para la **submisión double-blind**, mantener:
```latex
\author{Anonymous Author(s)}
\institute{}
```

### Paso 3: Formato de la bibliografía

CCIS usa `\bibliographystyle{splncs04}` con archivo `.bib`. Para camera-ready:
1. Mover las referencias del entorno `thebibliography` manual al archivo `references.bib`
2. Sustituir por:
```latex
\bibliographystyle{splncs04}
\bibliography{references}
```

### Paso 4: Verificar límite de páginas

Con `llncs`, los márgenes son más ajustados (paper A4, fuente 10pt). El documento
actual en `article` con 12pt ocupa ~7-8 páginas. Al cambiar a `llncs` con 10pt
puede comprimirse a 6-7 páginas, dejando margen para referencias más largas.

### Paso 5: Keywords en formato CCIS

```latex
\keywords{Quantum Machine Learning \and Continual Learning \and
          Variational Quantum Circuits \and Elastic Weight Consolidation \and
          Barren Plateaus \and NISQ}
```

### Paso 6: Eliminar packages incompatibles con llncs

`llncs` puede entrar en conflicto con algunos packages. Verificar:
- `geometry` — eliminar (llncs gestiona márgenes)
- `hyperref` — cargar al final, con `\usepackage[hidelinks]{hyperref}`
- `natbib` — sustituir por `splncs04`; en llncs se usa `\cite{}` estándar

---

## Figuras a usar

| Figura | Archivo | Sección | Prioridad |
|--------|---------|---------|-----------|
| Arquitectura QTCL | `architecture.pdf` | Method | ALTA — incluida |
| Métricas CL (barras de error) | `cl_metrics.pdf` | Results | ALTA — incluida |
| Matrices de accuracy | `accuracy_matrix.pdf` | Results | ALTA — incluida |
| Ablation lambda | `ablation_lambda.pdf` | Ablation | ALTA — incluida |
| Ablation rehearsal | `ablation_rehearsal.pdf` | Ablation | ALTA — incluida |
| Circuito VQC | `vqc_circuit.pdf` | Method (opcional) | MEDIA — omitida para espacio |
| Evolución accuracy | `acc_evolution.pdf` | Results (opcional) | MEDIA — omitida |
| Forgetting per-task | `forgetting.pdf` | Results (opcional) | MEDIA — omitida |
| PennyLane vs Qiskit | `pennylane_vs_qiskit.pdf` | — | NO — resultados pendientes |
| Qiskit metrics | `qiskit_cl_metrics.pdf` | — | NO — resultados pendientes |
| IBM hardware | — | — | NO — no ejecutado |

**Si hay espacio sobrante** al compilar con `llncs`, añadir `acc_evolution.pdf`
en la sección de resultados (entre la tabla y la figura de accuracy matrix).

---

## Claim central a preservar

> QTCL logra **AA = 0.849 ± 0.018** y **F = 0.130 ± 0.002** en Split-MNIST (PennyLane),
> frente al mejor baseline clásico (Classical EWC: AA = 0.636, F = 0.426):
> **+21.3 pp de accuracy** y reducción de olvido **3.3×** (3 seeds, 95% CI).

Este claim aparece en el abstract, en la tabla principal, y en las conclusiones.
No modificar estos números al ajustar el formato.

---

## Ventaja cuántica real a enfatizar

La QFIM se calcula **exactamente** vía diferenciación adjunta (PennyLane
`lightning.qubit`), no por aproximación finita. Esto significa:
- El regularizador EWC opera sobre una cantidad geométrica intrínseca al espacio
  de parámetros cuántico (Quantum Geometric Tensor).
- No es una mera adaptación heurística: la curvatura que penaliza EWC es la
  curvatura real del paisaje cuántico de log-verosimilitud.

Este punto está en el Background (§2) y en la Introduction (contribución 2).
Asegurarse de que survives al ajuste de formato.
