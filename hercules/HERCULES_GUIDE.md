# Guía de lanzamiento en Hércules (CICA)

Guía paso a paso para lanzar el experimento QUANTICS en el clúster Hércules del CICA.

---

## 1. Acceso al clúster

### 1.1. Conectar a la VPN

Antes de cualquier conexión, hay que activar la VPN FortiClient con las credenciales
proporcionadas por el CICA al registrarse. Sin VPN, el SSH no funciona.

### 1.2. Conectar por SSH

```bash
ssh usuario@login.spc.cica.es
```

Una vez dentro, el directorio de trabajo es `$HOME`. El sistema de ficheros es Lustre
con **1 TB de cuota** y **sin backup** — hacer copias locales de los resultados
cuando termine cada experimento.

### 1.3. Portal web (alternativa)

También se puede acceder vía navegador:
`https://ood-hercules.spc.cica.es`

---

## 2. Preparar el entorno

### 2.1. Subir el código

Desde la máquina local:

```bash
scp -r ~/qtcl-paper/ usuario@login.spc.cica.es:~/
```

O con rsync para sincronizaciones posteriores:

```bash
rsync -avz --exclude='.git' ~/qtcl-paper/ usuario@login.spc.cica.es:~/qtcl-paper/
```

### 2.2. Ver módulos disponibles

```bash
module avail                        # lista todos los módulos
module spider Python                # busca versiones de Python
module load Python/3.11.5-GCCcore-13.2.0
```

### 2.3. Crear el entorno virtual (solo la primera vez)

```bash
cd ~/qtcl-paper
python -m venv .venv
source .venv/bin/activate
pip install pennylane torch scikit-learn matplotlib numpy
```

Verificar que PennyLane funciona:

```bash
python -c "import pennylane as qml; print(qml.version())"
```

### 2.4. Crear el directorio de resultados

```bash
mkdir -p ~/qtcl-paper/results
```

---

## 3. Lanzar los experimentos

Hay dos modos: **job único** (más sencillo) o **array de jobs** (más rápido, 3 seeds en paralelo).

### Opción A — Job único (recomendado para pruebas)

Ejecuta main → ablation_lambda → ablation_rehearsal → figuras de forma secuencial en un solo nodo.

```bash
cd ~/qtcl-paper
sbatch hercules/submit_quantics_main.slurm
```

Recursos solicitados: 48h, 24 CPUs, 48 GB RAM, partición `standard`.

### Opción B — Array de jobs (recomendado para resultados finales)

Lanza los 3 seeds en paralelo. Cada seed tarda ~16h con 16 CPUs y 32 GB.

```bash
cd ~/qtcl-paper
sbatch hercules/submit_quantics_array.slurm
```

Cuando los 3 jobs terminen, agregar los resultados:

```bash
cd ~/qtcl-paper
python hercules/aggregate_results.py
```

Esto genera `results/quantics_main.json` y `results/quantics_ablation.json`.

---

## 4. Monitorizar los jobs

### Ver estado de los jobs propios

```bash
squeue -u $USER
```

Estados relevantes:
- `PD` — en cola (Pending)
- `R` — ejecutándose (Running)
- `CD` — completado (Completed)
- `CA` — cancelado (Cancelled)
- `F` — fallido (Failed)
- `TO` — tiempo agotado (Timeout)

Para un array job, los jobs aparecen como `JOBID_0`, `JOBID_1`, `JOBID_2`.

### Ver nodos disponibles

```bash
sinfo -s                            # resumen de particiones
sinfo -p standard -t idle           # nodos libres en standard
sinfo -p gpu                        # estado de la partición GPU
```

### Ver historial de jobs

```bash
sacct -u $USER -S $(date +%Y-%m-%d) --format=jobid,jobname,state,elapsed,ncpus
```

---

## 5. Revisar los logs

Los archivos de salida y error se guardan en `results/`:

```
results/
├── quantics_main_%j.out     # salida estándar (job único)
├── quantics_main_%j.err     # errores (job único)
├── quantics_seed_0_%j.out   # salida seed 0 (array)
├── quantics_seed_0_%j.err   # errores seed 0 (array)
├── ...
├── seed_0_main.json         # resultados seed 0
├── seed_1_main.json         # resultados seed 1
├── seed_2_main.json         # resultados seed 2
├── quantics_main.json       # resultados agregados (tras aggregate_results.py)
└── quantics_ablation.json   # ablación agregada
```

Seguir el progreso en tiempo real:

```bash
tail -f results/quantics_main_<JOBID>.out
```

### Cancelar un job

```bash
scancel <JOBID>             # cancelar job concreto
scancel -u $USER            # cancelar todos los jobs propios
```

---

## 6. Sesión interactiva (para depurar)

Para probar el código antes de lanzar el job completo, abrir una sesión interactiva
(máximo 24 horas):

```bash
salloc --mem=16G -c 4 -t 06:00:00 -p standard srun --pty /bin/bash -i
```

Dentro de la sesión:

```bash
module load Python/3.11.5-GCCcore-13.2.0
source ~/qtcl-paper/.venv/bin/activate
cd ~/qtcl-paper
python code/qtcl_quantics_experiment.py --mode main --seed 0
```

---

## 7. Bajar los resultados

Cuando los jobs terminen, copiar los resultados a la máquina local:

```bash
scp -r usuario@login.spc.cica.es:~/qtcl-paper/results/ ~/qtcl-paper/results/
```

---

## 8. Resumen de comandos

| Acción | Comando |
|--------|---------|
| Conectar | `ssh usuario@login.spc.cica.es` |
| Ver módulos Python | `module spider Python` |
| Cargar Python | `module load Python/3.11.5-GCCcore-13.2.0` |
| Lanzar job único | `sbatch hercules/submit_quantics_main.slurm` |
| Lanzar array | `sbatch hercules/submit_quantics_array.slurm` |
| Ver estado | `squeue -u $USER` |
| Ver nodos libres | `sinfo -p standard -t idle` |
| Seguir log | `tail -f results/*.out` |
| Cancelar | `scancel <JOBID>` |
| Agregar resultados | `python hercules/aggregate_results.py` |
| Bajar resultados | `scp -r usuario@login... ~/qtcl-paper/results/` |

---

## 9. Soporte

Cualquier problema con el clúster: **soporte@cica.es**
