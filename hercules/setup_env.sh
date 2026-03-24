#!/bin/bash
# Setup del entorno QTCL en Hercules
# Ejecutar UNA VEZ antes del primer sbatch:
#   bash hercules/setup_env.sh

set -e

echo "=== Configurando entorno QTCL en Hercules ==="

# Módulos
module purge
module load python/3.10
module load miniconda/23.5

# Crear entorno conda
conda create -n qtcl-env python=3.10 -y
source activate qtcl-env

# Instalar dependencias
pip install --upgrade pip
pip install pennylane pennylane-lightning \
            torch torchvision \
            qiskit==2.3.1 qiskit-machine-learning==0.9.0 \
            scikit-learn matplotlib seaborn pandas

# Verificar
python -c "
import pennylane, torch, qiskit
print('PennyLane:', pennylane.__version__)
print('PyTorch:  ', torch.__version__)
print('Qiskit:   ', qiskit.__version__)
print('OK — entorno listo')
"

# Pre-descargar MNIST
python -c "
import torchvision
torchvision.datasets.MNIST(root='data/', train=True,  download=True)
torchvision.datasets.MNIST(root='data/', train=False, download=True)
print('MNIST descargado')
"

echo ""
echo "=== Setup completo ==="
echo "Lanzar experimento completo:  sbatch hercules/submit_qtcl.slurm"
echo "Lanzar por seeds en paralelo: sbatch hercules/submit_array.slurm"
