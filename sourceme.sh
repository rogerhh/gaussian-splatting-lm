#!/bin/bash

conda activate gaussian_splatting-jvp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export CUDA_PATH=/usr/local/cuda-12.6
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/bin/lib64:$LD_LIBRARY_PATH"

if [ -z "$CONDA_PREFIX" ]; then
    echo "No conda environment found. Please activate a conda environment."
    return
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

export QT_QPA_PLATFORM=offscreen

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
