#!/bin/bash

echo "Setting up BERTopic environment..."

cd /scr

CONDA_DIR=/scr/eshagore_conda

# Remove existing bertopic env if it exists
if [ -d "$CONDA_DIR/envs/bertopic_env" ]; then
    rm -rf "$CONDA_DIR/envs/bertopic_env"
fi

# Download Anaconda if not already there
if [ ! -f "Anaconda3-2024.10-1-Linux-x86_64.sh" ]; then
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
fi

# Install conda if not already installed
if [ ! -d "$CONDA_DIR" ]; then
    bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p ${CONDA_DIR}
fi

source ${CONDA_DIR}/etc/profile.d/conda.sh

conda create -n bertopic_env python=3.10 -y

# Use env's pip/python directly instead of conda activate
ENV_PIP=$CONDA_DIR/envs/bertopic_env/bin/pip
ENV_PYTHON=$CONDA_DIR/envs/bertopic_env/bin/python

echo "Installing packages..."
$ENV_PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ENV_PIP install bertopic sentence-transformers plotly pandas

echo "Verifying install..."
$ENV_PYTHON -c "import bertopic; import sentence_transformers; import plotly; import pandas; print('All packages installed successfully!')"

echo "Setup complete!"
