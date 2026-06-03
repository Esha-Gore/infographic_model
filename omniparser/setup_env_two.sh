#!/bin/bash
echo "Setting up OmniParser environment"

# Persistent, shared location — every node can see this. NOT /scr.
CONDA_DIR=/gscratch/stf/eshagore/conda
cd /gscratch/stf/eshagore

export HF_HOME=/gscratch/stf/eshagore/hf_cache
export PIP_CACHE_DIR=/gscratch/stf/eshagore/.pip_cache
export CONDA_PKGS_DIRS=/gscratch/stf/eshagore/conda/pkgs
export XDG_CACHE_HOME=/gscratch/stf/eshagore/.cache

# Only build if it doesn't already exist (don't nuke it every run)
if [ ! -f "Anaconda3-2024.10-1-Linux-x86_64.sh" ]; then
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
fi
if [ ! -d "$CONDA_DIR" ]; then
    bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p ${CONDA_DIR}
fi

source ${CONDA_DIR}/etc/profile.d/conda.sh

# Only create the env if it's not already there
if [ ! -d "$CONDA_DIR/envs/omniparser" ]; then
    conda create -n omniparser python=3.10 -y
fi
conda activate omniparser

echo "Installing packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.37.2 accelerate pillow einops timm
pip install scipy matplotlib ultralytics
pip install supervision==0.18.0 openai gradio pandas

echo "Pinning numpy and opencv to compatible versions..."
pip uninstall -y numpy opencv-python opencv-python-headless opencv-contrib-python
pip install --no-cache-dir --force-reinstall "numpy==1.26.4" "opencv-python-headless==4.6.0.66"
echo "Verifying versions..."
python -c "import numpy; import cv2; print(f'numpy: {numpy.__version__}'); print(f'opencv: {cv2.__version__}')"

echo "Setup complete!"
