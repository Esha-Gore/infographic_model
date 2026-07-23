#!/usr/bin/env bash
#
# Portable conda-env setup for the icon pipeline. One script, one env per stage.
# Works on macOS and Linux with no hardcoded paths:
# it reuses whatever conda is already installed and auto-detects the torch build.
#
# Usage:
#   bash setup_env.sh <stage>       # stage: omniparser | resnet | translate | bertopic
#   bash setup_env.sh all           # build all four
#

set -eo pipefail

STAGE="${1:-}"
PYVER="${PYVER:-3.10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$STAGE" ]; then
  echo "usage: bash setup_env.sh <omniparser|resnet|translate|bertopic|all>" >&2
  exit 2
fi

# where is conda?
if [ -z "${CONDA_BASE:-}" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  elif [ -n "${CONDA_EXE:-}" ]; then
    CONDA_BASE="$("$CONDA_EXE" info --base)"
  else
    echo "ERROR: conda not found. Install Miniconda/Anaconda first, or set CONDA_BASE." >&2
    exit 1
  fi
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# pick the torch build
TORCH_VARIANT="${TORCH_VARIANT:-auto}"
if [ "$TORCH_VARIANT" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then TORCH_VARIANT="cu118"; else TORCH_VARIANT="cpu"; fi
fi

install_torch() {
  if [ "$TORCH_VARIANT" = "cu118" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  else
    pip install torch torchvision           # default wheels: macOS MPS/CPU or Linux CPU
  fi
}

# build one env 
build() {
  local stage="$1" default_name="$2"
  local env_name="${ENV_NAME:-$default_name}"
  echo ">>> [$stage] env='$env_name'  python=$PYVER  torch=$TORCH_VARIANT  conda_base=$CONDA_BASE"

  conda create -n "$env_name" "python=$PYVER" -y
  conda activate "$env_name"
  python -m pip install --upgrade pip >/dev/null

  case "$stage" in
    omniparser)
      install_torch
      pip install "transformers==4.38.2" accelerate pillow einops timm scipy matplotlib ultralytics "supervision==0.18.0" pandas
      pip install --force-reinstall "numpy==1.26.4" "opencv-python-headless==4.6.0.66"
      python -c "import torch,transformers,ultralytics,cv2,numpy; print('OK omniparser | torch',torch.__version__,'| transformers',transformers.__version__,'| numpy',numpy.__version__,'| cv2',cv2.__version__)"
      ;;
    resnet)
      install_torch
      pip install -r "$SCRIPT_DIR/requirements.txt"
      pip install tqdm
      python -c "import torch,torchvision,PIL,tqdm; print('OK resnet | torch',torch.__version__)"
      ;;
    translate)
      pip install pandas boto3 botocore langdetect     # no torch; AWS creds needed only to RUN the translate stage
      python -c "import pandas,boto3,langdetect; print('OK translate')"
      ;;
    bertopic)
      install_torch
      pip install -r "$SCRIPT_DIR/BERTopic/requirements.txt"     # matches the saved model
      python -c "import bertopic,sentence_transformers,sklearn,scipy; print('OK bertopic | bertopic',bertopic.__version__,'| sklearn',sklearn.__version__)"
      ;;
    *)
      echo "ERROR: unknown stage '$stage'" >&2; exit 2 ;;
  esac

  conda deactivate
  echo "    -> paste into pipeline_config.json  environments.$stage :"
  echo "       \"$CONDA_BASE/envs/$env_name/bin/python\""
  echo
}

case "$STAGE" in
  omniparser) build omniparser omniparser ;;
  resnet)     build resnet     resnet_env ;;
  translate)  build translate  translate_env ;;
  bertopic)   build bertopic   bertopic_env ;;
  all)
    unset ENV_NAME
    build omniparser omniparser
    build resnet     resnet_env
    build translate  translate_env
    build bertopic   bertopic_env
    ;;
  *) echo "usage: bash setup_env.sh <omniparser|resnet|translate|bertopic|all>" >&2; exit 2 ;;
esac

echo "Done :)"
