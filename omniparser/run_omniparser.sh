#!/bin/bash
#SBATCH --job-name=omniparser
#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/eshagore/omniparser_%j.out
#SBATCH --error=/gscratch/scrubbed/eshagore/omniparser_%j.err

source /gscratch/stf/eshagore/conda/etc/profile.d/conda.sh
conda activate omniparser

export HF_HOME=/gscratch/scrubbed/eshagore/hf_cache

cd /gscratch/stf/eshagore/OmniParser
python process_dataset_with_csv.py

echo "Processing complete!"
