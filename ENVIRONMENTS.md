# Pipeline environments

`run_pipeline.py` does not create any conda environments. Each stage is launched with its env's python, so the envs stay
isolated as Omniparser has issues with flash-attn and BERTopic is sensitive to the exact versions of packages. Thus, you must create each env and fill in the paths into `pipeline_config.json` then run the `run_pipeline.py` script.

The full pipeline has 4 envs, if you only need to run a certain stage you may only want to create a selection.

To set them up, use `setup_env.sh`. This one script builds the environment for a stage. Run it once for each stage you need (see the command below). It works the same on Mac and Linux, and it picks the right PyTorch for you (GPU version if you have an NVIDIA GPU, otherwise the CPU version). When it finishes, it prints the Python path for that env and you can copy that line into `pipeline_config.json`.

```bash
bash setup_env.sh all
```
For this script to work, you need to have conda installed first. For normal use that's all. If you want to change the defaults, set any of these before the command: `ENV_NAME=<name>` to give the env a different name, `TORCH_VARIANT=cpu|cu118` to force CPU or GPU PyTorch, and `CONDA_BASE=<dir>` if the script can't find your conda. 

## 1. Omniparser
Runs: `omniparser/process_image.py` (stage: omniparser; single-image entry point used by `run_pipeline.py`)

```bash
bash setup_env.sh omniparser
```
Skip this env if you already have `all_detections.json` (set `stages.omniparser=false`).

Note that `setup_env.sh omniparser` builds the conda env, but the omniparser stage needs three more one-time steps
beyond it (the conda env alone is not enough; see `omniparser/README.md`)


## 2. Resnet
Runs: `resnet_inference_processing/updated_icon_crop.py` (icon_crops),
`resnet_inference_processing/infer.py` (infer)

Use `bash setup_env.sh resnet`, or manually:
```bash
conda create -n resnet_env python=3.10 -y
conda activate resnet_env
pip install -r requirements.txt          
pip install tqdm
```

## 3. Translate
Runs: `omniparser/flatten_dataset.py` (flatten),
`omniparser/translation_aws.py` (translate),
`resnet_inference_processing/build_joined_filtered.py` (join)

Use `bash setup_env.sh translate`, or manually:
```bash
conda create -n translate_env python=3.10 -y
conda activate translate_env
pip install pandas boto3 botocore langdetect
```
The `translate` stage calls AWS Bedrock (Claude Haiku), so this env's machine needs
AWS credentials configured. Skip that stage (`stages.translate=false`) if you already
have `captions_translated.csv`; flatten + join still run in this env.


## 4. Bertopic
Runs: `BERTopic/bertopic_infer.py` (bertopic),
`BERTopic/fill_category.py` (fill_category),
`Metrics/count_embeddings.py` (count_embeddings)

```bash
bash setup_env.sh bertopic               # installs bertopic + sentence-transformers + scipy
```
(`count_embeddings.py` needs scipy, which the script includes.)
