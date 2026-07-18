# Pipeline environments

`run_pipeline.py` does not create any conda environments. Each stage is launched with its env's python, so the envs stay
isolated as Omniparser has issues with flash-attn and BERTopic is sensitive to the exact versions of packages. Thus, you must create each env and fill in the paths into `pipeline_config.json` then run the `run_pipeline.py` script.

The full pipeline has 4 envs, if you only need to run a certain stage you may only want to create a selection. Create each like this:

## 1. Omniparser
Runs: `omniparser/process_dataset_with_csv.py` (stage: omniparser)

```bash
bash omniparser/setup_env_two.sh   
```
Skip this env if you already have `all_detections.json` (set `stages.omniparser=false`).


## 2. Resnet
Runs: `resnet_inference_processing/updated_icon_crop.py` (icon_crops),
`resnet_inference_processing/infer.py` (infer)

Needs torch + image libs. `requirements.txt` at the repo root covers it:
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

Lightweight pandas + AWS glue. No torch needed.
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
bash BERTopic/bert_env.sh               
pip install scipy                        # count_embeddings.py uses scipy
```
