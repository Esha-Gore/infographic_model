# OmniParser pipeline for UI screenshot captioning

This subfolder contains a pipeline that runs Microsoft's [OmniParser v2](https://huggingface.co/microsoft/OmniParser-v2.0) on a dataset of website screenshots to extract bounding boxes and functional captions for UI elements (buttons, icons, menus, etc.).

## What this does

`process_dataset_with_csv.py` reads a CSV of image UUIDs grouped by country, runs each screenshot through OmniParser's YOLO detector + Florence-2 captioner, and writes per-country detection results plus annotated images.

## Setup

This was developed on UW's Klone HPC cluster (SLURM + GPFS). Paths in the scripts are absolute to that environment; adjust for your system.

### 1. Clone OmniParser

git clone https://github.com/microsoft/OmniParser.git

### 2. Download the model weights

You need **two sets of weights**, both from the OmniParser v2 release. The fine-tuned Florence-2 captioner is the critical one — base Florence-2 produces generic photo captions ("a person with a beard") instead of UI descriptions.

```bash
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do
  huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done
mv weights/icon_caption weights/icon_caption_florence
```

The fine-tuned captioner folder should have exactly three files: `config.json`, `generation_config.json`, `model.safetensors`. If you see `.py` files, tokenizer files, or a README, you have the base Florence-2 model, not the fine-tuned one.

### 3. Build the conda environment

```bash
bash setup_env.sh
```

This installs torch, transformers, Florence-2 dependencies, YOLO (ultralytics), and a numpy/opencv version pin. **Note:** the script intentionally does *not* install paddleocr or easyocr. The pipeline as configured here passes empty OCR boxes (icon captioning only), so those packages aren't needed.

### 4. Patch out unused imports

OmniParser's `util/utils.py` imports `openai`, `easyocr`, and `paddleocr` at the top and instantiates the OCR engines on import. Since we don't use them, comment out the following lines at the top of `OmniParser/util/utils.py`:

```python
# from openai import AzureOpenAI
# import easyocr
# from paddleocr import PaddleOCR
# reader = easyocr.Reader(['en'])
# paddle_ocr = PaddleOCR(...)  # all lines of this block
```

Without this, the file fails to import even if your pipeline never calls OCR.

### 5. Patch flash_attn check (Florence-2 quirk)

Florence-2's downloaded `modeling_florence2.py` declares a hard dependency on `flash_attn`, which is painful to install on most clusters. The check is overly strict — the model has a working fallback. Two ways to handle this:

1. Pass `attn_implementation="eager"` when loading the model (already set in `get_caption_model_processor` in our setup), AND
2. Comment out the two `from flash_attn import ...` lines in the cached `modeling_florence2.py` after first run. The cached file lives at:
