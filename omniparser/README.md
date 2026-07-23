# OmniParser pipeline

This subfolder contains a pipeline to run Microsoft [OmniParser v2](https://huggingface.co/microsoft/OmniParser-v2.0), a YOLO icon detector combined with a fine-tuned Florence-2 captioner. The input is a website screenshot to get an output of bounding boxes and short captions for UI elements (buttons, icons, menus). This is stage 1 of the overall pipeline.

## Setup

This was originally built on UW's Klone HPC cluster, so the old scripts have cluster paths in them but we have defined a portable version version below:

### 1. Clone OmniParser

```bash
git clone https://github.com/microsoft/OmniParser.git
```

### 2. Download the weights

You need two sets of weights from the OmniParser v2 release: the YOLO icon detector and the **fine-tuned** Florence-2 captioner. (The *base* Florence-2 gives junk captions like "a person with a beard" instead of UI labels, so make sure you get the fine-tuned one.)

```bash
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do
  huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done
mv weights/icon_caption weights/icon_caption_florence
```

To ensure this worked, check that  `weights/icon_caption_florence` should have exactly three files:  `config.json`, `generation_config.json`, `model.safetensors`. If you see `.py` or tokenizer files, you downloaded the base model by mistake.

### 3. Build the conda env

Use the portable script at the repo root. It works on Mac and Linux, picks the right torch (GPU vs CPU) for you, and prints the python path to paste into `pipeline_config.json`:

```bash
bash ../setup_env.sh omniparser    
```

It installs torch, transformers, the Florence-2 dependencies, YOLO, and pinned numpy/opencv. It does not install paddleocr or easyocr as this pipeline only runs icon captioning only (no OCR), so they aren't needed.


### 4. Use our `util/utils.py`, not the fresh clone's 

A fresh clone of OmniParser **won't work for us**: upstream changed `util/utils.py` after our run, and the new version crashes on our no-OCR setup (you'll get `'NoneType' object is not iterable`). Our working copy is saved here as `og_utils.py`. After cloning, swap it in:

```bash
cp OmniParser/util/utils.py OmniParser/util/utils.py.upstream.bak   # keep the original, just in case
cp infographic_model/omniparser/og_utils.py OmniParser/util/utils.py
```

Ours differs from upstream in a few ways that matter: it handles the empty-OCR case, uses the plain YOLO detector (not upstream's newer v3 default), and keeps the eager-attention setting. It also already has the unused OCR imports (`openai`, `easyocr`, `paddleocr`) commented out.

### 5. Turn off the flash_attn requirement

Florence-2's model file (`modeling_florence2.py`, downloaded into your Hugging Face cache) lists `flash_attn` as a required import. flash_attn is hard to install and isn't available on Mac/CPU. The model's `if is_flash_attn_2_available()` check is unreliable, because transformers scans the file for imports before that check runs and errors out with:

```
ImportError: This modeling file requires the following packages that were not found in your environment: flash_attn
```

The fix we used is to comment those imports out of the cached file. The easiest way is the helper script:

```bash
python omniparser/patch_flash_attn.py          # add --check to see the status
```

It finds every cached copy (usually three) and replaces each `from flash_attn ...` line with `pass`. The model then runs fine on its normal (non-flash) attention.

Note: this file lives in the Hugging Face cache (`~/.cache/huggingface`), so if that cache is cleared or the model re-downloads, the error comes back but you can just run the script again. 

Skip this step only if flash_attn is actually installed and working.

## Running

### Single image (what `run_pipeline.py` calls)

Stage 1 is `process_image.py` (one screenshot → `all_detections.json`). Run it from **inside the OmniParser clone** so `from util.utils` resolves:

```bash
cd /path/to/OmniParser
/path/to/envs/omniparser/bin/python /path/to/infographic_model/omniparser/process_image.py \
  --image   /path/to/screenshot.png \
  --uuid    test1 --country US --category beauty \
  --output-dir /tmp/omni_test \
  --yolo-weights     /path/to/weights/icon_detect/model.pt \
  --florence-weights /path/to/weights/icon_caption_florence \
  --device cpu
```

**Set `--device`.** It defaults to `cuda`. On a Mac, use `--device cpu` (reliable). When it works you'll see `OK: <img> - N elements (0 text, N icons)` and a non-empty `all_detections.json`. For the full pipeline, set `params.device` in `pipeline_config.json` and `run_pipeline.py` passes it through.

### Batch / cluster (original SLURM workflow)

Edit `process_dataset_with_csv.py` to set `CSV_FILE` and `OUTPUT_FOLDER`, set your weight paths in the model-loading section, then submit with `sbatch run_omniparser.sh`.

## Files

- `process_image.py` — single-image inference (stage 1 of the pipeline).
- `process_dataset_with_csv.py` — original batch inference (cluster).
- `og_utils.py` — our working `util/utils.py`; swap it into the OmniParser clone (step 4).
- `patch_flash_attn.py` — comments flash_attn out of the cached Florence-2 file (step 5).
- `run_omniparser.sh` — SLURM job script (cluster).
