# Icon Detection and Clustering Model
This is an icon detection and classification model. It can take in a website screenshot and output an embeddings of the topics of icons present in the website. 
## FLOWCHART OF DATA + FILES

```mermaid
flowchart TD
    C[/"single photo<br/>(+uuid, country, category)"/] --> P
    P["process_image.py<br/>(OmniParser: YOLO + Florence-2)"] --> BI[/"annotated image"/]
    P --> AD[/"all_detections.json"/]

    AD --> CR["updated_icon_crop.py<br/>(icon_crops)"]
    C --> CR
    CR --> CROPS[/"icon crops"/]
    CROPS --> RN["infer.py<br/>(ResNet50)"]
    RN --> PRED[/"predictions.json"/]

    AD --> CF["flatten_dataset.py<br/>(flatten)"]
    CF --> FLAT[/"captions_flat.csv"/]
    FLAT --> AWS["translation_aws.py<br/>(translate — AWS Bedrock, optional)"]
    AWS --> TRANS[/"captions_translated.csv"/]

    PRED --> BJ["build_joined_filtered.py<br/>(join)"]
    TRANS --> BJ
    AD --> BJ
    BJ --> JF[/"joined_filtered.json + .csv"/]
    JF --> BT["bertopic_infer.py<br/>(BERTopic)"]
    BT --> TOP[/"topics.csv"/]
    TOP --> FC["fill_category.py"]
    FC --> TWC[/"topics_with_categories.csv"/]
    TWC --> CE["build_embeddings.py<br/>(count_embeddings)"]
    CE --> EMB[/"embeddings_og.csv"/]

    classDef script fill:#cce5cc,stroke:#5a8a5a,color:#1b3b1b;
    classDef data fill:#cce0f0,stroke:#4a7aa5,color:#143047;
    class P,CR,RN,CF,AWS,BJ,BT,FC,CE script;
    class C,BI,AD,CROPS,PRED,FLAT,TRANS,JF,TOP,TWC,EMB data;
```

### Omniparser
We ran Microsoft OmniParser v2 (YOLO detector + finetuned Florence-2 captioner) on each screenshot. Omniparser is a vision language model that produces per-image bounding boxes, semantic captions, and annotated images. See `omniparser/README.md` for setup, weights download, and the flash_attn and `util/utils.py` patches required.

### RESNET50

We cropped each detected icon region from the source screenshots (`updated_icon_crop.py`) and classifies them with a finetuned ResNet50 (`infer.py`). This produces per-icon class predictions.

### BERTOPIC

We used the classifications from Resnet50 and the captions from Omniparser as test input into BERTopic to produce Topic clusters for icons.

## Running the pipeline

`run_pipeline.py` runs the whole flow for a **single photo**. Everything is configured in `pipeline_config.json`.

1. Each stage runs in its own conda env. Create them once by following `ENVIRONMENTS.md`.
2. Edit `pipeline_config.json`: fill in the `python` path for each env (under `environments`), the `single_image` block (see below), and your output folder. Also set `paths.omniparser_repo` to your cloned OmniParser directory (so stage 1 can find `util.utils`) and `params.device` to the torch device for the captioner.  More details in `ENVIRONMENTS.md`.
3. Model paths need to be imported under `Models/`. [TODO: Add public links somewhere?]
4. Run it:
   ```bash
   python run_pipeline.py
   ```

Each stage has an on/off switch under `stages` in the config. Turn a stage off to skip it and reuse existing output. For example, you can set `omniparser` and `translate` to `false` if you already have `all_detections.json` and `captions_translated.csv`.

### Input: a single photo

Fill in the `single_image` block in `pipeline_config.json` with the photo path plus its metadata:

```json
"single_image": {
  "image": "/path/to/shot.png",
  "uuid": "abc123",
  "country": "ar",
  "category": "Business and Finance"
}
```

`uuid` is the join key / crop-filename prefix; `country` and `category` are tags that ride into the outputs. The OmniParser stage script (`omniparser/process_image.py`) can also be run directly:

```bash
python omniparser/process_image.py --image shot.png \
  --uuid abc123 --country ar --category "Business and Finance" \
  --output-dir OUT --yolo-weights .../model.pt --florence-weights .../icon_caption_florence
```

### Calling it from your own Python code

`run_pipeline.py` has a `run_pipeline(...)` function, so another program can run the whole pipeline for one image and get the result back. It runs each stage as a subprocess in that stage's conda env. To run it, you need to setup the envs, OmniParser repo/weights, and a filled-in `pipeline_config.json` on the machine first (see setup above and `ENVIRONMENTS.md`).

If the config is filled in, the call takes no arguments and uses `pipeline_config.json`:

```python
import sys
sys.path.insert(0, "/path/to/infographic_model")   # so the import resolves
from run_pipeline import run_pipeline

out = run_pipeline()

# topic embedding
print(out["embeddings_og"])
```
Every argument is optional and overrides the corresponding config value for that call, so you can process many images without editing the config each time:

```python
out = run_pipeline(
    image="/data/shot.png",
    uuid="abc123",
    country="ar",
    category="Business and Finance",
    work_dir="/tmp/run_abc123",          
    device="cpu",                        # overrides params.device; if none,  use the config value
    omniparser_cwd="/path/to/OmniParser",# overrides paths.omniparser_repo; if none, use use the config value
    stages={"translate": False},         # skip any stage, e.g. translate if no AWS access
)
```

`python run_pipeline.py` still runs from the config as before.
