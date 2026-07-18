import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT, "pipeline_config.json")) as f:
    cfg = json.load(f)

envs   = cfg["environments"]
paths  = cfg["paths"]
params = cfg["params"]
stages = cfg["stages"]

work = paths["work_dir"]
os.makedirs(work, exist_ok=True)

# resolve all file locations
omni_out       = paths["omniparser_output"]
all_detections = os.path.join(omni_out, "all_detections.json")
crops_dir      = os.path.join(work, "crops")
predictions    = os.path.join(work, "predictions.json")
captions_flat  = os.path.join(work, "captions_flat.csv")
# if translate use translated.
if stages.get("translate"):
    captions_translated = os.path.join(work, "captions_translated.csv")
else:
    captions_translated = paths["captions_translated_csv"]
joined_json = os.path.join(work, "joined_filtered.json")
joined_csv  = os.path.join(work, "joined_filtered.csv")   
topics_csv  = os.path.join(work, "topics.csv")
topics_with_cat_csv = os.path.join(work, "topics_with_categories.csv")


def run(name, python, script, script_args, cwd=None):
    if not python or "FILL_ME" in python:
        sys.exit(f"[{name}] environments not set in pipeline_config.json")
    cmd = [python, script] + [str(a) for a in script_args]
    print(f"\n[{name}]\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def need(path, stage):
    if not os.path.exists(path):
        sys.exit(f"[{stage}] missing required input: {path}")


# 1. OmniParser: screenshots + csv -> all_detections.json
if stages.get("omniparser"):
    run("omniparser", envs["omniparser"],
        os.path.join(ROOT, "omniparser", "process_dataset_with_csv.py"),
        ["--csv", paths["countries_csv"],
         "--images-dir", paths["screenshots_dir"],
         "--output-dir", omni_out,
         "--yolo-weights", params["omniparser_weights"]["yolo"],
         "--florence-weights", params["omniparser_weights"]["florence"]])

# 2. Crop icons out of the screenshots -> crops/ + manifest.csv
if stages.get("icon_crops"):
    need(all_detections, "icon_crops")
    run("icon_crops", envs["resnet"],
        os.path.join(ROOT, "resnet_inference_processing", "updated_icon_crop.py"),
        ["--json", all_detections,
         "--images_dir", paths["screenshots_dir"],
         "--output_dir", work])

# 3. ResNet50 classify crops -> predictions.json (config generated on the fly)
if stages.get("infer"):
    need(crops_dir, "infer")
    infer_cfg_path = os.path.join(work, "infer_config.json")
    with open(infer_cfg_path, "w") as f:
        json.dump({
            "model_path": paths["resnet_model"],
            "crops_dir": crops_dir,
            "output_path": predictions,
            "batch_size": params.get("infer_batch_size", 64),
        }, f, indent=2)
    run("infer", envs["resnet"],
        os.path.join(ROOT, "resnet_inference_processing", "infer.py"),
        ["--config", infer_cfg_path])

# 4. Flatten detections -> captions_flat.csv
if stages.get("flatten"):
    need(all_detections, "flatten")
    run("flatten", envs["translate"],
        os.path.join(ROOT, "omniparser", "flatten_dataset.py"),
        ["--input", all_detections, "--output", captions_flat])

# 5. Translate captions -> captions_translated.csv
if stages.get("translate"):
    need(captions_flat, "translate")
    run("translate", envs["translate"],
        os.path.join(ROOT, "omniparser", "translation_aws.py"),
        ["--input", captions_flat, "--output", captions_translated,
         "--max-spend", params.get("translate_max_spend", 5.0)])

# 6. Join predictions + detections + translations -> joined_filtered.json/.csv
if stages.get("join"):
    need(predictions, "join")
    need(all_detections, "join")
    need(captions_translated, "join")
    run("join", envs["translate"],
        os.path.join(ROOT, "resnet_inference_processing", "build_joined_filtered.py"),
        ["--predictions", predictions,
         "--detections", all_detections,
         "--translations", captions_translated,
         "--output", joined_json,
         "--threshold", params.get("confidence_threshold", 0.8)])

# 7. BERTopic assign topics -> topics.csv (final_topic column)
if stages.get("bertopic"):
    need(joined_csv, "bertopic")
    run("bertopic", envs["bertopic"],
        os.path.join(ROOT, "BERTopic", "bertopic_infer.py"),
        ["--input", joined_csv, "--model", paths["bertopic_model"], "--output", topics_csv])

# 8. Fill category from the original csv -> topics_with_categories.csv
if stages.get("fill_category"):
    need(topics_csv, "fill_category")
    need(paths["countries_csv"], "fill_category")
    run("fill_category", envs["bertopic"],
        os.path.join(ROOT, "BERTopic", "fill_category.py"),
        [topics_csv, paths["countries_csv"], topics_with_cat_csv])

# 9. Count embeddings -> embeddings_og.csv + embeddings_filtered.csv
if stages.get("count_embeddings"):
    need(topics_with_cat_csv, "count_embeddings")
    run("count_embeddings", envs["bertopic"],
        os.path.join(ROOT, "Metrics", "count_embeddings.py"),
        ["--input", topics_with_cat_csv])

print("\nPipeline complete.")
