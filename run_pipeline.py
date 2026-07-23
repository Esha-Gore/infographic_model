import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def _run(name, python, script, script_args, cwd=None):
    if not python or "FILL_ME" in python:
        sys.exit(f"[{name}] environments not set in pipeline_config.json")
    cmd = [python, script] + [str(a) for a in script_args]
    print(f"\n[{name}]\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def _need(path, stage):
    if not os.path.exists(path):
        sys.exit(f"[{stage}] missing required input: {path}")


def run_pipeline(image=None, uuid=None, country=None, category=None,
                 config_path=None, stages=None, work_dir=None,
                 omniparser_cwd=None, device=None):
    """
    Run the single-image icon pipeline end to end.

    This can be called from another Python program. It does NOT run in your
    interpreter's environment: each stage is launched as a subprocess using its
    own conda-env python (from the config), so your caller's environment is fully
    isolated from the pipeline's 4 envs. Those envs (+ the OmniParser repo/weights,
    a GPU, and optional AWS creds for translate) must already exist on the machine.

    Args:
        image, uuid, country, category: override the `single_image` block in the
            config for this call. Any left as None fall back to the config value.
        config_path: path to pipeline_config.json (default: next to this file).
        stages: optional dict to override individual stage on/off switches,
            e.g. {"translate": False}.
        work_dir: optional override for the work/output dir. Pass a per-call dir
            (e.g. one per uuid) if you run many images so they don't clobber
            each other.
        omniparser_cwd: optional working dir for the OmniParser stage so that
            `from util.utils` resolves — point it at the cloned OmniParser repo.
            If None, falls back to paths.omniparser_repo in the config; if that is
            unset too, the stage runs in the current working directory.
        device: torch device for the OmniParser captioner ("cpu", "mps", or "cuda").
            If None, falls back to params.device in the config; if that is unset too,
            process_image.py's own default ("cuda") applies. On a machine without an
            NVIDIA GPU (e.g. any Mac) use "cpu".

    Returns:
        dict of the output file paths for the stages that were enabled. The final
        result is under "topics_with_categories".
    """
    with open(config_path or os.path.join(ROOT, "pipeline_config.json")) as f:
        cfg = json.load(f)

    envs   = cfg["environments"]
    paths  = cfg["paths"]
    params = cfg["params"]
    stage_flags = dict(cfg["stages"])
    if stages:
        stage_flags.update(stages)

    # per-call overrides for the single image + its fields
    single = dict(cfg["single_image"])
    for key, val in (("image", image), ("uuid", uuid),
                     ("country", country), ("category", category)):
        if val is not None:
            single[key] = val

    # OmniParser stage cwd (so `from util.utils` resolves) and captioner device,
    # each overridable per-call but otherwise taken from the config.
    if omniparser_cwd is None:
        omniparser_cwd = paths.get("omniparser_repo")
    if device is None:
        device = params.get("device")

    work = work_dir or paths["work_dir"]
    os.makedirs(work, exist_ok=True)

    # The crop stage scans a directory for the source image; point it at the
    # folder holding the one photo.
    images_dir = os.path.dirname(os.path.abspath(single["image"]))

    # resolve all file locations
    omni_out       = paths["omniparser_output"]
    all_detections = os.path.join(omni_out, "all_detections.json")
    crops_dir      = os.path.join(work, "crops")
    predictions    = os.path.join(work, "predictions.json")
    captions_flat  = os.path.join(work, "captions_flat.csv")
    # if translate use translated.
    if stage_flags.get("translate"):
        captions_translated = os.path.join(work, "captions_translated.csv")
    else:
        captions_translated = paths["captions_translated_csv"]
    joined_json = os.path.join(work, "joined_filtered.json")
    joined_csv  = os.path.join(work, "joined_filtered.csv")
    topics_csv  = os.path.join(work, "topics.csv")
    topics_with_cat_csv = os.path.join(work, "topics_with_categories.csv")
    embeddings_og       = os.path.join(work, "embeddings_og.csv")

    # 1. OmniParser: single photo -> all_detections.json
    if stage_flags.get("omniparser"):
        _run("omniparser", envs["omniparser"],
             os.path.join(ROOT, "omniparser", "process_image.py"),
             ["--image", single["image"],
              "--uuid", single["uuid"],
              "--country", single["country"],
              "--category", single["category"],
              "--output-dir", omni_out,
              "--yolo-weights", params["omniparser_weights"]["yolo"],
              "--florence-weights", params["omniparser_weights"]["florence"]]
             + (["--device", device] if device else []),
             cwd=omniparser_cwd)

    # 2. Crop icons out of the screenshots -> crops/ + manifest.csv
    if stage_flags.get("icon_crops"):
        _need(all_detections, "icon_crops")
        _run("icon_crops", envs["resnet"],
             os.path.join(ROOT, "resnet_inference_processing", "updated_icon_crop.py"),
             ["--json", all_detections,
              "--images_dir", images_dir,
              "--output_dir", work])

    # 3. ResNet50 classify crops -> predictions.json
    if stage_flags.get("infer"):
        _need(crops_dir, "infer")
        infer_cfg_path = os.path.join(work, "infer_config.json")
        with open(infer_cfg_path, "w") as f:
            json.dump({
                "model_path": paths["resnet_model"],
                "crops_dir": crops_dir,
                "output_path": predictions,
                "batch_size": params.get("infer_batch_size", 64),
            }, f, indent=2)
        _run("infer", envs["resnet"],
             os.path.join(ROOT, "resnet_inference_processing", "infer.py"),
             ["--config", infer_cfg_path])

    # 4. Flatten detections -> captions_flat.csv
    if stage_flags.get("flatten"):
        _need(all_detections, "flatten")
        _run("flatten", envs["translate"],
             os.path.join(ROOT, "omniparser", "flatten_dataset.py"),
             ["--input", all_detections, "--output", captions_flat])

    # 5. Translate captions -> captions_translated.csv
    if stage_flags.get("translate"):
        _need(captions_flat, "translate")
        _run("translate", envs["translate"],
             os.path.join(ROOT, "omniparser", "translation_aws.py"),
             ["--input", captions_flat, "--output", captions_translated,
              "--max-spend", params.get("translate_max_spend", 5.0)])

    # 6. Join predictions + detections + translations -> joined_filtered.json/.csv
    if stage_flags.get("join"):
        _need(predictions, "join")
        _need(all_detections, "join")
        _need(captions_translated, "join")
        _run("join", envs["translate"],
             os.path.join(ROOT, "resnet_inference_processing", "build_joined_filtered.py"),
             ["--predictions", predictions,
              "--detections", all_detections,
              "--translations", captions_translated,
              "--output", joined_json,
              "--threshold", params.get("confidence_threshold", 0.8)])

    # 7. BERTopic assign topics -> topics.csv (final_topic column)
    if stage_flags.get("bertopic"):
        _need(joined_csv, "bertopic")
        _run("bertopic", envs["bertopic"],
             os.path.join(ROOT, "BERTopic", "bertopic_infer.py"),
             ["--input", joined_csv, "--model", paths["bertopic_model"], "--output", topics_csv])

    # 8. Fill category -> topics_with_categories.csv
    if stage_flags.get("fill_category"):
        _need(topics_csv, "fill_category")
        category_lookup = os.path.join(work, "single_meta.csv")
        with open(category_lookup, "w") as f:
            f.write("uuid,country,category\n")
            f.write(f'{single["uuid"]},{single["country"]},"{single["category"]}"\n')
        _run("fill_category", envs["bertopic"],
             os.path.join(ROOT, "BERTopic", "fill_category.py"),
             [topics_csv, category_lookup, topics_with_cat_csv])

    # 9. Build per-uuid topic-count embedding vectors -> embeddings_og.csv
    #    (single all-uuid file, no topic-count filter, matching Metrics/new.py. The
    #    distance-analysis part of the old count_embeddings stage was dropped; it
    #    needed >=2 websites and does not apply to a single image.)
    if stage_flags.get("count_embeddings"):
        _need(topics_with_cat_csv, "count_embeddings")
        _run("count_embeddings", envs["bertopic"],
             os.path.join(ROOT, "Metrics", "build_embeddings.py"),
             ["--input", topics_with_cat_csv,
              "--output", embeddings_og])

    # return the paths for whatever stages ran. embeddings_og is the end goal.
    outputs = {
        "all_detections": all_detections,
        "crops_dir": crops_dir,
        "predictions": predictions,
        "captions_flat": captions_flat,
        "joined_csv": joined_csv,
        "topics_csv": topics_csv,
        "topics_with_categories": topics_with_cat_csv,
        "embeddings_og": embeddings_og,
    }
    return {k: v for k, v in outputs.items() if os.path.exists(v)}


if __name__ == "__main__":
    run_pipeline()
    print("\nPipeline complete.")
