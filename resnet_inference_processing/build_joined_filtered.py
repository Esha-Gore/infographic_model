"""
build_joined_filtered.py

Consolidates the three pipeline steps (join -> add translations -> confidence
filter) into one script. Intermediate forms are kept in memory only; the single
output written is joined_filtered.json (and a matching .csv).

Index convention (IMPORTANT):
    Crops are named "{uuid}_{el_idx}.png" where el_idx is the position of the
    element in the full  elements list. Every step below keys on that same full-list index so the join,
    translation match, and filter all line up.

Usage:
    python build_joined_filtered.py \
        --predictions predictions.json \
        --detections  detections.json \
        --translations captions_translated.csv \
        --output dataset_results/joined_filtered.json \
        --threshold 0.8
Data Needed:
    predictions.json -> ResNet50 output for cropped icons
    detections.json -> all_detections.json from the Omniparser output
    captions_translated.csv -> tranlsated captions from AWS output


"""

import os
import re
import json
import argparse

import pandas as pd

# Step 1: join predictions with detection captions (from join.py)
def join_predictions_detections(predictions, detections):
    print("Building lookup from detections...")
    lookup = {}
    # country is dropped by infer.py so add back here
    country_lookup = {}
    for entry in detections:
        uuid = entry.get("uuid", "")
        country_lookup[uuid] = entry.get("country", "")
        for el_idx, el in enumerate(entry.get("elements", [])):
            if el.get("type") == "icon":
                lookup[(uuid, el_idx)] = el.get("content", "")

    print(f"Lookup size: {len(lookup)}")

    results = []
    skipped = 0
    for pred in predictions:
        crop_filename = pred["crop_filename"]
        name = crop_filename.replace(".png", "")
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            skipped += 1
            continue

        uuid, el_idx = parts[0], int(parts[1])
        content = lookup.get((uuid, el_idx), "")

        if not content:
            skipped += 1
            continue

        combined = f"{pred['pred_class']}: {content}"

        results.append({
            "crop_filename": crop_filename,
            "uuid": uuid,
            "element_index": el_idx,
            "country": country_lookup.get(uuid, pred.get("country", "")),
            "pred_class": pred["pred_class"],
            "confidence": pred["confidence"],
            "content": content,
            "combined_text": combined,
        })

    print(f"Joined {len(results)} entries, skipped {skipped}")
    return results

# Step 2: attach English translations (from join_en.py)
def clean_translation(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'^.*?:\s*', '', text, count=1, flags=re.IGNORECASE)
    return text.strip()


def add_translations(joined, translated):
    translated = translated.copy()
    translated['content_en'] = translated['content_en'].apply(clean_translation)

    # full elemenrs index: cumcount over all rows per uuid (file order == element
    # order), then keep icons. 
    translated['element_index'] = translated.groupby('uuid').cumcount()
    icon_rows = translated[translated['element_type'] == 'icon']

    lookup = {(row['uuid'], int(row['element_index'])): row['content_en']
              for _, row in icon_rows.iterrows()}

    for item in joined:
        key = (item['uuid'], item['element_index'])
        item['content_en'] = lookup.get(key, '')
        item['combined_text_en'] = (
            f"{item['pred_class']}: {item['content_en']}"
            if item['content_en'] else item['combined_text']
        )

    matched = sum(1 for i in joined if i['content_en'])
    print(f"Attached translations: {matched}/{len(joined)} matched a translation")
    return joined


# Step 3: confidence filter (from filter.py)
# Prepend pred_class only when confidence >= threshold; otherwise content only.
def add_confidence_filter(joined, threshold):
    for item in joined:
        content = item.get('content_en') or item.get('content', '')
        if item.get('confidence', 0) >= threshold:
            item['combined_text_filtered'] = f"{item['pred_class']}: {content}"
        else:
            item['combined_text_filtered'] = content

    above = sum(1 for i in joined if i.get('confidence', 0) >= threshold)
    print(f"Filtered at threshold {threshold}: "
          f"{above} above / {len(joined) - above} below")
    return joined



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="predictions JSON")
    p.add_argument("--detections", required=True, help="detections JSON")
    p.add_argument("--translations", required=True, help="captions_translated.csv")
    p.add_argument("--output", required=True,
                   help="output path for joined_filtered.json (a .csv is written alongside)")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="confidence threshold for prepending pred_class (default 0.8)")
    p.add_argument("--save-intermediates", action="store_true",
                   help="also write joined.json and joined_en.json next to the output")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.detections) as f:
        detections = json.load(f)
    translated = pd.read_csv(args.translations)

    joined = join_predictions_detections(predictions, detections)
    if args.save_intermediates:
        with open(os.path.join(out_dir or ".", "joined.json"), "w") as f:
            json.dump(joined, f, indent=2)

    joined = add_translations(joined, translated)
    if args.save_intermediates:
        with open(os.path.join(out_dir or ".", "joined_en.json"), "w") as f:
            json.dump(joined, f, indent=2)

    joined = add_confidence_filter(joined, args.threshold)

    # single final output
    with open(args.output, "w") as f:
        json.dump(joined, f, indent=2)
    csv_path = args.output[:-5] + ".csv" if args.output.endswith(".json") else args.output + ".csv"
    pd.DataFrame(joined).to_csv(csv_path, index=False)

    print(f"\nDone. {len(joined)} entries saved to:")
    print(f"  {args.output}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
