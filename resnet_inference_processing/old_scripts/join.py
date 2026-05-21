import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--detections",  type=str, required=True)
    parser.add_argument("--output",      type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    with open(args.detections) as f:
        detections = json.load(f)

    print("Building lookup from detections...")
    lookup = {}
    for entry in detections:
        uuid = entry.get("uuid", "")
        icon_idx = 0
        for el in entry.get("elements", []):
            if el.get("type") == "icon":
                lookup[(uuid, icon_idx)] = el.get("content", "")
                icon_idx += 1   

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
            "country": pred.get("country", ""),
            "pred_class": pred["pred_class"],
            "confidence": pred["confidence"],
            "content": content,
            "combined_text": combined,
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} joined entries to {args.output}")
    print(f"Skipped {skipped} entries")

if __name__ == "__main__":
    main()
