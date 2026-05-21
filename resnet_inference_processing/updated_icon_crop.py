import os, json, csv, argparse
from PIL import Image
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    crops_dir = os.path.join(args.output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    with open(args.json) as f:
        annotations = json.load(f)

    csv_path = os.path.join(args.output_dir, "manifest.csv")
    fields = ["uuid", "country", "image_filename", "element_index",
              "crop_filename", "norm_x1", "norm_y1", "norm_x2", "norm_y2",
              "img_width", "img_height", "crop_width", "crop_height",
              "content", "interactivity", "skip_reason"]

    saved = skipped = 0
    with open(csv_path, "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fields)
        w.writeheader()

        for entry in tqdm(annotations):
            uuid = entry.get("uuid", "")
            country = entry.get("country", "")
            image_fn = entry.get("image", "")
            elements = entry.get("elements", [])
            icons = [(i, el) for i, el in enumerate(elements) if el.get("type") == "icon"]
            if not icons:
                continue

            # Find image (tries annotated_ prefix first, then original)
            img_path = None
            for fn in ["annotated_" + image_fn, image_fn]:
                p = os.path.join(args.images_dir, fn)
                if os.path.exists(p):
                    img_path = p
                    break

            base_row = {"uuid": uuid, "country": country, "image_filename": image_fn}

            if img_path is None:
                for el_idx, el in icons:
                    w.writerow({**base_row, "element_index": el_idx,
                                "content": el.get("content", ""),
                                "interactivity": el.get("interactivity", ""),
                                "skip_reason": "image_not_found"})
                    skipped += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                for el_idx, el in icons:
                    w.writerow({**base_row, "element_index": el_idx,
                                "content": el.get("content", ""),
                                "interactivity": el.get("interactivity", ""),
                                "skip_reason": f"image_load_error: {e}"})
                    skipped += 1
                continue

            img_w, img_h = img.size

            for el_idx, el in icons:
                bbox = el.get("bbox", [])
                row = {**base_row, "element_index": el_idx,
                       "img_width": img_w, "img_height": img_h,
                       "content": el.get("content", ""),
                       "interactivity": el.get("interactivity", "")}

                if len(bbox) != 4:
                    w.writerow({**row, "skip_reason": "invalid_bbox"})
                    skipped += 1
                    continue

                nx1, ny1, nx2, ny2 = bbox
                x1 = max(0, int(nx1 * img_w))
                y1 = max(0, int(ny1 * img_h))
                x2 = min(img_w, int(nx2 * img_w) + 1)  # +1 to avoid degenerate from truncation
                y2 = min(img_h, int(ny2 * img_h) + 1)

                if x2 <= x1 or y2 <= y1:
                    w.writerow({**row, "norm_x1": nx1, "norm_y1": ny1, "norm_x2": nx2, "norm_y2": ny2,
                                "skip_reason": "degenerate_bbox"})
                    skipped += 1
                    continue

                crop = img.crop((x1, y1, x2, y2))
                crop_fn = f"{uuid}_{el_idx}.png"
                crop.save(os.path.join(crops_dir, crop_fn))

                w.writerow({**row, "crop_filename": crop_fn,
                            "norm_x1": nx1, "norm_y1": ny1, "norm_x2": nx2, "norm_y2": ny2,
                            "crop_width": crop.size[0], "crop_height": crop.size[1],
                            "skip_reason": ""})
                saved += 1

    print(f"\nSaved: {saved}, Skipped: {skipped}")
    print(f"Manifest: {csv_path}")

if __name__ == "__main__":
    main()
