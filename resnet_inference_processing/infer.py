import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="infer_config.json")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    ckpt = torch.load(cfg["model_path"], map_location=device)
    selected_classes = ckpt["selected_classes"]
    num_classes = len(selected_classes)
    print(f"Loaded model with {num_classes} classes")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    crops_dir = cfg["crops_dir"]
    batch_size = cfg.get("batch_size", 64)
    image_files = [f for f in os.listdir(crops_dir) if f.endswith(".png")]
    print(f"Found {len(image_files)} crops")

    results = []
    for i in tqdm(range(0, len(image_files), batch_size), desc="Inferring"):
        batch_files = image_files[i:i+batch_size]
        imgs, valid_files = [], []

        for fn in batch_files:
            try:
                img = Image.open(os.path.join(crops_dir, fn)).convert("RGB")
                imgs.append(transform(img))
                valid_files.append(fn)
            except Exception as e:
                print(f"Skipping {fn}: {e}")

        if not imgs:
            continue

        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1)
            confs, preds = probs.max(1)

        for fn, pred, conf in zip(valid_files, preds.cpu(), confs.cpu()):
            uuid, el_idx = fn.replace(".png", "").rsplit("_", 1)
            results.append({
                "crop_filename": fn,
                "uuid": uuid,
                "element_index": int(el_idx),
                "pred_class": selected_classes[pred.item()],
                "pred_label": pred.item(),
                "confidence": round(conf.item(), 4),
            })

    with open(cfg["output_path"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {cfg['output_path']}")

if __name__ == "__main__":
    main()
