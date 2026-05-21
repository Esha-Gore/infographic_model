import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile, Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def load_config(config_path="combined_config.json"):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    print("LOADED CONFIGURATION")
    return cfg


def save_config(config, save_path):
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {save_path}")


def build_transforms(config):
    aug = config["data_augmentation"]
    norm = aug["normalize"]

    train_t = [transforms.Resize((224, 224))]

    if aug.get("train", {}).get("random_horizontal_flip", False):
        train_t.append(transforms.RandomHorizontalFlip())

    if "color_jitter" in aug.get("train", {}):
        j = aug["train"]["color_jitter"]
        train_t.append(
            transforms.ColorJitter(
                brightness=j.get("brightness", 0),
                contrast=j.get("contrast", 0),
                saturation=j.get("saturation", 0),
                hue=j.get("hue", 0),
            )
        )

    if "random_affine" in aug.get("train", {}):
        a = aug["train"]["random_affine"]
        train_t.append(
            transforms.RandomAffine(
                degrees=a.get("degrees", 0),
                translate=a.get("translate", None),
                scale=a.get("scale", None),
            )
        )

    if "random_crop" in aug.get("train", {}):
        c = aug["train"]["random_crop"]
        train_t.append(transforms.RandomCrop(c["size"], padding=c["padding"]))

    train_t.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ]
    )

    test_t = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ]

    return transforms.Compose(train_t), transforms.Compose(test_t)


# multi-level directory scanner

def _scan_root(root_path, class_depth, class_name_template):
    """
    Walk `root_path` to depth `class_depth`, building (image_path, class_name)
    pairs.  `class_name_template` uses {d1}, {d2}, ... placeholders for each
    directory level, e.g. "{d1}" for flat layouts, "{d1}_{d2}" for two-level.

    Returns: list of (abs_image_path: str, class_name: str)
    """
    root = Path(root_path)
    samples = []

    def _recurse(current_dir, depth, parts):
        if depth == class_depth:
            # At the class leaf — collect images directly inside
            class_name = class_name_template
            for i, p in enumerate(parts, start=1):
                class_name = class_name.replace(f"{{d{i}}}", p)
            for f in current_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS:
                    samples.append((str(f), class_name))
        else:
            for sub in sorted(current_dir.iterdir()):
                if sub.is_dir():
                    _recurse(sub, depth + 1, parts + [sub.name])

    _recurse(root, 0, [])
    return samples


def _discover_classes(root_configs):
    """
    Given a list of root config dicts (each with 'path', 'class_depth',
    'class_name_template'), return the sorted union of all class names.
    """
    all_classes = set()
    for rc in root_configs:
        samples = _scan_root(rc["path"], rc["class_depth"], rc["class_name_template"])
        all_classes.update(cls for _, cls in samples)
    return sorted(all_classes)

# Create Dataset from the dual roots

class MultiRootDataset(Dataset):
    """
    Reads images from one or more roots, each with its own class_depth and
    class_name_template.  Only keeps classes in `selected_classes`, remapped
    to indices per `new_class_to_idx`.

    Returns: (img_tensor, new_label, path_str)

    Also stores self.sample_meta: list of dicts with keys
        path, class_name, label, source, raw_parts
    so you can recover category/brand/source info post-hoc without re-parsing.
    """

    def __init__(self, root_configs, transform, selected_classes, new_class_to_idx):
        self.transform = transform
        self.classes = list(selected_classes)
        self.class_to_idx = dict(new_class_to_idx)

        selected_set = set(selected_classes)
        self.samples = []       # (path, label) — fast access for DataLoader
        self.sample_meta = []   # rich metadata — for analysis / debugging

        for rc in root_configs:
            path_cls_pairs = _scan_root(rc["path"], rc["class_depth"], rc["class_name_template"])
            for img_path, class_name in path_cls_pairs:
                if class_name not in selected_set:
                    continue
                label = new_class_to_idx[class_name]
                self.samples.append((img_path, label))

                # Re-derive directory components for metadata
                rel = Path(img_path).relative_to(rc["path"])
                raw_parts = list(rel.parts[:-1])  # drop filename
                self.sample_meta.append({
                    "path":       img_path,
                    "class_name": class_name,
                    "label":      label,
                    "source":     rc["path"],
                    "raw_parts":  raw_parts,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, path

    def save_class_metadata(self, save_path):
        """
        Saves a JSON mapping every class name to its label index and parsed
        directory components (d1, d2, ...).  Useful for post-hoc analysis
        by category or brand without re-scanning the filesystem.

        Example entry for a 2-level logo root:
          "transportation_Nike": { "label": 42, "d1": "transportation", "d2": "Nike" }
        Example entry for a 1-level icon root:
          "arrow":               { "label":  0, "d1": "arrow" }
        """
        meta = {}
        # Index one sample per class to read its raw_parts
        seen = {}
        for m in self.sample_meta:
            if m["class_name"] not in seen:
                seen[m["class_name"]] = m["raw_parts"]

        for cls_name, label in self.class_to_idx.items():
            parts_dict = {}
            if cls_name in seen:
                for j, part in enumerate(seen[cls_name], start=1):
                    parts_dict[f"d{j}"] = part
            meta[cls_name] = {"label": label, **parts_dict}

        with open(save_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        print(f"Class metadata saved to: {save_path}")


class SubsetWrap(Dataset):
    """Subset wrapper that preserves __getitem__ outputs from base dataset."""
    def __init__(self, base_ds, indices):
        self.base_ds = base_ds
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_ds[int(self.indices[i])]

def topk_correct(outputs, labels, k=5):
    _, topk = outputs.topk(k, dim=1)
    correct = topk.eq(labels.view(-1, 1)).any(dim=1).float().sum().item()
    return correct


def run_eval(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    top1 = 0.0
    top5 = 0.0
    n = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _paths in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            _, pred = outputs.max(1)
            top1 += pred.eq(labels).sum().item()
            top5 += topk_correct(outputs, labels, k=5)
            n += bs

            probs = F.softmax(outputs, dim=1)
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    num_classes = all_scores.shape[1]

    labels_onehot = np.zeros_like(all_scores)
    labels_onehot[np.arange(len(all_labels)), all_labels] = 1

    per_class_ap = []
    for c in range(num_classes):
        if labels_onehot[:, c].sum() == 0:
            continue
        ap = average_precision_score(labels_onehot[:, c], all_scores[:, c])
        per_class_ap.append(ap)

    mean_ap = float(np.mean(per_class_ap)) * 100.0 if per_class_ap else 0.0

    return total_loss / max(1, n), 100.0 * top1 / max(1, n), 100.0 * top5 / max(1, n), mean_ap


def log_image_grid(writer, tag, images, step, max_images=16):
    if writer is None:
        return
    k = min(max_images, images.size(0))
    grid = torchvision.utils.make_grid(images[:k].detach().cpu(), normalize=True)
    writer.add_image(tag, grid, step)


def log_val_predictions(writer, epoch, model, loader, class_names,
                         max_images=8, misclassified_only=False):
    if writer is None:
        return

    model.eval()
    imgs_to_log = []
    captions = []
    images_logged = 0

    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            conf, preds = probs.max(1)

            for i in range(inputs.size(0)):
                if images_logged >= max_images:
                    break
                t = int(labels[i].item())
                p = int(preds[i].item())
                if misclassified_only and (p == t):
                    continue
                imgs_to_log.append(inputs[i].detach().cpu())
                captions.append(
                    f"T:{class_names[t]} | P:{class_names[p]} | "
                    f"conf:{float(conf[i].item()):.3f} | {os.path.basename(paths[i])}"
                )
                images_logged += 1

            if images_logged >= max_images:
                break

    if imgs_to_log:
        grid = torchvision.utils.make_grid(imgs_to_log, normalize=True)
        writer.add_image("val/predictions_grid", grid, epoch)
        writer.add_text("val/predictions_text", "\n".join(captions), epoch)
    else:
        writer.add_text("val/predictions_text", "No images matched filter.", epoch)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)

    # dirs
    exp_name  = config.get("experiment_name", "resnet50_run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{exp_name}_{timestamp}"
    run_dir   = os.path.join("./runs", run_name)
    ckpt_dir  = os.path.join(run_dir, "checkpoints")
    tb_dir    = os.path.join(run_dir, "tensorboard")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    config.setdefault("paths", {})
    config["paths"]["checkpoint_dir"] = ckpt_dir
    config["paths"]["tensorboard_dir"] = tb_dir
    print(f"Run directory: {run_dir}")

    batch_size  = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 2))
    num_epochs  = int(config["training"]["num_epochs"])
    val_frac    = float(config["training"].get("val_frac", 0.10))
    seed        = int(config["training"].get("seed", 42))
    class_frac  = float(config["training"].get("class_frac", 1.0))
    select_mode = config["training"].get("class_select_mode", "first")

    tb_cfg = config.get("tensorboard", {})
    tb_enable                      = bool(tb_cfg.get("enabled", True))
    tb_log_train_images            = bool(tb_cfg.get("log_train_images", True))
    tb_train_images_per_epoch      = int(tb_cfg.get("train_images_per_epoch", 16))
    tb_log_val_predictions         = bool(tb_cfg.get("log_val_predictions", True))
    tb_val_pred_images_per_epoch   = int(tb_cfg.get("val_pred_images_per_epoch", 8))
    tb_val_pred_misclassified_only = bool(tb_cfg.get("val_pred_misclassified_only", False))

    # transforms
    transform_train, transform_test = build_transforms(config)

    # root configs: list of {path, class_depth, class_name_template}
    train_root_configs = config["paths"]["train_roots"]
    test_root_configs  = config["paths"]["test_roots"]

    # discover all classes across every train root
    all_classes   = _discover_classes(train_root_configs)
    total_classes = len(all_classes)
    k = max(1, int(round(class_frac * total_classes)))

    if select_mode != "first":
        raise ValueError("Only class_select_mode='first' is implemented.")

    selected_classes = all_classes[:k]
    new_class_to_idx = {c: i for i, c in enumerate(selected_classes)}

    print(f"Total classes across all train roots: {total_classes}")
    print(f"Using first {k} classes ({class_frac*100:.1f}%): "
          f"{selected_classes[:10]}{'...' if k > 10 else ''}")

    # datasets
    full_train_ds_aug = MultiRootDataset(
        root_configs=train_root_configs, transform=transform_train,
        selected_classes=selected_classes, new_class_to_idx=new_class_to_idx,
    )
    full_train_ds_noaug = MultiRootDataset(
        root_configs=train_root_configs, transform=transform_test,
        selected_classes=selected_classes, new_class_to_idx=new_class_to_idx,
    )
    test_ds = MultiRootDataset(
        root_configs=test_root_configs, transform=transform_test,
        selected_classes=selected_classes, new_class_to_idx=new_class_to_idx,
    )

    # Save class metadata (label + d1/d2/... parts) for post-hoc analysis
    full_train_ds_aug.save_class_metadata(os.path.join(ckpt_dir, "class_metadata.json"))

    num_classes = len(selected_classes)
    print(f"Number of selected classes: {num_classes}")
    print(f"Filtered train samples (before split): {len(full_train_ds_aug)}")
    print(f"Filtered test samples: {len(test_ds)}")

    # train/val split
    n_total = len(full_train_ds_aug)
    n_val   = int(val_frac * n_total)
    rng     = np.random.default_rng(seed)
    idx     = np.arange(n_total)
    rng.shuffle(idx)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = SubsetWrap(full_train_ds_aug,   train_idx)
    val_ds   = SubsetWrap(full_train_ds_noaug, val_idx)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    valloader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    testloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Train split: {len(train_ds)} | Val split: {len(val_ds)}")

    # model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # loss 
    loss_cfg           = config.get("loss", {})
    label_smoothing    = float(loss_cfg.get("label_smoothing", 0.0))
    class_weights_mode = loss_cfg.get("class_weights", "none")

    if class_weights_mode == "inverse_frequency":
        print("Computing inverse frequency class weights from training split...")
        counts = torch.zeros(num_classes, dtype=torch.float32)
        for idx_i in train_idx:
            _, label = full_train_ds_aug.samples[idx_i]
            counts[label] += 1
        weights = 1.0 / (counts + 1e-12)
        weights = weights / weights.mean()
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        print(f"Class weights: min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # optimizer
    opt_cfg      = config["optimizer"]
    lr           = float(opt_cfg["lr"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if opt_cfg["type"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_cfg["type"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=float(opt_cfg.get("momentum", 0.9)),
                              weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg['type']}")

    # scheduler
    sched_cfg  = config.get("scheduler", {})
    sched_type = sched_cfg.get("type", "OFF")

    if sched_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=int(sched_cfg["step_size"]), gamma=float(sched_cfg["gamma"]))
    elif sched_type in ("OFF", None, ""):
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {sched_type}")

    save_config(config, os.path.join(ckpt_dir, "run_config.json"))
    writer = SummaryWriter(tb_dir) if tb_enable else None

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val = -1.0

    print("STARTING TRAINING")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        top1 = 0.0
        top5 = 0.0
        n = 0
        logged_train_images = False

        for inputs, labels, _paths in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            _, pred = outputs.max(1)
            top1 += pred.eq(labels).sum().item()
            top5 += topk_correct(outputs, labels, k=5)
            n += bs

            if writer is not None and tb_log_train_images and not logged_train_images:
                log_image_grid(writer, "train/examples", inputs, epoch,
                               max_images=tb_train_images_per_epoch)
                logged_train_images = True

        train_loss = total_loss / max(1, n)
        train_acc1 = 100.0 * top1 / max(1, n)
        train_acc5 = 100.0 * top5 / max(1, n)

        val_loss, val_acc1, val_acc5, val_map = run_eval(model, valloader, criterion)

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc1)
        val_accs.append(val_acc1)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc@1: {train_acc1:.2f}% | Train Acc@5: {train_acc5:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc@1: {val_acc1:.2f}% | Val Acc@5: {val_acc5:.2f}% | "
            f"Val mAP: {val_map:.2f}% | LR: {lr_now:.6f}"
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("acc1/train", train_acc1, epoch)
            writer.add_scalar("acc5/train", train_acc5, epoch)
            writer.add_scalar("loss/val",   val_loss,   epoch)
            writer.add_scalar("acc1/val",   val_acc1,   epoch)
            writer.add_scalar("acc5/val",   val_acc5,   epoch)
            writer.add_scalar("map/val",    val_map,    epoch)
            writer.add_scalar("lr",         lr_now,     epoch)

            if tb_log_val_predictions:
                log_val_predictions(writer, epoch, model, valloader, selected_classes,
                                    max_images=tb_val_pred_images_per_epoch,
                                    misclassified_only=tb_val_pred_misclassified_only)

        save_every = int(config.get("logging", {}).get("save_checkpoint_every", 5))
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "train_loss": train_loss, "train_acc1": train_acc1, "train_acc5": train_acc5,
                "val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5,
                "val_map": val_map,
                "selected_classes": selected_classes,
                "new_class_to_idx": new_class_to_idx,
                "config": config,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if val_acc1 > best_val:
            best_val = val_acc1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "selected_classes": selected_classes,
                "new_class_to_idx": new_class_to_idx,
                "config": config,
                "val_acc1": val_acc1, "val_acc5": val_acc5, "val_map": val_map,
            }, os.path.join(ckpt_dir, "best_model.pth"))

    #  Test
    print("\nEvaluating on test set...")
    test_loss, test_acc1, test_acc5, test_map = run_eval(model, testloader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Acc@1: {test_acc1:.2f}% | "
          f"Test Acc@5: {test_acc5:.2f}% | Test mAP: {test_map:.2f}%")

    if writer is not None:
        writer.add_scalar("loss/test", test_loss, 0)
        writer.add_scalar("acc1/test", test_acc1, 0)
        writer.add_scalar("acc5/test", test_acc5, 0)
        writer.add_scalar("map/test",  test_map,  0)

    # predictions
    print("\nGenerating predictions on test set...")
    model.eval()
    preds = []
    with torch.no_grad():
        for inputs, labels, paths in testloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
            for i in range(labels.size(0)):
                t = int(labels[i].item())
                p = int(pred[i].item())
                preds.append({
                    "true_label": t,  "true_class": selected_classes[t],
                    "pred_label": p,  "pred_class": selected_classes[p],
                    "confidence": float(conf[i].item()),
                    "image_path": paths[i],
                })

    pred_path = os.path.join(ckpt_dir, "predictions_test.json")
    with open(pred_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Predictions saved to: {pred_path} (n={len(preds)})")

    
    final_path = os.path.join(ckpt_dir, "final_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "selected_classes": selected_classes,
        "new_class_to_idx": new_class_to_idx,
        "config": config,
    }, final_path)
    print(f"Final model saved to: {final_path}")

    # Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.legend(); plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc@1")
    plt.plot(val_accs,   label="Val Acc@1")
    plt.legend(); plt.title("Accuracy")

    plot_path = os.path.join(ckpt_dir, "training_plots.png")
    plt.savefig(plot_path)
    print(f"Training plots saved to: {plot_path}")

    if writer is not None:
        writer.close()

    print("TRAINING COMPLETE")
    print(f"TensorBoard: tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
