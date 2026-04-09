import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.config import DATASET, MODEL, PATHS, TRAINING
from src.model.yolo1d import yolo_1d_v11_n
from src.model.yolo2d import yolo_2d_v11_n
from src.training.annotation_parser import parse_annotations
from src.training.classification_dataset import (
    EEGWindowClassificationDataset,
    ID_TO_LABEL,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class YoloWindowClassifier(nn.Module):
    def __init__(self, model_mode="1d", in_channels=18, image_channels=1, grid_S=20):
        super().__init__()
        self.model_mode = str(model_mode).lower()
        self.num_event_classes = 3
        self.num_classes = 4  # [none, !, !start, !end]

        if self.model_mode == "2d":
            self.backbone = yolo_2d_v11_n(
                in_channels=int(image_channels),
                input_height=18,
                S=int(grid_S),
                num_classes=self.num_event_classes,
            )
        else:
            self.backbone = yolo_1d_v11_n(
                in_channels=int(in_channels),
                S=int(grid_S),
                num_classes=self.num_event_classes,
            )

    def forward(self, x):
        # Grid logits: [B, S, 2 + event_classes]
        grid = self.backbone(x)
        obj_logits = grid[..., 0]
        cls_logits = grid[..., 2 : 2 + self.num_event_classes]

        # Event class evidence combines class logit and objectness confidence proxy.
        event_logits = cls_logits + obj_logits.unsqueeze(-1)
        event_logits, _ = torch.max(event_logits, dim=1)  # [B, 3]

        # Background score is high when all objectness scores are low.
        bg_logit = -torch.max(obj_logits, dim=1).values.unsqueeze(-1)  # [B, 1]

        return torch.cat([bg_logit, event_logits], dim=-1)  # [B, 4]


def reprocess_annotations_for_classification(raw_events_dir, out_events_dir):
    os.makedirs(out_events_dir, exist_ok=True)
    csv_files = sorted(
        [f for f in os.listdir(raw_events_dir) if f.endswith("_events.csv")]
    )

    kept = 0
    for fname in csv_files:
        src = os.path.join(raw_events_dir, fname)
        dst = os.path.join(out_events_dir, fname)

        parsed = parse_annotations(src)
        if len(parsed) == 0:
            pd.DataFrame(columns=["t_center_abs", "class_id", "label"]).to_csv(dst, index=False)
            continue

        out = parsed[["t_center_abs", "class_id", "label"]].copy()
        out.to_csv(dst, index=False)
        kept += 1

    print(f"[reprocess] Wrote cleaned classification annotations for {kept}/{len(csv_files)} files to: {out_events_dir}")


def split_pids(seed=42):
    eeg_ids = list(range(1, 83))
    random.seed(seed)
    random.shuffle(eeg_ids)
    n = len(eeg_ids)
    tr = int(0.70 * n)
    va = int(0.85 * n)
    return eeg_ids[:tr], eeg_ids[tr:va], eeg_ids[va:]


def compute_class_weights_and_sampler(dataset):
    counts = dataset.class_counts().astype(np.float64)
    counts = np.maximum(counts, 1.0)

    # Inverse-frequency class weights for CE loss.
    inv = 1.0 / counts
    class_weights = inv / inv.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Per-sample weights for balanced sampling during training.
    sample_weights = np.array([inv[int(s["label_id"])] for s in dataset.samples], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return class_weights, sampler


def update_confusion(conf, y_true, y_pred, num_classes=4):
    for t, p in zip(y_true, y_pred):
        conf[int(t)][int(p)] += 1


def metrics_from_confusion(conf, num_classes=4):
    eps = 1e-12
    per_class = {}
    f1s = []
    supports = []

    total = 0
    correct = 0
    for c in range(num_classes):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(num_classes) if r != c)
        fn = sum(conf[c][r] for r in range(num_classes) if r != c)
        support = sum(conf[c][r] for r in range(num_classes))

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        per_class[c] = {
            "label": ID_TO_LABEL.get(c, str(c)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        f1s.append(f1)
        supports.append(support)
        total += support
        correct += tp

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=np.maximum(np.array(supports), 1))) if f1s else 0.0
    acc = float(correct / (total + eps))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    conf = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item())

            pred = torch.argmax(logits, dim=1)
            update_confusion(conf, y.detach().cpu().numpy(), pred.detach().cpu().numpy())

    n_batches = max(1, len(loader))
    metrics = metrics_from_confusion(conf)
    metrics["loss"] = total_loss / n_batches
    return metrics


def train_one_model(
    model_mode,
    data_dir,
    anno_dir,
    fs=500,
    window_size_sec=1.0,
    stride_sec=1.0,
    epochs=15,
    batch_size=256,
    lr=1e-3,
    patience=5,
    in_channels=18,
    image_channels=1,
    grid_S=20,
    results_file="results_classification.txt",
):
    train_pids, val_pids, test_pids = split_pids(seed=42)

    train_ds = EEGWindowClassificationDataset(
        data_dir=data_dir,
        anno_dir=anno_dir,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        allowed_pids=train_pids,
        input_mode=model_mode,
    )
    val_ds = EEGWindowClassificationDataset(
        data_dir=data_dir,
        anno_dir=anno_dir,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        allowed_pids=val_pids,
        input_mode=model_mode,
    )
    test_ds = EEGWindowClassificationDataset(
        data_dir=data_dir,
        anno_dir=anno_dir,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        allowed_pids=test_pids,
        input_mode=model_mode,
    )

    class_weights, sampler = compute_class_weights_and_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = YoloWindowClassifier(
        model_mode=model_mode,
        in_channels=in_channels,
        image_channels=image_channels,
        grid_S=grid_S,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val = -1.0
    wait = 0

    log_header = (
        f"\n=== Classification Run ({model_mode.upper()}) ===\n"
        f"window={window_size_sec}s stride={stride_sec}s fs={fs} grid_S={grid_S}\n"
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}\n"
        f"class_weights={class_weights.tolist()}\n"
    )
    print(log_header)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(log_header + "\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{model_mode}] epoch {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device, criterion)

        line = (
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_weighted_f1={val_metrics['weighted_f1']:.4f}"
        )
        print(line)
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            wait = 0
            torch.save(model.state_dict(), f"best_cls_{model_mode}.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"[{model_mode}] early stop at epoch {epoch}")
                break

    # Test best model.
    best_path = f"best_cls_{model_mode}.pt"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = evaluate(model, test_loader, device, criterion)

    summary = [
        f"\n--- TEST CLASSIFICATION METRICS ({model_mode.upper()}) ---",
        f"Loss: {test_metrics['loss']:.4f}",
        f"Accuracy: {test_metrics['accuracy']:.4f}",
        f"Macro F1: {test_metrics['macro_f1']:.4f}",
        f"Weighted F1: {test_metrics['weighted_f1']:.4f}",
    ]
    for c in range(4):
        m = test_metrics["per_class"][c]
        summary.append(
            f"Class {m['label']}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} support={m['support']}"
        )

    text = "\n".join(summary)
    print(text)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    return test_metrics


def main():
    set_seed(42)

    data_dir = PATHS.get("parquet_data_dir")
    raw_anno_dir = PATHS.get("events_dir")
    processed_anno_dir = PATHS.get("processed_events_dir", raw_anno_dir)

    # Explicit annotation reprocessing for classification.
    reprocess_annotations_for_classification(raw_anno_dir, processed_anno_dir)

    # Classification-oriented defaults (can be adjusted in config later).
    fs = int(DATASET.get("fs", 500))
    window_size_sec = 1.0
    stride_sec = 1.0
    epochs = int(TRAINING.get("epochs", 20))
    batch_size = int(TRAINING.get("batch_size", 256))
    lr = float(TRAINING.get("learning_rate", 1e-3))
    patience = int(TRAINING.get("patience", 5))

    in_channels = int(MODEL.get("in_channels", 18))
    image_channels = int(MODEL.get("image_channels", 1))

    # For 1-second windows, smaller temporal grid is sufficient.
    grid_S = 20

    results_file = "results_classification.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("EEG 1-second Window Multiclass Classification Benchmark\n")
        f.write("Labels: none, !, !start, !end\n")
        f.write("===============================================\n")

    m1 = train_one_model(
        model_mode="1d",
        data_dir=data_dir,
        anno_dir=processed_anno_dir,
        fs=fs,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        in_channels=in_channels,
        image_channels=image_channels,
        grid_S=grid_S,
        results_file=results_file,
    )

    m2 = train_one_model(
        model_mode="2d",
        data_dir=data_dir,
        anno_dir=processed_anno_dir,
        fs=fs,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        in_channels=in_channels,
        image_channels=image_channels,
        grid_S=grid_S,
        results_file=results_file,
    )

    final = (
        "\n=== FINAL BENCHMARK SUMMARY ===\n"
        f"1D  -> Acc={m1['accuracy']:.4f} MacroF1={m1['macro_f1']:.4f} WeightedF1={m1['weighted_f1']:.4f}\n"
        f"2D  -> Acc={m2['accuracy']:.4f} MacroF1={m2['macro_f1']:.4f} WeightedF1={m2['weighted_f1']:.4f}\n"
    )
    print(final)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(final)


if __name__ == "__main__":
    main()
