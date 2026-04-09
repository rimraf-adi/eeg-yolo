import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import DATASET, MODEL, PATHS, TRAINING
from src.model.yolo1d import yolo_1d_v11_n
from src.model.yolo2d import yolo_2d_v11_n
from src.training.annotation_parser import parse_annotations

BINARY_LABELS = {0: "none", 1: "event"}
EVENT_LABELS = {0: "!", 1: "!start", 2: "!end"}
EVENT_TO_ID = {"!": 0, "!start": 1, "!end": 2}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_pids(seed=42):
    eeg_ids = list(range(1, 83))
    random.seed(seed)
    random.shuffle(eeg_ids)
    n = len(eeg_ids)
    tr = int(0.70 * n)
    va = int(0.85 * n)
    return eeg_ids[:tr], eeg_ids[tr:va], eeg_ids[va:]


def reprocess_annotations_for_classification(raw_events_dir, out_events_dir):
    os.makedirs(out_events_dir, exist_ok=True)
    csv_files = sorted([f for f in os.listdir(raw_events_dir) if f.endswith("_events.csv")])

    wrote = 0
    for fname in csv_files:
        src = os.path.join(raw_events_dir, fname)
        dst = os.path.join(out_events_dir, fname)

        parsed = parse_annotations(src)
        if len(parsed) == 0:
            pd.DataFrame(columns=["t_center_abs", "class_id", "label"]).to_csv(dst, index=False)
            continue

        out = parsed[["t_center_abs", "class_id", "label"]].copy()
        out.to_csv(dst, index=False)
        wrote += 1

    print(f"[reprocess] wrote cleaned classification annotations for {wrote}/{len(csv_files)} files to {out_events_dir}")


def _resolve_event_label(events_df, t_start, t_end):
    if events_df is None or len(events_df) == 0:
        return None

    in_window = events_df[(events_df["t_center_abs"] >= t_start) & (events_df["t_center_abs"] < t_end)]
    if len(in_window) == 0:
        return None

    labels = set(in_window["label"].astype(str).str.strip().tolist())

    # Priority preserves boundary semantics in ambiguous multi-event windows.
    if "!start" in labels:
        return "!start"
    if "!end" in labels:
        return "!end"
    if "!" in labels:
        return "!"
    return None


class EEGWindowTwoStageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        anno_dir,
        allowed_pids,
        stage,
        window_size_sec=1.0,
        stride_sec=1.0,
        fs=500,
        input_mode="1d",
        split_name="train",
        max_neg_pos_ratio=3.0,
    ):
        self.data_dir = Path(data_dir)
        self.anno_dir = Path(anno_dir)
        self.allowed_pids = set(allowed_pids)
        self.stage = str(stage)
        self.window_size_sec = float(window_size_sec)
        self.stride_sec = float(stride_sec)
        self.fs = int(fs)
        self.input_mode = str(input_mode).lower()
        self.split_name = str(split_name)
        self.max_neg_pos_ratio = float(max_neg_pos_ratio)

        self.window_samples = int(self.window_size_sec * self.fs)

        self.bipolar_indices = [
            (1, 3), (3, 5), (5, 7), (7, 9),
            (0, 2), (2, 4), (4, 6), (6, 8),
            (1, 11), (11, 13), (13, 15), (15, 9),
            (0, 10), (10, 12), (12, 14), (14, 8),
            (16, 17), (17, 18),
        ]
        self.num_channels = len(self.bipolar_indices)

        self.eeg_cache = {}
        self.samples = []
        self._build_index()

    def _read_events(self, csv_path):
        df = pd.read_csv(csv_path)
        required_cols = {"t_center_abs", "class_id", "label"}
        if required_cols.issubset(set(df.columns)):
            out = df.copy()
            out["t_center_abs"] = pd.to_numeric(out["t_center_abs"], errors="coerce")
            out["class_id"] = pd.to_numeric(out["class_id"], errors="coerce").fillna(-1).astype(int)
            out["label"] = out["label"].astype(str).str.strip()
            out = out.dropna(subset=["t_center_abs"]).sort_values("t_center_abs").reset_index(drop=True)
            return out
        return parse_annotations(str(csv_path))

    def _build_index(self):
        parquet_files = sorted(self.data_dir.glob("*.parquet"))
        print(f"[*] building {self.stage} dataset ({self.split_name}, mode={self.input_mode})...")

        for pq_path in parquet_files:
            pid = pq_path.stem
            try:
                pid_int = int(pid.replace("P", ""))
            except Exception:
                continue
            if pid_int not in self.allowed_pids:
                continue

            events_csv = self.anno_dir / f"{pid}_events.csv"
            if not events_csv.exists():
                continue

            raw_data = pd.read_parquet(pq_path).values.T
            bipolar = np.zeros((self.num_channels, raw_data.shape[1]), dtype=np.float32)
            for c_idx, (anode, cathode) in enumerate(self.bipolar_indices):
                bipolar[c_idx, :] = raw_data[anode, :] - raw_data[cathode, :]
            self.eeg_cache[pid] = bipolar

            events_df = self._read_events(events_csv)
            max_duration = raw_data.shape[1] / self.fs

            start_time = 0.0
            while start_time + self.window_size_sec <= max_duration:
                end_time = start_time + self.window_size_sec
                event_label = _resolve_event_label(events_df, start_time, end_time)
                has_event = int(event_label is not None)

                if self.stage == "binary":
                    label_id = has_event
                    self.samples.append({
                        "pid": pid,
                        "start_time": start_time,
                        "end_time": end_time,
                        "label": int(label_id),
                    })
                elif self.stage == "event":
                    if event_label is not None:
                        label_id = EVENT_TO_ID[event_label]
                        self.samples.append({
                            "pid": pid,
                            "start_time": start_time,
                            "end_time": end_time,
                            "label": int(label_id),
                        })
                else:
                    raise ValueError(f"Unknown stage={self.stage}")

                start_time += self.stride_sec

        if self.stage == "binary" and self.split_name == "train" and self.max_neg_pos_ratio > 0:
            pos = [s for s in self.samples if s["label"] == 1]
            neg = [s for s in self.samples if s["label"] == 0]

            if len(pos) > 0:
                max_neg = int(self.max_neg_pos_ratio * len(pos))
                if len(neg) > max_neg:
                    random.shuffle(neg)
                    neg = neg[:max_neg]
                    self.samples = pos + neg
                    random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def class_counts(self, n_classes):
        counts = np.zeros(int(n_classes), dtype=np.int64)
        for s in self.samples:
            counts[int(s["label"])] += 1
        return counts

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample["pid"]
        start_time = sample["start_time"]
        end_time = sample["end_time"]

        start_idx = int(start_time * self.fs)
        end_idx = int(end_time * self.fs)

        signal = self.eeg_cache[pid]
        raw = signal[:, start_idx:min(end_idx, signal.shape[1])].copy()
        x = np.zeros((self.num_channels, self.window_samples), dtype=np.float32)
        x[:, :raw.shape[1]] = raw

        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-7
        x = (x - mean) / std

        xt = torch.tensor(x, dtype=torch.float32)
        if self.input_mode == "2d":
            xt = xt.unsqueeze(0)

        yt = torch.tensor(int(sample["label"]), dtype=torch.long)
        return xt, yt


class YoloClassifierHead(nn.Module):
    def __init__(self, model_mode="1d", out_classes=2, in_channels=18, image_channels=1, grid_S=20):
        super().__init__()
        self.model_mode = str(model_mode).lower()

        if self.model_mode == "2d":
            self.backbone = yolo_2d_v11_n(
                in_channels=int(image_channels),
                input_height=18,
                S=int(grid_S),
                num_classes=3,
            )
        else:
            self.backbone = yolo_1d_v11_n(
                in_channels=int(in_channels),
                S=int(grid_S),
                num_classes=3,
            )

        feature_dim = 2 * (2 + 3)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, int(out_classes)),
        )

    def forward(self, x):
        grid = self.backbone(x)  # [B,S,5]
        pooled_mean = torch.mean(grid, dim=1)
        pooled_max = torch.max(grid, dim=1).values
        feats = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.classifier(feats)


def build_class_weights(counts):
    counts = np.maximum(counts.astype(np.float64), 1.0)
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32)


def update_confusion(conf, y_true, y_pred, num_classes):
    for t, p in zip(y_true, y_pred):
        conf[int(t)][int(p)] += 1


def metrics_from_confusion(conf, labels_map, n_classes):
    eps = 1e-12
    per_class = {}
    f1s = []
    supports = []
    total = 0
    correct = 0

    for c in range(int(n_classes)):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(int(n_classes)) if r != c)
        fn = sum(conf[c][r] for r in range(int(n_classes)) if r != c)
        support = sum(conf[c][r] for r in range(int(n_classes)))

        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)

        per_class[c] = {
            "label": labels_map.get(c, str(c)),
            "precision": p,
            "recall": r,
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
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "per_class": per_class}


def evaluate(model, loader, device, criterion, labels_map, n_classes):
    model.eval()
    conf = defaultdict(lambda: defaultdict(int))
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item())

            pred = torch.argmax(logits, dim=1)
            update_confusion(conf, y.detach().cpu().numpy(), pred.detach().cpu().numpy(), n_classes)

    metrics = metrics_from_confusion(conf, labels_map, n_classes)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def train_stage(
    stage_name,
    model_mode,
    data_dir,
    anno_dir,
    train_pids,
    val_pids,
    test_pids,
    out_classes,
    labels_map,
    window_size_sec,
    stride_sec,
    fs,
    in_channels,
    image_channels,
    grid_S,
    epochs,
    batch_size,
    lr,
    patience,
    max_neg_pos_ratio,
    results_file,
):
    train_ds = EEGWindowTwoStageDataset(
        data_dir=data_dir,
        anno_dir=anno_dir,
        allowed_pids=train_pids,
        stage=stage_name,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        input_mode=model_mode,
        split_name="train",
        max_neg_pos_ratio=max_neg_pos_ratio,
    )
    val_ds = EEGWindowTwoStageDataset(
        data_dir=data_dir,
        anno_dir=anno_dir,
        allowed_pids=val_pids,
        stage=stage_name,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        input_mode=model_mode,
        split_name="val",
        max_neg_pos_ratio=max_neg_pos_ratio,
    )
    test_ds = EEGWindowTwoStageDataset(
        data_dir=data_dir,
        anno_dir=anno_dir,
        allowed_pids=test_pids,
        stage=stage_name,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        input_mode=model_mode,
        split_name="test",
        max_neg_pos_ratio=max_neg_pos_ratio,
    )

    counts = train_ds.class_counts(out_classes)
    class_weights = build_class_weights(counts)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = YoloClassifierHead(
        model_mode=model_mode,
        out_classes=out_classes,
        in_channels=in_channels,
        image_channels=image_channels,
        grid_S=grid_S,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    header = (
        f"\n=== {stage_name.upper()} STAGE ({model_mode.upper()}) ===\n"
        f"window={window_size_sec}s stride={stride_sec}s fs={fs} grid_S={grid_S}\n"
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}\n"
        f"class_counts={counts.tolist()} class_weights={class_weights.tolist()}\n"
    )
    print(header)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(header + "\n")

    best_val = -1.0
    wait = 0
    best_path = f"best_two_stage_{stage_name}_{model_mode}.pt"

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[{stage_name}-{model_mode}] epoch {epoch}/{epochs}")
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
        val_metrics = evaluate(model, val_loader, device, criterion, labels_map, out_classes)

        line = (
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f}"
        )
        print(line)
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= int(patience):
                print(f"[{stage_name}-{model_mode}] early stop at epoch {epoch}")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = evaluate(model, test_loader, device, criterion, labels_map, out_classes)
    summary = [
        f"\n--- TEST {stage_name.upper()} METRICS ({model_mode.upper()}) ---",
        f"Loss: {test_metrics['loss']:.4f}",
        f"Accuracy: {test_metrics['accuracy']:.4f}",
        f"Macro F1: {test_metrics['macro_f1']:.4f}",
        f"Weighted F1: {test_metrics['weighted_f1']:.4f}",
    ]
    for c in range(out_classes):
        m = test_metrics["per_class"][c]
        summary.append(
            f"Class {m['label']}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} support={m['support']}"
        )

    text = "\n".join(summary)
    print(text)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    return test_metrics


def run_for_mode(model_mode, data_dir, anno_dir, results_file):
    train_pids, val_pids, test_pids = split_pids(seed=42)

    fs = int(DATASET.get("fs", 500))
    window_size_sec = float(TRAINING.get("cls_window_size_sec", 1.0))
    stride_sec = float(TRAINING.get("cls_stride_sec", 1.0))

    in_channels = int(MODEL.get("in_channels", 18))
    image_channels = int(MODEL.get("image_channels", 1))
    grid_S = int(TRAINING.get("cls_grid_S", 20))

    batch_size = int(TRAINING.get("cls_batch_size", 256))
    lr = float(TRAINING.get("cls_learning_rate", TRAINING.get("learning_rate", 1e-3)))
    patience = int(TRAINING.get("cls_patience", 5))

    s1_epochs = int(TRAINING.get("cls_stage1_epochs", 15))
    s2_epochs = int(TRAINING.get("cls_stage2_epochs", 20))

    max_neg_pos_ratio = float(TRAINING.get("cls_max_neg_pos_ratio", 3.0))

    stage1_metrics = train_stage(
        stage_name="binary",
        model_mode=model_mode,
        data_dir=data_dir,
        anno_dir=anno_dir,
        train_pids=train_pids,
        val_pids=val_pids,
        test_pids=test_pids,
        out_classes=2,
        labels_map=BINARY_LABELS,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        in_channels=in_channels,
        image_channels=image_channels,
        grid_S=grid_S,
        epochs=s1_epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        max_neg_pos_ratio=max_neg_pos_ratio,
        results_file=results_file,
    )

    stage2_metrics = train_stage(
        stage_name="event",
        model_mode=model_mode,
        data_dir=data_dir,
        anno_dir=anno_dir,
        train_pids=train_pids,
        val_pids=val_pids,
        test_pids=test_pids,
        out_classes=3,
        labels_map=EVENT_LABELS,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        in_channels=in_channels,
        image_channels=image_channels,
        grid_S=grid_S,
        epochs=s2_epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        max_neg_pos_ratio=max_neg_pos_ratio,
        results_file=results_file,
    )

    return stage1_metrics, stage2_metrics


def main():
    set_seed(42)

    data_dir = PATHS.get("parquet_data_dir")
    raw_anno_dir = PATHS.get("events_dir")

    # Keep two-stage outputs isolated from the localization pipeline outputs.
    out_anno_dir = str(Path(PATHS.get("processed_events_dir", raw_anno_dir)) / "classification_two_stage")
    reprocess_annotations_for_classification(raw_anno_dir, out_anno_dir)

    results_file = "results_classification_two_stage.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("EEG Two-Stage Classification Benchmark\n")
        f.write("Stage1: binary (none/event), Stage2: event subclass (!,!start,!end)\n")
        f.write("===============================================================\n")

    s1_1d, s2_1d = run_for_mode("1d", data_dir, out_anno_dir, results_file)
    s1_2d, s2_2d = run_for_mode("2d", data_dir, out_anno_dir, results_file)

    final = (
        "\n=== FINAL TWO-STAGE SUMMARY ===\n"
        f"1D Stage1(binary): Acc={s1_1d['accuracy']:.4f} MacroF1={s1_1d['macro_f1']:.4f}\n"
        f"1D Stage2(event):  Acc={s2_1d['accuracy']:.4f} MacroF1={s2_1d['macro_f1']:.4f}\n"
        f"2D Stage1(binary): Acc={s1_2d['accuracy']:.4f} MacroF1={s1_2d['macro_f1']:.4f}\n"
        f"2D Stage2(event):  Acc={s2_2d['accuracy']:.4f} MacroF1={s2_2d['macro_f1']:.4f}\n"
    )
    print(final)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(final)


if __name__ == "__main__":
    main()
