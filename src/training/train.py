import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np

import sys
import os
# Allow executing gracefully from the root folder or deeply nested execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.yolo1d import yolo_1d_v11_n
from src.training.dataset import EEGRegressionDataset

def yolo_loss(preds, targets, obj_pos_weight=1.0):
    """
    Computes spatially optimized YOLO bounding losses natively spanning [B, 100, 1, 5] tensor allocations.
    """
    obj_mask = targets[..., 0] == 1.0
    
    bce = nn.BCELoss(reduction='none')
    bce_mean = nn.BCELoss(reduction='mean')
    mse = nn.MSELoss(reduction='mean')
    
    # 1. Objectness Loss (Evaluated across the entire grid universally)
    obj_target = targets[..., 0]
    obj_weight = torch.where(obj_target == 1.0, float(obj_pos_weight), 1.0)
    loss_obj = (bce(preds[..., 0], obj_target) * obj_weight).mean()
    
    # 2. Offset Loss & 3. Class Loss (Evaluated ONLY exactly where Events physically sit)
    if obj_mask.sum() > 0:
        loss_box = mse(preds[..., 1][obj_mask], targets[..., 1][obj_mask])
        loss_cls = bce_mean(preds[..., 2:][obj_mask], targets[..., 2:][obj_mask])
    else:
        loss_box = torch.tensor(0.0, device=preds.device)
        loss_cls = torch.tensor(0.0, device=preds.device)
        
    return loss_obj, loss_box, loss_cls

def extract_events_from_grid(tensor, conf_threshold=0.5, cell_duration=0.05, num_classes=3):
    """
    Dynamically reconstructs discrete temporal occurrences from a dense 1D structural grid.
    Returns: List of events per batch sample -> [[{'time': t, 'class': c, 'conf': p_c}, ...], ...]
    """
    B, S = tensor.size(0), tensor.size(1)
    batch_events = []
    
    for b in range(B):
        events = []
        for i in range(S):
            p_c = tensor[b, i, 0, 0].item()
            if p_c >= conf_threshold:
                t_x = tensor[b, i, 0, 1].item()
                # Determine absolute class classification natively logic
                cls_probs = tensor[b, i, 0, 2:2 + num_classes]
                class_id = torch.argmax(cls_probs).item()
                
                time_rel = (i * cell_duration) + (t_x * cell_duration)
                events.append({'time': time_rel, 'class': class_id, 'conf': p_c})
        batch_events.append(events)
    return batch_events

def calc_temporal_metrics(preds_tensor, trues_tensor, tau=0.25, conf_threshold=0.5, num_classes=3):
    """
    Computes TP, FP, and FN structurally bypassing generic 2D mapping overlapping dependencies (IoU).
    Instead, calculates point precision natively applying explicit temporal bound matching.
    """
    # 1. Reconstruct discrete timestamp items organically natively parsing grid
    pred_batch = extract_events_from_grid(
        preds_tensor,
        conf_threshold=conf_threshold,
        num_classes=num_classes,
    )
    true_batch = extract_events_from_grid(
        trues_tensor,
        conf_threshold=0.5,
        num_classes=num_classes,
    ) # Ground truth exactly outputs p_c = 1.0!
    
    tp_total, fp_total, fn_total = 0, 0, 0
    
    for preds, trues in zip(pred_batch, true_batch):
        matched_preds = set()
        matched_trues = set()
        
        # Sort predictions prioritizing tracking structurally by confidence locally
        preds = sorted(preds, key=lambda x: x['conf'], reverse=True)
        
        for p_idx, p in enumerate(preds):
            best_dist = float('inf')
            best_t_idx = -1
            
            for t_idx, t in enumerate(trues):
                if t_idx in matched_trues:
                    continue
                if p['class'] != t['class']: # Class MUST be identical dynamically aligning exactly
                    continue
                
                dist = abs(p['time'] - t['time'])
                if dist <= tau and dist < best_dist:
                    best_dist = dist
                    best_t_idx = t_idx
            
            if best_t_idx != -1:
                # Valid TP structurally connected locally!
                tp_total += 1
                matched_preds.add(p_idx)
                matched_trues.add(best_t_idx)
                
        fp_total += (len(preds) - len(matched_preds))
        fn_total += (len(trues) - len(matched_trues))
        
    return tp_total, fp_total, fn_total


def evaluate_loader_metrics(model, data_loader, device, tau=0.25, conf_threshold=0.5, num_classes=3):
    """Evaluate TP/FP/FN and derived metrics for a single confidence threshold."""
    tp_total, fp_total, fn_total = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            tp, fp, fn = calc_temporal_metrics(
                preds,
                y,
                tau=tau,
                conf_threshold=conf_threshold,
                num_classes=num_classes,
            )
            tp_total += tp
            fp_total += fp
            fn_total += fn

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "detections": tp_total + fp_total,
        "annotations": tp_total + fn_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def train(
    data_dir,
    anno_dir,
    epochs=50,
    batch_size=16,
    lr=1e-3,
    patience=5,
    results_file="results.txt",
    window_size_sec=10.0,
    stride_sec=10.0,
    fs=500,
    S=100,
    num_classes=3,
    conf_threshold=0.35,
    obj_pos_weight=12.0,
    threshold_sweep=None,
):
    # We natively isolated the dataset strictly from P001 to P082!
    EEG_IDX = list(range(1, 83))

    # Split patients into Train / Val / Test (~70-15-15 split)
    random.seed(42) # fixed seed for reproducibility
    shuffled_idx = list(EEG_IDX)
    random.shuffle(shuffled_idx)
    
    n_pids = len(shuffled_idx)
    train_split = int(0.7 * n_pids)
    val_split = int(0.85 * n_pids)
    
    train_pids = shuffled_idx[:train_split]
    val_pids = shuffled_idx[train_split:val_split]
    test_pids = shuffled_idx[val_split:]

    with open(results_file, "w") as f:
        f.write(f"EEG Seizure 1D Multi-Channel YOLO Regression Training\n")
        f.write(f"Total Patients Analyzed: {n_pids}\n")
        f.write(f"Train PIDs: {train_pids}\n")
        f.write(f"Val PIDs: {val_pids}\n")
        f.write(f"Test PIDs: {test_pids}\n")
        f.write("-" * 50 + "\n\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Assigning GPU Engine backend mapping natively: {device}")
    
    print("Loading TRAIN dataset...")
    train_ds = EEGRegressionDataset(
        data_dir,
        anno_dir,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        S=S,
        num_classes=num_classes,
        allowed_pids=train_pids,
    )
    print("Loading VAL dataset...")
    val_ds = EEGRegressionDataset(
        data_dir,
        anno_dir,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        S=S,
        num_classes=num_classes,
        allowed_pids=val_pids,
    )
    print("Loading TEST dataset...")
    test_ds = EEGRegressionDataset(
        data_dir,
        anno_dir,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        fs=fs,
        S=S,
        num_classes=num_classes,
        allowed_pids=test_pids,
    )
    
    print(f"Train Windows: {len(train_ds)}, Val Windows: {len(val_ds)}, Test Windows: {len(test_ds)}")
    with open(results_file, "a") as f:
        f.write(f"Train Windows: {len(train_ds)}, Val Windows: {len(val_ds)}, Test Windows: {len(test_ds)}\n\n")

    if len(train_ds) == 0:
        print("Train Dataset empty. Check paths and formats.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = yolo_1d_v11_n(in_channels=18, S=S, num_classes=num_classes).to(device)
    model_S = int(getattr(model.head, "S", 100))
    if int(S) != model_S:
        raise ValueError(
            f"Grid size mismatch: dataset S={S} but model head S={model_S}. "
            f"Set dataset.S to {model_S} in config.yaml or make model head configurable."
        )

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_obj = 0.0
        train_box = 0.0
        train_cls = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            
            l_obj, l_box, l_cls = yolo_loss(preds, y, obj_pos_weight=obj_pos_weight)
            
            # YOLO weighted heuristic: object detection logic dominates bounding offset shifts
            loss = 5.0 * l_obj + l_box + l_cls
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_obj += l_obj.item()
            train_box += l_box.item()
            train_cls += l_cls.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'obj': f"{l_obj.item():.4f}"})
            
        n_train = len(train_loader)
        train_loss /= n_train
        train_obj /= n_train
        train_box /= n_train
        train_cls /= n_train
        
        model.eval()
        val_loss = 0.0
        val_obj = 0.0
        val_box = 0.0
        val_cls = 0.0
        val_tp = 0
        val_fp = 0
        val_fn = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                
                l_obj, l_box, l_cls = yolo_loss(preds, y, obj_pos_weight=obj_pos_weight)
                loss = 5.0 * l_obj + l_box + l_cls
                
                val_loss += loss.item()
                val_obj += l_obj.item()
                val_box += l_box.item()
                val_cls += l_cls.item()
                
                tp, fp, fn = calc_temporal_metrics(
                    preds,
                    y,
                    tau=0.25,
                    conf_threshold=conf_threshold,
                    num_classes=num_classes,
                )
                val_tp += tp
                val_fp += fp
                val_fn += fn
                
        if len(val_loader) > 0:
            n_val = len(val_loader)
            val_loss /= n_val
            val_obj /= n_val
            val_box /= n_val
            val_cls /= n_val
            
        precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
        recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        log_line = (f"Epoch {epoch}/{epochs}:\n"
                    f"  [Train] Loss={train_loss:.4f} | Obj={train_obj:.4f}\n"
                    f"  [Val]   Loss={val_loss:.4f} | F1={f1:.4f} | P={precision:.4f} | R={recall:.4f}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
            log_line += "  -> BEST! Saved weights to best_model.pt"
        else:
            epochs_no_improve += 1
            log_line += f"  -> [No improvement: {epochs_no_improve}/{patience}]"

        print(log_line)
        with open(results_file, "a") as f:
            f.write(log_line + "\n")
            
        if epochs_no_improve >= patience:
            msg = f"\nEarly stopping triggered. Best Validation Loss: {best_val_loss:.4f}."
            print(msg)
            with open(results_file, "a") as f:
                f.write(msg + "\n")
            break

    with open(results_file, "a") as f:
        f.write(f"\nTraining completed.\n")
        
    print("\n" + "="*50)
    print("Evaluating Test Set with Best Model")
    print("="*50)
    
    try:
        model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True))
    except TypeError:
        # Fallback for older PyTorch versions where weights_only is not supported
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
    except Exception as e:
        print(f"Error loading best_model.pt: {e}")
        return
        
    model.eval()
    
    test_loss = 0.0
    test_obj = 0.0
    test_box = 0.0
    test_cls = 0.0
    test_tp = 0
    test_fp = 0
    test_fn = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            
            l_obj, l_box, l_cls = yolo_loss(preds, y, obj_pos_weight=obj_pos_weight)
            loss = 5.0 * l_obj + l_box + l_cls
            
            test_loss += loss.item()
            test_obj += l_obj.item()
            test_box += l_box.item()
            test_cls += l_cls.item()
            
            tp, fp, fn = calc_temporal_metrics(
                preds,
                y,
                tau=0.25,
                conf_threshold=conf_threshold,
                num_classes=num_classes,
            )
            test_tp += tp
            test_fp += fp
            test_fn += fn
            
    if len(test_loader) > 0:
        n_test = len(test_loader)
        test_loss /= n_test
        test_obj /= n_test
        test_box /= n_test
        test_cls /= n_test
        
    precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0.0
    recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
    test_msg = (f"\n--- TEST SET TEMPORAL METRICS (tau=0.25s) ---\n"
                f"Validation Box Detections: {test_tp + test_fp}\n"
                f"True Annotations Mapped:   {test_tp + test_fn}\n\n"
                f"True Positives (TP):  {test_tp}\n"
                f"False Positives (FP): {test_fp}\n"
                f"False Negatives (FN): {test_fn}\n\n"
                f"Precision: {precision:.4f}\n"
                f"Recall:    {recall:.4f}\n"
                f"F1-Score:  {f1:.4f}\n"
                f"-------------------------------------------\n"
                f"Total Objective Loss: {test_loss:.4f}\n")
    
    print(test_msg)
    with open(results_file, "a") as f:
        f.write(test_msg)

    if threshold_sweep:
        sweep_header = "\n--- CONFIDENCE THRESHOLD SWEEP (TEST) ---\n"
        print(sweep_header.strip())
        with open(results_file, "a") as f:
            f.write(sweep_header)

        best_row = None
        for th in threshold_sweep:
            metrics = evaluate_loader_metrics(
                model,
                test_loader,
                device,
                tau=0.25,
                conf_threshold=float(th),
                num_classes=num_classes,
            )
            line = (
                f"thr={float(th):.3f} | "
                f"P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}"
            )
            print(line)
            with open(results_file, "a") as f:
                f.write(line + "\n")

            if best_row is None or metrics["f1"] > best_row["f1"]:
                best_row = {"threshold": float(th), **metrics}

        if best_row is not None:
            best_line = (
                f"Best threshold by F1: {best_row['threshold']:.3f} "
                f"(F1={best_row['f1']:.4f}, P={best_row['precision']:.4f}, R={best_row['recall']:.4f})"
            )
            print(best_line)
            with open(results_file, "a") as f:
                f.write(best_line + "\n")

    print(f"Results stored in {results_file} in current directory.")

if __name__ == '__main__':
    from src.config import DATASET, MODEL, PATHS, TRAINING
    
    train(
        data_dir=PATHS["parquet_data_dir"],
        anno_dir=PATHS["events_dir"],
        epochs=TRAINING["epochs"],
        batch_size=TRAINING["batch_size"],
        lr=TRAINING["learning_rate"],
        patience=TRAINING["patience"],
        results_file=TRAINING["results_file"],
        window_size_sec=DATASET["window_size_sec"],
        stride_sec=DATASET["stride_sec"],
        fs=DATASET["fs"],
        S=DATASET["S"],
        num_classes=MODEL["num_classes"],
        conf_threshold=TRAINING.get("conf_threshold", 0.35),
        obj_pos_weight=TRAINING.get("obj_pos_weight", 12.0),
        threshold_sweep=TRAINING.get("threshold_sweep", [0.2, 0.3, 0.35, 0.4, 0.5, 0.6]),
    )
