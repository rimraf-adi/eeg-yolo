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
from src.model.yolo2d import yolo_2d_v11_n
from src.training.dataset import EEGRegressionDataset

CLASS_LABELS = {0: '!', 1: '!start', 2: '!end'}


def build_model(model_mode, in_channels, S, num_classes):
    model_mode = str(model_mode).lower()
    if model_mode == "2d":
        return yolo_2d_v11_n(in_channels=in_channels, S=S, num_classes=num_classes)
    return yolo_1d_v11_n(in_channels=in_channels, S=S, num_classes=num_classes)


def _expected_calibration_error(pred_probs, target_probs, n_bins=15):
    pred_probs = pred_probs.detach().flatten().float().cpu()
    target_probs = target_probs.detach().flatten().float().cpu()
    if pred_probs.numel() == 0:
        return 0.0

    bin_edges = torch.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    total = float(pred_probs.numel())

    for b in range(int(n_bins)):
        left = bin_edges[b]
        right = bin_edges[b + 1]
        if b == int(n_bins) - 1:
            mask = (pred_probs >= left) & (pred_probs <= right)
        else:
            mask = (pred_probs >= left) & (pred_probs < right)
        if mask.any():
            confidence = pred_probs[mask].mean().item()
            accuracy = target_probs[mask].mean().item()
            ece += (mask.float().sum().item() / total) * abs(confidence - accuracy)

    return float(ece)


def calc_regression_metrics(preds_tensor, targets_tensor, soft_offset_mask_threshold=0.1, n_bins=15):
    pred_obj = torch.sigmoid(preds_tensor[..., 0])
    pred_offset = torch.sigmoid(preds_tensor[..., 1])
    pred_cls = torch.sigmoid(preds_tensor[..., 2:])

    obj_target = targets_tensor[..., 0]
    offset_target = targets_tensor[..., 1]
    cls_target = targets_tensor[..., 2:]

    offset_mask = obj_target >= float(soft_offset_mask_threshold)
    if offset_mask.any():
        offset_diff = pred_offset[offset_mask] - offset_target[offset_mask]
        offset_mae = float(torch.mean(torch.abs(offset_diff)).item())
        offset_rmse = float(torch.sqrt(torch.mean(offset_diff ** 2)).item())
    else:
        offset_mae = 0.0
        offset_rmse = 0.0

    obj_mae = float(torch.mean(torch.abs(pred_obj - obj_target)).item())
    obj_brier = float(torch.mean((pred_obj - obj_target) ** 2).item())
    cls_brier = float(torch.mean((pred_cls - cls_target) ** 2).item())
    obj_ece = _expected_calibration_error(pred_obj, obj_target, n_bins=n_bins)

    return {
        'obj_mae': obj_mae,
        'obj_brier': obj_brier,
        'obj_ece': obj_ece,
        'cls_brier': cls_brier,
        'offset_mae': offset_mae,
        'offset_rmse': offset_rmse,
    }

def yolo_loss(preds, targets, obj_pos_weight=1.0):
    """
    Point-YOLO loss on logits with tensor shape [B, S, 2 + num_classes].
    """
    obj_target = targets[..., 0]
    soft_targets = bool(torch.any((obj_target > 0.0) & (obj_target < 1.0)).item())
    obj_mask = obj_target >= 0.1 if soft_targets else obj_target == 1.0
    
    bce = nn.BCEWithLogitsLoss(reduction='none')
    mse = nn.MSELoss(reduction='mean')
    
    # 1. Objectness Loss (Evaluated across the entire grid universally)
    obj_weight = torch.where(obj_target > 0.0, 1.0 + obj_target * (float(obj_pos_weight) - 1.0), 1.0)
    loss_obj = (bce(preds[..., 0], obj_target) * obj_weight).mean()
    
    # 2. Offset Loss & 3. Class Loss (evaluated only on positive cells)
    if obj_mask.sum() > 0:
        pred_offset = torch.sigmoid(preds[..., 1])
        offset_weight = obj_target[obj_mask]
        offset_error = (pred_offset[obj_mask] - targets[..., 1][obj_mask]) ** 2
        loss_offset = (offset_error * offset_weight).sum() / offset_weight.sum().clamp_min(1e-6)

        cls_loss = bce(preds[..., 2:][obj_mask], targets[..., 2:][obj_mask])
        cls_weight = obj_target[obj_mask].unsqueeze(-1)
        loss_cls = (cls_loss * cls_weight).sum() / cls_weight.sum().clamp_min(1e-6)
    else:
        loss_offset = torch.tensor(0.0, device=preds.device)
        loss_cls = torch.tensor(0.0, device=preds.device)
        
    return loss_obj, loss_offset, loss_cls

def extract_events_from_grid(tensor, conf_threshold=0.5, cell_duration=0.05, num_classes=3, is_logits=False):
    """
    Dynamically reconstructs discrete temporal occurrences from a dense 1D structural grid.
    Returns: List of events per batch sample -> [[{'time': t, 'class': c, 'conf': p_c}, ...], ...]
    """
    B, S = tensor.size(0), tensor.size(1)
    batch_events = []
    
    for b in range(B):
        events = []
        for i in range(S):
            obj_val = tensor[b, i, 0]
            p_c = torch.sigmoid(obj_val).item() if is_logits else obj_val.item()
            if p_c >= conf_threshold:
                offset_val = tensor[b, i, 1]
                t_x = torch.sigmoid(offset_val).item() if is_logits else offset_val.item()
                # Determine absolute class classification natively logic
                cls_probs = tensor[b, i, 2:2 + num_classes]
                class_id = torch.argmax(cls_probs).item()
                
                time_rel = (i * cell_duration) + (t_x * cell_duration)
                events.append({'time': time_rel, 'class': class_id, 'conf': p_c})
        batch_events.append(events)
    return batch_events


def extract_peak_events_from_grid(tensor, conf_threshold=0.5, cell_duration=0.05, num_classes=3, is_logits=False):
    """Extract one event per local peak in objectness for soft targets or softened predictions."""
    B, S = tensor.size(0), tensor.size(1)
    batch_events = []

    for b in range(B):
        events = []
        for i in range(S):
            obj_val = tensor[b, i, 0]
            p_c = torch.sigmoid(obj_val).item() if is_logits else obj_val.item()
            left_val = tensor[b, i - 1, 0] if i > 0 else obj_val
            right_val = tensor[b, i + 1, 0] if i + 1 < S else obj_val
            left_p = torch.sigmoid(left_val).item() if is_logits else left_val.item()
            right_p = torch.sigmoid(right_val).item() if is_logits else right_val.item()

            if p_c < conf_threshold:
                continue
            if p_c < left_p or p_c < right_p:
                continue

            offset_val = tensor[b, i, 1]
            t_x = torch.sigmoid(offset_val).item() if is_logits else offset_val.item()
            cls_probs = tensor[b, i, 2:2 + num_classes]
            class_id = torch.argmax(cls_probs).item()
            time_rel = (i * cell_duration) + (t_x * cell_duration)
            events.append({'time': time_rel, 'class': class_id, 'conf': p_c})

        batch_events.append(events)

    return batch_events


def _match_event_batches(pred_batch, true_batch, tau, num_classes):
    stats = _init_temporal_stats(num_classes)

    for preds, trues in zip(pred_batch, true_batch):
        matched_preds = set()
        matched_trues = set()

        preds = sorted(preds, key=lambda x: x['conf'], reverse=True)

        for p_idx, p in enumerate(preds):
            best_dist = float('inf')
            best_t_idx = -1

            for t_idx, t in enumerate(trues):
                if t_idx in matched_trues:
                    continue
                if p['class'] != t['class']:
                    continue

                dist = abs(p['time'] - t['time'])
                if dist <= tau and dist < best_dist:
                    best_dist = dist
                    best_t_idx = t_idx

            if best_t_idx != -1:
                stats['tp'] += 1
                matched_preds.add(p_idx)
                matched_trues.add(best_t_idx)
                stats['abs_errors'].append(best_dist)
                stats['sq_errors'].append(best_dist ** 2)
                stats['per_class'][p['class']]['tp'] += 1

        for p_idx, pred in enumerate(preds):
            if p_idx not in matched_preds:
                stats['fp'] += 1
                stats['per_class'][pred['class']]['fp'] += 1

        for t_idx, true_event in enumerate(trues):
            if t_idx not in matched_trues:
                stats['fn'] += 1
                stats['per_class'][true_event['class']]['fn'] += 1

    return stats

def _init_temporal_stats(num_classes):
    return {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'abs_errors': [],
        'sq_errors': [],
        'per_class': {cls_id: {'tp': 0, 'fp': 0, 'fn': 0} for cls_id in range(num_classes)},
    }


def _accumulate_temporal_stats(total_stats, batch_stats):
    total_stats['tp'] += batch_stats['tp']
    total_stats['fp'] += batch_stats['fp']
    total_stats['fn'] += batch_stats['fn']
    total_stats['abs_errors'].extend(batch_stats['abs_errors'])
    total_stats['sq_errors'].extend(batch_stats['sq_errors'])

    for cls_id, class_stats in batch_stats['per_class'].items():
        total_stats['per_class'][cls_id]['tp'] += class_stats['tp']
        total_stats['per_class'][cls_id]['fp'] += class_stats['fp']
        total_stats['per_class'][cls_id]['fn'] += class_stats['fn']

    return total_stats


def _finalize_temporal_stats(stats):
    precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0.0
    recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mae = float(np.mean(stats['abs_errors'])) if stats['abs_errors'] else 0.0
    rmse = float(np.sqrt(np.mean(stats['sq_errors']))) if stats['sq_errors'] else 0.0

    per_class_metrics = {}
    for cls_id, class_stats in stats['per_class'].items():
        cls_precision = class_stats['tp'] / (class_stats['tp'] + class_stats['fp']) if (class_stats['tp'] + class_stats['fp']) > 0 else 0.0
        cls_recall = class_stats['tp'] / (class_stats['tp'] + class_stats['fn']) if (class_stats['tp'] + class_stats['fn']) > 0 else 0.0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0.0
        per_class_metrics[cls_id] = {
            'label': CLASS_LABELS.get(cls_id, str(cls_id)),
            'tp': class_stats['tp'],
            'fp': class_stats['fp'],
            'fn': class_stats['fn'],
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1,
        }

    return {
        'tp': stats['tp'],
        'fp': stats['fp'],
        'fn': stats['fn'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mae': mae,
        'rmse': rmse,
        'per_class': per_class_metrics,
    }


def calc_temporal_metrics(preds_tensor, trues_tensor, tau=0.25, conf_threshold=0.5, num_classes=3, cell_duration=0.05, peak_mode=False):
    """Collect raw TP/FP/FN and localization errors for a batch."""
    decoder = extract_peak_events_from_grid if peak_mode else extract_events_from_grid
    pred_batch = decoder(
        preds_tensor,
        conf_threshold=conf_threshold,
        cell_duration=cell_duration,
        num_classes=num_classes,
        is_logits=True,
    )
    true_batch = decoder(
        trues_tensor,
        conf_threshold=0.5,
        cell_duration=cell_duration,
        num_classes=num_classes,
        is_logits=False,
    )
    return _match_event_batches(pred_batch, true_batch, tau=tau, num_classes=num_classes)


def evaluate_loader_metrics(model, data_loader, device, tau=0.25, conf_threshold=0.5, num_classes=3, cell_duration=0.05, peak_mode=False):
    """Evaluate detection and localization metrics for a single confidence threshold."""
    total_stats = _init_temporal_stats(num_classes)

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            batch_stats = calc_temporal_metrics(
                preds,
                y,
                tau=tau,
                conf_threshold=conf_threshold,
                num_classes=num_classes,
                cell_duration=cell_duration,
                peak_mode=peak_mode,
            )
            total_stats = _accumulate_temporal_stats(total_stats, batch_stats)

    return _finalize_temporal_stats(total_stats)


def evaluate_loader_regression_metrics(model, data_loader, device, soft_offset_mask_threshold=0.1, n_bins=15):
    totals = {
        'obj_mae': 0.0,
        'obj_brier': 0.0,
        'obj_ece': 0.0,
        'cls_brier': 0.0,
        'offset_mae': 0.0,
        'offset_rmse': 0.0,
    }
    batches = 0

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            metrics = calc_regression_metrics(
                preds,
                y,
                soft_offset_mask_threshold=soft_offset_mask_threshold,
                n_bins=n_bins,
            )
            for key in totals:
                totals[key] += metrics[key]
            batches += 1

    if batches == 0:
        return totals

    return {key: value / batches for key, value in totals.items()}

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
    val_metrics_every_n_epochs=1,
    model_mode="1d",
    model_in_channels=18,
    model_image_channels=1,
    event_supervision="hard",
    gaussian_sigma_cells=1.0,
    gaussian_radius_cells=3.0,
    soft_offset_mask_threshold=0.1,
    regression_metric_bins=15,
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

    model_mode = str(model_mode).lower()
    if model_mode not in {"1d", "2d"}:
        raise ValueError(f"Unsupported model_mode: {model_mode}. Expected '1d' or '2d'.")

    event_supervision = str(event_supervision).lower()
    if event_supervision not in {"hard", "soft"}:
        raise ValueError(f"Unsupported event_supervision: {event_supervision}. Expected 'hard' or 'soft'.")

    with open(results_file, "w") as f:
        f.write(f"EEG Seizure {model_mode.upper()} Multi-Channel YOLO Regression Training\n")
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
        input_mode=model_mode,
        target_mode=event_supervision,
        target_config={
            'gaussian_sigma_cells': gaussian_sigma_cells,
            'gaussian_radius_cells': gaussian_radius_cells,
            'soft_offset_mask_threshold': soft_offset_mask_threshold,
        },
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
        input_mode=model_mode,
        target_mode=event_supervision,
        target_config={
            'gaussian_sigma_cells': gaussian_sigma_cells,
            'gaussian_radius_cells': gaussian_radius_cells,
            'soft_offset_mask_threshold': soft_offset_mask_threshold,
        },
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
        input_mode=model_mode,
        target_mode=event_supervision,
        target_config={
            'gaussian_sigma_cells': gaussian_sigma_cells,
            'gaussian_radius_cells': gaussian_radius_cells,
            'soft_offset_mask_threshold': soft_offset_mask_threshold,
        },
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

    model_input_channels = int(model_image_channels if model_mode == "2d" else model_in_channels)
    model = build_model(model_mode, model_input_channels, S, num_classes).to(device)
    cell_duration = float(window_size_sec) / float(S)
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
        train_offset = 0.0
        train_cls = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            
            l_obj, l_offset, l_cls = yolo_loss(preds, y, obj_pos_weight=obj_pos_weight)
            
            # YOLO weighted heuristic: object detection logic dominates bounding offset shifts
            loss = 5.0 * l_obj + l_offset + l_cls
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_obj += l_obj.item()
            train_offset += l_offset.item()
            train_cls += l_cls.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'obj': f"{l_obj.item():.4f}"})
            
        n_train = len(train_loader)
        train_loss /= n_train
        train_obj /= n_train
        train_offset /= n_train
        train_cls /= n_train
        
        model.eval()
        val_loss = 0.0
        val_obj = 0.0
        val_offset = 0.0
        val_cls = 0.0
        val_stats = _init_temporal_stats(num_classes)
        val_reg_stats = {
            'obj_mae': 0.0,
            'obj_brier': 0.0,
            'obj_ece': 0.0,
            'cls_brier': 0.0,
            'offset_mae': 0.0,
            'offset_rmse': 0.0,
        }
        val_batches = 0
        compute_val_temporal = (
            int(val_metrics_every_n_epochs) > 0
            and (epoch == 1 or epoch == epochs or (epoch % int(val_metrics_every_n_epochs) == 0))
        )
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Val", leave=False)
            for x, y in val_pbar:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                
                l_obj, l_offset, l_cls = yolo_loss(preds, y, obj_pos_weight=obj_pos_weight)
                loss = 5.0 * l_obj + l_offset + l_cls
                
                val_loss += loss.item()
                val_obj += l_obj.item()
                val_offset += l_offset.item()
                val_cls += l_cls.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                if compute_val_temporal:
                    if event_supervision == "soft":
                        batch_reg = calc_regression_metrics(
                            preds,
                            y,
                            soft_offset_mask_threshold=soft_offset_mask_threshold,
                            n_bins=regression_metric_bins,
                        )
                        for key in val_reg_stats:
                            val_reg_stats[key] += batch_reg[key]
                        val_batches += 1
                    else:
                        batch_stats = calc_temporal_metrics(
                            preds,
                            y,
                            tau=0.25,
                            conf_threshold=conf_threshold,
                            num_classes=num_classes,
                            cell_duration=cell_duration,
                            peak_mode=False,
                        )
                        val_stats = _accumulate_temporal_stats(val_stats, batch_stats)
                
        if len(val_loader) > 0:
            n_val = len(val_loader)
            val_loss /= n_val
            val_obj /= n_val
            val_offset /= n_val
            val_cls /= n_val
            
        val_metrics = _finalize_temporal_stats(val_stats) if compute_val_temporal and event_supervision == "hard" else None
        val_reg_metrics = None
        if compute_val_temporal and event_supervision == "soft" and val_batches > 0:
            val_reg_metrics = {key: value / val_batches for key, value in val_reg_stats.items()}
        
        log_line = (f"Epoch {epoch}/{epochs}:\n"
                    f"  [Train] Loss={train_loss:.4f} | Obj={train_obj:.4f}\n"
                    f"  [Val]   Loss={val_loss:.4f}\n")
        if compute_val_temporal and event_supervision == "soft" and val_reg_metrics is not None:
            log_line += (
                f"  [Val:Regression] ObjMAE={val_reg_metrics['obj_mae']:.4f} | ObjBrier={val_reg_metrics['obj_brier']:.4f} | "
                f"ObjECE={val_reg_metrics['obj_ece']:.4f} | OffMAE={val_reg_metrics['offset_mae']:.4f} | "
                f"OffRMSE={val_reg_metrics['offset_rmse']:.4f} | ClsBrier={val_reg_metrics['cls_brier']:.4f}\n"
            )
        elif compute_val_temporal and val_metrics is not None:
            log_line += (
                f"  [Val:Temporal] F1={val_metrics['f1']:.4f} | P={val_metrics['precision']:.4f} | "
                f"R={val_metrics['recall']:.4f} | MAE={val_metrics['mae']:.4f}s | RMSE={val_metrics['rmse']:.4f}s\n"
            )
            for cls_id, cls_metrics in val_metrics['per_class'].items():
                log_line += (
                    f"    [Val:{cls_metrics['label']}] P={cls_metrics['precision']:.4f} "
                    f"R={cls_metrics['recall']:.4f} F1={cls_metrics['f1']:.4f} "
                    f"TP={cls_metrics['tp']} FP={cls_metrics['fp']} FN={cls_metrics['fn']}\n"
                )
        else:
            log_line += (
                f"  [Val:Temporal] Skipped (runs every {int(val_metrics_every_n_epochs)} epoch(s), "
                f"plus first/last epoch).\n"
            )
        
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
    test_offset = 0.0
    test_cls = 0.0
    test_stats = _init_temporal_stats(num_classes)
    test_reg_stats = {
        'obj_mae': 0.0,
        'obj_brier': 0.0,
        'obj_ece': 0.0,
        'cls_brier': 0.0,
        'offset_mae': 0.0,
        'offset_rmse': 0.0,
    }
    test_batches = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            
            l_obj, l_offset, l_cls = yolo_loss(preds, y, obj_pos_weight=obj_pos_weight)
            loss = 5.0 * l_obj + l_offset + l_cls
            
            test_loss += loss.item()
            test_obj += l_obj.item()
            test_offset += l_offset.item()
            test_cls += l_cls.item()

            if event_supervision == "soft":
                batch_reg = calc_regression_metrics(
                    preds,
                    y,
                    soft_offset_mask_threshold=soft_offset_mask_threshold,
                    n_bins=regression_metric_bins,
                )
                for key in test_reg_stats:
                    test_reg_stats[key] += batch_reg[key]
                test_batches += 1
            
            batch_stats = calc_temporal_metrics(
                preds,
                y,
                tau=0.25,
                conf_threshold=conf_threshold,
                num_classes=num_classes,
                cell_duration=cell_duration,
                peak_mode=(event_supervision == "soft"),
            )
            test_stats = _accumulate_temporal_stats(test_stats, batch_stats)
            
    if len(test_loader) > 0:
        n_test = len(test_loader)
        test_loss /= n_test
        test_obj /= n_test
        test_offset /= n_test
        test_cls /= n_test
        
    test_metrics = _finalize_temporal_stats(test_stats)
    test_reg_metrics = None
    if event_supervision == "soft" and test_batches > 0:
        test_reg_metrics = {key: value / test_batches for key, value in test_reg_stats.items()}
        
    test_msg = (f"\n--- TEST SET TEMPORAL METRICS (tau=0.25s) ---\n"
                f"Validation Box Detections: {test_metrics['tp'] + test_metrics['fp']}\n"
                f"True Annotations Mapped:   {test_metrics['tp'] + test_metrics['fn']}\n\n"
                f"True Positives (TP):  {test_metrics['tp']}\n"
                f"False Positives (FP): {test_metrics['fp']}\n"
                f"False Negatives (FN): {test_metrics['fn']}\n\n"
                f"Precision: {test_metrics['precision']:.4f}\n"
                f"Recall:    {test_metrics['recall']:.4f}\n"
                f"F1-Score:  {test_metrics['f1']:.4f}\n"
                f"MAE:       {test_metrics['mae']:.4f}s\n"
                f"RMSE:      {test_metrics['rmse']:.4f}s\n"
                f"-------------------------------------------\n"
                f"Total Objective Loss: {test_loss:.4f}\n")
    if test_reg_metrics is not None:
        test_msg += (
            f"Regression Metrics: ObjMAE={test_reg_metrics['obj_mae']:.4f} ObjBrier={test_reg_metrics['obj_brier']:.4f} "
            f"ObjECE={test_reg_metrics['obj_ece']:.4f} OffMAE={test_reg_metrics['offset_mae']:.4f} "
            f"OffRMSE={test_reg_metrics['offset_rmse']:.4f} ClsBrier={test_reg_metrics['cls_brier']:.4f}\n"
        )
    for cls_id, cls_metrics in test_metrics['per_class'].items():
        test_msg += (
            f"Class {cls_metrics['label']}: P={cls_metrics['precision']:.4f} "
            f"R={cls_metrics['recall']:.4f} F1={cls_metrics['f1']:.4f} "
            f"TP={cls_metrics['tp']} FP={cls_metrics['fp']} FN={cls_metrics['fn']}\n"
        )
    
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
                cell_duration=cell_duration,
                peak_mode=(event_supervision == "soft"),
            )
            line = (
                f"thr={float(th):.3f} | "
                f"P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"MAE={metrics['mae']:.4f}s RMSE={metrics['rmse']:.4f}s "
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
                f"(F1={best_row['f1']:.4f}, P={best_row['precision']:.4f}, R={best_row['recall']:.4f}, "
                f"MAE={best_row['mae']:.4f}s, RMSE={best_row['rmse']:.4f}s)"
            )
            print(best_line)
            with open(results_file, "a") as f:
                f.write(best_line + "\n")

    print(f"Results stored in {results_file} in current directory.")

if __name__ == '__main__':
    from src.config import DATASET, MODEL, PATHS, TRAINING

    model_mode = str(MODEL.get("mode", "1d")).lower()
    
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
        val_metrics_every_n_epochs=TRAINING.get("val_metrics_every_n_epochs", 1),
        model_mode=model_mode,
        model_in_channels=MODEL.get("in_channels", 18),
        model_image_channels=MODEL.get("image_channels", 1),
        event_supervision=TRAINING.get("event_supervision", "hard"),
        gaussian_sigma_cells=TRAINING.get("gaussian_sigma_cells", 1.0),
        gaussian_radius_cells=TRAINING.get("gaussian_radius_cells", 3.0),
        soft_offset_mask_threshold=TRAINING.get("soft_offset_mask_threshold", 0.1),
        regression_metric_bins=TRAINING.get("regression_metric_bins", 15),
    )
