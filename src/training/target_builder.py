import math
import torch
import numpy as np


def _gaussian_weight(distance_cells, sigma_cells):
    sigma_cells = max(float(sigma_cells), 1e-6)
    return float(np.exp(-0.5 * (distance_cells / sigma_cells) ** 2))

def build_target(annotations_df, t_win_start, t_win_end, config):
    """
    Constructs the formalized YOLO 1D target tensor using localized point annotations.
    Shape mappings align to: [S, 2 + num_classes]
    - [..., 0] = Objectness (1.0 vs 0.0)
    - [..., 1] = Cell Offset strictly bound [0.0, 1.0)
    - [..., 2:] = Classes One-hot explicitly.
    """
    S = int(config.get('S', 100))
    window_size_sec = float(config.get('window_size_sec', 10.0))
    num_classes = int(config.get('num_classes', 3))
    
    # Adhere strictly to the model's structural expected dimensions to prevent tensor conflicts
    target = torch.zeros((S, 2 + num_classes), dtype=torch.float32)
    
    if annotations_df is None or len(annotations_df) == 0:
        return target
        
    for _, event in annotations_df.iterrows():
        # Unpack event attributes safely
        t_center_abs = event['t_center_abs']
        class_id = int(event['class_id'])
        
        # 1. Filter events organically to window bounds constraints.
        if t_center_abs < t_win_start or t_center_abs >= t_win_end:
            continue
            
        # 2. Relativize mappings relative to the current window start.
        rel_center = (t_center_abs - t_win_start) / window_size_sec
        
        # Guard strictly
        if rel_center < 0.0 or rel_center >= 1.0:
            continue
            
        # 3. Discretize seamlessly into the structural 'S' grids
        cell_idx = int(math.floor(rel_center * S))
        
        # Safety bound guards natively checking
        if cell_idx >= S: cell_idx = S - 1
        if cell_idx < 0: cell_idx = 0
            
        # Offset cleanly strictly inside the cell's boundaries locally
        cell_offset = (rel_center * S) - cell_idx
        
        # Apply bounds limiting logically ensuring strict constraint
        cell_offset = max(0.0, min(0.99999, float(cell_offset)))
        
        # Map onto target ensuring objectness and classes align perfectly locally.
        if target[cell_idx, 0] == 1.0:
            print(f"Warning: Grid collision detected at index {cell_idx}. Overwriting older event.")
            target[cell_idx, 0] = 1.0
            target[cell_idx, 1] = cell_offset
            target[cell_idx, 2:] = 0.0
            target[cell_idx, 2 + class_id] = 1.0
            continue
            
        target[cell_idx, 0] = 1.0
        target[cell_idx, 1] = cell_offset
        target[cell_idx, 2 + class_id] = 1.0
        
    return target


def build_target_soft(annotations_df, t_win_start, t_win_end, config):
    """
    Gaussian-soft target builder that spreads supervision across nearby cells.
    Shape: [S, 2 + num_classes]
    - objectness is a soft peak in [0, 1]
    - offset is a weighted average of local offsets
    - classes are soft per-cell activations
    """
    S = int(config.get('S', 100))
    window_size_sec = float(config.get('window_size_sec', 10.0))
    num_classes = int(config.get('num_classes', 3))
    sigma_cells = float(config.get('gaussian_sigma_cells', 1.0))
    radius_cells = float(config.get('gaussian_radius_cells', 3.0))

    target = torch.zeros((S, 2 + num_classes), dtype=torch.float32)
    if annotations_df is None or len(annotations_df) == 0:
        return target

    objectness = np.zeros(S, dtype=np.float32)
    class_scores = np.zeros((S, num_classes), dtype=np.float32)
    offset_sum = np.zeros(S, dtype=np.float32)
    offset_weight_sum = np.zeros(S, dtype=np.float32)

    for _, event in annotations_df.iterrows():
        t_center_abs = float(event['t_center_abs'])
        class_id = int(event['class_id'])

        if t_center_abs < t_win_start or t_center_abs >= t_win_end:
            continue

        rel_center = (t_center_abs - t_win_start) / window_size_sec
        if rel_center < 0.0 or rel_center >= 1.0:
            continue

        cell_pos = rel_center * S
        cell_idx = int(math.floor(cell_pos))
        cell_idx = max(0, min(S - 1, cell_idx))
        cell_offset = float(cell_pos - cell_idx)
        cell_offset = max(0.0, min(0.99999, cell_offset))

        left = max(0, int(math.floor(cell_pos - radius_cells)))
        right = min(S - 1, int(math.ceil(cell_pos + radius_cells)))

        for grid_idx in range(left, right + 1):
            distance_cells = grid_idx - cell_pos
            weight = _gaussian_weight(distance_cells, sigma_cells)
            if weight <= 0.0:
                continue

            objectness[grid_idx] = max(objectness[grid_idx], weight)
            class_scores[grid_idx, class_id] = max(class_scores[grid_idx, class_id], weight)
            
            # Only set offset for the center cell where the event actually is.
            # Neighbors learn objectness and class but not offset to avoid incorrect localization.
            if grid_idx == cell_idx:
                offset_sum[grid_idx] += weight * cell_offset
                offset_weight_sum[grid_idx] += weight

    target[:, 0] = torch.from_numpy(objectness)
    offset_target = np.zeros(S, dtype=np.float32)
    positive = offset_weight_sum > 0.0
    offset_target[positive] = offset_sum[positive] / offset_weight_sum[positive]
    target[:, 1] = torch.from_numpy(np.clip(offset_target, 0.0, 0.99999))
    target[:, 2:] = torch.from_numpy(np.clip(class_scores, 0.0, 1.0))

    return target
