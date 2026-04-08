import math
import torch
import numpy as np

def build_target(annotations_df, t_win_start, t_win_end, config):
    """
    Constructs the formalized YOLO 1D target tensor using localized relativized timestamps.
    Shape mappings align to: [S, 1, 2 + num_classes]
    - [..., 0] = Objectness (1.0 vs 0.0)
    - [..., 1] = Cell Offset strictly bound [0.0, 1.0)
    - [..., 2:] = Classes One-hot explicitly.
    """
    S = int(config.get('S', 100))
    window_size_sec = float(config.get('window_size_sec', 10.0))
    num_classes = int(config.get('num_classes', 3))
    
    # Adhere strictly to the model's structural expected dimensions to prevent tensor conflicts
    target = torch.zeros((S, 1, 2 + num_classes), dtype=torch.float32)
    
    if annotations_df is None or len(annotations_df) == 0:
        return target
        
    for _, event in annotations_df.iterrows():
        # Unpack event attributes safely
        t_center_abs = event['t_center_abs']
        is_segment = event.get('is_segment', False)
        class_id = int(event['class_id'])
        rel_width_abs = event['rel_width_abs']
        
        # 1. Filter events organically to window bounds constraints
        if t_center_abs < t_win_start or t_center_abs >= t_win_end:
            # Segment overlaps outside the center are excluded per spec!
            # Sleeping & Points outside are excluded
            continue
            
        # 2. Relativize strictly mappings relative to the current window starting slice
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
        
        # Determine segment width mapping constraints locally (Though model is 1D 2+num_classes, we keep width ready if needed)
        # width = (rel_width_abs / window_size_sec) if is_segment else (1.0 / S) # Handled safely!
        
        # Map onto target ensuring objectness and classes align perfectly locally
        # If collision exists natively, overwrite offset and fuse classes defensively
        if target[cell_idx, 0, 0] == 1.0:
            target[cell_idx, 0, 2 + class_id] = 1.0
            continue
            
        target[cell_idx, 0, 0] = 1.0
        target[cell_idx, 0, 1] = cell_offset
        target[cell_idx, 0, 2 + class_id] = 1.0
        
    return target
