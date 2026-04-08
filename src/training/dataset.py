import os
import glob
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.training.annotation_parser import parse_annotations
from src.training.target_builder import build_target

logger = logging.getLogger(__name__)

class EEGRegressionDataset(Dataset):
    """
    1D Multi-Channel YOLO DataLoader mapping Continuous Parquets to YOLO format.
    Transforms raw 29-channel EEG signals into an 18-channel Bipolar Subtracted Tensor.
    Converts timestamp annotations into strict S-segment Grid bounding box targets.
    """
    def __init__(self, data_dir, anno_dir, window_size_sec=10.0, stride_sec=10.0, fs=500, S=100, num_classes=3, allowed_pids=None):
        self.data_dir = Path(data_dir)
        self.anno_dir = Path(anno_dir)
        self.window_size_sec = float(window_size_sec)
        self.stride_sec = float(stride_sec)
        self.fs = int(fs)
        self.S = int(S)
        self.cell_duration = self.window_size_sec / self.S
        
        # Exact mathematical samples mapped internally
        raw_samples = int(self.window_size_sec * self.fs)
        
        # PyTorch YOLO FPN inherently halves sizes across 5 pooling stages (2^5 = 32 grids). 
        # Sequences must natively align mathematically strictly maintaining multiples of 32 to skip tensor dimension crashes!
        self.window_samples = int(math.ceil(raw_samples / 32) * 32)
        self.stride_samples = int(self.stride_sec * self.fs)
        
        # 1. Global Setup — class mapping handled by annotation_parser:
        #    Sleeping=0, !=1, !start/!end segments=2
        self.num_classes = int(num_classes)
        
        # Explicit Bipolar Pair Matrix mapping directly from the 29-channel layout
        self.bipolar_indices = [
            (1, 3),   # Fp2-F4
            (3, 5),   # F4-C4
            (5, 7),   # C4-P4
            (7, 9),   # P4-O2
            (0, 2),   # Fp1-F3
            (2, 4),   # F3-C3
            (4, 6),   # C3-P3
            (6, 8),   # P3-O1
            (1, 11),  # Fp2-F8
            (11, 13), # F8-T4
            (13, 15), # T4-T6
            (15, 9),  # T6-O2
            (0, 10),  # Fp1-F7
            (10, 12), # F7-T3
            (12, 14), # T3-T5
            (14, 8),  # T5-O1
            (16, 17), # Fz-Cz
            (17, 18), # Cz-Pz
        ]
        self.num_channels = len(self.bipolar_indices)
        
        self.eeg_cache = {}    # pid -> memmapped fast numpy array [18, time]
        self.events_cache = {} # pid -> parsed annotations DataFrame [t_center_abs, rel_width_abs, class_id, is_segment]
        self.samples = []
        
        self._load_and_build(allowed_pids)

    def _load_and_build(self, allowed_pids):
        parquet_files = sorted(glob.glob(str(self.data_dir / "*.parquet")))
        
        print(f"🔵 Scanning Dataset... Analyzing extracted events natively.")
        for pq_path in parquet_files:
            pid = os.path.basename(pq_path).replace('.parquet', '')
            
            # Prune generic ID prefix 'P', allowing integer subset targeting if specified
            try:
                pid_int = int(pid.replace('P', ''))
            except:
                continue
                
            if allowed_pids is not None and pid_int not in allowed_pids:
                continue
                
            events_csv = self.anno_dir / f"{pid}_events.csv"
            if not events_csv.exists():
                continue
                
            """
            Optimized loading:
            1. Read PyArrow columnar formatting cleanly.
            2. Compute standard double banana transform directly across columns in RAM avoiding memory spikes.
            """
            try:
                # Transpose into shape (29, timestamps)
                raw_data = pd.read_parquet(pq_path).values.T
                
                # Perform the mathematical channel subtraction bridging identically
                bipolar_data = np.zeros((self.num_channels, raw_data.shape[1]), dtype=np.float32)
                for c_idx, (anode, cathode) in enumerate(self.bipolar_indices):
                    bipolar_data[c_idx, :] = raw_data[anode, :] - raw_data[cathode, :]
                    
                self.eeg_cache[pid] = bipolar_data
                
                # Parse annotations using dedicated parser (handles segment pairing,
                # correct class mapping: Sleeping=0, !=1, !start/!end segments=2)
                events_df = parse_annotations(str(events_csv))
                self.events_cache[pid] = events_df
                
                max_duration = raw_data.shape[1] / self.fs
                
                # Build Window Manifests
                start_time = 0.0
                while start_time + self.window_size_sec <= max_duration:
                    end_time = start_time + self.window_size_sec
                    
                    self.samples.append({
                        'pid': pid,
                        'start_time': start_time,
                        'end_time': end_time
                    })
                    start_time += self.stride_sec
                    
                # Handle trailing padding chunk
                if max_duration - (start_time - self.stride_sec) > self.window_size_sec + 0.1:
                    self.samples.append({
                        'pid': pid, 
                        'start_time': start_time, 
                        'end_time': start_time + self.window_size_sec
                    })
                    
            except Exception as e:
                print(f"Error buffering {pid}: {e}")

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        s = self.samples[idx]
        pid = s['pid']
        start_time = s['start_time']
        end_time = s['end_time']
        
        # ---------------------------------------------------------
        # Input Tensor Construction (X)
        # ---------------------------------------------------------
        start_idx = int(start_time * self.fs)
        end_idx = int(end_time * self.fs)
        
        full_signal = self.eeg_cache[pid]
        max_idx = full_signal.shape[1]
        
        actual_end_idx = min(end_idx, max_idx)
        raw_x = full_signal[:, start_idx:actual_end_idx].copy()
        
        # Edge sequence padding exactly maintaining [18, L] tensor
        padded_x = np.zeros((self.num_channels, self.window_samples), dtype=np.float32)
        padded_x[:, :raw_x.shape[1]] = raw_x
        
        # Z-score standardization locally
        mean = padded_x.mean(axis=1, keepdims=True)
        std = padded_x.std(axis=1, keepdims=True) + 1e-7
        norm_x = (padded_x - mean) / std
        
        X_tensor = torch.tensor(norm_x, dtype=torch.float32)
        
        # ---------------------------------------------------------
        # Target Tensor Construction (Y) via build_target
        # Uses parsed annotations with correct class mapping:
        #   Sleeping=0, !=1, !start/!end segments=2
        # All timestamps are relativized to the window start.
        # ---------------------------------------------------------
        config = {
            'S': self.S,
            'window_size_sec': self.window_size_sec,
            'num_classes': self.num_classes,
        }
        annotations_df = self.events_cache[pid]
        Y_tensor = build_target(annotations_df, start_time, end_time, config)
        
        return X_tensor, Y_tensor
