import os
import glob
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class EEGRegressionDataset(Dataset):
    """
    1D Multi-Channel YOLO DataLoader mapping Continuous Parquets to YOLO format.
    Transforms raw 29-channel EEG signals into an 18-channel Bipolar Subtracted Tensor.
    Converts timestamp annotations into strict S-segment Grid bounding box targets.
    """
    def __init__(self, data_dir, anno_dir, window_size_sec=10.0, stride_sec=10.0, fs=500, S=100, allowed_pids=None):
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
        
        # 1. Global Setup
        self.class_map = {'!': 0, '!start': 1, '!end': 2}
        self.num_classes = len(self.class_map)
        
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
        self.events_cache = {} # pid -> pandas dataframe holding times & classes
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
                
                # Load explicitly the surviving validated events
                events_df = pd.read_csv(events_csv)
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
        # Target Tensor Construction (Y) YOLO [S, 1, 2 + num_classes]
        # ---------------------------------------------------------
        # Shape: [100, 1, 5] since we have 3 classes mapped dynamically.
        Y_tensor = torch.zeros((self.S, 1, 2 + self.num_classes), dtype=torch.float32)
        
        events_df = self.events_cache[pid]
        # Strict window constraint bounding
        window_events = events_df[(events_df['timestamp_sec'] >= start_time) & (events_df['timestamp_sec'] < end_time)]
        
        for _, row in window_events.iterrows():
            label = str(row['label']).strip()
            if label not in self.class_map:
                continue
                
            relative_time = row['timestamp_sec'] - start_time
            i = int(math.floor(relative_time / self.cell_duration))
            
            # Grid bounds limiting
            if i >= self.S: i = self.S - 1
            if i < 0: i = 0
            
            t_x = (relative_time % self.cell_duration) / self.cell_duration
            
            # Warn structural collisions safely
            if Y_tensor[i, 0, 0] == 1.0:
                logger.warning(f"Grid collision detected at index {i} in Patient {pid}. Consider increasing S hyperparameter!")
                
            # Populate Vector
            Y_tensor[i, 0, 0] = 1.0                   # Objectness
            Y_tensor[i, 0, 1] = float(t_x)            # Scaled Offset
            
            # Assign One-Hot Class dynamically mapping classes uniformly
            class_idx = self.class_map[label]
            Y_tensor[i, 0, 2 + class_idx] = 1.0
            
        return X_tensor, Y_tensor
