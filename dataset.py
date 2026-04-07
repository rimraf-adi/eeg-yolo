import os
import glob
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import mne

mne.set_log_level('WARNING')

class EEGRegressionDataset(Dataset):
    """
    Dataset to yield 10-second sliding windows of 18-channel EEG data.
    Uses specific bipolar references and preloads everything into RAM.
    Targets are [start_time, end_time, label_fraction] for the window.
    """
    def __init__(self, data_dir, anno_dir, window_sec=10, stride_sec=1, fs=256, allowed_pids=None):
        self.data_dir = Path(data_dir)
        self.anno_dir = Path(anno_dir)
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.fs = fs
        self.window_samples = int(window_sec * fs)
        self.stride_samples = int(stride_sec * fs)
        
        self.bipolar_pairs = [
            ('EEG Fp1-REF', 'EEG F7-REF'), ('EEG F7-REF',  'EEG T3-REF'),
            ('EEG T3-REF',  'EEG T5-REF'), ('EEG T5-REF',  'EEG O1-REF'),
            ('EEG Fp1-REF', 'EEG F3-REF'), ('EEG F3-REF',  'EEG C3-REF'),
            ('EEG C3-REF',  'EEG P3-REF'), ('EEG P3-REF',  'EEG O1-REF'),
            ('EEG Fz-REF',  'EEG Cz-REF'), ('EEG Cz-REF',  'EEG Pz-REF'),
            ('EEG Fp2-REF', 'EEG F4-REF'), ('EEG F4-REF',  'EEG C4-REF'),
            ('EEG C4-REF',  'EEG P4-REF'), ('EEG P4-REF',  'EEG O2-REF'),
            ('EEG Fp2-REF', 'EEG F8-REF'), ('EEG F8-REF',  'EEG T4-REF'),
            ('EEG T4-REF',  'EEG T6-REF'), ('EEG T6-REF',  'EEG O2-REF'),
        ]
        
        self.desired_order = [
            'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
            'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
            'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
            'Fz-Cz', 'Cz-Pz',
        ]
        
        self._prepare_channel_mappings()
        
        self.eeg_data = {} # pid -> numpy array (18, n_samples)
        self.samples = []
        
        # Load indices and data
        self._load_and_build(allowed_pids)

    def _prepare_channel_mappings(self):
        def normalize(ch): return ch.strip().upper()
        def pair_name(p): 
            l = p[0].replace('EEG ', '').replace('-REF', '')
            r = p[1].replace('EEG ', '').replace('-REF', '')
            return f"{l}-{r}"

        clean_pairs = [(normalize(a), normalize(b)) for a, b in self.bipolar_pairs]
        name_map = {pair_name(p): cp for p, cp in zip(self.bipolar_pairs, clean_pairs)}
        
        self.reordered_pairs = [name_map[name] for name in self.desired_order if name in name_map]
        self.anode = [a for a, _ in self.reordered_pairs]
        self.cathode = [b for _, b in self.reordered_pairs]
        
        def pretty(ch): return ch.replace('EEG ', '').replace('-REF', '').capitalize()
        self.ch_names = [f"{pretty(a)}-{pretty(b)}" for a, b in self.reordered_pairs]

    def _get_array(self, filename: str):
        try:
            raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
            raw.rename_channels(lambda ch: ch.upper())
            
            drop = ['ECG EKG', 'RESP EFFORT', 'ECG EKG-REF', 'RESP EFFORT-REF']
            raw.drop_channels([c for c in drop if c in raw.ch_names])
            
            # Ensure required channels exist
            missing_anodes = [a for a in self.anode if a not in raw.ch_names]
            missing_cathodes = [c for c in self.cathode if c not in raw.ch_names]
            
            if missing_anodes or missing_cathodes:
                print(f"File {filename} is missing some channels. Skipping referencing fallback.")
                return None
                
            raw = mne.set_bipolar_reference(raw, anode=self.anode, cathode=self.cathode, ch_name=self.ch_names, copy=False, verbose=False)
            
            # Reorder to match desired exactly
            raw.pick_channels(self.ch_names)
            
            data = raw.get_data().astype(np.float32)
            return data
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None

    def _load_and_build(self, allowed_pids):
        patient_folders = glob.glob(str(self.anno_dir / 'patient_*'))
        patient_folders.sort()

        print("🔵 Loading EEG data into RAM...")
        for p_folder in patient_folders:
            p_name = os.path.basename(p_folder)
            pid = int(p_name.split('_')[1])
            
            if allowed_pids is not None and pid not in allowed_pids:
                continue
                
            edf_path = self.data_dir / f'eeg{pid}.edf'
            if not edf_path.exists():
                print(f"Skipping {p_name}, no EDF found at {edf_path}")
                continue
                
            csv_path = Path(p_folder) / 'annotations_w256_s256_strict.csv'
            if not csv_path.exists():
                continue
                
            # Preload the array fully 
            data_arr = self._get_array(str(edf_path))
            if data_arr is None:
                continue
                
            self.eeg_data[pid] = data_arr
            df = pd.read_csv(csv_path)
            raw_labels = df['label'].values
            
            max_samples = data_arr.shape[1]
            num_1s_blocks = len(raw_labels)
            
            # Generate overlapping windows
            for start_idx in range(0, num_1s_blocks - self.window_sec + 1, self.stride_sec):
                end_idx = start_idx + self.window_sec
                
                sample_start = start_idx * self.fs
                sample_end = end_idx * self.fs
                if sample_end > max_samples:
                    break
                    
                window_labels = raw_labels[start_idx:end_idx]
                
                pos_indices = np.where(window_labels > 0)[0]
                if len(pos_indices) > 0:
                    start_time = float(pos_indices[0])
                    end_time = float(pos_indices[-1] + 1)
                    label_frac = float(len(pos_indices)) / self.window_sec
                else:
                    start_time = 0.0
                    end_time = 0.0
                    label_frac = 0.0
                    
                self.samples.append({
                    'pid': pid,
                    'sample_start': sample_start,
                    'sample_end': sample_end,
                    'target': [start_time, end_time, label_frac]
                })
        
        # Cleanup any memory leftover
        gc.collect()

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        s = self.samples[idx]
        pid = s['pid']
        
        # Super fast array slicing from RAM
        data = self.eeg_data[pid][:, s['sample_start']:s['sample_end']].copy()
        
        # Standardization across time for each channel
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True) + 1e-6
        data = (data - mean) / std
        
        x = torch.from_numpy(data)
        y = torch.tensor(s['target'], dtype=torch.float32)
        return x, y
