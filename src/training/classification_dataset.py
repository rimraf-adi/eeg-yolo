import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.training.annotation_parser import parse_annotations

# Window-level multiclass labels.
# 0: no event in window
# 1: !
# 2: !start
# 3: !end
WINDOW_CLASS_MAP = {
    "none": 0,
    "!": 1,
    "!start": 2,
    "!end": 3,
}

ID_TO_LABEL = {v: k for k, v in WINDOW_CLASS_MAP.items()}


def _window_label_from_events(events_df: pd.DataFrame, t_start: float, t_end: float) -> int:
    if events_df is None or len(events_df) == 0:
        return WINDOW_CLASS_MAP["none"]

    in_window = events_df[(events_df["t_center_abs"] >= t_start) & (events_df["t_center_abs"] < t_end)]
    if len(in_window) == 0:
        return WINDOW_CLASS_MAP["none"]

    # Use earliest event in the window for deterministic multiclass labeling.
    first = in_window.sort_values("t_center_abs").iloc[0]
    label = str(first.get("label", "")).strip()
    if label in WINDOW_CLASS_MAP:
        return WINDOW_CLASS_MAP[label]

    class_id = int(first.get("class_id", -1))
    if class_id == 0:
        return WINDOW_CLASS_MAP["!"]
    if class_id == 1:
        return WINDOW_CLASS_MAP["!start"]
    if class_id == 2:
        return WINDOW_CLASS_MAP["!end"]
    return WINDOW_CLASS_MAP["none"]


class EEGWindowClassificationDataset(Dataset):
    def __init__(
        self,
        data_dir,
        anno_dir,
        window_size_sec=1.0,
        stride_sec=1.0,
        fs=500,
        allowed_pids=None,
        input_mode="1d",
    ):
        self.data_dir = Path(data_dir)
        self.anno_dir = Path(anno_dir)
        self.window_size_sec = float(window_size_sec)
        self.stride_sec = float(stride_sec)
        self.fs = int(fs)
        self.input_mode = str(input_mode).lower()

        self.window_samples = int(self.window_size_sec * self.fs)

        # Double banana bipolar mapping (18 channels from 29-channel raw).
        self.bipolar_indices = [
            (1, 3),
            (3, 5),
            (5, 7),
            (7, 9),
            (0, 2),
            (2, 4),
            (4, 6),
            (6, 8),
            (1, 11),
            (11, 13),
            (13, 15),
            (15, 9),
            (0, 10),
            (10, 12),
            (12, 14),
            (14, 8),
            (16, 17),
            (17, 18),
        ]
        self.num_channels = len(self.bipolar_indices)

        self.eeg_cache = {}
        self.events_cache = {}
        self.samples = []

        self._load_and_index(allowed_pids)

    def _read_events(self, csv_path: Path) -> pd.DataFrame:
        # Reprocessed annotation format.
        df = pd.read_csv(csv_path)
        required_reprocessed = {"t_center_abs", "class_id", "label"}
        if required_reprocessed.issubset(set(df.columns)):
            out = df.copy()
            out["t_center_abs"] = pd.to_numeric(out["t_center_abs"], errors="coerce")
            out["class_id"] = pd.to_numeric(out["class_id"], errors="coerce").fillna(-1).astype(int)
            out["label"] = out["label"].astype(str).str.strip()
            out = out.dropna(subset=["t_center_abs"]).sort_values("t_center_abs").reset_index(drop=True)
            return out

        # Fallback to raw parser.
        return parse_annotations(str(csv_path))

    def _load_and_index(self, allowed_pids):
        parquet_files = sorted(glob.glob(str(self.data_dir / "*.parquet")))
        print("[*] Building classification windows (1-second framing)...")

        for pq_path in parquet_files:
            pid = os.path.basename(pq_path).replace(".parquet", "")
            try:
                pid_int = int(pid.replace("P", ""))
            except Exception:
                continue

            if allowed_pids is not None and pid_int not in allowed_pids:
                continue

            events_csv = self.anno_dir / f"{pid}_events.csv"
            if not events_csv.exists():
                continue

            try:
                raw_data = pd.read_parquet(pq_path).values.T
                bipolar_data = np.zeros((self.num_channels, raw_data.shape[1]), dtype=np.float32)
                for c_idx, (anode, cathode) in enumerate(self.bipolar_indices):
                    bipolar_data[c_idx, :] = raw_data[anode, :] - raw_data[cathode, :]
                self.eeg_cache[pid] = bipolar_data

                events_df = self._read_events(events_csv)
                self.events_cache[pid] = events_df

                max_duration = raw_data.shape[1] / self.fs
                start_time = 0.0
                while start_time + self.window_size_sec <= max_duration:
                    end_time = start_time + self.window_size_sec
                    label_id = _window_label_from_events(events_df, start_time, end_time)
                    self.samples.append(
                        {
                            "pid": pid,
                            "start_time": start_time,
                            "end_time": end_time,
                            "label_id": int(label_id),
                        }
                    )
                    start_time += self.stride_sec

            except Exception as ex:
                print(f"Error processing {pid}: {ex}")

    def __len__(self):
        return len(self.samples)

    def class_counts(self):
        counts = np.zeros(len(WINDOW_CLASS_MAP), dtype=np.int64)
        for sample in self.samples:
            counts[int(sample["label_id"])] += 1
        return counts

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample["pid"]
        start_time = sample["start_time"]
        end_time = sample["end_time"]
        label_id = int(sample["label_id"])

        start_idx = int(start_time * self.fs)
        end_idx = int(end_time * self.fs)

        signal = self.eeg_cache[pid]
        max_idx = signal.shape[1]
        actual_end_idx = min(end_idx, max_idx)
        x_raw = signal[:, start_idx:actual_end_idx].copy()

        x = np.zeros((self.num_channels, self.window_samples), dtype=np.float32)
        x[:, : x_raw.shape[1]] = x_raw

        # Per-window per-channel z-score normalization.
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-7
        x = (x - mean) / std

        x_tensor = torch.tensor(x, dtype=torch.float32)
        if self.input_mode == "2d":
            x_tensor = x_tensor.unsqueeze(0)

        y_tensor = torch.tensor(label_id, dtype=torch.long)
        return x_tensor, y_tensor
