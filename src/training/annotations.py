"""
Dense YOLO-style Annotation Generator for EEG Seizure Dataset
=============================================================
Produces sliding-window annotations from the TUH-style dataset used in V_One.

Each row in the output represents one (patient, window) pair with:
  - patient_id, window_idx, start_sample, end_sample
  - label          : hard label (0/1) based on --label-mode
  - seizure_frac   : fraction of samples in window that are ictal (always stored)

Usage
-----
python annotate_dense.py \
    --window   512   \          # window size in samples  (default 512 = 2 s @ 256 Hz)
    --stride   256   \          # stride in samples        (default 256 = 50 % overlap)
    --label-mode majority \     # any | majority | strict | soft
    --output-mode both \        # per_patient | merged | both
    --data-root   ./data \
    --anno-dir    .    \
    --output-root ./annotations_dense
"""

import os
import csv
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne

mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

FS = 256  # sampling frequency (Hz)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def load_annotations(anno_dir: str):
    """
    Load the three annotation CSVs and majority-vote merge them.
    Returns a dict  {patient_id (1-based int): np.ndarray of per-second labels}.
    Columns in each CSV are named '1' … '79' (patient index).
    """
    paths = [
        Path(anno_dir) / 'annotations_2017_A_fixed.csv',
        Path(anno_dir) / 'annotations_2017_B.csv',
        Path(anno_dir) / 'annotations_2017_C.csv',
    ]
    dfs = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Annotation file not found: {p}")
        dfs.append(pd.read_csv(p))

    result = {}
    for col in dfs[0].columns:
        try:
            pid = int(col)
        except ValueError:
            continue
        try:
            s = dfs[0][col].fillna(0) + dfs[1][col].fillna(0) + dfs[2][col].fillna(0)
            result[pid] = (s >= 2).astype(int).values  # majority vote
        except KeyError:
            pass
    return result


def per_second_to_per_sample(second_labels: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Expand per-second annotation labels to per-sample resolution.
    second_labels[t] = 0 or 1  →  repeated fs times each.
    """
    return np.repeat(second_labels, fs)


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def label_window(sample_labels: np.ndarray, start: int, end: int, mode: str):
    """
    Assign a label to the window [start, end).

    Parameters
    ----------
    sample_labels : per-sample label array (0/1)
    start, end    : window boundaries (sample indices)
    mode          : 'any' | 'majority' | 'strict' | 'soft'

    Returns
    -------
    hard_label    : int (0 or 1); for 'soft' mode this equals round(frac)
    seizure_frac  : float  fraction of ictal samples in the window
    """
    window_labels = sample_labels[start:end]
    frac = float(np.mean(window_labels))

    if mode == 'any':
        hard = int(frac > 0)
    elif mode == 'majority':
        hard = int(frac >= 0.5)
    elif mode == 'strict':
        hard = int(frac == 1.0)
    elif mode == 'soft':
        hard = int(round(frac))
    else:
        raise ValueError(f"Unknown label mode: {mode}")

    return hard, frac


def generate_windows(n_samples: int, window_size: int, stride: int):
    """
    Yield (start, end, window_idx) tuples for non-overlapping or
    overlapping windows that fit fully within n_samples.
    """
    idx = 0
    w = 0
    while True:
        end = idx + window_size
        if end > n_samples:
            break
        yield idx, end, w
        idx += stride
        w += 1


# ---------------------------------------------------------------------------
# EDF loading  (minimal — we only need n_samples, no channel data)
# ---------------------------------------------------------------------------

def get_n_samples(edf_path: str) -> int:
    """Return the number of samples in the first EEG channel of an EDF."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    return raw.n_times


# ---------------------------------------------------------------------------
# Core annotation loop
# ---------------------------------------------------------------------------

def annotate_dataset(
    window_size: int,
    stride: int,
    label_mode: str,
    output_mode: str,
    data_root: str,
    anno_dir: str,
    output_root: str,
    skip_no_seizure: bool = True,
):
    """
    Main annotation routine.

    Parameters
    ----------
    window_size     : window length in samples
    stride          : stride in samples
    label_mode      : 'any' | 'majority' | 'strict' | 'soft'
    output_mode     : 'per_patient' | 'merged' | 'both'
    data_root       : directory containing eeg1.edf … eeg79.edf
    anno_dir        : directory containing annotation CSVs
    output_root     : where to write output CSVs
    skip_no_seizure : if True, skip patients with no seizure seconds
    """
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[config] window={window_size} samples ({window_size/FS:.3f} s)  "
          f"stride={stride} samples ({stride/FS:.3f} s)  "
          f"overlap={100*(1 - stride/window_size):.1f}%  "
          f"label_mode={label_mode}")

    # --- Load annotations ---
    print("[info] Loading annotation CSVs …")
    anno_map = load_annotations(anno_dir)   # {pid: per-second array}
    print(f"[info] Annotations loaded for {len(anno_map)} patients.")

    # --- CSV columns ---
    fieldnames = [
        'patient_id', 'window_idx',
        'start_sample', 'end_sample',
        'start_sec', 'end_sec',
        'label', 'seizure_frac',
    ]

    merged_rows = []        # collected only when output_mode in ('merged','both')
    total_windows = 0
    total_seizure_windows = 0

    # --- Per-patient loop ---
    for pid in range(1, 80):
        edf_path = Path(data_root) / f'eeg{pid}.edf'
        if not edf_path.exists():
            continue

        if pid not in anno_map:
            print(f"[warn] Patient {pid}: no annotation column. Skipping.")
            continue

        sec_labels = anno_map[pid]
        has_seizure = int(1 in sec_labels)

        if skip_no_seizure and not has_seizure:
            print(f"[skip] Patient {pid}: no seizure seconds.")
            continue

        # Expand annotations to per-sample resolution
        sample_labels = per_second_to_per_sample(sec_labels, FS)
        n_anno_samples = len(sample_labels)

        # Get actual recording length (samples) from EDF header
        try:
            n_rec_samples = get_n_samples(str(edf_path))
        except Exception as e:
            print(f"[error] Patient {pid}: could not read EDF – {e}. Skipping.")
            continue

        # Use the shorter of annotation vs recording (they should match)
        n_samples = min(n_rec_samples, n_anno_samples)

        if n_samples < window_size:
            print(f"[warn] Patient {pid}: recording too short ({n_samples} samples < window {window_size}). Skipping.")
            continue

        print(f"[proc] Patient {pid}:  {n_samples} samples  |  "
              f"{int(np.sum(sec_labels))} seizure-seconds")

        rows = []
        for start, end, widx in generate_windows(n_samples, window_size, stride):
            hard_label, frac = label_window(sample_labels, start, end, label_mode)
            row = {
                'patient_id'   : pid,
                'window_idx'   : widx,
                'start_sample' : start,
                'end_sample'   : end,
                'start_sec'    : round(start / FS, 6),
                'end_sec'      : round(end   / FS, 6),
                'label'        : hard_label,
                'seizure_frac' : round(frac, 6),
            }
            rows.append(row)
            total_windows += 1
            total_seizure_windows += hard_label

        n_pos = sum(r['label'] for r in rows)
        n_neg = len(rows) - n_pos
        print(f"         → {len(rows)} windows  |  pos={n_pos}  neg={n_neg}  "
              f"imbalance={n_pos/len(rows)*100:.2f}%")

        # --- Per-patient output ---
        if output_mode in ('per_patient', 'both'):
            p_dir = out_root / f"patient_{pid:03d}"
            p_dir.mkdir(parents=True, exist_ok=True)
            out_csv = p_dir / f"annotations_w{window_size}_s{stride}_{label_mode}.csv"
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"         → saved: {out_csv}")

        if output_mode in ('merged', 'both'):
            merged_rows.extend(rows)

    # --- Merged output ---
    if output_mode in ('merged', 'both') and merged_rows:
        merged_csv = out_root / f"all_patients_w{window_size}_s{stride}_{label_mode}.csv"
        with open(merged_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)
        print(f"\n[done] Merged CSV saved: {merged_csv}")

    print(f"\n[summary]")
    print(f"  Total windows      : {total_windows}")
    print(f"  Seizure windows    : {total_seizure_windows}")
    if total_windows:
        print(f"  Class imbalance    : {total_seizure_windows/total_windows*100:.2f}% positive")
    print(f"  Window  duration   : {window_size/FS:.3f} s")
    print(f"  Stride  duration   : {stride/FS:.3f} s")
    print(f"  Overlap            : {100*(1-stride/window_size):.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dense YOLO-style EEG annotation generator")

    p.add_argument('--window',      type=int,   default=256,
                   help='Window size in samples (default: 512 = 2 s @ 256 Hz)')
    p.add_argument('--stride',      type=int,   default=None,
                   help='Stride in samples. Default = window size (no overlap)')
    p.add_argument('--label-mode',  type=str,   default='strict',
                   choices=['any', 'majority', 'strict', 'soft'],
                   help='Window labelling strategy (default: strict — fully ictal windows only)')
    p.add_argument('--output-mode', type=str,   default='per_patient',
                   choices=['per_patient', 'merged', 'both'],
                   help='Output format (default: per_patient)')
    p.add_argument('--data-root',   type=str,   default='./data',
                   help='Directory with eeg1.edf … eeg79.edf')
    p.add_argument('--anno-dir',    type=str,   default='.',
                   help='Directory with annotation CSVs')
    p.add_argument('--output-root', type=str,   default='./annotations_dense',
                   help='Output directory')
    p.add_argument('--include-no-seizure', action='store_true',
                   help='Also annotate patients with no seizure activity')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    stride = args.stride if args.stride is not None else args.window

    annotate_dataset(
        window_size      = args.window,
        stride           = stride,
        label_mode       = args.label_mode,
        output_mode      = args.output_mode,
        data_root        = args.data_root,
        anno_dir         = args.anno_dir,
        output_root      = args.output_root,
        skip_no_seizure  = not args.include_no_seizure,
    )