import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.dataset import EEGRegressionDataset

print("Checking Dataset initialization...")
ds = EEGRegressionDataset(
    data_dir='/Volumes/WORKSPACE/opensource-dataset/processed/parquet_data',
    anno_dir='/Volumes/WORKSPACE/opensource-dataset/processed/extracted_events',
    allowed_pids=[1, 2]
)

print(f"Num samples: {len(ds)}")
if len(ds) > 0:
    x, y = ds[0]
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
