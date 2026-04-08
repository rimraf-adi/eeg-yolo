import os
import glob
import scipy.io as sio
import pandas as pd
from tqdm import tqdm

mat_dir = "/Volumes/WORKSPACE/opensource-dataset/MAT_Files"
out_dir = "/Volumes/WORKSPACE/opensource-dataset/parquet_data"

os.makedirs(out_dir, exist_ok=True)

mat_files = glob.glob(os.path.join(mat_dir, "*.mat"))
mat_files = [f for f in mat_files if not os.path.basename(f).startswith("._")]

print(f"Found {len(mat_files)} MAT files. Converting 'eeg_data' to Parquet in '{out_dir}'...")

processed_count = 0

for mat_file in tqdm(mat_files, desc="Converting MAT to Parquet"):
    filename = os.path.basename(mat_file)
    pid = filename.replace(".mat", "")
    out_pq = os.path.join(out_dir, f"{pid}.parquet")
    
    if os.path.exists(out_pq):
        processed_count += 1
        continue
        
    try:
        # Load only 'eeg_data' matrix to optimize RAM
        mat = sio.loadmat(mat_file, variable_names=['eeg_data'])
        if 'eeg_data' in mat:
            eeg_data = mat['eeg_data']
            
            # EEG in MAT is typically (channels, time).
            # Parquet is a columnar format. We must transpose to (time_steps, channels)
            # so that Parquet can compress and store each channel as an independent column.
            if eeg_data.shape[0] < eeg_data.shape[1]:
                eeg_data = eeg_data.T
                
            channels = eeg_data.shape[1]
            cols = [f"Ch_{i+1:02d}" for i in range(channels)]
            
            df = pd.DataFrame(eeg_data, columns=cols)
            
            # Save to Parquet using the widely supported PyArrow engine
            df.to_parquet(out_pq, engine='pyarrow', compression='snappy')
            processed_count += 1
        else:
            print(f"\nWarning: No 'eeg_data' array found in {filename}")
            
    except ImportError:
        print("\nError: pyarrow or fastparquet is not installed. Please pip install pyarrow pandas")
        break
    except Exception as e:
        print(f"\nError converting {filename}: {e}")

print(f"\nBatch conversion complete. Processed {processed_count} files into '{out_dir}'.")
