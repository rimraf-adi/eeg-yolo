import os
import glob
import scipy.io as sio
import csv
from tqdm import tqdm
from src.config import PATHS

mat_dir = PATHS["mat_dir"]
out_dir = PATHS["events_dir"]

os.makedirs(out_dir, exist_ok=True)

mat_files = glob.glob(os.path.join(mat_dir, "*.mat"))
mat_files = [f for f in mat_files if not os.path.basename(f).startswith("._")]

print(f"Found {len(mat_files)} valid MAT files. Extracting semantic events to CSV in '{out_dir}'...")

extracted_count = 0

for mat_file in tqdm(mat_files, desc="Extracting"):
    filename = os.path.basename(mat_file)
    pid = filename.replace(".mat", "")
    
    try:
        # Faster load by only grabbing the 'events' variable instead of all 140MB of eeg_data
        mat = sio.loadmat(mat_file, variable_names=['events'])
        
        if 'events' in mat:
            events = mat['events']
            
            # Save to semantic format (CSV)
            out_csv = os.path.join(out_dir, f"{pid}_events.csv")
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_sec', 'duration', 'label'])
                
                for row in events:
                    # Strip any padded spaces from MATLAB strings
                    cleaned_row = [str(item).strip() for item in row]
                    writer.writerow(cleaned_row)
            extracted_count += 1
        else:
            print(f"Warning: No 'events' array found in {filename}")
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")

print(f"\nSuccessfully extracted isolated event logs for {extracted_count} files into '{out_dir}'.")
