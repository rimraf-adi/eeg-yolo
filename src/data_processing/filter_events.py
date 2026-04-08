import os
import glob
import pandas as pd
import shutil
from src.config import PATHS

EVENTS_DIR = PATHS["events_dir"]
PARQUET_DIR = PATHS["parquet_data_dir"]
DISCARD_DIR = PATHS["discard_dir"]

os.makedirs(DISCARD_DIR, exist_ok=True)
os.makedirs(os.path.join(DISCARD_DIR, "events"), exist_ok=True)
os.makedirs(os.path.join(DISCARD_DIR, "parquet"), exist_ok=True)

csv_files = glob.glob(os.path.join(EVENTS_DIR, "*_events.csv"))

discarded_count = 0
kept_count = 0

print("Scanning all validated patients for annotation mismatches...")

for csv_file in csv_files:
    pid = os.path.basename(csv_file).replace("_events.csv", "")
    
    try:
        df = pd.read_csv(csv_file)
        if 'label' not in df.columns:
            continue
            
        # Count occurences of start and end blocks
        counts = df['label'].value_counts()
        start_count = counts.get('!start', 0)
        end_count = counts.get('!end', 0)
        
        # Check rule: cardinality(start) == cardinality(stop)
        if start_count != end_count:
            print(f" -> Discarding Patient [ {pid} ] (Mismatched annotations: {start_count} '!start', {end_count} '!end')")
            
            # Subtly move (discard) CSV
            shutil.move(csv_file, os.path.join(DISCARD_DIR, "events", f"{pid}_events.csv"))
            
            # Move (discard) Parquet if it exists
            pq_file = os.path.join(PARQUET_DIR, f"{pid}.parquet")
            if os.path.exists(pq_file):
                shutil.move(pq_file, os.path.join(DISCARD_DIR, "parquet", f"{pid}.parquet"))
            
            discarded_count += 1
        else:
            kept_count += 1
            
    except Exception as e:
        print(f"Error validating {pid}: {e}")

print(f"\nCleanup complete. Retained {kept_count} structurally sound profiles. Discarded {discarded_count} buggy profiles into `{DISCARD_DIR}`.")
