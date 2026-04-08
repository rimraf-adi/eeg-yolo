import os
import glob
import pandas as pd
from tqdm import tqdm
from src.config import PATHS

EVENTS_DIR = PATHS["processed_events_dir"]
csv_files = glob.glob(os.path.join(EVENTS_DIR, "*_events.csv"))

print(f"Filtering {len(csv_files)} event files to discard 'Waking' and 'Sleeping' markers...")

removed_total = 0

for file in tqdm(csv_files, desc="Cleaning Events"):
    try:
        df = pd.read_csv(file)
        if 'label' in df.columns:
            original_len = len(df)
            
            # Use strict lowercase string matching to prune them seamlessly
            mask = df['label'].astype(str).str.strip().str.lower().isin(['waking', 'sleeping'])
            filtered_df = df[~mask]
            
            removed = original_len - len(filtered_df)
            removed_total += removed
            
            if removed > 0:
                # Overwrite directly to keep the dataset pristine
                filtered_df.to_csv(file, index=False)
                
    except Exception as e:
        print(f"Oops, failed to process {os.path.basename(file)}: {e}")

print(f"\n✅ Cleaned up! Successfully purged {removed_total} Waking/Sleeping state markers seamlessly across all files.")
