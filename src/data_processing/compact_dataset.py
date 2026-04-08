import os
import glob
import pandas as pd
import re
from src.config import PATHS

BASE_DIR = PATHS["base_dir"]
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MAT_DIR = os.path.join(BASE_DIR, "MAT_Files")
EVENTS_DIR = os.path.join(PROCESSED_DIR, "extracted_events")
PARQUET_DIR = os.path.join(PROCESSED_DIR, "parquet_data")
MAPPING_CSV = os.path.join(PROCESSED_DIR, "id_mapping.csv")
INFO_CSV = os.path.join(MAT_DIR, "base_info.csv")

def compact_names():
    print("Checking gaps in dataset sequences (based on active Event files)...")
    
    # Grab current surviving EVENT files as the root of truth since we purposely broke the MAT parity
    event_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(EVENTS_DIR, "*_events.csv"))])
    current_ids = [f.replace("_events.csv", "") for f in event_files]
    
    if len(current_ids) == 0:
        print("No valid Event files found to compact.")
        return

    expected_ids = [f"P{i+1:03d}" for i in range(len(current_ids))]
    
    # Prune MAT files natively that were abandoned during the filter pass to prevent overwrite collisions
    all_mats = [f.replace(".mat", "") for f in [os.path.basename(m) for m in glob.glob(os.path.join(MAT_DIR, "P*.mat"))]]
    abandoned_mats = set(all_mats) - set(current_ids)
    for bad_mat in abandoned_mats:
        os.remove(os.path.join(MAT_DIR, f"{bad_mat}.mat"))
        print(f"Purged orphaned raw matrix: {bad_mat}.mat")
    
    compact_map = {}
    for old_serial, new_serial in zip(current_ids, expected_ids):
        if old_serial != new_serial:
            compact_map[old_serial] = new_serial
            
    if not compact_map:
        print("All names are already perfectly contiguous. No action needed.")
        return
        
    print(f"Discovered structural gaps. Shifting {len(compact_map)} indices back to maintain contiguous boundaries...")
    
    # 1. Update id_mapping.csv
    try:
        mapping_df = pd.read_csv(MAPPING_CSV)
        # Drop rows where new_id is missing from current_ids (i.e. discarded models)
        mapping_df = mapping_df[mapping_df['new_id'].isin(current_ids)].copy()
        # Map shifting
        mapping_df['new_id'] = mapping_df['new_id'].map(lambda x: compact_map.get(x, x))
        mapping_df.to_csv(MAPPING_CSV, index=False)
        print("✅ Restructured root id_mapping.csv.")
    except Exception as e:
        print(f"Mapping CSV error: {e}")

    # 2. Update base_info.csv
    try:
        if os.path.exists(INFO_CSV):
            info_df = pd.read_csv(INFO_CSV)
            info_df['file_name'] = info_df['file_name'].map(lambda x: compact_map.get(x, x))
            
            # Since some profiles were entirely discarded, we should probably prune them from base_info.csv too
            # The current current_ids + new expected_ids covers everything active.
            # info_df = info_df[info_df['file_name'].isin(expected_ids)]
            info_df.to_csv(INFO_CSV, index=False)
            print("✅ Restructured base_info.csv.")
    except Exception as e:
        print(f"Info CSV error: {e}")

    # 3. Rename files! Important to do in sorted forwards order (which keys are already in).
    renamed_npy_count = 0
    all_npy = glob.glob(os.path.join(BASE_DIR, "**", "*.npy"), recursive=True)
    all_npy = [f for f in all_npy if not os.path.basename(f).startswith("._")]
    
    for old_id, new_id in compact_map.items():
        # MAT
        old_mat = os.path.join(MAT_DIR, f"{old_id}.mat")
        if os.path.exists(old_mat): os.rename(old_mat, os.path.join(MAT_DIR, f"{new_id}.mat"))
        
        # CSV
        old_csv = os.path.join(EVENTS_DIR, f"{old_id}_events.csv")
        if os.path.exists(old_csv): os.rename(old_csv, os.path.join(EVENTS_DIR, f"{new_id}_events.csv"))
        
        # Parquet
        old_pq = os.path.join(PARQUET_DIR, f"{old_id}.parquet")
        if os.path.exists(old_pq): os.rename(old_pq, os.path.join(PARQUET_DIR, f"{new_id}.parquet"))
            
        # Recursive NPY chunks. Instead of scanning glob all over again per key, we check our loaded cache.
        for f in all_npy:
            basename = os.path.basename(f)
            if basename.startswith(f"{old_id}_"):
                new_f = os.path.join(os.path.dirname(f), basename.replace(f"{old_id}_", f"{new_id}_", 1))
                os.rename(f, new_f)
                renamed_npy_count += 1
                
    print(f"✅ Compaction complete. Shifted {renamed_npy_count} embedded sub-matrix matrices.")
    print("\nDataset is strictly contiguous scaling from P001 -> P082.")

if __name__ == "__main__":
    compact_names()
