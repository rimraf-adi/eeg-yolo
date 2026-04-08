import os
import glob
import pandas as pd
import csv
from src.config import PATHS

BASE_DIR = PATHS["base_dir"]
MAT_DIR = os.path.join(BASE_DIR, "MAT_Files")
EVENTS_DIR = os.path.join(BASE_DIR, "extracted_events")
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_data")
MAPPING_CSV = os.path.join(BASE_DIR, "id_mapping.csv")
INFO_CSV = os.path.join(MAT_DIR, "base_info.csv")

def rename_dataset():
    print("Step 1: Identifying unique patient IDs from MAT_Files...")
    
    mat_files = [f for f in glob.glob(os.path.join(MAT_DIR, "*.mat")) if not os.path.basename(f).startswith("._")]
    # Extract unique IDs
    old_ids = sorted(list(set([os.path.basename(f).replace(".mat", "") for f in mat_files])))
    
    if not old_ids:
        print("No valid MAT files found. Exiting.")
        return
        
    print(f"Found {len(old_ids)} unique patient IDs.")
    
    # Create mapping dictionary
    mapping = {}
    for i, old_id in enumerate(old_ids):
        # Format P001, P002...
        new_id = f"P{i+1:03d}"
        mapping[old_id] = new_id
        
    # Save mapping DataFrame
    mapping_df = pd.DataFrame(list(mapping.items()), columns=['old_id', 'new_id'])
    mapping_df.to_csv(MAPPING_CSV, index=False)
    print(f"✅ Saved id_mapping.csv directly inside {BASE_DIR}")
    
    # ---------------------------------------------------------
    # Update base_info.csv
    # ---------------------------------------------------------
    if os.path.exists(INFO_CSV):
        print("Step 2: Updating base_info.csv...")
        try:
            info_df = pd.read_csv(INFO_CSV)
            if 'file_name' in info_df.columns:
                # Map old_ids directly to new_ids
                info_df['file_name'] = info_df['file_name'].map(mapping).fillna(info_df['file_name'])
                info_df.to_csv(INFO_CSV, index=False)
                print(f"✅ base_info.csv rewritten with new serial numbers.")
            else:
                print("Warning: 'file_name' column not found in base_info.csv")
        except Exception as e:
            print(f"Error handling base_info.csv: {e}")
            
    # ---------------------------------------------------------
    # Rename Top-level directories (MAT, CSV, Parquet)
    # ---------------------------------------------------------
    print("\nStep 3: Renaming root aggregated MAT / CSV / Parquet files...")
    
    def process_renames(folder, old_id, new_id, suffix):
        old_path = os.path.join(folder, f"{old_id}{suffix}")
        new_path = os.path.join(folder, f"{new_id}{suffix}")
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            
    for old_id, new_id in mapping.items():
        # MAT
        process_renames(MAT_DIR, old_id, new_id, ".mat")
        # EVENTS CSV
        process_renames(EVENTS_DIR, old_id, new_id, "_events.csv")
        # PARQUET
        process_renames(PARQUET_DIR, old_id, new_id, ".parquet")
        
    print("✅ Completed root renames.")

    # ---------------------------------------------------------
    # Rename Deep NPY Matrices
    # ---------------------------------------------------------
    print("\nStep 4: Recursively renaming extracted NPY subsets (*-IED) ...")
    renamed_npy_count = 0
    
    # Find all NPYs in BASE_DIR (excluding the meta mac files ._ )
    # glob iter recursive available in 3.5+
    all_npy = glob.glob(os.path.join(BASE_DIR, "**", "*.npy"), recursive=True)
    valid_npy = [f for f in all_npy if not os.path.basename(f).startswith("._")]
    
    for f in valid_npy:
        basename = os.path.basename(f)
        directory = os.path.dirname(f)
        
        # Check if filename starts with any old_id
        for old_id, new_id in mapping.items():
            # Example filename: DA00100S_246000_248000.npy -> must start with old_id + "_"
            if basename.startswith(f"{old_id}_"):
                new_basename = basename.replace(f"{old_id}_", f"{new_id}_", 1)
                new_f = os.path.join(directory, new_basename)
                
                os.rename(f, new_f)
                renamed_npy_count += 1
                break  # Stop checking other old_ids if matched

    print(f"✅ Renamed {renamed_npy_count} deeply-nested .npy arrays successfully.")
    
    print("\n🎉 ENTIRE DATASET RENAMING COMPLETED SUCCESSFULLY. IDs HAVE BEEN ANONYMIZED.")

if __name__ == "__main__":
    rename_dataset()
