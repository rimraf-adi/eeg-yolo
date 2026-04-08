# EEG 1D Point-YOLO Model

This repository builds a 1D sequence mapping Point-YOLO architecture for Neonatal EEG seizure detection dynamically optimized around a 18-Channel Double Banana temporal mapping limit natively mapping bounded points bypassing spatial dimensions tracking Temporal tolerance ($\tau$).

## Execution Workflow (Reproducibility)

To successfully reproduce the dataset translation exactly from raw legacy clinical `.mat` file vectors into PyArrow Parquet formats directly streaming into PyTorch limits, strictly execute the underlying scripts in the following order:

### Phase 1: Data Standardization & Unification
The raw datasets historically contained unsynchronized identifiers and spatial format errors.

**1. Rename Dataset to Standard Sequencing**
```bash
uv run python -m src.data_processing.rename_dataset
```
_Function:_ Uniformly renames arbitrary filenames sequentially mapping exactly from `P001` upwards. Generates a master `id_mapping.csv` tracing operations dynamically.

**2. Compact the Dataset Structure**
```bash
uv run python -m src.data_processing.compact_dataset
```
_Function:_ Purges orphaned labels without sequence definitions logically shifting directories closing numeric sequence gaps automatically cleanly (e.g., if P020 is empty, P021 becomes P020 smoothly).

### Phase 2: Signal Serialization and Bounding Output

**3. Extract CSV Bounds from MATLAB arrays**
```bash
uv run python -m src.data_processing.extract_events
```
_Function:_ Converts bounding chronological annotations securely out of nested proprietary `.mat` arrays strictly into uniform columnar `.csv` sheets natively cleanly generating representations inside `/processed/extracted_events/`.

**4. Eliminate Passive Masks**
```bash
uv run python -m src.data_processing.remove_sleep_markers
```
_Function:_ Iteratively scrubs the continuous bounding outputs securely eliminating background non-seizure metrics internally (e.g., removing `Waking` & `Sleeping` tracking vectors strictly preserving exact `!` bounds).

**5. Compress Waveforms into Parquet Columnar Storage**
```bash
uv run python -m src.data_processing.mat_to_parquet
```
_Function:_ Extracts raw dense numerical matrices recursively dynamically loading internal arrays tracking strictly saving them tightly mapped locally inside PyArrow `.parquet` arrays. Substantially boosts streaming batch sequences bypassing native disk IO bottlenecks.

### Phase 3: Train Evaluation Network

**6. Model Generation Evaluation**
```bash
uv run python -m src.training.train
```
_Function:_ Instantiates PyTorch sequence operations dynamically natively extracting Double Banana sequences on-the-fly (`18 channels, 5024 samples`), processing spatial limits structurally via `yolo1d.py`, securely tracking limits generating exactly continuous F1-scores mapping localized Object configurations sequentially over 50 epochs natively cleanly evaluated dynamically against Test sets logically.
