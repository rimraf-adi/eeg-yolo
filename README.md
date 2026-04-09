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

**6. Filter Discarded Parquet Bounds**

```bash
uv run python -m src.data_processing.filter_events
```

_Function:_ Verifies exact matching Start/Stop cardinalities inside the `_events.csv` files natively comparing boundaries. Discards unaligned Parquet data securely preventing training crashes dynamically routing broken sequence arrays into `/discarded`.

### Phase 3: Train Evaluation Network

**7. Model Generation Evaluation**

```bash
uv run python -m src.training.train
```

_Function:_ Instantiates PyTorch sequence operations dynamically natively extracting Double Banana sequences on-the-fly (`18 channels, 5024 samples`), processing spatial limits structurally via `yolo1d.py`, securely tracking limits generating exactly continuous F1-scores mapping localized Object configurations sequentially over 50 epochs natively cleanly evaluated dynamically against Test sets logically.

## Implemented Methodology

The current pipeline is built around a sliding-window 1D detection problem rather than a standard image detector. The implemented methodology is:

### 1. Annotation Cleanup and Dataset Compaction

- Empty annotation CSV files are detected recursively under the active events directory and removed.
- Paired raw files are removed alongside empty annotations so the dataset stays aligned.
- `Sleeping` and `Waking` rows are removed from the event stream completely.
- After cleanup, patient IDs are compacted so the remaining files stay contiguous from `P001` upward.

### 2. Annotation Reconstruction

- The active labels are kept as three point classes: `!`, `!start`, and `!end`.
- `Sleeping` and `Waking` rows are removed before training.
- The annotation parser converts each valid timestamp into a point event with a class id.

### 3. Sliding-Window Target Construction

- EEG recordings are loaded into fixed windows of `window_size_sec` seconds.
- Each window is divided into `S` temporal grid cells.
- Events are mapped to the current window using window-relative coordinates instead of global timestamps.
- For each valid point event, the target builder computes the cell index and the normalized offset inside that cell.
- If two events fall into the same cell, the newer one overwrites the older one and a warning is emitted.

### 4. Model and Training Flow

- The model consumes 18 bipolar EEG channels and predicts 1D temporal detections across the `S` grid.
- The training loop uses objectness, offset, and class losses.
- Validation and test evaluation are driven by temporal tolerance `tau`, not IoU.
- The evaluation loop reports precision, recall, and F1, and it can sweep confidence thresholds to find the best operating point.

### 5. Current Label Set

- After cleanup, the active training annotations are the three EEG point labels: `!`, `!start`, and `!end`.
- `Sleeping` and `Waking` are treated as preprocessing noise and are excluded from training.

## Hyperparameter Note: `S` (Temporal Grid Size)

`S` is the number of temporal grid cells used by YOLO inside each input window.

- Window duration is `window_size_sec` seconds.
- Each grid cell covers `window_size_sec / S` seconds.

- Example with `window_size_sec = 10`:
  - `S = 100` -> each cell is `0.10s`
  - `S = 200` -> each cell is `0.05s`

Why this matters:

- If multiple events fall into the same cell, you can get **grid collisions**.
- Increasing `S` reduces collisions by using finer temporal bins.
- Very large `S` can increase sparsity and make optimization harder, so tune it with validation metrics.

Where to set it:

- In `config.yaml` under `dataset.S`.
- This repository now forwards that value to both dataset target generation and model head output shape, so they stay consistent.

Practical starting range:

- Try `S` in `[150, 300]` for dense event timelines.
- Keep `window_size_sec` fixed first, tune `S`, then tune `conf_threshold` and `obj_pos_weight`.

## Relative Annotation Mapping (How It Works)

The training pipeline stores annotation timestamps as absolute seconds, then converts them to window-relative positions at target build time.

1. Parsing step (`parse_annotations`):

- Keeps only `!`, `!start`, `!end`.
- Stores absolute time in `t_center_abs`.

2. Window filtering step (`build_target`):

- For a window `[t_win_start, t_win_end)`, an event is used only if:
  - `t_win_start <= t_center_abs < t_win_end`

3. Relative conversion:

- `rel_center = (t_center_abs - t_win_start) / window_size_sec`
- This gives a normalized position in `[0, 1)` inside the current window.

4. Grid assignment:

- `cell_idx = floor(rel_center * S)`
- `cell_offset = (rel_center * S) - cell_idx`
- Target values set at `target[cell_idx]`:
  - `target[cell_idx, 0] = 1.0` (objectness)
  - `target[cell_idx, 1] = cell_offset`
  - one-hot class in `target[cell_idx, 2:]`

### Worked Example

Given:

- `window_size_sec = 10.0`
- `S = 200` cells (`0.05s` per cell)
- Event timestamp: `t_center_abs = 11.842s`
- Window: `[10.0, 20.0)`

Compute:

- `rel_center = (11.842 - 10.0) / 10.0 = 0.1842`
- `cell_idx = floor(0.1842 * 200) = floor(36.84) = 36`
- `cell_offset = 36.84 - 36 = 0.84`

So the target row for this event is written to cell `36` with:

- objectness `1.0`
- offset `0.84`
- class one-hot for label `!` (class id `0`)

Because windows overlap (for example stride < window size), the same absolute event can appear in multiple windows, but at different relative cell locations in each window. This behavior is expected and is covered by tests.
