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

## Hyperparameter Note: `S` (Temporal Grid Size)

`S` is the number of temporal grid cells used by YOLO inside each input window.

- Window duration is `window_size_sec` seconds.
- Each grid cell covers:

$$
	ext{cell\_duration} = \frac{\text{window\_size\_sec}}{S}
$$

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
