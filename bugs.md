# EEG-YOLO: IED Detection — Conversation Specfile

**Project:** Automated IED (Interictal Epileptiform Discharge) Detection from Neonatal EEG  
**Repo:** https://github.com/rimraf-adi/eeg-yolo  
**Date:** April 2026

---

## 1. Problem Statement

Detect three point-event classes in multi-channel neonatal EEG (18-channel Double Banana montage, 500 Hz):

| Label | Meaning |
|---|---|
| `!` | Spike (IED) — frequent |
| `!start` | Seizure onset — very rare |
| `!end` | Seizure offset — very rare |

**Key challenge:** Extreme class imbalance. In the test set: 10,777 `none` windows vs. 219 `!`, 6 `!start`, 5 `!end`.

---

## 2. Approach Attempted

### 2.1 YOLO-style Point Detection (Primary Repo)

- Sliding window over EEG → fixed-length windows (1–10 s)
- Divide each window into `S` temporal grid cells
- Per cell: predict objectness, offset within cell, and class one-hot
- Loss: objectness + offset + classification
- Evaluation: temporal tolerance `tau` (match within τ seconds = TP)

### 2.2 Two-Stage Binary + Event Classifier (Benchmark)

- Stage 1: Binary classifier → `none` vs `event`
- Stage 2: Event classifier → `!` vs `!start` vs `!end`
- Separate models (1D CNN and 2D CNN variants)

### 2.3 Flat Multiclass Classifier (Latest Benchmark)

- Direct 4-class classification: `none`, `!`, `!start`, `!end`
- 1-second windows, stride=1s, fs=500, grid_S=20
- Class weights: `[0.0056, 0.0963, 1.867, 2.031]`

---

## 3. Experimental Results

### 3.1 Two-Stage Results

```
1D Stage1 (binary): Acc=0.9791  MacroF1=0.4947
  → Class event: P=0.00  R=0.00  F1=0.00  (model predicts all-none)

1D Stage2 (event):  Acc=0.0261  MacroF1=0.0169
  → Class !: F1=0.00, !start: F1=0.05, !end: F1=0.00

2D Stage1 (binary): Acc=0.7699  MacroF1=0.4597
  → Class event: P=0.03  R=0.29  F1=0.05  (some recall)

2D Stage2 (event):  Acc=0.0217  MacroF1=0.0142
```

### 3.2 Flat Multiclass Results

```
1D Model (19 epochs):
  val_loss: 11.08 → ~9.2  (never converges)
  train_loss: 0.25 → 0.12 (converges fine)
  TEST: MacroF1=0.0118, WeightedF1=0.0008
  Class none: F1=0.00 (predicts zero none)

2D Model (50 epochs):
  val_loss: 10.30 → ~7.8  (slow, noisy descent)
  train_loss: 0.24 → 0.12
  TEST: MacroF1=0.0203, WeightedF1=0.0022
  Class none: P=1.00  R=0.0004  F1=0.0007 (predicts almost all none)
```

**Key anomaly:** Train loss ~0.12, val loss ~8–11 → **~70x gap** indicates a data pipeline bug, not model failure.

---

## 4. Bugs Identified

### 4.1 Bugs in YOLO Repo (documented in README)

| # | Bug | Severity | Fix |
|---|---|---|---|
| 1 | `tau` hardcoded to 0.25s everywhere, config says 1.0s | **Critical** | Read tau from config |
| 2 | Soft target offset copied to all neighbor cells (wrong) | **High** | Guard offset update to center cell only |
| 3 | Stride 0.5s on 10s windows → 95% overlap, 20x redundancy | **High** | stride_sec: 0.5 → 2.0 |
| 4 | Target shape `[B,S,1,5]` in test script, should be `[B,S,5]` | Medium | Fix indexing |
| 5 | `*.pt` model files committed to git | Medium | Update .gitignore |

### 4.2 Bugs in Flat Classifier (Diagnosed from Results)

**Bug A — Normalization mismatch (most likely)**
- Training data normalized with train-set mean/std
- Val/test data likely not normalized, or normalized with different stats
- Explains 70x train/val loss gap
- **Fix:** Compute mean/std on train set, apply same stats to val and test

**Bug B — Class weights applied inconsistently**
- Weights: `[0.0056, 0.0963, 1.867, 2.031]`
- If applied only during train but not val loss → incomparable loss values
- **Fix:** Use same criterion (with or without weights) consistently

**Bug C — S value mismatch**
- Config says `S=200`, benchmark prints `grid_S=20`
- If model outputs 200 cells but targets built with S=20 → broken loss
- **Fix:** Verify S is propagated consistently to dataset and model head

**Smoking gun:** 1D model predicts zero `none` (all event), 2D model predicts almost all `none`. Opposite failure modes on same data = broken loss surface from pipeline bug, not architecture.

---

## 5. Fundamental Architecture Question

### 5.1 Why YOLO-style is Novel (No Prior Work)

The field has not applied YOLO-style 1D point detection to EEG because:
- Prior work frames seizure detection as **binary segmentation** (seizure/no-seizure)
- Onset/offset are derived as **boundaries** of detected intervals, not classified separately
- Detecting `!`, `!start`, `!end` jointly as three point-event classes is unprecedented

### 5.2 What the Literature Actually Does

| Approach | Architecture | Task |
|---|---|---|
| SZTrack | CNN encoder + BiLSTM | Binary seizure tracking, onset/offset from boundaries |
| IEDnet | CNN + GRU, AC-GAN augmentation | Binary IED (spike) detection |
| AugUNet1D | Residual 1D U-Net | Dense temporal segmentation of spike-wave discharges |
| InceptionTime / Minirocket | TSC algorithms | Binary IED classification, F1=0.80, AUPRC=0.98 |
| VGG on 2s epochs | CNN | Binary IED, AUC=0.96, class weight 100:1 |

**Key finding:** Best architecture for dense temporal event detection in EEG is **1D U-Net** (residual), not YOLO-style regression. LSTM-only architectures **completely failed** (F1=0.00) in cross-subject settings.

### 5.3 Critical Observations from Literature

- Class weight ratio should be **~100:1** for IED detection (current: ~18:1 — too mild)
- Seizure **offset** is consistently the hardest event to detect across all papers
- Nobody jointly classifies onset + offset as separate learned classes — it's always derived post-hoc from binary mask boundaries
- Data augmentation (amplitude scaling, Gaussian noise, signal inversion) significantly improves cross-subject generalization

---

## 6. Recommended Path Forward

### Option A: Fix the YOLO Model (Preserve Novel Contribution)

1. Apply all 5 repo bugs fixes (tau, offset guard, stride, shape, gitignore)
2. Verify normalization: print train/val mean+std before training
3. Increase `obj_pos_weight` drastically (try 50–100x, not 1.5x)
4. Decouple: get **objectness-only** detection working first, ignore class head
5. Add class head only after objectness F1 > 0.3
6. Use `S=150–300`, `window_size_sec=10`, tune `conf_threshold` via sweep

### Option B: Switch to Literature-Proven Architecture

1. **Train binary seizure segmentor**: 1D U-Net or CNN-BiLSTM  
   - Input: `[18, T]` EEG segment  
   - Output: per-timestep binary label (event / no-event)  
   - Loss: Binary focal loss (not weighted CE)
   
2. **Derive onset/offset from mask boundaries**:  
   - Rising edge of predicted binary mask → `!start`  
   - Falling edge → `!end`  
   - Any positive prediction → `!` (spike)

3. **Handle imbalance properly**:  
   - Focal loss with γ=2 instead of weighted CE  
   - Or class weight ratio ≥ 50:1 for positive class  
   - Data augmentation: amplitude scale, Gaussian noise, signal inversion

### Option C: Hybrid (Recommended)

Train binary U-Net as baseline → use its confident positive predictions as **hard negatives for the YOLO model** → bootstrapped YOLO training with far better positive/negative ratio.

---

## 7. Open Questions

1. **Total event count in full dataset?** — With only 5 `!end` in test set, the problem may be data-limited regardless of architecture.
2. **Is normalization per-recording or global?** — Critical to check immediately.
3. **What is `grid_S=20` vs config `S=200`?** — Potential mismatch that would break everything.
4. **Are val/test patients seen during training?** — Cross-subject generalization is the hardest unsolved problem in this field.

---

## 8. Debugging Checklist (Immediate Next Steps)

```python
# Step 1: Normalization sanity check
print("Train X mean/std:", train_X.mean(), train_X.std())
print("Val   X mean/std:", val_X.mean(),   val_X.std())

# Step 2: Disable class weights for one run
criterion = nn.CrossEntropyLoss()  # no weights — does val_loss drop to ~1.5?

# Step 3: Check S consistency
print("Model output shape:", model(dummy_x).shape)  # should be [B, S, num_classes]
print("Target shape:", target.shape)                # must match

# Step 4: Tau check (YOLO model)
# grep -r "tau=0.25" src/  → should return zero results after fix
```

---

*Generated from conversation session — April 2026*
