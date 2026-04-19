
---

# Architectural Specification: Anchor-Free Temporal Localization of Interictal Epileptiform Discharges in Neonatal EEG via 1D Feature Pyramid Networks

## 1. Executive Summary
Traditional approaches to epileptic event detection in continuous EEG rely on rigid sliding windows and multi-stage classification pipelines, leading to severe performance degradation when defining exact onset and offset boundaries. This document outlines a novel adaptation of Temporal Action Localization (TAL)—specifically anchor-free, single-stage architectures like ActionFormer—to 1D multichannel biosignals. The proposed **EEG-ActionFormer** network replaces ad-hoc boundary smoothing with direct temporal regression, optimizing simultaneously for event presence, classification, and boundary localization natively at multiple timescales.

## 2. The TAL to EEG Isomorphism
The fundamental structure of video-based TAL translates identically to continuous EEG monitoring. This direct mapping allows the deployment of state-of-the-art TAL loss functions and evaluation metrics directly to biosignal processing.

| Feature Area | TAL (Video Domain) | Proposed (EEG Domain) |
| :--- | :--- | :--- |
| **Input Modality** | Untrimmed long-form video | Continuous untrimmed EEG recording |
| **Target Entity** | Action instance (e.g., "running") | Clinical event (e.g., IED, Seizure) |
| **Boundary Target** | Action start / end frames | Temporal onset / offset (ms precision) |
| **Categorization** | Action class | Event morphology / Seizure type |
| **Negative Space** | Background / unannotated frames | Normal background EEG activity |
| **Primary Metric** | mAP @ tIoU threshold | mAP @ tIoU threshold (replacing F1) |

## 3. Methodology Evaluation
Three mature families of TAL architectures were evaluated for adaptation to the neonatal EEG domain:

1.  **Anchor-Based (e.g., YOLO-style temporal grids):** Requires predefined interval priors. While clinically justifiable (e.g., setting anchors to standard 10s, 30s durations), it struggles with the extreme scale variance between sub-second IED spikes and multi-minute electrographic seizures.
2.  **Query-Based (e.g., DETR-style Set Prediction):** Highly novel and eliminates Non-Maximum Suppression (NMS) via bipartite matching. However, cross-attention over long biosignals is computationally expensive and requires large-scale datasets to converge effectively.
3.  **Anchor-Free / Boundary Detection (e.g., ActionFormer):** Selected as the optimal architecture. It frames localization as a dense sequence-labeling problem. Every timestep predicts its distance to the nearest event boundary, naturally accommodating the extreme duration variance of neonatal EEG events without rigid anchor definitions.

## 4. EEG-ActionFormer: System Architecture
The proposed architecture is a single-stage, anchor-free 1D neural network operating directly on multichannel EEG.

### 4.1. Input Representation
* **Tensor Shape:** `[B, C, T]` where `B` is batch size, `C = 18` (standard 10-20 montage channels), and `T` is the temporal sequence length at a sampling rate of **500Hz** or **256Hz**.

### 4.2. Stage 1: Multi-Scale 1D Encoder (Feature Pyramid)
To capture both high-frequency spikes and low-frequency rhythmic buildups, the encoder utilizes a Temporal Convolutional Network (TCN) structured as a Feature Pyramid Network (FPN).
* **Channel Mixing:** Initial depthwise separable 1D convolutions project the 18 channels into a dense latent space.
* **Temporal Hierarchy:** A block of dilated 1D convolutions (e.g., dilation rates of 1, 2, 4, 8) expands the receptive field.
* **Pyramid Generation:** Downsampling produces a feature pyramid at multiple resolutions: `[T, T/2, T/4, T/8]`. Short IEDs are detected at the `T` resolution; minutes-long seizures are detected at the `T/8` resolution.

### 4.3. Stage 2: Dense Prediction Heads
For every timestep $t$ across all pyramid levels, the network applies shared 1D convolutional heads to output four parallel predictions:
* **Objectness Head:** A single unit with a Sigmoid activation predicting the probability $\hat{p}_t$ that timestep $t$ lies within a clinically relevant event.
* **Classification Head:** A Softmax distribution over $K$ event classes (e.g., background, spike, sharp wave, seizure).
* **Start Regression Head:** A ReLU-activated scalar $\hat{d}_{start}$ predicting the temporal distance (in seconds or samples) from $t$ to the event's onset.
* **End Regression Head:** A ReLU-activated scalar $\hat{d}_{end}$ predicting the temporal distance from $t$ to the event's offset.

The predicted temporal interval for any valid timestep $t$ is thus recovered as:
$$\hat{v}_t = [t - \hat{d}_{start}, t + \hat{d}_{end}]$$

### 4.4. Stage 3: Loss Formulation
The network optimizes a multi-task loss function. By utilizing Focal Loss, the architecture natively handles the extreme class imbalance between rare IEDs and predominant background EEG.

$$\mathcal{L}_{total} = \lambda_{obj} \mathcal{L}_{focal} + \lambda_{reg} \mathcal{L}_{tIoU} + \lambda_{cls} \mathcal{L}_{CE}$$

* **Objectness (Focal Loss):** Penalizes easy background examples while focusing on ambiguous boundaries.
    $$\mathcal{L}_{focal} = - \alpha (1 - \hat{p}_t)^\gamma \log(\hat{p}_t)$$
* **Regression (Temporal IoU Loss):** Directly optimizes the temporal overlap between the predicted interval $\hat{v} = [\hat{s}, \hat{e}]$ and the ground truth interval $v = [s, e]$.
    $$\text{tIoU} = \frac{\max(0, \min(\hat{e}, e) - \max(\hat{s}, s))}{\max(\hat{e}, e) - \min(\hat{s}, s)}$$
    $$\mathcal{L}_{tIoU} = 1 - \text{tIoU}$$
* **Classification (Cross-Entropy):** Standard categorical cross-entropy $\mathcal{L}_{CE}$, computed *only* on timesteps where ground-truth objectness is positive.

### 4.5. Stage 4: Post-Processing
1.  **Thresholding:** Discard any predictions where the objectness score $\hat{p}_t < \tau_{obj}$.
2.  **Soft-NMS:** Apply 1D Soft Non-Maximum Suppression. Instead of greedily discarding overlapping predicted intervals (which might represent rapid successive spikes), Soft-NMS smoothly decays the confidence scores of highly overlapping intervals.

## 5. Evaluation Methodology: mAP @ tIoU
Traditional sliding-window F1-scores fail to measure boundary precision. We will adopt the standard TAL evaluation metric: **mean Average Precision (mAP) at various temporal Intersection-over-Union (tIoU) thresholds**.

* Reporting mAP at strict thresholds (e.g., **mAP@0.7**, **mAP@0.9**) proves the model's ability to precisely localize the start and end of seizures.
* Reporting mAP at looser thresholds (e.g., **mAP@0.3**) proves the model's ability to detect the general presence of an event, analogous to traditional clinical tolerance windows.

## 6. Implementation Roadmap

| Priority | Component | Complexity | Strategic Value |
| :--- | :--- | :--- | :--- |
| **1** | Anchor-free 1D regression heads (ActionFormer-EEG) | Medium | Core novelty; completely solves the boundary regression issue. |
| **2** | 1D Feature Pyramid Network (Encoder) | Low | Allows single-pass detection of both short spikes and long seizures. |
| **3** | Focal Loss + tIoU Loss Implementation | Low | Trivial drop-in replacement that natively fixes class imbalance. |
| **4** | Metric pipeline overhaul (mAP@tIoU) | Low | Essential for proving the efficacy of the method to reviewers. |
| **5** | EEG-DETR (Query-based architecture) | High | Future work / secondary paper once dataset size is scaled. |

Let me think about this properly before searching.Good, now I have a solid picture. Here's a structured breakdown of genuinely novel directions you could take, grounded in what the literature is and isn't doing.

---

## The Core Insight First

Before novel ideas: your labels are actually very valuable in a way you haven't exploited yet. You have **three annotation types that carry structural clinical information**:

- `!` = "something epileptiform happened here" (dense, local)
- `!start` = "a seizure is beginning" (extremely sparse, transitional)
- `!end` = "a seizure is ending" (extremely sparse, transitional)

The relationship between these three is **sequential and causal** — a `!start` must precede `!end`, spikes often cluster before `!start` and thin out after `!end`. No current model exploits this structure. That's your white space.

---

## Novel Idea 1: Spike Rate as a Seizure Precursor Signal

**What it is:** Don't try to detect `!start` directly. Instead, model the **density of `!` spikes over time** as a continuous signal, and detect the phase transitions in that signal.

**The insight from literature:** Hybrid CNN-LSTM models assign epileptogenic zone likelihood scores by linking high-frequency oscillation density with seizure onset propagation. The same logic applies to your data — spike density is a proxy for seizure state.

**How to implement it:**
1. Train a reliable binary `!` spike detector (simpler problem, more data)
2. Compute a **spike rate timeseries** over a rolling window (e.g. spikes/10s)
3. Train a second model on the spike rate signal to detect the phase transition — sudden rise = `!start`, gradual fall = `!end`
4. This is a 1D changepoint detection problem on a derived signal, which is far simpler than raw EEG

**Why it's novel:** Nobody in the IED literature uses spike rate dynamics as the input to onset/offset detection. They either do raw EEG → binary seizure, or raw EEG → onset. The two-stage spike-rate approach is new.

---

## Novel Idea 2: Exploit Label Noise as a Feature, Not a Problem

**What it is:** Inter-rater agreement among clinicians interpreting EEG data hovers around 60%, and research showed that the performance of AI models may plateau on widely used datasets due to this variability. Your labels have the same uncertainty — but you can turn this into a feature.

BUNDL (Bayesian Uncertainty-aware Deep Learning) introduces a probabilistic training framework that models uncertainty-informed label transitions. Around the onset region — 30 seconds before annotated onset — the framework determines clinician labels to be false positives with 0.21 probability, reflecting the ambiguous seizure start.

**Your novel angle:** Rather than treating `!start` as a hard point label, model it as a **soft temporal distribution** — a Gaussian centered at the annotated time with learned width. This reframes the impossibly hard "predict the exact onset second" into the more tractable "predict that onset is likely somewhere in this 5-second window." The width of the learned distribution is itself a clinically meaningful output (uncertainty quantification).

**Why it's novel:** BUNDL applies this to seizure intervals. Nobody has applied Bayesian soft labels specifically to **onset/offset point events** in neonatal EEG, which is your exact setting.

---

## Novel Idea 3: Structured State Machine Loss

**What it is:** Add a **temporal consistency loss** that enforces the biological constraint that events must follow the order: `background → !start → (spikes) → !end → background`.

Current models treat every window independently. They can predict `!end` before `!start`, which is clinically impossible. You can penalize this.

**How to implement it:**
- Train a model that outputs a state probability at each timestep: `P(background)`, `P(interictal)`, `P(ictal)`
- Add a loss term penalizing invalid state transitions (e.g. `ictal → background` without `!end`, `!end` without a prior `!start`)
- This is essentially a Neural HMM or a CRF layer on top of a CNN encoder

**Why it's novel:** SZTrack learns solely from annotations of seizure onset and offset intervals, producing temporally contiguous predictions, but does not enforce structural constraints on the valid ordering of events. Explicit structural constraints on event ordering is not done in any current EEG seizure paper.

---

## Novel Idea 4: Semi-supervised Learning from the Unlabeled Majority

**What it is:** You have a massive amount of background EEG (`none` windows = 98% of data) that is labeled but effectively unused as a learning signal. Use it for **self-supervised pretraining**.

**How to implement it:**
1. Train a masked autoencoder on raw EEG (mask random time segments, reconstruct them)
2. The model learns what "normal" EEG looks like
3. Fine-tune the encoder on your labeled event data — now the model already knows what background looks like, so the event signal is much clearer
4. High reconstruction error at inference time = anomaly = candidate event

A 2024 study on neonatal EEG found that greater performance gains come from harnessing the latest deep learning architectural advances than from using a larger training dataset when using AUC as a metric — but the best performance combines both.

**Why it's novel:** Masked autoencoders for EEG pretraining followed by point-event fine-tuning hasn't been applied to the neonatal IED detection problem. The reconstruction-error-as-anomaly-signal is a clean, publishable contribution.

---

## Novel Idea 5: Reframe as Duration Regression, Not Boundary Classification

**What it is:** Instead of classifying `!start` and `!end` separately, train a model that, **given a detected spike cluster**, predicts:
- **Seizure duration** (single scalar regression)
- **Time to next spike** (inter-spike interval regression)

This completely avoids the sparse class problem. Duration regression from spike context is a much better-posed problem than "find the exact onset second."

**How to implement:**
- Window centered on each detected `!` spike
- Predict: "is this spike part of a seizure? If yes, how far are we from the start? How far are the we from the end?"
- These are regression targets, not classification, so they don't suffer from class imbalance

**Why it's novel:** The field does binary seizure detection or onset localization. **Relative temporal regression from detected spikes** — predicting seizure phase as a continuous variable — is not done in any paper in the current literature.

---

## Honest Ranking by Feasibility vs. Novelty

| Idea | Feasibility | Novelty | Best for |
|---|---|---|---|
| Spike rate → transition detection | ⭐⭐⭐⭐ | ⭐⭐⭐ | Getting something working fast |
| Soft Bayesian labels | ⭐⭐⭐ | ⭐⭐⭐⭐ | Publication |
| State machine / CRF loss | ⭐⭐⭐ | ⭐⭐⭐⭐ | Publication |
| Masked autoencoder pretraining | ⭐⭐⭐ | ⭐⭐⭐ | If you have compute |
| Duration regression from spikes | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Strongest novel framing |

**If I had to pick one:** Idea 5 (duration regression) completely sidesteps your class imbalance problem, produces a clinically meaningful output, and has no direct prior work. Combine it with Idea 1 (spike rate) as the input signal and you have a genuinely novel two-stage system that doesn't rely on ever detecting `!start` and `!end` as independent classes.
