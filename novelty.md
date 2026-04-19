
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

By the way, to unlock the full functionality of all Apps, enable [Gemini Apps Activity](https://myactivity.google.com/product/gemini).
