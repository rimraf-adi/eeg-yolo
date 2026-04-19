

### Group A: Temporal & Structural Logic (Exploiting Annotation Causality)

**1. Relative Temporal Regression from Spike Anchors**
* **The Concept:** Instead of predicting `!start` and `!end` as sparse isolated classes, predict them as continuous temporal distances ($\Delta t_{start}$, $\Delta t_{end}$) conditioned on—and anchored to—detected local spikes (`!`).
* **The Value:** Completely bypasses the class-imbalance problem of sparse boundary labels by reframing them as regression targets from data-rich anchor points.

**2. Spike Density as a Phase Transition Signal**
* **The Concept:** Derive a continuous "spike-rate" timeseries (e.g., spikes per 10s) from a binary spike detector. Train a secondary 1D changepoint model to identify seizures based on the velocity and acceleration of this derived density signal.
* **The Value:** Simplifies the boundary detection task by operating on an engineered, low-frequency signal rather than the chaotic, high-frequency raw EEG.

**3. Biologically Constrained State Machine Optimization**
* **The Concept:** Integrate a Conditional Random Field (CRF) or a Neural Hidden Markov Model (HMM) loss term that mathematically penalizes biologically impossible sequences (e.g., jumping from `ictal` to `background` without predicting an `!end`).
* **The Value:** Enforces strict clinical logic during the optimization phase, preventing the model from making temporal predictions that contradict the sequential nature of seizures.

### Group B: Data Uncertainty & Label Noise

**4. Probabilistic Boundaries via Soft Labels**
* **The Concept:** Model `!start` and `!end` not as hard points, but as Gaussian distributions $\mathcal{N}(\mu, \sigma^2)$ centered at the clinician's annotation. The model outputs both the predicted boundary and the variance ($\sigma^2$).
* **The Value:** Treats the notorious 60% inter-rater agreement in EEG interpretation as a feature rather than a bug, providing clinical uncertainty quantification out of the box.

**5. Weakly Supervised Multi-Instance Learning (MIL) for Annotation Refinement**
* **The Concept:** Treat a 1-minute EEG window as a "bag" of 1-second "instances." If a `!start` is annotated somewhere inside, the bag is positive. Use an attention mechanism to let the model discover the *true* boundary within the bag.
* **The Value:** Bypasses rigid boundary definitions early in training. The model's temporal attention weights act as an automated label-correction mechanism, finding the true physiological onset even if the clinician was off by a few seconds.

### Group C: Spatial & Feature Representation

**6. Spatial-Temporal Graph Convolutional Networks (ST-GCN) for Topographical Propagation**
* **The Concept:** Standard 1D FPNs treat the 18 EEG channels as a flat vector, mixing them arbitrarily. ST-GCNs map the channels to their physical 10-20 scalp topology using a Graph layer before the temporal pyramid.
* **The Value:** Captures the physical spatial propagation of seizures (e.g., focal evolving to bilateral), providing a biological spatial heuristic to the temporal boundary detector.

**7. Contrastive Representation Learning for State Disentanglement**
* **The Concept:** Apply a contrastive loss (e.g., InfoNCE) at the bottleneck of the encoder to force latent representations of `interictal` segments to cluster together and actively repel `background` segments.
* **The Value:** Forces the FPN to create a highly linearly separable latent space *before* boundary regression occurs, drastically reducing the burden on the final dense prediction heads.

### Group D: Training Dynamics & Deployment Pre-training

**8. Self-Supervised Pretraining on Background EEG (Masked Autoencoders)**
* **The Concept:** Train a 1D Masked Autoencoder to reconstruct masked segments of the abundant, unlabeled background EEG. Fine-tune this encoder on your sparse event boundaries.
* **The Value:** Utilizes the 98% "negative space" data to learn the baseline manifold of normal neonatal brain activity, providing an anomaly-detection foundation for the supervised localization task.

**9. Curriculum Learning via Density-Based Sampling**
* **The Concept:** Directly training on the 98% background / 2% event distribution often collapses gradients, even with Focal Loss. Implement a training curriculum that starts with an artificial 50/50 event-to-background ratio and anneals to the true sparse reality over time.
* **The Value:** A tailored data-loading architecture that guides the network from learning basic morphological features to mastering extreme temporal sparsity without stalling.

**10. Knowledge Distillation for Real-Time Edge Deployment**
* **The Concept:** Cross-attention models (like DETR) are highly accurate but computationally prohibitive for real-time neonatal ICU (NICU) monitoring. Train a massive offline Transformer as a "Teacher," and distill its rich spatial-temporal feature maps into your lightweight, 1D ActionFormer "Student."
* **The Value:** Achieves the theoretical accuracy limits of heavy query-based architectures while maintaining the zero-latency execution required for real-world clinical edge devices.

---

### Strategic Summary Matrix

| Strategy | Primary Domain | Complexity | Target Problem Solved |
| :--- | :--- | :--- | :--- |
| **1. Spike Anchor Regression** | Temporal | High | Severe class imbalance of boundaries |
| **2. Spike Density Signal** | Temporal | Low | Raw EEG high-frequency noise |
| **3. State Machine Loss** | Temporal | High | Impossible clinical predictions |
| **4. Probabilistic Labels** | Labels | Medium | Inter-rater disagreement / fuzzy onsets |
| **5. MIL Refinement** | Labels | Medium | Inaccurate clinician timestamping |
| **6. ST-GCN Topology** | Spatial | High | Loss of physical brain mapping |
| **7. Contrastive Pre-clustering**| Features | Low | Feature overlap between classes |
| **8. Autoencoder Pretraining** | Data | Medium | Unused background data (98%) |
| **9. Curriculum Sampling** | Data | Low | Initial gradient collapse |
| **10. Knowledge Distillation** | Deployment | High | Edge compute latency constraints |

