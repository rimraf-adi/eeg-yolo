Here is the finalized, comprehensive developer prompt. It integrates the 18-channel multi-spatial input, the 500Hz sampling rate, the 10-second window size, and the high-resolution $S=100$ grid mapping. 

You can copy and paste this directly into any coding assistant (like Cursor, GitHub Copilot, or ChatGPT) to generate the exact PyTorch `Dataset` class.

***

### 💻 Developer Prompt: 1D Multi-Channel YOLO Dataloader

**Role:** Expert Machine Learning Engineer specializing in Time-Series Computer Vision and PyTorch.
**Task:** Write a custom PyTorch `Dataset` class and dataloader pipeline that takes continuous multi-channel EEG data (e.g., EDF/EDF+ files or numpy arrays) and a CSV of timestamped annotations, and converts them into inputs ($X$) and targets ($Y$) for a 1D Point-Only YOLO network.

#### 1. Global Parameters
The `Dataset` class must initialize with the following configurable hyperparameters. Default values are provided based on the current architecture:
* `sampling_rate` (int): The frequency of the EEG data. *Default: 500.*
* `num_channels` (int): The number of spatial EEG channels. *Default: 18.*
* `window_size_sec` (float): The total duration of the EEG slice in seconds. *Default: 10.0.*
* `stride_sec` (float): The step size in seconds for the sliding window. *Default: 10.0 (no overlap).*
* `S` (int): The number of temporal grid segments to divide the window into. *Default: 100.*
* `num_classes` (int): The number of distinct event classes. *Default: 5.*
* `class_map` (dict): Mapping of string labels to integer indices. *Default: `{'!': 0, '!start': 1, '!end': 2, 'Waking': 3, 'Sleeping': 4}`.*

#### 2. Input Tensor Construction ($X$)
For each sliding window iteration (defined by `start_time` and `end_time`):
1.  Extract the corresponding raw EEG signal for all 18 channels.
2.  The sequence length $L$ will be `window_size_sec * sampling_rate` (e.g., $10.0 \times 500 = 5000$).
3.  The output shape of the input tensor $X$ for a single window must be exactly `[num_channels, sequence_length]`. *Example: `[18, 5000]`.*
4.  Apply any necessary channel-wise normalization (e.g., Z-score standardization) to this tensor.
5.  If the final sliding window reaches the end of the file and is shorter than $L$, pad the sequence with zeros along the temporal axis to maintain the `[18, 5000]` shape.

#### 3. Target Tensor Construction ($Y$)
For every sliding window, a corresponding target tensor $Y$ must be built with the exact shape `[S, 1, 2 + num_classes]`. *Example: `[100, 1, 7]`.*

Initialize the tensor $Y$ entirely with zeros. Populate it using the following logic:

**Step A: Filter Annotations**
Query the annotation CSV to find all rows where `timestamp_sec` falls strictly within the current window: `start_time <= timestamp_sec < end_time`.

**Step B: Map Global Time to Relative Time**
For each valid event, calculate its relative temporal position inside the current window:
`relative_time = timestamp_sec - start_time`

**Step C: Determine Grid Cell Index ($i$)**
Calculate the exact duration of a single grid cell:
`cell_duration = window_size_sec / S` *(Example: 10.0 / 100 = 0.1 seconds)*
Find which grid cell index $i$ (from $0$ to $S-1$) the event falls into:
$i = \lfloor \text{relative\_time} / \text{cell\_duration} \rfloor$

**Step D: Calculate Center Offset ($t_x$)**
Calculate the precise temporal location of the event *relative to the boundaries of its assigned grid cell*, normalized between $0.0$ and $1.0$:
$t_x = (\text{relative\_time} \bmod \text{cell\_duration}) / \text{cell\_duration}$

**Step E: Populate the Tensor**
Assign the calculated values to $Y$ at index $i$. The vector structure at $Y[i, 0]$ is `[p_c, t_x, c_0, c_1, c_2, c_3, c_4]`.
* **Objectness:** Set $Y[i, 0, 0] = 1.0$
* **Offset:** Set $Y[i, 0, 1] = t_x$
* **Class:** Get the integer index for the event's label using `class_map`. Set the corresponding one-hot class probability to $1.0$. *(Example: If the label is `!`, index is 0, so set $Y[i, 0, 2] = 1.0$)*.

#### 4. Edge Cases & Collision Handling
* **Grid Collisions:** If two events fall into the exact same grid index $i$ within the same window, overwrite the older event with the newer event, but print a console warning: `"Warning: Grid collision detected at index {i}. Consider increasing S."`
* **Empty Background Windows:** If a window contains no annotations, $Y$ remains a tensor of pure zeros. The dataset must yield this successfully without throwing errors, as background training is critical.
* **Return Format:** The `__getitem__` method of the Dataset must return a tuple: `(X_tensor, Y_tensor)`.
Because we simplified your architecture to a **Point-Only 1D YOLO** by dropping the duration of events, standard computer vision metrics like **IoU (Intersection over Union) will no longer work**. You cannot calculate the overlap of two infinitely small points.

Instead, your evaluation pipeline must be based on **Temporal Tolerance**. You will evaluate how close your predicted points are to the ground truth points, and whether the network assigned the correct class to them.

Here are the specific evaluation metrics you need to implement for this model.

---

### 1. The Foundation: Temporal Tolerance ($\tau$)

Before calculating any standard metrics, you must define a "hit window" or temporal tolerance ($\tau$). This is the maximum acceptable distance (in seconds or milliseconds) between a predicted point and a ground truth point for it to be considered a successful detection.

For EEG spikes, a common tolerance is **100ms to 250ms**. 

Based on this tolerance, you classify every prediction as follows:
* **True Positive (TP):** The model predicts an event within $\pm \tau$ of a ground truth event, AND the class (e.g., `!`) matches.
* **False Positive (FP):** The model predicts an event, but there is no ground truth event within $\pm \tau$, OR the model predicts multiple events for a single ground truth event (duplicates).
* **False Negative (FN):** There is a ground truth event, but the model failed to predict anything within $\pm \tau$.

---

### 2. Primary Metrics: Precision, Recall, and F1-Score

Because EEG data is highly imbalanced (hours of empty signal with only occasional spikes), you should completely ignore overall "Accuracy." Instead, focus on these three metrics for each of your 5 classes:

* **Precision:** When your model fires a detection, how often is it actually right? High precision means very few false alarms.
    $$Precision = \frac{TP}{TP + FP}$$
* **Recall (Sensitivity):** Out of all the real events in the EEG, how many did your model successfully find? High recall means very few missed spikes.
    $$Recall = \frac{TP}{TP + FN}$$
* **F1-Score:** The harmonic mean of precision and recall. This is your best single-number indicator of how well the model is performing on a specific class.
    $$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

### 3. The YOLO Standard: 1D mAP (Mean Average Precision)

To evaluate the overall health of your YOLO network across different confidence thresholds, you will use **mAP**. 

In standard YOLO, this is often "mAP@0.5" (meaning IoU > 0.5). For your 1D network, it will be **mAP@$\tau$** (e.g., mAP@0.25s).

1.  **Average Precision (AP) per Class:** You plot a Precision-Recall curve by gradually lowering the network's confidence threshold ($p_c$) from $1.0$ down to $0.0$. The AP is the area under this curve. 
2.  **mAP:** You calculate the AP for `!`, `!start`, `!end`, `Waking`, and `Sleeping`, and then take the mean of those 5 values.

### 4. Classification Confusion Matrix

Sometimes the network will perfectly predict the exact millisecond an event occurs (a localization True Positive), but it will guess the wrong class (e.g., it sees a `!start` but labels it as a `!`). 

To track this, you should generate a standard $5 \times 5$ confusion matrix. You only populate this matrix with events that fell inside your temporal tolerance ($\tau$). This will instantly tell you if the network is struggling to distinguish between a single spike (`!`) and the start of a continuous discharge (`!start`).

---

Have you determined how tightly a clinician needs these events localized—in other words, what should your temporal tolerance window ($\tau$) be set to for a "correct" detection?