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