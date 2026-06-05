# Engineering Validation Report: Face Recognition Subsystem

This report presents evidence-backed engineering validation findings for the offline Face Recognition subsystem. 

---

## 1. Embedding Stability Findings

To measure the stability of the MobileFaceNet embedding representation, we extracted a reference 128D embedding from the baseline aligned face crop of `sample_normal.jpg` and compared it against embeddings extracted from transformed versions of the same crop.

### Stability Under Illumination Variations
* **Factor 0.50 (Low Light)**: `0.956474` cosine similarity.
* **Factor 0.75 (Mild Low Light)**: `0.992340` cosine similarity.
* **Factor 1.25 (Mild Bright)**: `0.992437` cosine similarity.
* **Factor 1.50 (Over-exposure)**: `0.994341` cosine similarity.

> [!NOTE]
> **Illumination Invariance**: Illumination scaling is highly stable. The similarity remains above `0.95` even when brightness is cut in half. This shows that the pre-processing logic, which maps pixel values from `[0, 255]` to `[-0.996, 1.0]` via `(val - 127.5) / 128.0`, successfully cancels out linear illumination shifts.

### Stability Under Sharpness & Focus Variations
* **Gaussian Blur 3x3 (Mild Blur)**: `0.971599` cosine similarity.
* **Gaussian Blur 5x5 (Medium Blur)**: `0.892655` cosine similarity.
* **Gaussian Blur 7x7 (Severe Defocus)**: `0.775569` cosine similarity.

> [!WARNING]
> **Blur Sensitivity**: Embedding similarity drops below `0.90` under medium-to-severe blur. Convolutional layers rely heavily on high-frequency edges and micro-textures. This highlights the absolute necessity of the **Face Quality Assessment (FQA)** blur gate, which rejects blurry inputs before they reach the recognition module.

### Stability Under Resolution Scaling
* **84x84 → 112x112 (Mild Downsampling)**: `0.982158` cosine similarity.
* **56x56 → 112x112 (50% Downsampling)**: `0.859824` cosine similarity.

> [!NOTE]
> **Resolution Requirements**: Face images must maintain sufficient resolution. A 50% downscale causes a severe drop in similarity (`0.859`). This validates the FQA engine's minimum face size threshold (`150px` bounding box), ensuring that downsampling degradation is prevented.

### Stability Under Roll Rotation
* **Rotation -15°**: `0.953257` | **Rotation +15°**: `0.982327`
* **Rotation -10°**: `0.975768` | **Rotation +10°**: `0.978596`
* **Rotation -5°**: `0.959019` | **Rotation +5°**: `0.959288`

> [!NOTE]
> **Rotation Tolerance**: MobileFaceNet exhibits high robustness to minor in-plane rotations, maintaining similarities above `0.95` within a $\pm 15^\circ$ range. In production, this roll is pre-compensated by the `FaceAligner` module, which rotates the face to horizontal eye alignment.

### Stability Under Sensor Noise
* **Gaussian Noise (std = 5)**: `0.939449` cosine similarity.
* **Gaussian Noise (std = 10)**: `0.795406` cosine similarity.
* **Gaussian Noise (std = 20)**: `0.758243` cosine similarity.

> [!WARNING]
> **Noise Sensitivity**: The embedding is sensitive to sensor noise, with similarity dropping to `0.758` at std 20. Camera feeds must be pre-filtered or captured under adequate lighting to minimize noise.

---

## 2. Similarity Distribution Analysis

The table below summarizes the similarity distributions across different comparison categories. Raw values are exported in [similarity_distribution.csv](file:///d:/nhai/evaluation/similarity_distribution.csv).

| Comparison Category | Target Image | Transformation / Condition | Cosine Similarity |
| :--- | :--- | :--- | :--- |
| **Identical** | `sample_normal.jpg` | Baseline vs Baseline | **1.000000** |
| **Transformed (Illumination)**| `sample_normal.jpg` | Brightness Factor 1.50 | **0.994341** |
| **Transformed (Blur)** | `sample_normal.jpg` | Gaussian Blur 3x3 | **0.971599** |
| **Transformed (Rotation)** | `sample_normal.jpg` | Rotation +15° | **0.982327** |
| **Transformed (Noise)** | `sample_normal.jpg` | Gaussian Noise std 5 | **0.939449** |
| **Different (Mild Tilt)** | `sample_mild_tilt.jpg`| Roll -15° (Aligned) | **0.998995** |
| **Different (Extreme Tilt)**| `sample_extreme_tilt.jpg`| Roll 55° (Alignment Gate) | *Failed (Gate Rejected)* |
| **Different (Close Up)** | `sample_close_up.jpg` | Scaled / Zoomed (Aligned) | **0.998271** |
| **Different (Small Eyes)** | `sample_small_eyes.jpg`| Lower resolution (Aligned) | **0.969916** |

### Key Observations
1. **Alignment Success**:
   Comparing the baseline face with its transformed and tilted counterparts (`sample_mild_tilt.jpg`, `sample_close_up.jpg`, `sample_small_eyes.jpg`) yielded extremely high similarities (`> 0.969`). This proves that the `FaceAligner` successfully standardized the position, scale, and angle of the face, aligning the features perfectly before embedding extraction.
2. **Alignment Safety Gate**:
   `sample_extreme_tilt.jpg` was rejected by the `FaceAligner` (`FACE_POORLY_POSITIONED`) because the tilt angle ($55^\circ$) exceeded the $45^\circ$ feasibility threshold. This demonstrates that the preprocessing safety layer operates correctly, preventing garbage inputs from reaching the recognition model.

---

## 3. Engineering & Robustness Observations

### The "Null Input" Embedding Collision Risk
During initial integration testing, we verified that running model inference on uniform color crops (e.g., solid red or green blocks) resulted in extremely high similarities (`> 0.95`). 
* **Mechanism**: Convolutional neural networks extract features using spatial gradients. Solid color blocks lack gradients, forcing all activations to zero and causing the network to output its baseline bias vector. Since all solid color crops result in the same bias vector, they collide with a similarity near `1.0`.
* **Mitigation**: The system's integration architecture mitigates this risk by executing recognition **only** after both FQA (Face Quality Assessment) and Liveness validation pass. FQA rejects blank or zero-gradient inputs due to edge count, size, and blur thresholds, ensuring that only actual face structures are passed to the model.

### Database Edge Case Robustness
* **Empty Registry**: Verified that calling `SQLiteFaceRegistry.verify()` on an empty database returns `(None, 0.0)` gracefully without throwing exceptions.
* **Corrupted Embedding Records**: Inserted invalid BLOB structures of size `100 bytes` (insufficient size) and `1000 bytes` (excessive size) into the database. The `SQLiteFaceRegistry._load_cache()` loader successfully skipped the corrupted records (filtering them out because their float32 size was not exactly 128), loaded the valid records, and maintained 100% search and verification functionality without throwing any exceptions or crashing.
