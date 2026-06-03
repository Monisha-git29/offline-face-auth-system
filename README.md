# Offline Facial Recognition Preprocessing: Preprocessing Pipeline

This workspace provides lightweight, production-grade, and fully-offline preprocessing modules using **OpenCV** and **NumPy**. Optimized for mobile deployment (React Native integration), this repository includes **Face Alignment**, **CLAHE Contrast Enhancement**, **Laplacian Variance Blur Detection**, the unified **Face Quality Assessment (FQA) Engine**, and the **EAR Blink Liveness Detection** engine.

---

## 1. Directory Structure

```
d:\nhai\
├── opencv_module/
│   ├── __init__.py          # Package initialization, clean exports
│   ├── alignment.py         # FaceAligner class (Validations, checks, single-pass warp)
│   ├── enhancement.py       # CLAHEEnhancer class (Grayscale prioritization + quality hints)
│   ├── blur.py              # BlurDetector class (Laplacian variance calculations)
│   ├── fqa.py               # FaceQualityEngine class (Compilation, scoring & recommendations)
│   ├── blink.py             # BlinkDetector class (MediaPipe EAR calculations)
│   └── utils.py             # Math helper functions, debug visualizers
├── examples/
│   ├── align_face_demo.py   # Demonstration loading & aligning 5 samples from disk
│   ├── enhance_face_demo.py # Demonstration applying CLAHE, quality gateways & metrics
│   ├── blur_demo.py         # Complete integrated pipeline demo (Align -> CLAHE -> Blur)
│   ├── fqa_demo.py          # Unified FQA Engine demo (Excellent, warning & recapture cases)
│   ├── blink_demo.py        # Stream simulation verifying time-based blinks & challenges
│   └── benchmark.py         # Latency, FPS, and throughput performance profiling
├── requirements.txt         # Package dependencies (opencv-python, numpy)
└── README.md                # Detailed pipeline guide and mathematical documentation (This File)
```

---

## 2. Pipeline Integration Architecture

The modules form the core preprocessing blocks of our secure offline facial recognition and liveness detection pipeline:

```
Camera Feed 
  │
  ▼
[BlazeFace Face Detection]  --> Returns bounding box & eye coordinates
  │
  ▼
[Face Validation]           --> Checks inputs and bounds
  │
  ▼
[Face Alignment] (Module 1) --> Performs single-pass affine warp to 112x112
  │
  ▼
[CLAHE Enhancement] (Module 2) --> Normalizes lighting and mitigates shadows
  │
  ▼
[Blur Detection] (Module 3) --> Laplacian variance check to reject blurry frames
  │
  ▼
[Face Quality Assessment]   --> Compiles Quality Hints & Blur Status (FQA Engine)
  │
  ▼
[Liveness Detection] (Mod 4)--> MediaPipe Face Mesh checks (Blink, Smile, Turn)
  │
  ▼
[MobileFaceNet Recognition] --> Standardized 112x112 crop used to extract 128D embeddings
```

---

## 3. Module 1: Face Alignment

To align the face, we perform an **affine similarity transformation** (rotation, scaling, and translation) in a **single pass**.
- **Angle Calculation**: Given left eye $E_L = (x_L, y_L)$ and right eye $E_R = (x_R, y_R)$:
  $$\theta = \arctan2(y_R - y_L, x_R - x_L) \times \frac{180}{\pi}$$
- **Scale Factor ($s$)**: Derived from current eye distance ($d_{\text{input}}$) and target crop distance ($d_{\text{target}}$):
  $$s = \frac{d_{\text{target}}}{d_{\text{input}}}$$
- **Translation Adjustments ($tx, ty$)**: Adjusted directly within the rotation matrix $R$ to map the eye midpoint $M_{\text{input}}$ to the desired output crop midpoint $M_{\text{target}} = (W/2, H \cdot r_y)$.

---

## 4. Module 2: CLAHE Contrast Enhancement

CLAHE divides the face into $8 \times 8$ local tiles, clips histograms at `clipLimit=2.0` (redistributing clipped pixels uniformly to prevent noise amplification in shadows), equalizes local contrast, and blends neighboring tiles using bilinear interpolation.

---

## 5. Module 3: Blur Detection

Convolving the grayscale face image with a standard $3 \times 3$ Laplacian kernel detects high-frequency edge gradients:
$$K_{\text{Laplacian}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
The variance of the Laplacian response acts as the blur score:
$$\text{Blur Score} = \text{Var}(\Delta I) = \frac{1}{N}\sum_{i=1}^N (\Delta I_i - \mu_{\Delta I})^2$$

---

## 6. Module 4: Face Quality Assessment (FQA) Engine

The `FaceQualityEngine` orchestrates the entire preprocessing flow, normalizes metrics into sub-scores from $0$ to $100$, and computes a weighted average.

### 6.1 Mathematical Scoring Formulations
- **Brightness**: Optimal mean range $[100, 180]$ maps to $100$. Underexposure and overexposure decay linearly down to $30$ and up to $240$.
- **Blur**: Scales linearly from $0$ up to $b_{\text{target}} = 150.0$.
- **Pose**: Decreases linearly from $100$ down to $0$ as face tilt roll reaches $\theta_{\text{max}} = 45^\circ$.
- **Face Size**: Scales linearly between $50\text{px}$ and $150\text{px}$ bounding box dimensions.

---

## 7. Module 5: Blink Detection (Liveness Check)

To detect eye blinks fully offline using MediaPipe Face Mesh landmarks, we apply the Eye Aspect Ratio (EAR) method coupled with high-precision time-based safety windows.

### 7.1 Mathematical EAR Formulation
For a single eye, given horizontal inner/outer corners ($p_1, p_4$) and vertical eyelid points ($p_2, p_6$ and $p_3, p_5$):
$$\text{EAR} = \frac{||p_2 - p_6||_2 + ||p_3 - p_5||_2}{2 \cdot ||p_1 - p_4||_2}$$

In MediaPipe 468/478 landmark coordinate space, the indices used are:
- **Left Eye**: corners `(33, 133)`, verticals `(160, 144)` and `(158, 145)`.
- **Right Eye**: corners `(362, 263)`, verticals `(385, 380)` and `(387, 373)`.

Average EAR is used to determine closure:
$$\text{EAR}_{\text{avg}} = \frac{\text{EAR}_{\text{left}} + \text{EAR}_{\text{right}}}{2.0}$$

### 7.2 Robust Validation Layer

1.  **Face Presence Check**: If landmarks are null or contain $<400$ entries, the assessment is aborted and returns `{"success": False, "error": "FACE_NOT_DETECTED"}`.
2.  **Landmark stability Gate**: MediaPipe Face Mesh outputs a tracking confidence score. If the confidence falls below `min_confidence = 0.5`, the frame is ignored to avoid false blinks from rapid head shakes, multiple faces, or reflections on glasses. It returns `{"success": False, "error": "LANDMARKS_UNSTABLE"}`.
3.  **Aspect-Ratio Pixel Scaling**: Landmarks are scaled to absolute pixels using the camera frame dimensions $(W, H)$ first:
    $$x_{\text{pixel}} = x_{\text{normalized}} \cdot W,\quad y_{\text{pixel}} = y_{\text{normalized}} \cdot H$$
    This guarantees consistent EAR metrics when switching between portrait and landscape modes.

### 7.3 Time-Based Blink Verification State Machine
To maintain consistency across diverse camera frame rates (e.g. $15\text{ FPS}$ on budget Android devices vs. $60\text{ FPS}$ on high-end iOS devices), the state machine tracks eye closure duration using timestamps:
- Upon transition to the `CLOSED` state, the engine records the start timestamp: $T_{\text{start}} = \text{time.perf\_counter()}$.
- Upon opening, the duration is calculated: $\Delta T = T_{\text{end}} - T_{\text{start}}$.
- **Realistic Blink Gate**: A blink is verified only if $\Delta T$ falls in a realistic human blink interval:
  $$50\text{ ms} \le \Delta T \le 500\text{ ms}$$
  *(i.e. $0.05\text{s} \le \Delta T \le 0.5\text{s}$)*
- If the eyes remain closed longer than $500\text{ms}$, it is classified as a prolonged squint or sleep nod and rejected.

### 7.4 Challenge System Compatibility
The API is fully compliant with liveness challenge engines:
- Pass the challenge type `active_challenge = "BLINK"`.
- If a valid blink is verified during the active blink challenge, the output dictionary includes `"challenge_verified": True`.

---

## 8. Parameter Calibration Guidance

For optimal performance in various production hardware, calibrate the thresholds as follows:

| Parameter | Recommended Default | Calibration Guide |
| :--- | :--- | :--- |
| **`ear_threshold`** | **`0.22`** | Decrease (e.g., `0.18`) for small eyes or squinting users; increase (e.g., `0.24`) for users wearing thick glasses to ensure complete closures trigger properly. |
| **`min_consecutive_frames`** | **`2`** | Increase to `3` in low-light environments (high noise) to prevent micro-jitters from triggering false closures. |
| **`min_confidence`** | **`0.5`** | Increase to `0.7` in secure facial recognition zones (banks, e-gates) to ensure landmarks are perfectly locked. |
| **`max_blink_duration`** | **`0.50s`** | Standard is `500ms`. Extend to `0.75s` (750ms) if users are in cold remote locations where eye movements can be slightly slower. |

---

## 10. Module 6: Head Turn Detection (Liveness Check)

To verify spatial head movement without heavy computational overhead, we calculate a **Symmetry Deviation Ratio ($R$)** using stable facial landmarks (Nose tip `1`, and Left/Right eye centers) scaled to aspect-ratio corrected absolute pixel coordinates:
- **Left Eye Center ($E_L$)**: Midpoint of Left Eye corners `33` and `133`.
- **Right Eye Center ($E_R$)**: Midpoint of Right Eye corners `362` and `263`.
- **Nose Tip ($N$)**: Landmark `1`.

Horizontal spans from the nose tip to each eye center are:
$$dx_L = |x_{E_L} - x_N|, \quad dx_R = |x_{E_R} - x_N|$$

The Symmetry Deviation Ratio ($R$) is defined as:
$$R = \frac{dx_L - dx_R}{dx_L + dx_R}$$

Head Yaw is mapped as:
$$\text{Yaw} = R \times 80.0$$

- **Positive Yaw (>18°)**: Head is turned LEFT.
- **Negative Yaw (<-18°)**: Head is turned RIGHT.
- **Near Zero (<8°)**: Head is centered.

An **Exponential Moving Average (EMA)** filter ($\alpha=0.35$) is applied to smooth out high-frequency noise. A 3-frame turn lock, 3-frame center lock, and a 300ms cooldown lockout are enforced.

---

## 11. Module 7: Unified Challenge Engine Orchestrator

The `ChallengeEngine` statefully coordinates sequential or randomized liveness challenges. It manages:
1.  **Session & Expiry Management**: Generates a unique `session_id` offline and appends `verified_at` and `expires_at` timestamps upon completion.
2.  **Difficulty Presets**:
    - `EASY`: `["BLINK"]`
    - `MEDIUM`: `["BLINK", "TURN_LEFT"]`
    - `HIGH`: `["BLINK", "TURN_LEFT", "TURN_RIGHT"]`
3.  **Anti-Repetition Sequence Logic**: Shuffles sequence challenges without generating identical patterns on consecutive runs.
4.  **Anti-Spoofing & Replay Rejection**:
    - **Monotonic Frame Freshness Check**: Validates timestamps increase monotonically.
    - **Landmark Displacement Check**: Calculates average Euclidean (L2) distance displacement of all 468 landmarks across consecutive frames:
      $$\text{displacement} = \frac{1}{468} \sum_{i=1}^{468} \sqrt{(x_i^{(t)} - x_i^{(t-1)})^2 + (y_i^{(t)} - y_i^{(t-1)})^2}$$
      If displacement remains below `1e-5` for 3 consecutive frames, the session fails with `FRAME_STALLED`.
5.  **Explicit Return-to-Center**: Head turns require the user to turn and then return to center before the challenge is marked complete.
6.  **Dual Timeout**: Independent challenge timeout (5s) and session-wide timeout (`max(15s, len(challenges) * timeout)`).
7.  **Structured Fail Codes**: `TIMEOUT`, `FACE_NOT_DETECTED`, `LANDMARKS_UNSTABLE`, `FRAME_STALLED`, `CHALLENGE_NOT_COMPLETED`.
8.  **Session Analytics**: Returns total duration, completed challenges, failed attempts, average tracking confidence, and a detailed challenge history log (`challenge_history`).

---

## 12. Module 8: Passive Anti-Spoofing Engine

The `PassiveAntiSpoofEngine` is a multi-frame temporal engine that detects presentation attacks (printed photos, screen replays, frozen frames) over a rolling history buffer ($10-15$ frames):
1.  **LBP Texture Entropy**: Vectorized $3 \times 3$ Local Binary Pattern histograms calculated on a standardized $112 \times 112$ grayscale crop. Skin displays moderate entropy ($5.0 \le H \le 7.0$). Moire screen grids or high-frequency print noise spike the entropy ($H > 7.2$), whereas blur drops it ($H < 4.5$).
2.  **Scale-Invariant depth protrusion ratio ($R_{\text{depth}}$)**: Measures relative 3D nose protrusion normalized by the 3D inter-eye Euclidean span:
    $$R_{\text{depth}} = \frac{|z_{\text{nose\_tip}} - z_{\text{eyes\_mid}}|}{D_{\text{eyes}}}$$
    Human faces typically sit in a tight range $R_{\text{depth}} \in [0.18, 0.42]$. Flat spoof targets remain compressed near $0.0$.
3.  **Jitter & Motion Consistency**: Tracks standard deviation of landmarks ($\sigma_{\text{lms}}$) and box centers ($\sigma_{\text{box}}$). Rejects rigid translation bypasses (where landmark-to-box distance variance is $< 0.05$ pixels$^2$) and static pictures.
4.  **Temporal stability**: Tracks Standard Deviation of LBP bin values over the window to detect screen refresh flickers or freeze attacks.
5.  **Adaptive weight tuning**: Dynamic adjustments of signal weights based on bounding box width and yaw angles.
6.  **Spoof Accumulator**: Uses an EMA filter ($\beta = 0.85$) to aggregate deviations, creating a smooth liveness confidence $C_{\text{passive}}$.

---

## 13. Module 9: Fused Liveness Decision Engine

The `LivenessDecisionEngine` fuses active challenges and passive liveness signals into a final verdict:
- **Fusion Equation**:
  $$\text{Final Liveness Score} = \alpha_F S_{\text{active}} + (1 - \alpha_F) S_{\text{passive}}$$
  Where $S_{\text{active}}$ is challenge completion progress ($0.0 - 1.0$), and $S_{\text{passive}}$ is the passive liveness rating ($1.0 - A_{\text{spoof}}$).
- **Hard Security Overrides**:
  - Monotonic/Stall failure $\implies$ immediate `SPOOF` (`final_liveness_score = 0.0`).
  - Passive liveness rating drops below `0.40` $\implies$ immediate `SPOOF`.
  - Active liveness timeout $\implies$ immediate `TIMEOUT`.

---

## 14. Latency Benchmarking (Complete Preprocessing & Fused Liveness)

Using high-precision CPU performance counters over 2,000 iterations:

| Stage / Component | Average Latency | Throughput (FPS) | Latency % Contribution |
| :--- | :--- | :--- | :--- |
| **Face Alignment** | **0.354 ms** | 2,823 FPS | 1.4% |
| **Grayscale CLAHE** | **0.210 ms** | 4,767 FPS | 0.8% |
| **Blur Detection** | **0.062 ms** | 16,127 FPS | 0.2% |
| **Integrated FQA Engine** | **0.490 ms** | 2,042 FPS | 1.9% |
| **Blink Detection** | **0.021 ms** | 47,545 FPS | 0.1% |
| **Head Turn Detection** | **0.004 ms** | 228,699 FPS | < 0.1% |
| **Unified Challenge Engine** | **0.028 ms** | 35,887 FPS | 0.1% |
| **Unified LivenessDecisionEngine** | **0.704 ms** | 1,419 FPS | 2.7% |
| **MediaPipe Face Mesh Inference** (Est) | **22.000 ms** | 45 FPS | 84.1% |
| **Total Integrated Mobile Pipeline** | **26.180 ms** | 38 FPS | 100.0% |

> [!IMPORTANT]
> The active + passive liveness decision fusion engine executes in **704 microseconds** per frame, adding negligible overhead. The entire integrated mobile pipeline (FQA + MediaPipe + Fused Decision) operates at **26.18 ms**, comfortably passing the **$<35\text{ms}$** mobile latency budget.


