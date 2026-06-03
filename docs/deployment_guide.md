# Production Deployment Guide: Offline Facial Authentication

This guide outlines deployment recommendations, configurations, and performance optimizations for hosting the liveness SDK inside mobile environments and offline client punch-in terminals.

---

## 1. Selecting Security Presets

Select the liveness security preset based on the performance and security requirements of your target devices:

| Preset Name | Recommended Device / Context | Target Mobile CPU Overhead | Security Description |
| :--- | :--- | :--- | :--- |
| **`LOW_SECURITY`** | Legacy mobile devices (e.g. Android Go, older iPads) | $< 0.8$ ms | Blink challenge only. Lenient passive thresholds. Fast validation, suitable for low-fraud environments. |
| **`MEDIUM_SECURITY`**| Standard modern smartphones (Android 9+, iOS 13+) | $< 1.2$ ms | Default setting. Sequential Blink and Head Turn Left verification. Prevents photo bypasses. |
| **`HIGH_SECURITY`** | High-security attendance terminals and flagship phones | $< 2.0$ ms | Blink, Smile, and bidirectional Head Turns. Full passive sliding-window verification. Prevents screen replays. |

---

## 2. Performance Optimizations

To keep execution speeds within the target budget ($<35$ ms mobile latency), follow these platform optimizations:

### 2.1 OpenCV Thread Pool Settings
Limit OpenCV's default thread pool allocations to prevent CPU core oversaturation on mobile:

```python
import cv2
cv2.setNumThreads(2) # Limit to 2 processing threads
```

### 2.2 Vectorized NumPy Operations
The passive LBP texture extractor uses vectorized array slicing. Ensure the mobile framework uses a NumPy backend linked with optimized BLAS libraries (e.g., Apple Accelerate on iOS, OpenBLAS on Android).

### 2.3 Memory Pre-allocation
The FQA and Alignment modules reuse internal canvas buffers. Do not call `.realloc()` or re-initialize `LivenessSDK` inside your frame loop. Instantiate it once and invoke `reset()` between sessions:

```python
# GOOD: Reuses allocations across updates
sdk = LivenessSDK(preset="MEDIUM_SECURITY")

for frame in camera_stream:
    res = sdk.process_frame(frame, bbox, landmarks)
    if res["success"]:
        sdk.reset()  # Prepares for next user
```

---

## 3. Passive Anti-Spoofing Calibration

For high-profile enterprise punch-ins, we recommend calibrating thresholds locally using the evaluation framework. 

* Run `evaluation/calibrate.py` on your validation dataset to sweep for optimal `passive_spoof_threshold` settings.
* Standard ISO/IEC 30107-3 metrics target:
  * **APCER (Spoof False Accept Rate)**: $< 1.0\%$
  * **BPCER (Live False Reject Rate)**: $< 2.5\%$

---

## 4. Hard Security Rules

To prevent bypasses from compromised hardware streams, the SDK enforces two offline overrides:

1. **Replay Rejection (`FRAME_STALLED`)**: If the landmark displacement falls below $1 \times 10^{-5}$ pixels over 3 consecutive frames, the session is failed immediately. This prevents static photograph holding and video injection.
2. **Flat Depth Rejection (`PASSIVE_SPOOF_DETECTED`)**: If the scale-invariant nose protrusion ratio drops below $0.15$ (i.e. flat screen or paper printout), the session is failed, even if the user successfully completed active blink/smile challenges.
