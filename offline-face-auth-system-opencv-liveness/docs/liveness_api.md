# Liveness SDK Developer API Guide

The `LivenessSDK` is the unified entry point for the offline face liveness verification pipeline. It encapsulates Face Quality Assessment (FQA), Active Challenge verification, and Passive Presentation Attack Detection (PAD).

---

## 1. Class Interface

```python
from opencv_module.sdk import LivenessSDK

sdk = LivenessSDK(preset="MEDIUM_SECURITY")
```

### Constructor Parameters

* `preset` (str | dict, default `"MEDIUM_SECURITY"`):
  * `"LOW_SECURITY"`: 1 challenge (Blink), lenient thresholds, fast validation.
  * `"MEDIUM_SECURITY"`: 2 challenges (Blink -> Turn Left), standard thresholds.
  * `"HIGH_SECURITY"`: 4 challenges (Blink -> Smile -> Turn Left -> Turn Right), strict thresholds.
  * Can also accept a custom dictionary matching the config schema.

---

## 2. API Reference

### `process_frame`

Processes a single camera frame and drives the liveness state machine.

```python
def process_frame(
    self,
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    landmarks: Optional[np.ndarray],
    tracking_confidence: float = 1.0,
    timestamp: Optional[float] = None
) -> Dict[str, Any]
```

#### Inputs

1. **`frame`** (`np.ndarray`): BGR camera frame in OpenCV format.
2. **`bbox`** (`Tuple[int, int, int, int]`): Face bounding box formatted as `(x, y, w, h)`.
3. **`landmarks`** (`np.ndarray`): Face Mesh landmark array of shape `(468, 3)` or `(478, 3)`. Coordinates must be normalized (0.0 to 1.0) relative to frame dimensions.
4. **`tracking_confidence`** (`float`): MediaPipe landmark tracking confidence score (0.0 to 1.0).
5. **`timestamp`** (`float`, optional): Precision epoch or monotonic clock timestamp in seconds.

#### Output Schema

Returns a standardized dictionary representation of the liveness session state:

```json
{
  "success": false,
  "decision": "SUSPECT",
  "trust_score": 0.25,
  "session_id": "8a3d...",
  "current_challenge": "BLINK",
  "active_score": 0.25,
  "passive_score": 0.88,
  "final_liveness_score": 0.565,
  "verified_at": null,
  "expires_at": null,
  "error": null
}
```

* **`success`** (`bool`): `true` only when the active challenge sequence completes, the decision is `"LIVE"`, and no quality/security overrides triggered.
* **`decision`** (`str`): `"LIVE"` | `"SUSPECT"` | `"SPOOF"`.
* **`trust_score`** (`float`): Same as `final_liveness_score`. Fused active and passive value (0.0 to 1.0).
* **`session_id`** (`str`): Unique hexadecimal identifier for the current session.
* **`current_challenge`** (`str` | `null`): `"BLINK"`, `"SMILE"`, `"TURN_LEFT"`, `"TURN_RIGHT"`, or `null`.
* **`active_score`** (`float`): Active challenge completion progress (0.0 to 1.0).
* **`passive_score`** (`float`): Passive anti-spoofing score (0.0 to 1.0).
* **`final_liveness_score`** (`float`): Weighted linear fusion score (0.0 to 1.0).
* **`verified_at`** (`float` | `null`): Session validation timestamp.
* **`expires_at`** (`float` | `null`): Validation expiration timestamp.
* **`error`** (`str` | `null`): Standardized error code if liveness fails or is blocked.

---

## 3. Standardized Error Codes

The SDK centralizes error codes inside `opencv_module.errors` to notify developers of exact failure modes:

| Error Code | Trigger Condition | Recommended User Instruction |
| :--- | :--- | :--- |
| `FACE_NOT_DETECTED` | Face Mesh landmarks or bounding box missing from frame. | "Place your face inside the frame" |
| `FACE_TOO_SMALL` | Bounding box dimensions fall below the preset minimum. | "Move closer to the camera" |
| `LANDMARKS_UNSTABLE` | MediaPipe tracking confidence is low or face tilt exceeds 45°. | "Hold your device steady and look straight" |
| `FRAME_STALLED` | Frozen camera feed or video replay frame duplication detected. | "Verification failed: Feed stalled" |
| `TIMEOUT` | Challenge timer exceeded without completion. | "Time limit exceeded. Please try again" |
| `CHALLENGE_NOT_COMPLETED`| Sub-score quality checks fail (e.g. high blur, bad lighting). | "Ensure good lighting and remove blur" |
| `PASSIVE_SPOOF_DETECTED` | Passive texture, depth, or motion anomaly detected. | "Bypass detected: Please use a live face" |

---

## 4. Basic Integration Example

```python
import cv2
from opencv_module.sdk import LivenessSDK
from opencv_module import errors

# Initialize the SDK
sdk = LivenessSDK(preset="MEDIUM_SECURITY")

# In your frame processing loop:
def on_camera_frame(bgr_frame, bbox, landmarks, tracking_conf):
    res = sdk.process_frame(
        frame=bgr_frame,
        bbox=bbox,
        landmarks=landmarks,
        tracking_confidence=tracking_conf
    )
    
    if res["success"]:
        print(f"Liveness verified! Trust Score: {res['trust_score']}")
        # Proceed with attendance mark
        return
        
    if res["error"] == errors.PASSIVE_SPOOF_DETECTED:
        print("Spoof attempt blocked!")
        sdk.reset()
        # Prompt security alert
        
    elif res["error"] == errors.TIMEOUT:
        print("Session expired: prompt retry button.")
        sdk.reset()
```
