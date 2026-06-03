# Real-World Test Suite: Face Liveness & PAD System

This test suite contains 52 structured test cases designed to validate the face preprocessing, active liveness challenges, passive presentation attack detection (PAD), and error reporting of the integrated `LivenessSDK`.

---

## 1. Lighting Conditions (Test Cases 1–10)

| ID | Input Condition | Expected Output | Handling Module | Description / Expected Error |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | Low-light toll booth ($< 10$ lux), high sensor noise | **FAIL** | **FQA** | Recapture prompt: `CHALLENGE_NOT_COMPLETED` (under-exposed) |
| **TC-02** | Normal indoor office lighting ($300$ lux), neutral pose | **LIVE** | **SDK** | Success: completes active blink/turn, decision `LIVE` |
| **TC-03** | Strong backlight (sun behind operator head), face in shadow | **FAIL** | **FQA** | Recapture prompt: `CHALLENGE_NOT_COMPLETED` (under-exposed face) |
| **TC-04** | Direct outdoor sunlight ($> 20,000$ lux), high contrast | **LIVE** | **SDK** | Success: CLAHE normalizes exposure, completes challenges |
| **TC-05** | Fluorescent light flicker ($50$ Hz), minor intensity waves | **LIVE** | **Passive** | Success: LBP temporal filter smooths out minor flicker |
| **TC-06** | Severe strobe flashing light ($> 5$ Hz frequency) | **FAIL** | **FQA** | Recapture prompt: `CHALLENGE_NOT_COMPLETED` (unstable brightness) |
| **TC-07** | Side spotlight, half-shadow face (high lighting asymmetry) | **FAIL** | **FQA** | Recapture prompt: `CHALLENGE_NOT_COMPLETED` (poor contrast) |
| **TC-08** | Complete pitch darkness ($0$ lux), screen glow only | **FAIL** | **SDK** | Rejection: `FACE_NOT_DETECTED` (landmarks extraction fails) |
| **TC-09** | Gradual sunrise transition (light levels rising slowly) | **LIVE** | **SDK** | Success: preprocessing dynamically adapts to mean shift |
| **TC-10** | Moving shadows across face (operator sitting in vehicle) | **LIVE** | **SDK** | Success: tracker maintains lock, active challenges complete |

---

## 2. Face Variations (Test Cases 11–20)

| ID | Input Condition | Expected Output | Handling Module | Description / Expected Error |
| :--- | :--- | :--- | :--- | :--- |
| **TC-11** | Thick, dense beard covering jawline and lips | **LIVE** | **Active** | Success: routes to blink/turn, adjusts smile to mouth width |
| **TC-12** | Heavy black-rimmed eyeglasses (no glare) | **LIVE** | **Active** | Success: EAR threshold registers blinks |
| **TC-13** | Prescription glasses with severe overhead light glare | **FAIL** | **Active** | Timeout: glare blocks eye corners; `TIMEOUT` error |
| **TC-14** | Polarized dark sunglasses | **FAIL** | **SDK** | Rejection: `LANDMARKS_UNSTABLE` or `TIMEOUT` (no eye landmarks) |
| **TC-15** | Standard surgical face mask covering mouth/nose | **FAIL** | **SDK** | Rejection: `FACE_NOT_DETECTED` |
| **TC-16** | Operator wearing high-visibility cap tilted down | **LIVE** | **SDK** | Success: coordinates normalize, landmarks stable |
| **TC-17** | Religious face veil (Niqab) exposing eyes only | **FAIL** | **SDK** | Rejection: `FACE_NOT_DETECTED` |
| **TC-18** | Bandage / gauze patch on one cheek (partial occlusion) | **LIVE** | **SDK** | Success: face alignment and active challenge paths bypass patch |
| **TC-19** | Extreme face tilt ($&gt; 50^{\circ}$ roll angle) | **FAIL** | **FQA** | Recapture prompt: `LANDMARKS_UNSTABLE` (fails alignment) |
| **TC-20** | Heavy makeup or face paint | **LIVE** | **SDK** | Success: geometric landmarks and active challenges remain stable |

---

## 3. Attack Scenarios (Test Cases 21–32)

| ID | Input Condition | Expected Output | Handling Module | Description / Expected Error |
| :--- | :--- | :--- | :--- | :--- |
| **TC-21** | Matte paper printout face held static | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (zero motion variance) |
| **TC-22** | Matte paper printout face translated/shaken manually | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (flat depth ratio) |
| **TC-23** | Curved paper printout face (attempting to fake depth) | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (LBP texture entropy anomaly) |
| **TC-24** | 2D Photo printout with eye holes cut out (attacker blinking behind) | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (flat depth and LBP mismatch) |
| **TC-25** | High-res iPad replay attack ($120$ Hz screen video) | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (moire texture + flat depth) |
| **TC-26** | Smartphone video replay attack ($60$ Hz screen) | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (moire texture + flat depth) |
| **TC-27** | Laptop screen replay under direct overhead sun glare | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (specular screen glare peaks) |
| **TC-28** | Frozen frame injection (identical landmarks/timestamps) | **SPOOF** | **SDK** | Rejection: `FRAME_STALLED` |
| **TC-29** | 3D physical mannequin mask (plastic/silicone face model) | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (LBP entropy targets mismatch) |
| **TC-30** | Video replay with eye blinks cut/synced to prompts | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (flat depth override triggers) |
| **TC-31** | Printed photo wrapped around a cylinder | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (rigid motion consistency fail) |
| **TC-32** | Digital photo zoom/pan animation loop | **SPOOF** | **Passive** | Rejection: `PASSIVE_SPOOF_DETECTED` (planar parallax ratio is 0.0) |

---

## 4. Camera & Stream Issues (Test Cases 33–42)

| ID | Input Condition | Expected Output | Handling Module | Description / Expected Error |
| :--- | :--- | :--- | :--- | :--- |
| **TC-33** | Heavy camera shake / vibrations (shaky toll booth wall) | **LIVE** | **SDK** | Success: motion normalization isolates background shake |
| **TC-34** | Severe motion blur (fast camera panning) | **FAIL** | **FQA** | Recapture prompt: `LANDMARKS_UNSTABLE` (blur score too low) |
| **TC-35** | Moderate motion blur (head moving fast during turns) | **LIVE** | **SDK** | Success: active engine tracking handles moderate blur |
| **TC-36** | Camera frame stall (feed frozen for $&gt; 200$ ms) | **FAIL** | **SDK** | Rejection: `FRAME_STALLED` |
| **TC-37** | Intermittent dropped frames (FPS drops from 15 to 3) | **LIVE** | **SDK** | Success: timers adapt, buffer size holds data longer |
| **TC-38** | Dirty camera lens (dust, fingerprint smudge) | **FAIL** | **FQA** | Recapture prompt: `LANDMARKS_UNSTABLE` (contrast score too low) |
| **TC-39** | Out-of-focus camera (fixed focus misaligned) | **FAIL** | **FQA** | Recapture prompt: `LANDMARKS_UNSTABLE` (severe blur score) |
| **TC-40** | Non-monotonic timestamps (out of order frames) | **FAIL** | **SDK** | Rejection: `FRAME_STALLED` |
| **TC-41** | Camera aspect ratio squashed (image stretched horizontally) | **FAIL** | **FQA** | Recapture prompt: `LANDMARKS_UNSTABLE` (alignment fails) |
| **TC-42** | Low-quality video stream ($160 \times 120$ resolution) | **FAIL** | **FQA** | Recapture prompt: `FACE_TOO_SMALL` |

---

## 5. Edge Cases (Test Cases 43–52)

| ID | Input Condition | Expected Output | Handling Module | Description / Expected Error |
| :--- | :--- | :--- | :--- | :--- |
| **TC-43** | Multiple faces in camera view (crowd behind toll operator) | **LIVE** | **SDK** | Success: tracks primary face in bbox; ignores background |
| **TC-44** | No face present (camera pointing at empty toll booth) | **FAIL** | **SDK** | Rejection: `FACE_NOT_DETECTED` |
| **TC-45** | Face too far away (face dimension $&lt; 40$ pixels) | **FAIL** | **FQA** | Recapture prompt: `FACE_TOO_SMALL` |
| **TC-46** | Face extremely close (crops eyes/mouth outside camera view) | **FAIL** | **SDK** | Rejection: `FACE_NOT_DETECTED` (landmarks out of boundaries) |
| **TC-47** | Partial face visible (half covered by toll window frame) | **FAIL** | **SDK** | Rejection: `FACE_NOT_DETECTED` |
| **TC-48** | Rapidly switching faces (different user cuts in mid-session) | **FAIL** | **SDK** | Rejection: `FRAME_STALLED` (large coordinate jumps) |
| **TC-49** | User wearing eye patch on one eye | **LIVE** | **Active** | Success: EAR handles single eye blinks if configured |
| **TC-50** | Non-human face (printed animal mask or pet dog) | **FAIL** | **SDK** | Rejection: `FACE_NOT_DETECTED` (no human landmarks matched) |
| **TC-51** | User starting session during an active head turn | **FAIL** | **Active** | Rejection: `LANDMARKS_UNSTABLE` (requires return-to-center first) |
| **TC-52** | Bounding box coordinates out of frame boundaries | **FAIL** | **SDK** | Rejection: `FACE_NOT_DETECTED` |
