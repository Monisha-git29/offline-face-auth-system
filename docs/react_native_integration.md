# React Native Integration Workflow Guide

This guide details how to integrate the offline `LivenessSDK` into a React Native camera stream for securing the Attendance marking workflow.

---

## 1. Pipeline Architecture

```
┌─────────────────┐      ┌─────────────┐      ┌────────────────┐      ┌─────────────┐
│  Camera Frame   │ ──►  │  BlazeFace  │ ──►  │ MediaPipe Mesh │ ──►  │ LivenessSDK │
│ (Native Buffer) │      │ (Face BBox) │      │  (Landmarks)   │      │  (Bridge)   │
└─────────────────┘      └─────────────┘      └────────────────┘      └─────────────┘
                                                                             │
                                                                             ▼
┌──────────────────┐      ┌───────────────┐      ┌───────────────┐     ┌─────────────┐
│ Attendance Punch │ ◄──  │ Verify Status │ ◄──  │  Trust Score  │ ◄── │  Response   │
│    [Allowed]     │      │   == "LIVE"   │      │    >= 0.75    │     │ Dictionary  │
`──────────────────`      `───────────────`      `───────────────`     `─────────────`
```

---

## 2. Integration Stages

### Stage A: Camera Frame Capture & Downscaling
1. Use `react-native-vision-camera` to obtain native frame buffers at $\ge 15$ FPS.
2. Target resolution: **720x1280** (or 480x640) BGR/RGB matrices to keep mobile CPU overhead minimal.

### Stage B: BlazeFace & Face Mesh Extraction
1. Pass the frame to the lightweight **BlazeFace** model to determine the bounding box `(x, y, w, h)`.
2. Feed the cropped area to the **MediaPipe Face Mesh** model to retrieve 468 (or 478) 3D landmarks.
3. Normalize coordinates between `0.0` and `1.0` using the frame dimensions.

### Stage C: Forwarding data to LivenessSDK Bridge
Write a Native Module bridge (Python-C++-JNI for Android, Objective-C for iOS) to forward the arguments to the `LivenessSDK.process_frame` call:

```javascript
import { NativeModules } from 'react-native';
const { LivenessBridge } = NativeModules;

// In your camera frame processor:
async function onFrameProcessor(frameData) {
  try {
    const result = await LivenessBridge.processFrame(
      frameData.base64Image, // BGR frame matrix representation
      frameData.bbox,        // [x, y, width, height]
      frameData.landmarks,   // [[x1,y1,z1], ...]
      frameData.trackingConf, // MediaPipe tracking confidence
      frameData.timestamp    // Monotonic timestamp
    );
    
    handleLivenessResult(result);
  } catch (err) {
    console.error("SDK Bridge failed", err);
  }
}
```

---

## 3. UI and Challenge Prompts Integration

The React Native UI should dynamically adapt to guide the user based on the SDK's output state:

1. **Active Challenge Banner**:
   Read `result.current_challenge`. Render the appropriate instruction:
   * `"BLINK"`: "Blink your eyes"
   * `"SMILE"`: "Smile for the camera"
   * `"TURN_LEFT"`: "Turn your head to the left"
   * `"TURN_RIGHT"`: "Turn your head to the right"
   * `null`: "Hold still..."

2. **Error Prompt Routing**:
   Read `result.error`. Map the error code to localized instructions (see `docs/liveness_api.md`). Do not allow markings if `result.error` is present.

---

## 4. Securing the Attendance Punch-In

To prevent attendance spoofing (such as bypassing using deepfakes, picture printouts, or pre-recorded videos), enforce a strict verification checklist in the mobile authentication controller:

```javascript
function handleLivenessResult(res) {
  if (res.success && res.decision === "LIVE") {
    // 1. Double check security fields
    if (res.trust_score >= 0.75 && res.error === null) {
      enableAttendanceMarkingButton();
    } else {
      rejectAttendancePunch("Security criteria not met.");
    }
  } else if (res.decision === "SPOOF") {
    rejectAttendancePunch("Spoof attempt detected!");
    alertUser("Bypass detected. Please use a live face.");
  } else if (res.error) {
    updateInstructionBanner(res.error);
  }
}
```
> [!IMPORTANT]
> **Cryptographic Session Signing**: For absolute production security, the mobile app should transmit the completed session payload (`session_id`, `trust_score`, `verified_at`) signed with the mobile secure enclave's private key to the attendance server to prevent transaction interception.
