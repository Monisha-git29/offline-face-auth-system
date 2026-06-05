"""
Liveness SDK Comprehensive Demonstration.

Validates the LivenessSDK API and demonstrates:
1. Successful LIVE session with active challenge progression (Blink, Smile, Turn Left, Turn Right).
2. Active challenge progression transitions.
3. Passive spoof detection & rejection (using moire / flat depth).
4. Timeout rejection.
5. Early Face Quality Assessment recapture (e.g. face too small).
"""

import os
import sys
import time
import numpy as np
import cv2

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.sdk import LivenessSDK
from opencv_module import errors


def create_base_skin_frame(width: int = 400, height: int = 400, draw_moire: bool = False, noise_std: float = 2.0) -> np.ndarray:
    """
    Creates a synthetic BGR camera frame.
    Draws a face ellipse filled with skin tone and minor random Gaussian noise
    to simulate organic skin sub-surface scattering texture.
    If draw_moire is True, overlays high-frequency grid lines to simulate screen replays.
    """
    # 1. Neutral gray background canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 128

    # 2. Skin tone face ellipse (User's face bounding box ~ [100, 80, 200, 240])
    # Ellipse center (200, 200), axes (70, 100)
    cv2.ellipse(canvas, (200, 200), (70, 100), 0, 0, 360, (180, 200, 235), -1)

    # 3. Add skin micro-texture Gaussian noise
    if not draw_moire and noise_std > 0:
        noise = np.random.normal(0, noise_std, canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. If moire requested, overlay dark screen scanlines grid
    if draw_moire:
        for x in range(0, width, 3):
            cv2.line(canvas, (x, 0), (x, height), (0, 0, 0), 1)
        for y in range(0, height, 3):
            cv2.line(canvas, (0, y), (width, y), (0, 0, 0), 1)

    return canvas


def create_synthetic_landmarks(
    pose_type: str = "CENTER",
    eye_state: str = "OPEN",
    smile_state: str = "NEUTRAL",
    depth_flat: bool = False,
    jitter_scale: float = 1e-4
) -> np.ndarray:
    """
    Generates a 468x3 coordinate array representing MediaPipe Face Mesh.
    Includes eye corners and nose tip points to test EAR/Yaw.
    If depth_flat is True, nose Z protrusion is flattened to 0.0, simulating screens/paper.
    """
    landmarks = np.ones((468, 3), dtype=np.float32) * 0.5  # neutral midpoint

    # Standard eye x coordinate placement
    landmarks[33] = [0.62, 0.40, -0.02]   # Left Eye Outer
    landmarks[133] = [0.58, 0.40, -0.02]  # Left Eye Inner
    landmarks[160] = [0.60, 0.39, -0.02]  # Vertical 1
    landmarks[144] = [0.60, 0.41, -0.02]
    
    landmarks[362] = [0.42, 0.40, -0.02]  # Right Eye Inner
    landmarks[263] = [0.38, 0.40, -0.02]  # Right Eye Outer
    landmarks[385] = [0.40, 0.39, -0.02]
    landmarks[380] = [0.40, 0.41, -0.02]

    # Adjust eye vertical height for closed state
    if eye_state == "CLOSED":
        # Narrow verticals to EAR < 0.22
        landmarks[160] = [0.60, 0.399, -0.02]
        landmarks[144] = [0.60, 0.401, -0.02]
        landmarks[385] = [0.40, 0.399, -0.02]
        landmarks[380] = [0.40, 0.401, -0.02]

    # Nose Tip index 1
    z_nose = -0.02 if depth_flat else -0.08  # Real nose protrudes Z closer to camera (around -0.08)
    
    if pose_type == "CENTER":
        landmarks[1] = [0.50, 0.40, z_nose]
    elif pose_type == "LEFT":
        landmarks[1] = [0.47, 0.40, z_nose]
    elif pose_type == "RIGHT":
        landmarks[1] = [0.53, 0.40, z_nose]

    # Smile landmarks:
    # 61, 291: Left and right mouth corners
    # 13, 14: Upper and lower lip centers
    if smile_state == "SMILING":
        landmarks[61] = [0.57, 0.52, -0.03]
        landmarks[291] = [0.43, 0.52, -0.03]
        landmarks[13] = [0.50, 0.51, -0.03]
        landmarks[14] = [0.50, 0.55, -0.03]
    else:
        landmarks[61] = [0.55, 0.52, -0.03]
        landmarks[291] = [0.45, 0.52, -0.03]
        landmarks[13] = [0.50, 0.52, -0.03]
        landmarks[14] = [0.50, 0.54, -0.03]

    # Add minor Gaussian coordinate jitter to simulate live tracking tremours
    if jitter_scale > 0:
        landmarks += np.random.normal(0, jitter_scale, landmarks.shape)

    return landmarks


def run_happy_path_live_session():
    print("\n=======================================================")
    print("DEMO 1 & 2: Happy Path - Live Session (HIGH_SECURITY)")
    print("Challenges: BLINK -> SMILE -> TURN_LEFT -> TURN_RIGHT")
    print("=======================================================")
    sdk = LivenessSDK(preset="HIGH_SECURITY")
    bbox = (100, 80, 200, 240)
    dt = 0.033
    curr_time = 1000.0

    # Interaction sequence simulating organic responses to successive challenges:
    # 1. BLINK:
    #    - Frame 1-3: Open
    #    - Frame 4-6: Closed
    #    - Frame 7-9: Open -> Completes BLINK
    # 2. SMILE:
    #    - Frame 10-12: Neutral
    #    - Frame 13-16: Smiling -> Completes SMILE
    # 3. TURN_LEFT:
    #    - Frame 17-19: Center
    #    - Frame 20-24: Left
    #    - Frame 25-28: Center -> Completes TURN_LEFT
    # 4. TURN_RIGHT:
    #    - Frame 29-31: Center
    #    - Frame 32-36: Right
    #    - Frame 37-40: Center -> Completes TURN_RIGHT
    sequence = [
        # (pose, eye, smile)
        # BLINK stage
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),
        ("CENTER", "CLOSED", "NEUTRAL"), ("CENTER", "CLOSED", "NEUTRAL"), ("CENTER", "CLOSED", "NEUTRAL"),
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),
        
        # SMILE stage
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),
        ("CENTER", "OPEN", "SMILING"), ("CENTER", "OPEN", "SMILING"), ("CENTER", "OPEN", "SMILING"),
        ("CENTER", "OPEN", "SMILING"), ("CENTER", "OPEN", "SMILING"),
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),

        # TURN_LEFT stage
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),
        ("LEFT", "OPEN", "NEUTRAL"), ("LEFT", "OPEN", "NEUTRAL"), ("LEFT", "OPEN", "NEUTRAL"),
        ("LEFT", "OPEN", "NEUTRAL"), ("LEFT", "OPEN", "NEUTRAL"), ("LEFT", "OPEN", "NEUTRAL"),
        ("LEFT", "OPEN", "NEUTRAL"), ("LEFT", "OPEN", "NEUTRAL"), ("LEFT", "OPEN", "NEUTRAL"),
        ("LEFT", "OPEN", "NEUTRAL"),
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),

        # TURN_RIGHT stage
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),
        ("RIGHT", "OPEN", "NEUTRAL"), ("RIGHT", "OPEN", "NEUTRAL"), ("RIGHT", "OPEN", "NEUTRAL"),
        ("RIGHT", "OPEN", "NEUTRAL"), ("RIGHT", "OPEN", "NEUTRAL"), ("RIGHT", "OPEN", "NEUTRAL"),
        ("RIGHT", "OPEN", "NEUTRAL"), ("RIGHT", "OPEN", "NEUTRAL"), ("RIGHT", "OPEN", "NEUTRAL"),
        ("RIGHT", "OPEN", "NEUTRAL"),
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL"),
        ("CENTER", "OPEN", "NEUTRAL"), ("CENTER", "OPEN", "NEUTRAL")
    ]

    for idx, (pose, eye, smile) in enumerate(sequence):
        curr_time += dt
        frame = create_base_skin_frame(draw_moire=False)
        lms = create_synthetic_landmarks(pose_type=pose, eye_state=eye, smile_state=smile, depth_flat=False)

        res = sdk.process_frame(
            frame=frame,
            bbox=bbox,
            landmarks=lms,
            tracking_confidence=0.96,
            timestamp=curr_time
        )
        
        # Log challenge progression
        print(f"Frame {idx+1:02d} | Challenge: {str(res['current_challenge']):10s} | "
              f"Decision: {res['decision']:7s} | Active: {res['active_score']:.2f} | "
              f"Passive: {res['passive_score']:.2f} | Trust Score: {res['trust_score']:.3f}")

        if res["success"]:
            print(f"\n[SUCCESS] Liveness session verified successfully!")
            print(f"Session ID:         {res['session_id']}")
            print(f"Verified At:        {res['verified_at']:.3f}")
            print(f"Expires At:         {res['expires_at']:.3f}")
            print(f"Final Trust Score:  {res['trust_score']:.3f}")
            break
    else:
        print("\n[FAILED] Happy path failed to complete. Final Response:")
        print(res)


def run_spoof_rejection():
    print("\n=======================================================")
    print("DEMO 3: Passive Spoof Detection & Rejection")
    print("Scenario: Flat Screen Replay with scanlines grid (HIGH_SECURITY)")
    print("=======================================================")
    sdk = LivenessSDK(preset="HIGH_SECURITY")
    bbox = (100, 80, 200, 240)
    dt = 0.033
    curr_time = 2000.0

    # We feed 35 frames so that the rolling history window (size 15) is fully filled
    # and the spoof accumulator converges to trigger a hard SPOOF decision.
    sequence = [("CENTER", "OPEN")] * 35

    for idx, (pose, eye) in enumerate(sequence):
        curr_time += dt
        
        # Moire grid frame and flat depth coordinates
        frame = create_base_skin_frame(draw_moire=True, noise_std=0.0)
        lms = create_synthetic_landmarks(pose_type=pose, eye_state=eye, depth_flat=True)

        res = sdk.process_frame(
            frame=frame,
            bbox=bbox,
            landmarks=lms,
            tracking_confidence=0.95,
            timestamp=curr_time
        )

        print(f"Frame {idx+1:02d} | Decision: {res['decision']:7s} | Trust Score: {res['trust_score']:.2f} | Error: {res['error']}")

        if res["error"] == errors.PASSIVE_SPOOF_DETECTED:
            print("\n[SPOOF REJECTED OK] Presentation Attack correctly flagged!")
            print(f"Final Response Decision: {res['decision']} | Error Code: {res['error']}")
            break
    else:
        print("\n[FAILED] Presentation attack was not flagged.")


def run_timeout_rejection():
    print("\n=======================================================")
    print("DEMO 4: Active Challenge Timeout Rejection")
    print("=======================================================")
    # 1.0s timeout per challenge to speed up simulation
    custom_preset = {
        "challenges": ["BLINK"],
        "timeout_per_challenge": 1.0,
        "session_timeout": 5.0,
        "passive_history_size": 8,
        "passive_live_threshold": 0.70,
        "passive_spoof_threshold": 0.40,
        "min_face_size_target": 120.0,
        "critical_score_threshold": 15.0,
        "min_confidence": 0.50,
    }
    sdk = LivenessSDK(preset=custom_preset)
    bbox = (100, 80, 200, 240)
    dt = 0.200  # 200ms frame intervals
    curr_time = 3000.0

    # User remains centered with eyes open (never completes blink)
    # After 6 frames (1.2 seconds elapsed), timeout will trigger
    for idx in range(7):
        curr_time += dt
        frame = create_base_skin_frame(draw_moire=False)
        lms = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN", depth_flat=False)

        res = sdk.process_frame(
            frame=frame,
            bbox=bbox,
            landmarks=lms,
            tracking_confidence=0.96,
            timestamp=curr_time
        )
        print(f"Frame {idx+1:02d} | Time Elapsed: {dt*(idx+1):.2f}s | Success: {res['success']} | Error: {res['error']}")

        if res["error"] == errors.TIMEOUT:
            print("\n[TIMEOUT OK] SDK Session correctly rejected due to active challenge timeout!")
            break


def run_fqa_early_rejection():
    print("\n=======================================================")
    print("DEMO 5: Early Face Quality Recapture Rejection")
    print("=======================================================")
    sdk = LivenessSDK(preset="MEDIUM_SECURITY")
    
    # Face bbox width and height is 30px (critical size target is 150px)
    tiny_bbox = (100, 80, 30, 30)
    
    frame = create_base_skin_frame(draw_moire=False)
    lms = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN", depth_flat=False)

    res = sdk.process_frame(
        frame=frame,
        bbox=tiny_bbox,
        landmarks=lms,
        tracking_confidence=0.96,
        timestamp=4000.0
    )
    print("Response for small face bounding box:")
    for key, val in res.items():
        print(f"  - {key}: {val}")

    if res["error"] == errors.FACE_TOO_SMALL:
        print("\n[FQA REJECTED OK] SDK correctly rejected the low-quality input with FACE_TOO_SMALL!")


if __name__ == "__main__":
    print("=======================================================")
    print("       STARTING INTEGRATED LIVENESS SDK DEMONSTRATION  ")
    print("=======================================================")
    
    run_happy_path_live_session()
    run_spoof_rejection()
    run_timeout_rejection()
    run_fqa_early_rejection()
    
    print("\n=======================================================")
    print("       INTEGRATED LIVENESS SDK DEMONSTRATION COMPLETE  ")
    print("=======================================================")
