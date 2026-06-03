"""
Unified Passive Anti-Spoofing and Liveness Fusion Decision Simulator.

Demonstrates:
1. Scenario 1: Live User (micro-jitters, natural skin texture, 3D facial depth curvature).
2. Scenario 2: Static Photograph (rigid bounding box shifts, zero landmark micro-movements).
3. Scenario 3: Screen Replay (flat depth ratio, high-frequency moire noise grid pattern).
4. Scenario 4: Frozen Frame Attack (identical landmarks, identical timestamps).
"""

import os
import sys
import numpy as np
import cv2

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.decision import LivenessDecisionEngine


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
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. If moire requested, overlay dark screen scanlines grid
    if draw_moire:
        for x in range(0, width, 3):
            cv2.line(canvas, (x, 0), (x, height), (70, 70, 70), 1)
        for y in range(0, height, 3):
            cv2.line(canvas, (0, y), (width, y), (70, 70, 70), 1)

    return canvas


def create_synthetic_landmarks(pose_type: str = "CENTER", eye_state: str = "OPEN", depth_flat: bool = False, jitter_scale: float = 1e-4) -> np.ndarray:
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

    # Add minor Gaussian coordinate jitter to simulate live tracking tremours
    if jitter_scale > 0:
        landmarks += np.random.normal(0, jitter_scale, landmarks.shape)

    return landmarks


def simulate_live_user():
    print("\n=======================================================")
    print("SCENARIO 1: Live User (Healthy active challenges + 3D passive liveness)")
    print("=======================================================")
    engine = LivenessDecisionEngine(preset="MEDIUM", passive_history_size=10)
    bbox = (100, 80, 200, 240)  # Face crop ROI bounds
    dt = 0.033
    curr_time = 1000.0

    # User performs:
    # 1. BLINK challenge:
    #    - Frame 1-3: Open eyes
    #    - Frame 4-6: Closed eyes
    #    - Frame 7-9: Open eyes -> completes BLINK challenge
    # 2. TURN_LEFT challenge:
    #    - Frame 10-12: Look Center
    #    - Frame 13-17: Turn Left (EMA satisfies turn lock)
    #    - Frame 18-22: Return Center (completes active turn sequence)
    sequence = [
        # (pose, eye)
        ("CENTER", "OPEN"), ("CENTER", "OPEN"), ("CENTER", "OPEN"),
        ("CENTER", "CLOSED"), ("CENTER", "CLOSED"), ("CENTER", "CLOSED"),
        ("CENTER", "OPEN"), ("CENTER", "OPEN"), ("CENTER", "OPEN"),
        
        ("CENTER", "OPEN"), ("CENTER", "OPEN"), ("CENTER", "OPEN"),
        ("LEFT", "OPEN"), ("LEFT", "OPEN"), ("LEFT", "OPEN"),
        ("LEFT", "OPEN"), ("LEFT", "OPEN"), ("LEFT", "OPEN"),
        ("LEFT", "OPEN"), ("LEFT", "OPEN"),
        ("CENTER", "OPEN"), ("CENTER", "OPEN"), ("CENTER", "OPEN"),
        ("CENTER", "OPEN"), ("CENTER", "OPEN"), ("CENTER", "OPEN"),
        ("CENTER", "OPEN"), ("CENTER", "OPEN")
    ]

    print(f"Active challenges: {engine.challenge_engine.challenges}")
    print("Feeding organic skin frame crops + 3D depth landmarks...")

    for idx, (pose, eye) in enumerate(sequence):
        curr_time += dt
        
        # Generate organic BGR skin frames and 3D depth coordinate meshes
        frame = create_base_skin_frame(draw_moire=False)
        lms = create_synthetic_landmarks(pose_type=pose, eye_state=eye, depth_flat=False)

        res = engine.update(
            frame=frame,
            bbox=bbox,
            landmarks=lms,
            tracking_confidence=0.96,
            timestamp=curr_time
        )
        print(f"Frame {idx+1:02d} | Pose: {pose:6s} | Liveness Decision: {res['decision']:7s} | Fused Score: {res['final_liveness_score']:.2f} | Active: {res['active_score']:.2f} | Passive: {res['passive_score']:.2f}")

    print(f"\n>> Final Session Verdict: {res['decision']}")
    print(f"   Success Status:  {res['success']}")
    print(f"   Verification:    Verified At: {res['verified_at']} | Expires At: {res['expires_at']}")
    print("   Sub-signals analytics:")
    for sig, score in res["analytics"]["passive_signals"].items():
        print(f"     - {sig}: {score:.4f}")
    print(f"   Average Tracking confidence: {res['analytics']['average_tracking_confidence']:.2f}")


def simulate_static_photograph():
    print("\n=======================================================")
    print("SCENARIO 2: Static Printed Photograph Attack")
    print("=======================================================")
    # Evaluates print attack where landmarks move 100% rigidly with no micro-variance
    engine = LivenessDecisionEngine(preset="EASY", passive_history_size=10)
    bbox = (100, 80, 200, 240)
    dt = 0.033
    curr_time = 2000.0

    # User presents a printed cut-out paper photo
    # Bbox shifts rigidly across frames due to hand tremor, but landmarks have ZERO organic micro-jitters
    frame = create_base_skin_frame(draw_moire=False)
    lms_static = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN", jitter_scale=0.0)

    print("Feeding perfectly static landmark coordinates (zero micro-deformations)...")
    for idx in range(12):
        curr_time += dt
        
        # Bounding box shifts rigidly by 1 pixel (simulating slight hand translations of the page)
        offset = idx % 2
        rigid_bbox = (100 + offset, 80 + offset, 200, 240)
        
        res = engine.update(
            frame=frame,
            bbox=rigid_bbox,
            landmarks=lms_static, # Exact static landmarks
            tracking_confidence=0.94,
            timestamp=curr_time
        )
        print(f"Frame {idx+1:02d} | Decision: {res['decision']:7s} | Fused: {res['final_liveness_score']:.2f} | Passive: {res['passive_score']:.2f} | Motion Score: {res['analytics']['passive_signals']['motion_score']:.2f}")

        if res["decision"] == "SPOOF":
            print("\n[REJECTED OK] Static printed photograph bypass successfully blocked via zero-motion micro-check!")
            break


def simulate_screen_replay():
    print("\n=======================================================")
    print("SCENARIO 3: HD Mobile Screen Replay Attack")
    print("=======================================================")
    # Simulates an HD video playback on a screen, which exhibits flat depth ratios and grid moire lines
    engine = LivenessDecisionEngine(preset="EASY", passive_history_size=10)
    bbox = (100, 80, 200, 240)
    dt = 0.033
    curr_time = 3000.0

    print("Feeding flat depth ratio (0.01) + screen grid moire textures...")
    
    # We feed frames overlaid with dark scanline patterns
    frame_moire = create_base_skin_frame(draw_moire=True)

    for idx in range(12):
        curr_time += dt
        
        # Flat coordinates (Z is flat, nose protrusion is zero)
        lms_flat = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN", depth_flat=True)

        res = engine.update(
            frame=frame_moire,
            bbox=bbox,
            landmarks=lms_flat,
            tracking_confidence=0.95,
            timestamp=curr_time
        )
        print(f"Frame {idx+1:02d} | Decision: {res['decision']:7s} | Fused Score: {res['final_liveness_score']:.2f} | Texture Score: {res['analytics']['passive_signals']['texture_score']:.2f} | Depth Score: {res['analytics']['passive_signals']['depth_score']:.2f}")

        if res["decision"] == "SPOOF":
            print("\n[REJECTED OK] Screen playback attack successfully blocked via combined texture/depth checks!")
            print(f"   Calculated skin scale-depth ratio: {res['analytics']['scale_depth_ratio']:.4f} (Target ~0.30)")
            print(f"   Calculated LBP texture entropy:    {res['analytics']['lbp_entropy']:.4f} (Target ~6.0)")
            break


def simulate_frozen_frame():
    print("\n=======================================================")
    print("SCENARIO 4: Frozen Frame / Stalled Stream Replay")
    print("=======================================================")
    # Frozen frame attack where timestamps don't increase or coordinates are 100% identical
    engine = LivenessDecisionEngine(preset="EASY", passive_history_size=10)
    bbox = (100, 80, 200, 240)
    
    # Static inputs
    frame = create_base_skin_frame(draw_moire=False)
    lms = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN", jitter_scale=0.0)
    
    print("Feeding identical frame and landmark coordinates across consecutive time cycles...")
    curr_time = 4000.0
    dt = 0.033
    
    for idx in range(5):
        curr_time += dt
        res = engine.update(
            frame=frame,
            bbox=bbox,
            landmarks=lms,
            tracking_confidence=0.95,
            timestamp=curr_time
        )
        print(f"Frame {idx+1:02d} | Decision: {res['decision']:7s} | Success: {res['success']} | Error Code: {res['error']}")
        
        if res["error"] == "FRAME_STALLED":
            print("\n[REJECTED OK] Frozen frame attack successfully blocked!")
            break


if __name__ == "__main__":
    print("=== STARTING ACTIVE & PASSIVE LIVENESS FUSION DECISION TESTS ===")
    simulate_live_user()
    simulate_static_photograph()
    simulate_screen_replay()
    simulate_frozen_frame()
    print("\n=== TESTS COMPLETED ===")
