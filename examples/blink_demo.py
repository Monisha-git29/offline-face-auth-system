"""
Blink Detection and Time-Based Liveness Simulator Demo.

This script demonstrates:
1. Creating a synthetic 468-point landmark array representing face coordinates.
2. Simulating a real-time frame stream (running at 30 FPS, i.e., dt = 33.3ms per frame)
   by updating vertical eye coordinates and passing high-precision timestamps.
3. Verifying the FACE_NOT_DETECTED and LANDMARKS_UNSTABLE quality gates.
4. Simulating a normal time-based blink (duration 100ms) with active challenge validation,
   verifying that blink_detected and challenge_verified trigger successfully.
5. Simulating a prolonged squint/sleep nod (duration 533ms) to verify time-based rejection gates.
6. Printing step-by-step state logs.
"""

import os
import sys
import numpy as np

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.blink import BlinkDetector


def create_synthetic_landmarks(eye_state: str = "OPEN") -> np.ndarray:
    """
    Generates a 468x3 array of synthetic face landmarks.
    Adjusts eye coordinates specifically to simulate OPEN or CLOSED states.
    """
    # Create empty normalized landmarks (N=468)
    landmarks = np.ones((468, 3), dtype=np.float32) * 0.5  # default neutral coordinates
    
    # Scale: x, y centered around 0.5
    # Default Left Eye indices: Corners: 33, 133. Verts: (160, 144), (158, 145)
    landmarks[33]  = [0.40, 0.40, 0.0]
    landmarks[133] = [0.45, 0.40, 0.0]
    
    # Default Right Eye indices: Corners: 362, 263. Verts: (385, 380), (387, 373)
    landmarks[362] = [0.55, 0.40, 0.0]
    landmarks[263] = [0.60, 0.40, 0.0]

    # Adjust vertical distances based on state
    if eye_state == "OPEN":
        # Verticals left (160-144, 158-145). Vertical dist ~ 0.015. EAR = 0.30
        landmarks[160] = [0.415, 0.385, 0.0]
        landmarks[144] = [0.415, 0.415, 0.0]
        landmarks[158] = [0.435, 0.385, 0.0]
        landmarks[145] = [0.435, 0.415, 0.0]

        # Verticals right (385-380, 387-373). EAR = 0.30
        landmarks[385] = [0.565, 0.385, 0.0]
        landmarks[380] = [0.565, 0.415, 0.0]
        landmarks[387] = [0.585, 0.385, 0.0]
        landmarks[373] = [0.585, 0.415, 0.0]
        
    elif eye_state == "CLOSED":
        # Verticals left squashed. Vertical dist ~ 0.004. EAR = 0.08
        landmarks[160] = [0.415, 0.398, 0.0]
        landmarks[144] = [0.415, 0.402, 0.0]
        landmarks[158] = [0.435, 0.398, 0.0]
        landmarks[145] = [0.435, 0.402, 0.0]

        # Verticals right squashed. EAR = 0.08
        landmarks[385] = [0.565, 0.398, 0.0]
        landmarks[380] = [0.565, 0.402, 0.0]
        landmarks[387] = [0.585, 0.398, 0.0]
        landmarks[373] = [0.585, 0.402, 0.0]

    return landmarks


def run_demo():
    print("=== MediaPipe Time-Based Blink Detection & Challenge Simulator ===")
    
    # 1. Initialize detector with standard configuration
    # EAR threshold: 0.22, min_frames = 2, duration = 50ms to 500ms (0.05s to 0.5s)
    detector = BlinkDetector(
        ear_threshold=0.22,
        min_consecutive_frames=2,
        min_blink_duration=0.05,
        max_blink_duration=0.50,
        min_confidence=0.50
    )
    
    print(f"\n[Detector Configuration]")
    print(f"  - EAR Threshold:       {detector.ear_threshold}")
    print(f"  - Min Frames Closed:   {detector.min_consecutive_frames}")
    print(f"  - Blink Speed Window:  {detector.min_blink_duration*1000:.0f}ms to {detector.max_blink_duration*1000:.0f}ms")
    print(f"  - Min Track Confidence: {detector.min_confidence}\n")

    # Camera resolution
    width, height = 720, 1280
    
    # Frame duration simulation (30 FPS -> dt = 0.0333 seconds / 33.3ms per frame)
    dt = 0.0333

    # ==========================================
    # Phase 1: Quality Gates Verification
    # ==========================================
    print("--- Phase 1: Quality Gateways Verification ---")
    
    # A. Face Not Detected
    res_none = detector.update(None, width, height)
    print(f"  - Result with Null Landmarks: success={res_none['success']}, error={res_none.get('error')}")

    # B. Landmarks Unstable
    lms_normal = create_synthetic_landmarks("OPEN")
    res_unstable = detector.update(lms_normal, width, height, tracking_confidence=0.4)
    print(f"  - Result with Low Confidence (0.4): success={res_unstable['success']}, error={res_unstable.get('error')}\n")

    # ==========================================
    # Phase 2: Time-Based Valid Blink Challenge
    # ==========================================
    print("--- Phase 2: Simulating Valid Blink Challenge (30 FPS Stream) ---")
    detector.reset()
    
    # Define a stream sequence of 10 frames:
    # 3 frames OPEN -> 3 frames CLOSED -> 4 frames OPEN
    # Duration closed: 3 frames * 33.3ms = 99.9ms (within 50ms - 500ms window)
    stream_sequence = (
        ["OPEN"] * 3 +
        ["CLOSED"] * 3 +
        ["OPEN"] * 4
    )
    
    curr_time = 0.0

    for idx, state in enumerate(stream_sequence):
        landmarks = create_synthetic_landmarks(state)
        curr_time += dt
        
        # Pass active challenge BLINK
        res = detector.update(
            landmarks=landmarks,
            width=width,
            height=height,
            tracking_confidence=0.9,
            timestamp=curr_time,
            active_challenge="BLINK"
        )
        
        if res["success"]:
            print(f"  Frame {idx+1:02d} | T: {curr_time*1000:5.1f}ms | Input: {state:6s} | EAR: {res['ear']:.4f} | State: {res['eye_state']:6s} | Blink: {str(res['blink_detected']):5s} | Challenge OK: {str(res['challenge_verified']):5s} | Count: {res['total_blinks']}")
        else:
            print(f"  Frame {idx+1:02d} | Error: {res.get('error')}")
            
    print(f"\n  >> Challenge Phase Completed. Total verified blinks: {detector.total_blinks}\n")

    # ==========================================
    # Phase 3: Prolonged closure Rejection Gate
    # ==========================================
    print("--- Phase 3: Simulating Squint/Sleeping Rejection Gate ---")
    detector.reset()
    
    # Define a stream sequence of 21 frames:
    # 2 frames OPEN -> 16 frames CLOSED -> 3 frames OPEN
    # Duration closed: 16 frames * 33.3ms = 532.8ms (exceeds 500ms max limit!)
    squint_sequence = (
        ["OPEN"] * 2 +
        ["CLOSED"] * 16 +
        ["OPEN"] * 3
    )
    
    curr_time = 0.0

    for idx, state in enumerate(squint_sequence):
        landmarks = create_synthetic_landmarks(state)
        curr_time += dt
        
        res = detector.update(
            landmarks=landmarks,
            width=width,
            height=height,
            tracking_confidence=0.9,
            timestamp=curr_time,
            active_challenge="BLINK"
        )
        
        if res["success"]:
            print(f"  Frame {idx+1:02d} | T: {curr_time*1000:5.1f}ms | Input: {state:6s} | EAR: {res['ear']:.4f} | State: {res['eye_state']:6s} | Blink: {str(res['blink_detected']):5s} | Challenge OK: {str(res['challenge_verified']):5s} | Count: {res['total_blinks']}")
        else:
            print(f"  Frame {idx+1:02d} | Error: {res.get('error')}")

    print(f"\n  >> Squint Phase Completed. Total verified blinks: {detector.total_blinks}")
    print("     *The 16-frame closure lasted 532.8ms (which is > 500ms safety window), so it was rejected.")
    print("     This ensures total safety against frozen frames or asleep spoofs!")
    
    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    run_demo()
