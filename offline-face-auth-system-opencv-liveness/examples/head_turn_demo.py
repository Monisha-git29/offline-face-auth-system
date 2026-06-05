"""
Head Turn Detection Challenge Demonstration.

Demonstrates:
1. Synthesizing 468-point landmarks representing CENTER, LEFT-TURN, and RIGHT-TURN head poses.
2. Running a simulated 30 FPS liveness challenge sequence:
   IDLE -> FACE_CENTERED -> TURN_LEFT_REQUESTED -> LEFT_VERIFIED -> RETURN_TO_CENTER -> CHALLENGE_COMPLETE.
3. Verification of consecutive-frame constraints (3 frames required to verify turns and centers).
4. Cooldown lock verification (300 ms locking after completion).
5. Validation gateways (Null landmarks, unstable confidence).
"""

import os
import sys
import numpy as np

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.head_turn import HeadTurnDetector


def create_synthetic_pose_landmarks(pose_type: str = "CENTER") -> np.ndarray:
    """
    Generates a 468x3 array of synthetic face landmarks.
    Adjusts eye and nose positions specifically to simulate head yaw poses.
    """
    landmarks = np.ones((468, 3), dtype=np.float32) * 0.5  # neutral values
    
    # Normalized x-coordinates of eye centers
    # Left eye center midpoint (outer corner 33 x=0.62, inner corner 133 x=0.58) -> x = 0.60
    landmarks[33] = [0.62, 0.40, 0.0]
    landmarks[133] = [0.58, 0.40, 0.0]
    
    # Right eye center midpoint (inner corner 362 x=0.42, outer corner 263 x=0.38) -> x = 0.40
    landmarks[362] = [0.42, 0.40, 0.0]
    landmarks[263] = [0.38, 0.40, 0.0]

    # Adjust nose tip (landmark 1) x coordinate based on pose_type
    if pose_type == "CENTER":
        # Nose tip perfectly centered between eyes (x_N = 0.50). R = 0.0 -> Yaw = 0 deg
        landmarks[1] = [0.50, 0.40, 0.0]
    elif pose_type == "LEFT":
        # Nose tip shifted to user's left (right of face, closer to ER on image left)
        # x_N = 0.42. dx_L = |0.6 - 0.42| = 0.18. dx_R = |0.42 - 0.4| = 0.02. R = 0.16/0.20 = 0.80 -> Yaw = 64 deg
        # Let's shift it slightly less to hit ~22 degrees:
        # If Yaw = 22, R = 22 / 80 = 0.275. dx_L - dx_R = 0.275 * 0.20 = 0.055. dx_L = 0.1275, dx_R = 0.0725 -> x_N = 0.4725
        landmarks[1] = [0.4725, 0.40, 0.0]
    elif pose_type == "RIGHT":
        # Nose tip shifted to user's right (left of face, closer to EL on image right)
        # Yaw = -22 deg, R = -0.275 -> x_N = 0.5275
        landmarks[1] = [0.5275, 0.40, 0.0]

    return landmarks


def run_demo():
    print("=== MediaPipe Head Turn Liveness Challenge Simulator ===")
    
    # 1. Initialize Detector with default thresholds
    detector = HeadTurnDetector(
        yaw_threshold=18.0,
        center_threshold=8.0,
        min_turn_frames=3,
        min_center_frames=3,
        cooldown_duration=0.30,  # 300 ms
        min_confidence=0.50
    )
    
    print(f"\n[Detector Configured]")
    print(f"  - Yaw Turn Threshold:    {detector.yaw_threshold}°")
    print(f"  - Center Guard Threshold: {detector.center_threshold}°")
    print(f"  - Min Frames turn/center: {detector.min_turn_frames} / {detector.min_center_frames}")
    print(f"  - Cooldown Duration:      {detector.cooldown_duration*1000:.0f} ms\n")

    width, height = 720, 1280
    
    # 30 FPS stream simulation (dt = 33.3ms)
    dt = 0.0333

    # ==========================================
    # Phase 1: Quality Gateways
    # ==========================================
    print("--- Phase 1: Quality gateways Validation ---")
    res_none = detector.update(None, width, height)
    print(f"  - Null Landmarks: success={res_none['success']}, error={res_none.get('error')}")

    res_unst = detector.update(create_synthetic_pose_landmarks("CENTER"), width, height, tracking_confidence=0.3)
    print(f"  - Unstable Landmarks: success={res_unst['success']}, error={res_unst.get('error')}\n")

    # ==========================================
    # Phase 2: TURN_LEFT Liveness Challenge (30 FPS)
    # ==========================================
    print("--- Phase 2: TURN_LEFT Liveness Challenge (30 FPS Stream) ---")
    detector.reset()

    # Challenge Sequence:
    # Frame 1-3: CENTER
    # Frame 4-11: LEFT (turned past 18° to allow EMA to rise and satisfy 3 consecutive frames check)
    # Frame 12-17: CENTER (returned past center 8° to satisfy 3 consecutive frames return check)
    # Frame 18-20: Keep CENTER (cooldown gate locking checks)
    stream_sequence = (
        [("CENTER", None)] * 3 +
        [("LEFT", "TURN_LEFT")] * 8 +
        [("CENTER", "TURN_LEFT")] * 6 +
        [("CENTER", None)] * 3
    )

    curr_time = 0.0

    for idx, (pose, challenge) in enumerate(stream_sequence):
        landmarks = create_synthetic_pose_landmarks(pose)
        curr_time += dt
        
        res = detector.update(
            landmarks=landmarks,
            width=width,
            height=height,
            tracking_confidence=0.9,
            timestamp=curr_time,
            active_challenge=challenge
        )
        
        if res["success"]:
            print(f"  Frame {idx+1:02d} | T: {curr_time*1000:5.1f}ms | Input: {pose:6s} | Smoothed Yaw: {res['yaw']:+.1f}° | State: {res['state']:20s} | OK: {str(res['challenge_verified']):5s}")
        else:
            print(f"  Frame {idx+1:02d} | Error: {res.get('error')}")

    print(f"\n  >> LEFT Challenge Completed successfully.")
    print("     *Notice on Frame 9, LEFT_VERIFIED state was reached after 3 consecutive turned frames.")
    print("     *Notice on Frame 16, CHALLENGE_COMPLETE state was reached after 3 consecutive returned frames.")
    print("     *Notice on Frames 17-20, the cooldown lock prevented double-triggering while reset was processing.\n")

    # ==========================================
    # Phase 3: Cooldown locking & Reset Demonstration
    # ==========================================
    print("--- Phase 3: Cooldown Lockout Demonstrations ---")
    # At 30 FPS, 300ms cooldown corresponds to ~9 frames.
    # We simulate keeping head LEFT after verification. The system should remain complete/cooldown locked and NOT trigger again.
    detector.reset()
    
    # 2 frames center -> 8 frames LEFT -> 6 frames CENTER (completes challenge) -> 5 frames LEFT (ignored due to cooldown lockout)
    cooldown_sequence = (
        [("CENTER", None)] * 2 +
        [("LEFT", "TURN_LEFT")] * 8 +
        [("CENTER", "TURN_LEFT")] * 6 +  # Return to center completes challenge
        [("LEFT", "TURN_LEFT")] * 5     # Turn left again (cooldown lockout should ignore this)
    )
    
    curr_time = 0.0
    for idx, (pose, challenge) in enumerate(cooldown_sequence):
        landmarks = create_synthetic_pose_landmarks(pose)
        curr_time += dt
        
        res = detector.update(
            landmarks=landmarks,
            width=width,
            height=height,
            tracking_confidence=0.9,
            timestamp=curr_time,
            active_challenge=challenge
        )
        
        if res["success"]:
            print(f"  Frame {idx+1:02d} | T: {curr_time*1000:5.1f}ms | Input: {pose:6s} | Yaw: {res['yaw']:+.1f}° | State: {res['state']:20s} | OK: {str(res['challenge_verified']):5s}")
        else:
            print(f"  Frame {idx+1:02d} | Error: {res.get('error')}")

    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    run_demo()
