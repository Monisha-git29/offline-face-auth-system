"""
Unified Challenge Engine Demonstration and Verification.

Demonstrates:
1. Happy Path (EASY preset): Blink liveness challenge.
2. Happy Path (MEDIUM preset): Blink then Head Turn Left liveness challenge.
3. Individual Challenge Timeout: Fails if current challenge exceeds its limit.
4. Global Session Timeout: Fails if total session time exceeds session_timeout.
5. Replay / Stalled Frame Rejection: Detects frame freezes using average displacement.
6. Anti-Repetition Logic: Ensures randomized challenges generate distinct sequences.
"""

import os
import sys
import time
import numpy as np

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.challenge import ChallengeEngine


def create_synthetic_landmarks(pose_type: str = "CENTER", eye_state: str = "OPEN") -> np.ndarray:
    """
    Generates a 468x3 array of synthetic face landmarks.
    Adjusts eye and nose positions specifically to simulate blink and yaw head turn.
    """
    landmarks = np.ones((468, 3), dtype=np.float32) * 0.5  # neutral values
    
    # 1. Blink coordinates (scaled to width=720, height=1280)
    dy = 0.007 if eye_state == "OPEN" else 0.001
    
    # Left eye vertical: 160 is top, 144 is bottom. Midpoint x = 0.60
    landmarks[33] = [0.62, 0.40, 0.0]
    landmarks[133] = [0.58, 0.40, 0.0]
    landmarks[160] = [0.60, 0.40 - dy, 0.0]
    landmarks[144] = [0.60, 0.40 + dy, 0.0]
    landmarks[158] = [0.60, 0.40 - dy, 0.0]
    landmarks[145] = [0.60, 0.40 + dy, 0.0]

    # Right eye: corners 362 and 263. Midpoint x = 0.40
    landmarks[362] = [0.42, 0.40, 0.0]
    landmarks[263] = [0.38, 0.40, 0.0]
    landmarks[385] = [0.40, 0.40 - dy, 0.0]
    landmarks[380] = [0.40, 0.40 + dy, 0.0]
    landmarks[387] = [0.40, 0.40 - dy, 0.0]
    landmarks[373] = [0.40, 0.40 + dy, 0.0]

    # Nose tip (landmark 1) x coordinate for yaw pose_type
    if pose_type == "CENTER":
        landmarks[1] = [0.50, 0.40, 0.0]
    elif pose_type == "LEFT":
        # Shift nose left in image coordinate space (closer to image left = ER = smaller x)
        landmarks[1] = [0.4725, 0.40, 0.0]
    elif pose_type == "RIGHT":
        # Shift nose right in image coordinate space (closer to image right = EL = larger x)
        landmarks[1] = [0.5275, 0.40, 0.0]

    return landmarks


def run_happy_path_easy():
    print("\n==============================================")
    print("DEMO 1: Happy Path (EASY Preset: BLINK only)")
    print("==============================================")
    engine = ChallengeEngine(preset="EASY")
    width, height = 720, 1280
    dt = 0.033  # 30 FPS stream simulation (33ms interval)
    curr_time = 1000.0

    print(f"Session ID: {engine.session_id}")
    print(f"Challenges Queue: {engine.challenges}")

    # Sequence of frames:
    # Frame 1-3: Eyes open
    # Frame 4-6: Eyes closed (Blink triggered)
    # Frame 7-9: Eyes open (Blink completed)
    sequence = ["OPEN"] * 3 + ["CLOSED"] * 3 + ["OPEN"] * 3
    
    for idx, eye in enumerate(sequence):
        curr_time += dt
        lms = create_synthetic_landmarks(pose_type="CENTER", eye_state=eye)
        # Shift landmarks slightly so they are not 100% static (prevents FRAME_STALLED)
        lms = lms + np.random.normal(0, 1e-4, lms.shape)

        res = engine.update(
            landmarks=lms,
            width=width,
            height=height,
            tracking_confidence=0.95,
            timestamp=curr_time
        )
        print(f"Frame {idx+1:02d} | Eye: {eye:6s} | Status: {res['session_status']:11s} | Liveness OK: {str(res['success']):5s} | Active Challenge: {res['current_challenge']}")

    if engine.status == "SUCCESS":
        print("\n[SUCCESS] Liveness session verified successfully!")
        print(f"Verified At: {engine.verified_at}")
        print(f"Expires At:  {engine.expires_at}")
        print(f"Analytics:   {engine.challenge_history}")
    else:
        print(f"\n[FAILED] Session ended with status: {engine.status}, error: {engine.error_code}")


def run_happy_path_medium():
    print("\n==============================================")
    print("DEMO 2: Happy Path (MEDIUM Preset: BLINK, TURN_LEFT)")
    print("==============================================")
    engine = ChallengeEngine(preset="MEDIUM")
    width, height = 720, 1280
    dt = 0.033
    curr_time = 2000.0

    print(f"Session ID: {engine.session_id}")
    print(f"Challenges Queue: {engine.challenges}")

    # Sequence of events:
    # 1. BLINK challenge:
    #    - 3 frames Open
    #    - 3 frames Closed
    #    - 3 frames Open -> completes BLINK, shifts to TURN_LEFT
    # 2. TURN_LEFT challenge:
    #    - 3 frames Center (stabilize)
    #    - 6 frames Left (turn verified)
    #    - 5 frames Center (return-to-center verification completes challenge)
    sequence = [
        # (pose, eye)
        ("CENTER", "OPEN"), ("CENTER", "OPEN"), ("CENTER", "OPEN"),  # Open
        ("CENTER", "CLOSED"), ("CENTER", "CLOSED"), ("CENTER", "CLOSED"),  # Closed
        ("CENTER", "OPEN"), ("CENTER", "OPEN"), ("CENTER", "OPEN"),  # Open (completes BLINK)
        
        ("CENTER", "OPEN"), ("CENTER", "OPEN"), ("CENTER", "OPEN"),  # Look Center for turn
        ("LEFT", "OPEN"), ("LEFT", "OPEN"), ("LEFT", "OPEN"),        # Turn Left
        ("LEFT", "OPEN"), ("LEFT", "OPEN"), ("LEFT", "OPEN"),
        ("CENTER", "OPEN"), ("CENTER", "OPEN"), ("CENTER", "OPEN"),  # Return Center
        ("CENTER", "OPEN"), ("CENTER", "OPEN")
    ]

    for idx, (pose, eye) in enumerate(sequence):
        curr_time += dt
        lms = create_synthetic_landmarks(pose_type=pose, eye_state=eye)
        lms = lms + np.random.normal(0, 1e-4, lms.shape)  # Jitter to avoid static stall detection

        res = engine.update(
            landmarks=lms,
            width=width,
            height=height,
            tracking_confidence=0.92,
            timestamp=curr_time
        )
        print(f"Frame {idx+1:02d} | Pose: {pose:6s} | Eye: {eye:6s} | Status: {res['session_status']:11s} | Current Challenge: {res['current_challenge']}")

    if engine.status == "SUCCESS":
        print("\n[SUCCESS] Liveness session verified successfully!")
        print("Analytics:")
        for key, val in res["analytics"].items():
            print(f"  - {key}: {val}")
    else:
        print(f"\n[FAILED] Session ended with status: {engine.status}, error: {res['error']}")


def run_challenge_timeout():
    print("\n==============================================")
    print("DEMO 3: Challenge Timeout Scenario")
    print("==============================================")
    # Configure short challenge timeout of 0.2s (200ms)
    engine = ChallengeEngine(preset="EASY", timeout_per_challenge=0.2)
    width, height = 720, 1280
    dt = 0.050  # 50ms frames
    curr_time = 3000.0

    # Feed open eyes frames. Since blink is requested, it will never complete.
    # 5 frames at 50ms = 250ms elapsed (> 200ms timeout threshold)
    for idx in range(6):
        curr_time += dt
        lms = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN")
        lms = lms + np.random.normal(0, 1e-4, lms.shape)

        res = engine.update(
            landmarks=lms,
            width=width,
            height=height,
            tracking_confidence=0.92,
            timestamp=curr_time
        )
        print(f"Frame {idx+1:02d} | Time Elapsed: {(curr_time - 3000.0)*1000:4.0f}ms | Status: {res['session_status']:11s} | Error: {res['error']}")

        if res["session_status"] == "TIMEOUT":
            print("\n[TIMEOUT OK] Liveness session successfully timed out as expected!")
            break


def run_session_timeout():
    print("\n==============================================")
    print("DEMO 4: Global Session Timeout Scenario")
    print("==============================================")
    # Set timeout_per_challenge = 5.0, but global session_timeout = 0.5s (500ms)
    engine = ChallengeEngine(preset="MEDIUM", timeout_per_challenge=5.0, session_timeout=0.5)
    width, height = 720, 1280
    dt = 0.100  # 100ms frames
    curr_time = 4000.0

    # User stays centered, eyes open.
    # Total elapsed time will exceed 500ms by Frame 6.
    for idx in range(7):
        curr_time += dt
        lms = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN")
        lms = lms + np.random.normal(0, 1e-4, lms.shape)

        res = engine.update(
            landmarks=lms,
            width=width,
            height=height,
            tracking_confidence=0.95,
            timestamp=curr_time
        )
        print(f"Frame {idx+1:02d} | Session Time: {(curr_time - 4000.0)*1000:4.0f}ms | Status: {res['session_status']:11s} | Error: {res['error']}")

        if res["session_status"] == "TIMEOUT":
            print("\n[SESSION TIMEOUT OK] Session-wide timeout triggered correctly!")
            break


def run_stall_detection():
    print("\n==============================================")
    print("DEMO 5: Frame Stall & Replay Rejection")
    print("==============================================")
    # movement_threshold = 1e-5. We feed identical coordinates to verify freeze.
    engine = ChallengeEngine(preset="EASY", movement_threshold=1e-5, min_stall_frames=3)
    width, height = 720, 1280
    dt = 0.033
    curr_time = 5000.0

    # Create exact identical landmarks without jitter
    static_lms = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN")

    for idx in range(5):
        curr_time += dt
        res = engine.update(
            landmarks=static_lms,  # Exact same landmark object
            width=width,
            height=height,
            tracking_confidence=0.95,
            timestamp=curr_time
        )
        print(f"Frame {idx+1:02d} | Status: {res['session_status']:11s} | Error: {res['error']}")

        if res["error"] == "FRAME_STALLED":
            print("\n[STALL OK] Stalled/Frozen frame attack detected and session rejected!")
            break


def run_anti_repetition():
    print("\n==============================================")
    print("DEMO 6: Anti-Repetition Randomized Sequences")
    print("==============================================")
    engine = ChallengeEngine(preset="HIGH", randomize=True)
    
    sequences = []
    for run in range(5):
        engine.reset()
        sequences.append(list(engine.challenges))
        print(f"Run {run+1}: {engine.challenges}")

    # Check that not all generated sequences are identical
    unique_seqs = {tuple(s) for s in sequences}
    print(f"\nTotal runs: 5 | Unique sequences: {len(unique_seqs)}")
    if len(unique_seqs) > 1:
        print("[ANTI-REPETITION OK] Anti-repetition logic successfully generated varied sequences!")
    else:
        print("[WARNING] Only one sequence pattern generated. Check if this is expected for short lists.")


if __name__ == "__main__":
    print("=== STARTING UNIFIED CHALLENGE ENGINE DEMONSTRATION ===")
    run_happy_path_easy()
    run_happy_path_medium()
    run_challenge_timeout()
    run_session_timeout()
    run_stall_detection()
    run_anti_repetition()
    print("\n=== DEMONSTRATIONS COMPLETED ===")
