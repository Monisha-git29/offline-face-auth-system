"""
Real-World Evaluation and Metrics Validation System.

Performs liveness pipeline testing (FQA -> LivenessDecisionEngine)
on webcam input (if camera and mediapipe are available) or falls back
to a simulated multi-scenario test suite (LIVE, PHOTO, REPLAY, STALL).
Computes FAR, FRR, SDR, latency, and writes results to CSV.
"""

import os
import sys
import time
import argparse
import uuid
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.fqa import FaceQualityEngine
from opencv_module.decision import LivenessDecisionEngine
from evaluation.metrics import MetricsCollector
from examples.antispoof_demo import create_base_skin_frame, create_synthetic_landmarks


def run_simulation_suite(metrics: MetricsCollector) -> None:
    """
    Executes a complete simulated validation suite covering 5 different liveness runs:
    - 2 Live runs (EASY and MEDIUM presets)
    - 3 Spoof runs (Static photo, screen replay, frozen frame)
    """
    print("\n--- Starting Headless Simulation Evaluation Suite ---")
    
    # Initialize components
    fqa_engine = FaceQualityEngine(target_size=(112, 112))
    bbox = (100, 80, 200, 240)
    left_eye_fqa = (165, 175)
    right_eye_fqa = (235, 175)
    dt = 0.033

    # ==========================================
    # Scenario 1: True LIVE Session (EASY preset)
    # ==========================================
    print("\nRunning Scenario 1: True LIVE (BLINK challenge)")
    engine = LivenessDecisionEngine(preset="EASY", passive_history_size=10)
    session_id = engine.session_id
    curr_time = 1000.0
    sequence_1 = ["OPEN"] * 3 + ["CLOSED"] * 3 + ["OPEN"] * 5
    
    t_start = time.perf_counter()
    fqa_res = {}
    
    for idx, eye in enumerate(sequence_1):
        curr_time += dt
        frame = create_base_skin_frame(draw_moire=False, noise_std=2.0)
        lms = create_synthetic_landmarks(pose_type="CENTER", eye_state=eye, depth_flat=False)

        # FQA
        fqa_res = fqa_engine.assess_raw_face(frame, left_eye_fqa, right_eye_fqa, bbox)
        # Liveness
        res = engine.update(frame, bbox, lms, tracking_confidence=0.96, timestamp=curr_time)
        
    duration_ms = (time.perf_counter() - t_start) * 1000.0
    metrics.add_session(
        session_id=session_id,
        true_label="LIVE",
        predicted_decision=res["decision"],
        success=res["success"],
        active_score=res["active_score"],
        passive_score=res["passive_score"],
        final_score=res["final_liveness_score"],
        duration_ms=duration_ms,
        avg_fps=1.0 / dt,
        avg_confidence=0.96,
        error=res["error"],
        fqa_metrics=fqa_res
    )
    print(f"  Result: Decision={res['decision']} | Success={res['success']} | Error={res['error']}")

    # ==========================================
    # Scenario 2: True LIVE Session (MEDIUM preset)
    # ==========================================
    print("\nRunning Scenario 2: True LIVE (BLINK & TURN_LEFT challenge)")
    engine = LivenessDecisionEngine(preset="MEDIUM", passive_history_size=10)
    session_id = engine.session_id
    curr_time = 2000.0
    
    # Sequence of 28 frames satisfying EAR and EMA Yaw turns
    sequence_2 = [
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
    
    t_start = time.perf_counter()
    for idx, (pose, eye) in enumerate(sequence_2):
        curr_time += dt
        frame = create_base_skin_frame(draw_moire=False, noise_std=2.0)
        lms = create_synthetic_landmarks(pose_type=pose, eye_state=eye, depth_flat=False)

        fqa_res = fqa_engine.assess_raw_face(frame, left_eye_fqa, right_eye_fqa, bbox)
        res = engine.update(frame, bbox, lms, tracking_confidence=0.95, timestamp=curr_time)

    duration_ms = (time.perf_counter() - t_start) * 1000.0
    metrics.add_session(
        session_id=session_id,
        true_label="LIVE",
        predicted_decision=res["decision"],
        success=res["success"],
        active_score=res["active_score"],
        passive_score=res["passive_score"],
        final_score=res["final_liveness_score"],
        duration_ms=duration_ms,
        avg_fps=1.0 / dt,
        avg_confidence=0.95,
        error=res["error"],
        fqa_metrics=fqa_res
    )
    print(f"  Result: Decision={res['decision']} | Success={res['success']} | Error={res['error']}")

    # ==========================================
    # Scenario 3: True SPOOF Session (Static photo)
    # ==========================================
    print("\nRunning Scenario 3: True SPOOF (Static printed photo)")
    engine = LivenessDecisionEngine(preset="EASY", passive_history_size=10)
    session_id = engine.session_id
    curr_time = 3000.0
    frame = create_base_skin_frame(draw_moire=False, noise_std=2.0)
    lms_static = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN", jitter_scale=0.0)

    t_start = time.perf_counter()
    for idx in range(12):
        curr_time += dt
        offset = idx % 2
        rigid_bbox = (100 + offset, 80 + offset, 200, 240)
        
        fqa_res = fqa_engine.assess_raw_face(frame, left_eye_fqa, right_eye_fqa, rigid_bbox)
        res = engine.update(frame, rigid_bbox, lms_static, tracking_confidence=0.95, timestamp=curr_time)
        if res["decision"] == "SPOOF":
            break

    duration_ms = (time.perf_counter() - t_start) * 1000.0
    metrics.add_session(
        session_id=session_id,
        true_label="SPOOF",
        predicted_decision=res["decision"],
        success=res["success"],
        active_score=res["active_score"],
        passive_score=res["passive_score"],
        final_score=res["final_liveness_score"],
        duration_ms=duration_ms,
        avg_fps=1.0 / dt,
        avg_confidence=0.95,
        error=res["error"],
        fqa_metrics=fqa_res
    )
    print(f"  Result: Decision={res['decision']} | Success={res['success']} | Error={res['error']}")

    # ==========================================
    # Scenario 4: True SPOOF Session (Screen replay)
    # ==========================================
    print("\nRunning Scenario 4: True SPOOF (Mobile screen replay)")
    engine = LivenessDecisionEngine(preset="EASY", passive_history_size=10)
    session_id = engine.session_id
    curr_time = 4000.0
    frame_moire = create_base_skin_frame(draw_moire=True)

    t_start = time.perf_counter()
    for idx in range(12):
        curr_time += dt
        lms_flat = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN", depth_flat=True)

        fqa_res = fqa_engine.assess_raw_face(frame_moire, left_eye_fqa, right_eye_fqa, bbox)
        res = engine.update(frame_moire, bbox, lms_flat, tracking_confidence=0.95, timestamp=curr_time)
        if res["decision"] == "SPOOF":
            break

    duration_ms = (time.perf_counter() - t_start) * 1000.0
    metrics.add_session(
        session_id=session_id,
        true_label="SPOOF",
        predicted_decision=res["decision"],
        success=res["success"],
        active_score=res["active_score"],
        passive_score=res["passive_score"],
        final_score=res["final_liveness_score"],
        duration_ms=duration_ms,
        avg_fps=1.0 / dt,
        avg_confidence=0.95,
        error=res["error"],
        fqa_metrics=fqa_res
    )
    print(f"  Result: Decision={res['decision']} | Success={res['success']} | Error={res['error']}")

    # ==========================================
    # Scenario 5: True SPOOF Session (Frozen frame)
    # ==========================================
    print("\nRunning Scenario 5: True SPOOF (Frozen frame/stream stall)")
    engine = LivenessDecisionEngine(preset="EASY", passive_history_size=10)
    session_id = engine.session_id
    curr_time = 5000.0
    frame = create_base_skin_frame(draw_moire=False, noise_std=2.0)
    lms = create_synthetic_landmarks(pose_type="CENTER", eye_state="OPEN", jitter_scale=0.0)

    t_start = time.perf_counter()
    for idx in range(5):
        curr_time += dt
        fqa_res = fqa_engine.assess_raw_face(frame, left_eye_fqa, right_eye_fqa, bbox)
        res = engine.update(frame, bbox, lms, tracking_confidence=0.95, timestamp=curr_time)
        if res["error"] == "FRAME_STALLED":
            break

    duration_ms = (time.perf_counter() - t_start) * 1000.0
    metrics.add_session(
        session_id=session_id,
        true_label="SPOOF",
        predicted_decision=res["decision"],
        success=res["success"],
        active_score=res["active_score"],
        passive_score=res["passive_score"],
        final_score=res["final_liveness_score"],
        duration_ms=duration_ms,
        avg_fps=1.0 / dt,
        avg_confidence=0.95,
        error=res["error"],
        fqa_metrics=fqa_res
    )
    print(f"  Result: Decision={res['decision']} | Success={res['success']} | Error={res['error']}")


def run_webcam_evaluation(metrics: MetricsCollector) -> None:
    """
    Attempts to read live webcam frames, extracts MediaPipe Face Mesh landmarks,
    runs FQA + Liveness Decision Engine, and registers session details.
    """
    print("\n--- Initializing Webcam Real-World Evaluation ---")
    
    # 1. Attempt imports
    try:
        import mediapipe as mp
    except ImportError:
        print("\n[WARNING] MediaPipe Python package is not installed.")
        print("          Headless simulation will be run instead. Use 'pip install mediapipe'.")
        run_simulation_suite(metrics)
        return

    # 2. Check webcam availability
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n[WARNING] No active camera device detected on cv2.VideoCapture(0).")
        print("          Headless simulation will be run instead.")
        run_simulation_suite(metrics)
        return

    # 3. Setup MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    fqa_engine = FaceQualityEngine(target_size=(112, 112))
    engine = LivenessDecisionEngine(preset="MEDIUM")

    print("\nWebcam successfully opened. Instructions:")
    print("  - A liveness session will begin. Complete 'BLINK' and 'TURN_LEFT' active prompts.")
    print("  - Real-time HUD will show scores.")
    print("  - Press 'q' to abort current run.")

    # Configure session run
    session_id = engine.session_id
    print(f"\nActive Session: {session_id}")
    
    # User prompt
    true_label = input("Enter true label for this evaluation run (LIVE / SPOOF): ").strip().upper()
    if true_label not in ["LIVE", "SPOOF"]:
        true_label = "LIVE"

    t_start = time.perf_counter()
    frame_count = 0
    fqa_res = {}
    res = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame.")
            break

        frame_count += 1
        h_frame, w_frame, _ = frame.shape
        timestamp = time.perf_counter()

        # Flip horizontally for natural mirror feel
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Process Face Mesh
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmarks coords
            lms_list = []
            for lm in face_landmarks.landmark:
                lms_list.append((lm.x, lm.y, lm.z))
            lms_array = np.array(lms_list, dtype=np.float32)

            # Bounding box bounds
            x_coords = lms_array[:, 0] * w_frame
            y_coords = lms_array[:, 1] * h_frame
            x = int(np.min(x_coords))
            y = int(np.min(y_coords))
            w = int(np.max(x_coords) - x)
            h = int(np.max(y_coords) - y)
            bbox = (x, y, w, h)

            # Eye positions for FQA (corners 33 and 263)
            left_eye = (int(lms_array[33, 0] * w_frame), int(lms_array[33, 1] * h_frame))
            right_eye = (int(lms_array[263, 0] * w_frame), int(lms_array[263, 1] * h_frame))

            # Update FQA
            fqa_res = fqa_engine.assess_raw_face(frame_bgr, left_eye, right_eye, bbox)
            
            # Update Liveness Engine
            res = engine.update(
                frame=frame_bgr,
                bbox=bbox,
                landmarks=lms_array,
                tracking_confidence=0.90,
                timestamp=timestamp
            )

            # Draw HUD
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Decision: {res['decision']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Active Challenge: {res['current_challenge']}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"FQA Score: {fqa_res.get('overall_quality_score', 0)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_bgr, f"Passive Score: {res['passive_score']:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            if res["session_status"] in ["SUCCESS", "TIMEOUT", "FAILED"]:
                print(f"\nSession terminated: {res['session_status']}, Decision: {res['decision']}")
                break
        else:
            cv2.putText(frame_bgr, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("Real-World Liveness Evaluation Console", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nSession aborted by user.")
            res = {"decision": "SPOOF", "success": False, "active_score": 0.0, "passive_score": 0.0, "final_liveness_score": 0.0, "error": "ABORTED"}
            break

    # Release webcam
    cap.release()
    cv2.destroyAllWindows()

    duration_ms = (time.perf_counter() - t_start) * 1000.0
    fps = float(frame_count / (duration_ms / 1000.0)) if duration_ms > 0 else 30.0

    # Add to MetricsCollector
    metrics.add_session(
        session_id=session_id,
        true_label=true_label,
        predicted_decision=res.get("decision", "SPOOF"),
        success=res.get("success", False),
        active_score=res.get("active_score", 0.0),
        passive_score=res.get("passive_score", 0.0),
        final_score=res.get("final_liveness_score", 0.0),
        duration_ms=duration_ms,
        avg_fps=fps,
        avg_confidence=0.90,
        error=res.get("error"),
        fqa_metrics=fqa_res
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-World Liveness Evaluation tool.")
    parser.add_argument("--simulate", action="store_true", help="Forces simulated headless evaluations.")
    parser.add_argument("--runs", type=int, default=1, help="Number of simulated runs iterations (for --simulate).")
    args = parser.parse_args()

    print("=== Liveness Preprocessing & Fusion Decision Evaluation Tool ===")
    
    collector = MetricsCollector()

    if args.simulate:
        # Run simulation suite multiple times if configured
        for r in range(args.runs):
            print(f"\n--- Suite iteration {r+1}/{args.runs} ---")
            run_simulation_suite(collector)
    else:
        # Interactive mode
        run_webcam_evaluation(collector)

    # Save and output reports
    csv_path = collector.save_to_csv("evaluation_results.csv")
    print(f"\nCSV report successfully written to: {csv_path}")
    collector.print_summary()
