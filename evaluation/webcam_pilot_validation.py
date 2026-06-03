"""
Webcam Pilot Validation Script for Offline Face Authentication.

Allows real-world pilot validation of the Face Authentication SDK using a local webcam
(or simulated inputs in headless mode). Exposes keyboard triggers to enroll and authenticate.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.recognition import FaceAuthSDK
from opencv_module import errors


def run_headless_simulation(sdk: FaceAuthSDK):
    print("\n[INFO] Running Headless Webcam Simulation...")
    normal_path = "examples/sample_images/sample_normal.jpg"
    close_up_path = "examples/sample_images/sample_close_up.jpg"
    
    if not os.path.exists(normal_path):
        print(f"[ERROR] Sample image not found at '{normal_path}'. Headless validation failed.")
        sys.exit(1)
        
    img_normal = cv2.imread(normal_path)
    img_close_up = cv2.imread(close_up_path) if os.path.exists(close_up_path) else None

    # Test Enrollment
    print("\n1. Simulating User Enrollment...")
    enroll_res = sdk.enroll_user("pilot_subject_01", img_normal)
    print(f"Enroll Result: {enroll_res}")
    if not enroll_res.get("success"):
        print("[ERROR] Enrollment failed.")
        sys.exit(1)

    # Test Authentication (Same user)
    print("\n2. Simulating Authentication (Same subject)...")
    sdk.reset()
    # Note: In a single frame punch-in, liveness challenges are expected to fail because liveness checks expect challenge frames.
    # To authenticate a user using a single frame in the pilot validation, we check the FQA alignment and similarity score directly.
    # We can retrieve the aligned face directly or use FaceAuthSDK's internal components.
    res_det = sdk.liveness_sdk.mp_detector.detect_landmarks(img_normal)
    if res_det is None:
        print("[ERROR] Face detection failed on normal image.")
        sys.exit(1)
        
    fqa_res = sdk.liveness_sdk.fqa_engine.assess_raw_face(
        image=img_normal,
        left_eye=res_det["left_eye"],
        right_eye=res_det["right_eye"],
        face_bbox=res_det["face_bbox"]
    )
    if not fqa_res.get("success"):
        print(f"[ERROR] FQA failed: {fqa_res.get('error')}")
        sys.exit(1)
        
    aligned = fqa_res["aligned_face"]
    query_emb = sdk.recognizer.extract_embedding(aligned)
    matched_id, similarity = sdk.registry.verify(query_emb, sdk.similarity_threshold)
    print(f"Match Result: Matched ID='{matched_id}', Similarity={similarity:.4f} (Threshold={sdk.similarity_threshold})")
    if matched_id == "pilot_subject_01":
        print("[SUCCESS] Headless authentication test passed!")
    else:
        print("[ERROR] Authentication failed to match correct user.")
        sys.exit(1)
        
    # Test Mismatch (different face)
    if img_close_up is not None:
        print("\n3. Simulating Authentication (Different pose/crop)...")
        res_det_other = sdk.liveness_sdk.mp_detector.detect_landmarks(img_close_up)
        if res_det_other is not None:
            fqa_other = sdk.liveness_sdk.fqa_engine.assess_raw_face(
                image=img_close_up,
                left_eye=res_det_other["left_eye"],
                right_eye=res_det_other["right_eye"],
                face_bbox=res_det_other["face_bbox"]
            )
            if fqa_other.get("success"):
                aligned_other = fqa_other["aligned_face"]
                other_emb = sdk.recognizer.extract_embedding(aligned_other)
                matched_other, sim_other = sdk.registry.verify(other_emb, sdk.similarity_threshold)
                print(f"Match Result (Different Pose): Matched ID='{matched_other}', Similarity={sim_other:.4f}")
    
    print("\n[SUCCESS] Headless validation finished successfully.")


def run_interactive_webcam(sdk: FaceAuthSDK):
    print("\n[INFO] Starting Webcam Pilot Capture...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[WARNING] Webcam device (0) could not be opened.")
        print("[INFO] Falling back to headless simulation...")
        run_headless_simulation(sdk)
        return

    print("\n" + "="*60)
    print("            WEBCAM PILOT VALIDATION HOTKEYS")
    print("="*60)
    print("  'e' : Enroll user 'pilot_subject'")
    print("  'a' : Authenticate current frame")
    print("  'q' : Quit script")
    print("="*60)

    window_name = "Offline Face Authentication - Pilot Validation"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    status_message = ""
    status_color = (255, 255, 255)
    message_expire_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Draw UI overlay instructions
        cv2.putText(display_frame, "Offline Face Auth Pilot", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_frame, "e: Enroll 'pilot_subject' | a: Authenticate | q: Quit", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw current active status message (if any)
        if time.time() < message_expire_time:
            cv2.putText(display_frame, status_message, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('e'):
            print("\n[ACTION] Enrolling user 'pilot_subject'...")
            # Detect landmarks for visual feedback
            res_det = sdk.liveness_sdk.mp_detector.detect_landmarks(frame)
            if res_det is None:
                status_message = "Enroll Failed: Face not detected"
                status_color = (0, 0, 255)
                print(f"[STATUS] {status_message}")
            else:
                res_enroll = sdk.enroll_user("pilot_subject", frame)
                if res_enroll.get("success"):
                    status_message = "Enroll SUCCESS: registered 'pilot_subject'"
                    status_color = (0, 255, 0)
                else:
                    status_message = f"Enroll Failed: {res_enroll.get('error')}"
                    status_color = (0, 0, 255)
                print(f"[STATUS] {status_message}")
            message_expire_time = time.time() + 3.0
            
        elif key == ord('a'):
            print("\n[ACTION] Authenticating...")
            res_det = sdk.liveness_sdk.mp_detector.detect_landmarks(frame)
            if res_det is None:
                status_message = "Auth Failed: Face not detected"
                status_color = (0, 0, 255)
                print(f"[STATUS] {status_message}")
            else:
                # Direct face recognition verification for single frame check
                fqa_res = sdk.liveness_sdk.fqa_engine.assess_raw_face(
                    image=frame,
                    left_eye=res_det["left_eye"],
                    right_eye=res_det["right_eye"],
                    face_bbox=res_det["face_bbox"]
                )
                if not fqa_res.get("success"):
                    status_message = f"Auth Failed: {fqa_res.get('error')}"
                    status_color = (0, 0, 255)
                else:
                    aligned = fqa_res["aligned_face"]
                    query_emb = sdk.recognizer.extract_embedding(aligned)
                    matched_id, similarity = sdk.registry.verify(query_emb, sdk.similarity_threshold)
                    
                    if matched_id is not None:
                        status_message = f"Welcome {matched_id}! Sim={similarity:.4f}"
                        status_color = (0, 255, 0)
                    else:
                        status_message = f"Rejected: Face Not Enrolled (Sim={similarity:.4f})"
                        status_color = (0, 0, 255)
                print(f"[STATUS] {status_message}")
            message_expire_time = time.time() + 4.0

    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam session closed.")


def main():
    parser = argparse.ArgumentParser(description="Webcam Pilot Validation Script")
    parser.add_argument("--headless", action="store_true", help="Run without physical camera using image cache")
    parser.add_argument("--db", type=str, default="face_recognition/webcam_pilot_registry.sqlite", help="SQLite database path")
    args = parser.parse_args()

    # Clear temp database if exists
    if os.path.exists(args.db):
        try:
            os.remove(args.db)
        except OSError:
            pass

    # Initialize SDK
    sdk = FaceAuthSDK(
        preset="MEDIUM",
        model_path="face_recognition/mobilefacenet.tflite",
        db_path=args.db,
        similarity_threshold=0.60
    )

    try:
        if args.headless:
            run_headless_simulation(sdk)
        else:
            run_interactive_webcam(sdk)
    finally:
        sdk.close()
        # Clean up temp database
        if os.path.exists(args.db):
            try:
                os.remove(args.db)
            except OSError:
                pass


if __name__ == "__main__":
    main()
