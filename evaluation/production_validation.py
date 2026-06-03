"""
Production SDK Engineering Validation and Latency Study.

Validates face detection edge cases (present, absent, partial, multiple),
landmark stability (roll, yaw, lighting), enrollment/authentication APIs,
and logs subcomponent and end-to-end latencies.
"""

import os
import sys
import time
import numpy as np
import cv2
from typing import Dict, Any, List

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.recognition import FaceAuthSDK, FaceRecognizer, SQLiteFaceRegistry
from opencv_module.landmarks import MediaPipeLandmarkDetector
from opencv_module.alignment import FaceAligner
from opencv_module.fqa import FaceQualityEngine
from opencv_module import errors


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def run_latency_benchmark(sdk: FaceAuthSDK, detector: MediaPipeLandmarkDetector, aligner: FaceAligner, recognizer: FaceRecognizer, frame: np.ndarray, num_iterations: int = 50):
    print(f"\n--- Running Latency Benchmark ({num_iterations} iterations) ---")
    
    fmesh_times = []
    align_times = []
    inf_times = []
    e2e_times = []
    
    # Warm-up run
    detector.detect_landmarks(frame)
    res_det = detector.detect_landmarks(frame)
    if res_det is not None:
        aligner.align(frame, res_det["left_eye"], res_det["right_eye"], res_det["face_bbox"])
        aligned = aligner.align(frame, res_det["left_eye"], res_det["right_eye"], res_det["face_bbox"])["aligned_face"]
        recognizer.extract_embedding(aligned)
    sdk.authenticate(frame)
    
    # Main benchmark loop
    for _ in range(num_iterations):
        # 1. Face Mesh Latency
        t0 = time.perf_counter()
        res_det = detector.detect_landmarks(frame)
        fmesh_times.append((time.perf_counter() - t0) * 1000.0)
        
        if res_det is not None:
            # 2. Alignment Latency
            t0 = time.perf_counter()
            align_res = aligner.align(frame, res_det["left_eye"], res_det["right_eye"], res_det["face_bbox"])
            align_times.append((time.perf_counter() - t0) * 1000.0)
            
            aligned = align_res["aligned_face"]
            if aligned is not None:
                # 3. Model Inference Latency
                t0 = time.perf_counter()
                recognizer.extract_embedding(aligned)
                inf_times.append((time.perf_counter() - t0) * 1000.0)
        
        # 4. End-to-End Authentication Latency
        t0 = time.perf_counter()
        sdk.authenticate(frame)
        e2e_times.append((time.perf_counter() - t0) * 1000.0)
        
    def print_stats(name: str, times: List[float]):
        if not times:
            print(f"{name:<25} : No successful runs")
            return
        arr = np.array(times)
        print(f"{name:<25} : Mean={np.mean(arr):.2f}ms | Median={np.median(arr):.2f}ms | P95={np.percentile(arr, 95):.2f}ms | P99={np.percentile(arr, 99):.2f}ms")

    print_stats("MediaPipe Face Mesh", fmesh_times)
    print_stats("Face Alignment", align_times)
    print_stats("MobileFaceNet Inference", inf_times)
    print_stats("End-to-End Auth", e2e_times)
    print("-" * 60)


def main():
    print("=" * 70)
    print("      PRODUCTION SDK INTEGRATION VALIDATION & LATENCY STUDY")
    print("=" * 70)

    # Initialize components
    db_path = "face_recognition/test_validation_registry.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    sdk = FaceAuthSDK(
        preset="MEDIUM",
        model_path="face_recognition/mobilefacenet.tflite",
        db_path=db_path,
        similarity_threshold=0.60
    )
    
    detector = MediaPipeLandmarkDetector(model_path="face_recognition/face_landmarker.task")
    aligner = FaceAligner(target_size=(112, 112))
    recognizer = FaceRecognizer(model_path="face_recognition/mobilefacenet.tflite")

    # Load normal sample image
    normal_img_path = "examples/sample_images/sample_normal.jpg"
    img_normal = load_image(normal_img_path)

    # ==========================================
    # 1. FACE DETECTION EDGE CASES
    # ==========================================
    print("\n--- Testing Face Detection Edge Cases ---")
    
    # A. Face Present
    res_present = detector.detect_landmarks(img_normal)
    print(f"Face Present Test: {'PASS' if res_present is not None else 'FAIL'}")
    
    # B. Face Absent (blank frame)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_absent = detector.detect_landmarks(blank_frame)
    print(f"Face Absent Test: {'PASS' if res_absent is None else 'FAIL'}")
    
    # C. Partial Face (crop bottom half)
    partial_frame = img_normal.copy()
    partial_frame[int(partial_frame.shape[0] * 0.6):, :] = 0
    res_partial = detector.detect_landmarks(partial_frame)
    print(f"Partial Face Test: Detected={res_partial is not None} (Expected to fail/poor position or not detected)")
    
    # D. Multiple Faces (concat side by side)
    multi_frame = np.hstack([img_normal, img_normal])
    res_multi = detector.detect_landmarks(multi_frame)
    print(f"Multiple Faces Test: {'PASS' if res_multi is not None else 'FAIL'} (Detected primary face bbox={res_multi['face_bbox'] if res_multi else None})")

    # ==========================================
    # 2. LANDMARK DETECTION UNDER VARIATION
    # ==========================================
    print("\n--- Testing Landmark Detection Under Visual Variations ---")
    
    # A. Mild Roll
    img_mild = load_image("examples/sample_images/sample_mild_tilt.jpg")
    res_mild = detector.detect_landmarks(img_mild)
    print(f"Mild Roll Pose: {'PASS' if res_mild is not None else 'FAIL'}")
    
    # B. Extreme Pose / Yaw / Roll
    img_extreme = load_image("examples/sample_images/sample_extreme_tilt.jpg")
    res_extreme = detector.detect_landmarks(img_extreme)
    print(f"Extreme Pose: Detected={res_extreme is not None}")

    # C. Low Light
    img_dark = np.clip(img_normal.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)
    res_dark = detector.detect_landmarks(img_dark)
    print(f"Low Light (30% brightness): {'PASS' if res_dark is not None else 'FAIL'}")
    
    # D. Bright Light
    img_bright = np.clip(img_normal.astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
    res_bright = detector.detect_landmarks(img_bright)
    print(f"Bright Light (180% brightness): {'PASS' if res_bright is not None else 'FAIL'}")

    # ==========================================
    # 3. HIGH-LEVEL ENROLLMENT & AUTHENTICATION API
    # ==========================================
    print("\n--- Testing High-level SDK Enrollment and Authentication APIs ---")
    
    # Enroll user
    enroll_res = sdk.enroll_user("user_007", img_normal)
    print(f"Enroll User: {enroll_res}")
    
    # Authenticate (should match)
    sdk.reset()
    auth_res = sdk.authenticate(img_normal)
    print(f"Authenticate (Same Face): success={auth_res.get('success')}, identity={auth_res.get('identity')}, similarity={auth_res.get('similarity_score')}, error={auth_res.get('error')}")
    
    # Mismatch Rejection (using a different face)
    # We can load a simulated different face or use a blank frame
    sdk.reset()
    auth_mismatch = sdk.authenticate(img_dark) # Darkened face might still match depending on threshold, let's use another image
    img_other = load_image("examples/sample_images/sample_close_up.jpg")
    sdk.reset()
    auth_other = sdk.authenticate(img_other)
    print(f"Authenticate (Different Face / Pose): success={auth_other.get('success')}, identity={auth_other.get('identity')}, similarity={auth_other.get('similarity_score')}, error={auth_other.get('error')}")

    # ==========================================
    # 4. PERFORMANCE LATENCY
    # ==========================================
    run_latency_benchmark(sdk, detector, aligner, recognizer, img_normal, num_iterations=50)

    # Cleanup
    sdk.close()
    detector.close()
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print("\nProduction validation testing complete. Exiting successfully.")


if __name__ == "__main__":
    main()
