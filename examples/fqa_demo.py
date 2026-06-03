"""
Face Quality Assessment (FQA) Engine Demonstration.

Demonstrates:
1. A standard clean frontal capture (yielding EXCELLENT or PROCEED).
2. An extreme tilt capture (triggering early alignment rejection).
3. A dark, blurred, and small face capture (triggering RECAPTURE via critical score overrides).
4. Full printouts of normalized sub-scores.
"""

import os
import cv2
import sys
import numpy as np

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.fqa import FaceQualityEngine
from opencv_module.utils import create_synthetic_face


def run_demo():
    print("=== Face Quality Assessment (FQA) Engine Demonstration ===")

    # Initialize FQA Engine
    fqa_engine = FaceQualityEngine(
        w_brightness=0.25,
        w_blur=0.35,
        w_pose=0.20,
        w_face_size=0.20,
        blur_target=150.0,            # Standard crop target
        min_face_size_target=150.0,    # Target resolution
        critical_score_threshold=20.0
    )

    print(f"\n[FQA Engine Initialized]")
    print(f"  - Weights: Brightness={fqa_engine.w_brightness}, Blur={fqa_engine.w_blur}, Pose={fqa_engine.w_pose}, Size={fqa_engine.w_face_size}")
    print(f"  - Critical Threshold: {fqa_engine.critical_score_threshold}\n")

    # ==========================================
    # Test Case 1: Standard Clean Frontal Face
    # ==========================================
    print("--- Test Case 1: Standard Clean Frontal Face ---")
    img_normal, le_n, re_n = create_synthetic_face(size=(400, 400), rotation_angle=0.0)
    bbox_normal = (100, 100, 200, 200)  # w=200, h=200 >= 150px
    
    res1 = fqa_engine.assess_raw_face(
        image=img_normal,
        left_eye=le_n,
        right_eye=re_n,
        face_bbox=bbox_normal
    )

    if res1["success"]:
        print(f"  - Overall Score:  {res1['overall_quality_score']}")
        print(f"  - Recommendation: {res1['recommendation']}")
        print(f"  - Sub-scores:")
        print(f"    * Brightness:   {res1['brightness_score']}/100")
        print(f"    * Blur Sharp:   {res1['blur_score']}/100")
        print(f"    * Pose Tilt:    {res1['pose_score']}/100")
        print(f"    * Face Scale:   {res1['face_size_score']}/100")
    else:
        print(f"  - [FAIL] Assessment failed: {res1.get('error')}")
    print()

    # ==========================================
    # Test Case 2: Extreme Tilt Face (>45 degrees)
    # ==========================================
    print("--- Test Case 2: Extreme Tilt Face (55 degrees) ---")
    img_tilt, le_t, re_t = create_synthetic_face(size=(400, 400), rotation_angle=55.0)
    bbox_tilt = (100, 100, 200, 200)

    res2 = fqa_engine.assess_raw_face(
        image=img_tilt,
        left_eye=le_t,
        right_eye=re_t,
        face_bbox=bbox_tilt
    )

    print(f"  - Assessment Success: {res2['success']}")
    print(f"  - Recommendation:     {res2['recommendation']}")
    print(f"  - Warning Details:    {res2.get('reason_details')}")
    print()

    # ==========================================
    # Test Case 3: Blurry, Dark, and Far Face Crop
    # ==========================================
    print("--- Test Case 3: Blurry, Dark, and Far Face Crop ---")
    # Generate tilted face, darken it, blur it, and give a tiny face bbox
    img_raw, le_r, re_r = create_synthetic_face(size=(400, 400), rotation_angle=-10.0)
    
    # Darken image: divide pixel intensities by 4 (mean will drop to ~15)
    dark_img = (img_raw // 4).astype(np.uint8)
    
    # Blur image severely: convolve with a 15x15 Gaussian
    blurry_dark_img = cv2.GaussianBlur(dark_img, (15, 15), 3.0)
    
    # Bounding box is small but valid (80x80 >= 50px limit for face_size)
    bbox_medium = (100, 100, 80, 80)

    res3 = fqa_engine.assess_raw_face(
        image=blurry_dark_img,
        left_eye=le_r,
        right_eye=re_r,
        face_bbox=bbox_medium
    )

    if res3["success"]:
        print(f"  - Overall Score:  {res3['overall_quality_score']}")
        print(f"  - Recommendation: {res3['recommendation']}")
        print(f"  - Warning Details: {res3['reason_details']}")
        print(f"  - Sub-scores:")
        print(f"    * Brightness:   {res3['brightness_score']}/100")
        print(f"    * Blur Sharp:   {res3['blur_score']}/100")
        print(f"    * Pose Tilt:    {res3['pose_score']}/100")
        print(f"    * Face Scale:   {res3['face_size_score']}/100")
    else:
        print(f"  - [Early Rejection Triggered]")
        print(f"    * Error:          {res3.get('error')}")
        print(f"    * Recommendation: {res3['recommendation']}")
        print(f"    * Reason Details: {res3.get('reason_details')}")
        
    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    run_demo()
