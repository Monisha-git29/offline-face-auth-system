"""
Blur Detection and Pipeline Integration Demonstration.

This script shows how to:
1. Load sample face images from disk.
2. Apply the FaceAligner module to align them.
3. Apply the CLAHEEnhancer module to equalize contrast in grayscale.
4. Apply the BlurDetector module to verify focus sharpness.
5. Simulate artificial blur (motion/defocus) using Gaussian filters
   to verify that quality status thresholds trigger correctly.
6. Print the structured dictionary metrics.
"""

import os
import cv2
import sys
import numpy as np

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.alignment import FaceAligner
from opencv_module.enhancement import CLAHEEnhancer
from opencv_module.blur import BlurDetector


def run_demo():
    print("=== Preprocessing Pipeline Integration & Blur Detection Demo ===")

    # Paths
    examples_dir = os.path.abspath(os.path.dirname(__file__))
    sample_dir = os.path.join(examples_dir, "sample_images")
    output_dir = os.path.join(examples_dir, "output_results")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize all Pipeline Preprocessing Modules
    print("\n[Step 1] Initializing Pipeline Preprocessing Modules...")
    aligner = FaceAligner(target_size=(112, 112), interpolation=cv2.INTER_LINEAR)
    enhancer = CLAHEEnhancer(clip_limit=2.0, tile_grid_size=(8, 8))
    detector = BlurDetector(sharp_threshold=100.0, blurry_threshold=50.0)
    
    print(f"  - Face Aligner:  target_size={aligner.target_width}x{aligner.target_height}")
    print(f"  - CLAHE Enhancer: clip_limit={enhancer.clip_limit}, tile={enhancer.tile_grid_size}")
    print(f"  - Blur Detector: sharp={detector.sharp_threshold}, blurry={detector.blurry_threshold}\n")

    # 2. Simulate BlazeFace output coordinates and load BGR original face
    # We will use the 'sample_normal.jpg' generated in previous steps
    filename = "sample_normal.jpg"
    filepath = os.path.join(sample_dir, filename)
    
    if not os.path.exists(filepath):
        # Fallback in case alignment demo hasn't been run
        from opencv_module.utils import create_synthetic_face
        img, left_eye, right_eye = create_synthetic_face(size=(400, 400), rotation_angle=0.0)
        os.makedirs(sample_dir, exist_ok=True)
        cv2.imwrite(filepath, img)
    else:
        img = cv2.imread(filepath)
        # Coordinates for unrotated synthetic face midpoint
        # cx, cy = 200, 200. Eyes were at cx-35, cy-25 and cx+35, cy-25 -> (165, 175), (235, 175)
        left_eye = (165.0, 175.0)
        right_eye = (235.0, 175.0)

    print(f"[Step 2] Processing standard image: {filename}...")

    # A. Pipeline Stage 1: Face Alignment & Crop (BGR)
    align_res = aligner.align(img, left_eye, right_eye)
    if not align_res["success"]:
        print("  - Alignment failed!")
        return
        
    aligned_face = align_res["aligned_face"]
    print(f"  - [Aligned] Success. Crop size: {aligned_face.shape[1]}x{aligned_face.shape[0]}")

    # B. Pipeline Stage 2: CLAHE Grayscale Contrast Enhancement
    enhance_res = enhancer.enhance(aligned_face, color_mode=False, compute_entropy=False)
    if not enhance_res["success"]:
        print("  - Contrast enhancement failed!")
        return
        
    enhanced_face = enhance_res["enhanced_image"]
    quality_hint = enhance_res["quality_hint"]
    print(f"  - [Enhanced] Success. Quality Hint: {quality_hint}")

    # C. Pipeline Stage 3: Blur Detection
    blur_res = detector.detect(enhanced_face)
    if not blur_res["success"]:
        print("  - Blur detection failed!")
        return

    print(f"  - [Blur Check] Original Aligned Face:")
    print(f"    * Blur Score: {blur_res['blur_score']:.2f}")
    print(f"    * Status:     {blur_res['status']}")

    # Save original enhanced output
    out_sharp_path = os.path.join(output_dir, "pipeline_sharp_grayscale.jpg")
    cv2.imwrite(out_sharp_path, enhanced_face)
    print(f"    * Saved Grayscale Face to: {os.path.basename(out_sharp_path)}\n")

    # 3. Simulate Artificial Focus/Motion Blur to Verify Gateway Statuses
    print("[Step 3] Simulating Focus / Motion Blur to verify quality gateways...")
    
    # A. Mild Blur (Simulating slight motion or lens smudge)
    mild_blurry_face = cv2.GaussianBlur(enhanced_face, (5, 5), 1.2)
    mild_blur_res = detector.detect(mild_blurry_face)
    
    print(f"  - [Blur Check] Simulated Mild Blur (Smudge):")
    print(f"    * Blur Score: {mild_blur_res['blur_score']:.2f}")
    print(f"    * Status:     {mild_blur_res['status']}")
    
    out_mild_path = os.path.join(output_dir, "pipeline_acceptable_grayscale.jpg")
    cv2.imwrite(out_mild_path, mild_blurry_face)
    print(f"    * Saved Grayscale Face to: {os.path.basename(out_mild_path)}\n")

    # B. Severe Blur (Simulating fast movement or drop focus)
    severe_blurry_face = cv2.GaussianBlur(enhanced_face, (13, 13), 3.0)
    severe_blur_res = detector.detect(severe_blurry_face)
    
    print(f"  - [Blur Check] Simulated Severe Blur (Motion):")
    print(f"    * Blur Score: {severe_blur_res['blur_score']:.2f}")
    print(f"    * Status:     {severe_blur_res['status']}")

    out_severe_path = os.path.join(output_dir, "pipeline_blurry_grayscale.jpg")
    cv2.imwrite(out_severe_path, severe_blurry_face)
    print(f"    * Saved Grayscale Face to: {os.path.basename(out_severe_path)}\n")

    # 4. Error Handling Verification
    print("[Step 4] Verifying Error Handling...")
    err_res = detector.detect(None)
    print(f"  - Result with None input: success = {err_res['success']}, error = {err_res.get('error')}")

    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    run_demo()
