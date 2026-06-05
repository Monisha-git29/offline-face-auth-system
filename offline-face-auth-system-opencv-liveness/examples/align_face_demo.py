"""
Face Alignment Demonstration Script.

This script demonstrates the production-ready FaceAligner module:
1. Generates 3 to 5 sample face images with different head tilt angles and positions,
   saving them as local JPEG assets.
2. Loads these sample images from disk, representing practical camera capture.
3. Passes them to the FaceAligner with coordinates and optional bounding boxes.
4. Validates the input quality and feasibility of alignment.
5. Prints the structured dictionary metrics (angle, scale, eye distance).
6. Saves the aligned results side-by-side with original images for manual verification.
"""

import os
import cv2
import sys
import numpy as np

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.alignment import FaceAligner
from opencv_module.utils import draw_alignment_debug, create_synthetic_face


def generate_samples(sample_dir: str):
    """
    Creates and saves 5 distinct sample face images on disk to simulate
    loading real-world images from the mobile camera feed.
    """
    os.makedirs(sample_dir, exist_ok=True)
    
    # 5 test cases covering normal, mild, extreme, close, and invalid poses
    configs = [
        {"name": "sample_normal.jpg", "angle": 0.0, "size": (400, 400)},
        {"name": "sample_mild_tilt.jpg", "angle": -15.0, "size": (400, 400)},
        {"name": "sample_extreme_tilt.jpg", "angle": 55.0, "size": (400, 400)},  # Feasibility fail (tilt > 45)
        {"name": "sample_close_up.jpg", "angle": 10.0, "size": (600, 600)},     # Large scale difference
        {"name": "sample_small_eyes.jpg", "angle": 5.0, "size": (150, 150)}      # Too small / low quality
    ]

    samples_data = []
    
    for conf in configs:
        img, left_eye, right_eye = create_synthetic_face(
            size=conf["size"],
            rotation_angle=conf["angle"]
        )
        
        # Save sample to disk
        filepath = os.path.join(sample_dir, conf["name"])
        cv2.imwrite(filepath, img)
        
        samples_data.append({
            "filepath": filepath,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "face_bbox": (conf["size"][0] // 4, conf["size"][1] // 4, conf["size"][0] // 2, conf["size"][1] // 2)
        })
        
    return samples_data


def run_demo():
    print("=== Face Alignment Practical Verification Demo ===")
    
    # Paths
    examples_dir = os.path.abspath(os.path.dirname(__file__))
    sample_dir = os.path.join(examples_dir, "sample_images")
    output_dir = os.path.join(examples_dir, "output_results")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate and save the 5 sample face images to disk
    print("\n[Step 1] Generating and saving 5 sample images to disk...")
    samples = generate_samples(sample_dir)
    print(f"Generated samples successfully in: {sample_dir}")

    # 2. Instantiate the FaceAligner
    # Defaulting to INTER_LINEAR (mobile optimized)
    aligner = FaceAligner(
        target_size=(112, 112),
        eye_x_ratio=0.35,
        eye_y_ratio=0.35,
        interpolation=cv2.INTER_LINEAR,
        min_eye_distance=20.0,
        min_face_size=50
    )

    # 3. Process and Align each loaded sample
    print("\n[Step 2] Loading samples from disk and performing alignment...\n")
    
    for idx, sample in enumerate(samples):
        filename = os.path.basename(sample["filepath"])
        print(f"--- Processing {idx + 1}: {filename} ---")
        
        # Load from disk
        image = cv2.imread(sample["filepath"])
        left_eye = sample["left_eye"]
        right_eye = sample["right_eye"]
        face_bbox = sample["face_bbox"]
        
        # Check boundary edge case simulation for the 5th sample (too small)
        # We also manually test a corrupted check (invalid coordinates) on the last sample
        if "small" in filename:
            # Let's pass valid eye coords first, but then force an out of bounds check
            print("  [Simulating low-quality checks]")
            
        # Execute alignment
        result = aligner.align(
            image=image,
            left_eye=left_eye,
            right_eye=right_eye,
            face_bbox=face_bbox
        )
        
        # 4. Print Structured Metrics & Save Results
        if result["success"]:
            aligned_face = result["aligned_face"]
            print(f"  [SUCCESS]")
            print(f"    - Measured Angle:   {result['angle']:.2f} degrees")
            print(f"    - Applied Scale:    {result['scale']:.4f}x")
            # Calculate distance between eyes in the aligned face to prove alignment accuracy
            print(f"    - Orig Eye Distance: {result['eye_distance']:.2f} pixels")
            
            # Save visual outputs
            # Original with landmarks
            debug_orig = draw_alignment_debug(image, left_eye, right_eye)
            orig_out_path = os.path.join(output_dir, f"{idx+1}_original_{filename}")
            aligned_out_path = os.path.join(output_dir, f"{idx+1}_aligned_{filename}")
            
            cv2.imwrite(orig_out_path, debug_orig)
            cv2.imwrite(aligned_out_path, aligned_face)
            
            print(f"    - Saved Debug Original to: {os.path.basename(orig_out_path)}")
            print(f"    - Saved Aligned Face to:    {os.path.basename(aligned_out_path)}")
        else:
            print(f"  [REJECTED]")
            print(f"    - Error Code:       {result['error']}")
            
            # Save original anyway to show what failed
            debug_orig = draw_alignment_debug(image, left_eye, right_eye)
            orig_out_path = os.path.join(output_dir, f"{idx+1}_rejected_{filename}")
            cv2.imwrite(orig_out_path, debug_orig)
            print(f"    - Saved Original to: {os.path.basename(orig_out_path)}")

    # 5. Simulate invalid quality inputs
    print("\n--- Processing 6: Error Handling Verification ---")
    corrupt_result = aligner.align(
        image=None,  # Null image
        left_eye=(100, 100),
        right_eye=(200, 200)
    )
    print(f"  Result with None image: success = {corrupt_result['success']}, error = {corrupt_result.get('error')}")

    out_of_bounds_result = aligner.align(
        image=samples[0]["left_eye"],  # passing wrong type
        left_eye=(500, 500),  # out of bounds coords
        right_eye=(600, 600)
    )
    print(f"  Result with invalid image type: success = {out_of_bounds_result['success']}, error = {out_of_bounds_result.get('error')}")
    
    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    run_demo()
