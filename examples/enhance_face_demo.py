"""
CLAHE Enhancement Demonstration Script (Simplified).

This script demonstrates how to:
1. Load sample face images from disk.
2. Instantiate the CLAHEEnhancer.
3. Apply Grayscale CLAHE (prioritized default).
4. Verify brightness, contrast statistics, and quality_hint outputs.
5. Simulate extreme environment conditions (too dark, overexposed, low contrast)
   to verify quality gateways.
"""

import os
import cv2
import sys
import numpy as np

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.enhancement import CLAHEEnhancer


def run_demo():
    print("=== CLAHE Contrast Enhancement & Quality Gateways ===")

    # Paths
    examples_dir = os.path.abspath(os.path.dirname(__file__))
    sample_dir = os.path.join(examples_dir, "sample_images")
    output_dir = os.path.join(examples_dir, "output_results")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Look for sample images from the previous demo run
    sample_files = [
        "sample_normal.jpg",
        "sample_mild_tilt.jpg",
        "sample_close_up.jpg",
    ]
    
    loaded_images = []
    for filename in sample_files:
        filepath = os.path.join(sample_dir, filename)
        if os.path.exists(filepath):
            loaded_images.append((filename, cv2.imread(filepath)))
        else:
            # Fallback in case alignment demo hasn't been run
            from opencv_module.utils import create_synthetic_face
            img, _, _ = create_synthetic_face(size=(400, 400), rotation_angle=10.0)
            os.makedirs(sample_dir, exist_ok=True)
            cv2.imwrite(filepath, img)
            loaded_images.append((filename, img))

    # 2. Instantiate CLAHEEnhancer with Recommended Parameters for Variable Environments
    enhancer = CLAHEEnhancer(
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        dark_threshold=40.0,
        bright_threshold=220.0,
        low_contrast_threshold=15.0
    )

    print(f"\n[Parameters Configured]")
    print(f"  - Clip Limit:             {enhancer.clip_limit}")
    print(f"  - Tile Grid Size:         {enhancer.tile_grid_size}")
    print(f"  - Quality Thresholds:     Dark < {enhancer.dark_threshold}, Bright > {enhancer.bright_threshold}, Low Contrast < {enhancer.low_contrast_threshold}\n")

    # 3. Process each standard sample
    for filename, img in loaded_images:
        print(f"--- Processing: {filename} ---")
        
        # Apply Grayscale CLAHE (Primary default, entropy disabled to optimize CPU)
        result = enhancer.enhance(img, color_mode=False, compute_entropy=False)

        if result["success"]:
            gray_enhanced = result["enhanced_image"]
            m_before = result["metrics"]["before"]
            m_after = result["metrics"]["after"]
            quality_hint = result["quality_hint"]
            
            print(f"  - Quality Gateway Hint: {quality_hint}")
            print(f"  - Brightness Mean:      {m_before['mean']:.2f}  -->  {m_after['mean']:.2f}")
            print(f"  - Contrast (StdDev):    {m_before['std']:.2f}  -->  {m_after['std']:.2f}")
            print(f"  - Entropy (Details):    {m_before['entropy']} (Optional / Disabled for Speed)")
            
            # Save simple enhanced image
            out_path = os.path.join(output_dir, f"enhanced_{filename}")
            cv2.imwrite(out_path, gray_enhanced)
            print(f"  - Saved Grayscale Output to: {os.path.basename(out_path)}\n")
        else:
            print(f"  - [ERROR] {result.get('error')}\n")

    # 4. Simulate Extreme Environments to Verify Quality Gateways
    print("--- Simulating Extreme Lighting Environments ---")
    
    # Create simple 112x112 canvases for extreme cases
    too_dark = np.ones((112, 112, 3), dtype=np.uint8) * 25  # Dark image (mean ~25)
    overexposed = np.ones((112, 112, 3), dtype=np.uint8) * 235  # Bright image (mean ~235)
    low_contrast = np.ones((112, 112, 3), dtype=np.uint8) * 128  # Solid gray (mean ~128, std ~0)
    # Add minor noise to solid gray so std is slightly above 0 but under 15
    noise = np.random.normal(0, 2.0, (112, 112, 3)).astype(np.int16)
    low_contrast = np.clip(low_contrast.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    extreme_cases = [
        ("Simulated Dark Face", too_dark),
        ("Simulated Overexposed Face", overexposed),
        ("Simulated Low Contrast Face", low_contrast)
    ]

    for label, img in extreme_cases:
        res = enhancer.enhance(img, color_mode=False)
        if res["success"]:
            print(f"  - {label}:")
            print(f"    * Mean:          {res['metrics']['before']['mean']:.2f}")
            print(f"    * Contrast (std):{res['metrics']['before']['std']:.2f}")
            print(f"    * Quality Hint:  {res['quality_hint']}")
            
    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    run_demo()
