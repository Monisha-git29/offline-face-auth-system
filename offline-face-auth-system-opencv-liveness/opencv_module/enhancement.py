"""
CLAHE Enhancement Module using OpenCV.

This module provides a production-grade CLAHE (Contrast Limited Adaptive Histogram
Equalization) processor designed to improve face visibility in variable lighting,
such as harsh outdoor sunlight, deep shadows, and low-light remote locations.

It prioritizes grayscale processing, supports optional entropy calculation (disabled by default
for mobile speed), and provides quality hints (GOOD, TOO_DARK, OVEREXPOSED, LOW_CONTRAST)
for downstream Face Quality Assessment.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Union, Optional


class CLAHEEnhancer:
    """
    Production-ready CLAHE enhancement engine for offline mobile pipelines.
    Optimized to improve local contrast and detail without magnifying noise.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        dark_threshold: float = 40.0,
        bright_threshold: float = 220.0,
        low_contrast_threshold: float = 15.0
    ):
        """
        Initializes the CLAHEEnhancer.

        Args:
            clip_limit (float): Threshold for contrast limiting. Standard default is 2.0.
                                Lower values (e.g. 1.5) are better for harsh outdoor light.
            tile_grid_size (Tuple[int, int]): Grid size for local histogram equalizations.
                                             Standard is (8, 8). (4, 4) provides fine grain.
            dark_threshold (float): Mean brightness threshold below which a face is "TOO_DARK".
            bright_threshold (float): Mean brightness threshold above which a face is "OVEREXPOSED".
            low_contrast_threshold (float): Standard deviation threshold below which a face is "LOW_CONTRAST".
        """
        if clip_limit <= 0.0:
            raise ValueError(f"Invalid clip_limit {clip_limit}. Must be positive.")
        if len(tile_grid_size) != 2 or tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
            raise ValueError(f"Invalid tile_grid_size {tile_grid_size}. Must be positive integers.")

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.low_contrast_threshold = low_contrast_threshold
        
        # Create standard OpenCV CLAHE operator
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

    def _calculate_metrics(
        self,
        gray_img: np.ndarray,
        compute_entropy: bool = False
    ) -> Dict[str, Union[float, Optional[float]]]:
        """
        Calculates brightness, contrast, and optional information entropy of a grayscale image.

        Args:
            gray_img (np.ndarray): 1-channel (grayscale) image.
            compute_entropy (bool): Whether to calculate texture entropy (expensive on mobile).

        Returns:
            Dict[str, Union[float, Optional[float]]]: Brightness, contrast (stddev), and optional entropy.
        """
        mean_brightness = float(np.mean(gray_img))
        contrast = float(np.std(gray_img))
        entropy = None

        if compute_entropy:
            # Calculate Information Entropy (P(intensity) distribution)
            hist, _ = np.histogram(gray_img, bins=256, range=(0, 256))
            probabilities = hist / float(gray_img.size)
            # Keep non-zero values to avoid log2(0)
            probabilities = probabilities[probabilities > 0]
            entropy = float(-np.sum(probabilities * np.log2(probabilities)))

        return {
            "mean": mean_brightness,
            "std": contrast,
            "entropy": entropy
        }

    def _determine_quality_hint(self, mean: float, std: float) -> str:
        """
        Determines the quality warning based on mean brightness and standard deviation.

        Args:
            mean (float): Mean brightness of the face crop.
            std (float): Standard deviation (contrast) of the face crop.

        Returns:
            str: "TOO_DARK" | "OVEREXPOSED" | "LOW_CONTRAST" | "GOOD"
        """
        if mean < self.dark_threshold:
            return "TOO_DARK"
        elif mean > self.bright_threshold:
            return "OVEREXPOSED"
        elif std < self.low_contrast_threshold:
            return "LOW_CONTRAST"
        return "GOOD"

    def enhance(
        self,
        image: np.ndarray,
        color_mode: bool = False,
        compute_entropy: bool = False
    ) -> Dict[str, Any]:
        """
        Applies Contrast Limited Adaptive Histogram Equalization to the input image,
        mitigating shadows, glare, and low-light conditions.

        Args:
            image (np.ndarray): Input face image (BGR or Grayscale).
            color_mode (bool): If True, equalizes only the luminance (L) channel in LAB space,
                               returning a color BGR image. If False (default), returns a
                               grayscale output image prioritizing mobile pipeline efficiency.
            compute_entropy (bool): If True, calculates texture entropy. Defaults to False
                                    to optimize performance for resource-constrained devices.

        Returns:
            Dict[str, Any]: Structured output.
                On failure:
                    {"success": False, "error": "INVALID_IMAGE_INPUT"}
                On success:
                    {
                        "success": True,
                        "enhanced_image": np.ndarray,  # 1-channel Grayscale (or 3-channel BGR)
                        "quality_hint": "GOOD" | "TOO_DARK" | "OVEREXPOSED" | "LOW_CONTRAST",
                        "metrics": {
                            "before": {"mean": float, "std": float, "entropy": Optional[float]},
                            "after": {"mean": float, "std": float, "entropy": Optional[float]}
                        }
                    }
        """
        # ==========================================
        # 1. Validation
        # ==========================================
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {"success": False, "error": "INVALID_IMAGE_INPUT"}

        # Validate dimensions
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return {"success": False, "error": "INVALID_IMAGE_INPUT"}

        channels = 1 if len(image.shape) == 2 else image.shape[2]

        try:
            # ==========================================
            # 2. Processing (Grayscale or LAB Color Mode)
            # ==========================================
            if not color_mode or channels == 1:
                # Grayscale enhancement (Primary pipeline path)
                if channels > 1:
                    gray_in = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_in = image.copy()
                
                # Compute before metrics
                metrics_before = self._calculate_metrics(gray_in, compute_entropy=compute_entropy)
                
                # Apply CLAHE
                enhanced_img = self.clahe.apply(gray_in)
                
                # Compute after metrics
                metrics_after = self._calculate_metrics(enhanced_img, compute_entropy=compute_entropy)
            
            else:
                # Optional Color enhancement using LAB color space (preserves natural hues)
                # Convert to LAB space: L (Luminance), A (Green-Red), B (Blue-Yellow)
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                
                # Compute before metrics using the L (luminance) channel
                metrics_before = self._calculate_metrics(l_channel, compute_entropy=compute_entropy)
                
                # Apply CLAHE only on the L channel
                l_enhanced = self.clahe.apply(l_channel)
                
                # Compute after metrics on L channel
                metrics_after = self._calculate_metrics(l_enhanced, compute_entropy=compute_entropy)
                
                # Merge back the enhanced luminance with color channels
                lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
                
                # Convert back to standard BGR color space
                enhanced_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            # Determine quality hint using the "before" metrics
            quality_hint = self._determine_quality_hint(
                mean=metrics_before["mean"],
                std=metrics_before["std"]
            )

            return {
                "success": True,
                "enhanced_image": enhanced_img,
                "quality_hint": quality_hint,
                "metrics": {
                    "before": metrics_before,
                    "after": metrics_after
                }
            }

        except Exception as e:
            return {"success": False, "error": f"ENHANCEMENT_FAILURE: {str(e)}"}
