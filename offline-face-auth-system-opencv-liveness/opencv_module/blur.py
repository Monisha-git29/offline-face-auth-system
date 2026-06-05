"""
Blur Detection Module using OpenCV.

This module provides a production-grade Blur Detection processor based on the
Laplacian Variance method. It serves as an early filter in our preprocessing
pipeline to ensure that only sharp, readable facial images are passed downstream,
saving mobile battery, memory, and computational overhead.
"""

import cv2
import numpy as np
from typing import Dict, Any, Union


class BlurDetector:
    """
    Production-ready Blur Detector using the Laplacian Variance method.
    Optimized for high-throughput mobile and offline environments.
    """

    def __init__(
        self,
        sharp_threshold: float = 100.0,
        blurry_threshold: float = 50.0
    ):
        """
        Initializes the BlurDetector.

        Args:
            sharp_threshold (float): Variance score above which an image is considered "SHARP".
                                     Standard default is 100.0.
            blurry_threshold (float): Variance score below which an image is rejected as "BLURRY".
                                      Standard default is 50.0.
        """
        if sharp_threshold <= blurry_threshold:
            raise ValueError("sharp_threshold must be strictly greater than blurry_threshold.")
        if blurry_threshold <= 0.0:
            raise ValueError("blurry_threshold must be positive.")

        self.sharp_threshold = sharp_threshold
        self.blurry_threshold = blurry_threshold

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Calculates the Laplacian variance of the image and classifies it as
        SHARP, ACCEPTABLE, or BLURRY.

        Args:
            image (np.ndarray): Input face image (Grayscale or BGR).

        Returns:
            Dict[str, Any]: Structured output.
                On failure:
                    {"success": False, "error": "INVALID_IMAGE_INPUT"}
                On success:
                    {
                        "success": True,
                        "blur_score": float,
                        "status": "SHARP" | "ACCEPTABLE" | "BLURRY"
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
            # 2. Grayscale Conversion (Mobile Optimization)
            # ==========================================
            if channels > 1:
                # Internal grayscale conversion for BGR images
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = image

            # ==========================================
            # 3. Laplacian Variance Computation
            # ==========================================
            # Convolve the image with the Laplacian kernel
            laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
            
            # Compute the variance of the edge map
            blur_score = float(laplacian.var())

            # ==========================================
            # 4. Status Classification
            # ==========================================
            if blur_score >= self.sharp_threshold:
                status = "SHARP"
            elif blur_score >= self.blurry_threshold:
                status = "ACCEPTABLE"
            else:
                status = "BLURRY"

            return {
                "success": True,
                "blur_score": blur_score,
                "status": status
            }

        except Exception as e:
            return {"success": False, "error": f"BLUR_DETECTION_FAILURE: {str(e)}"}
