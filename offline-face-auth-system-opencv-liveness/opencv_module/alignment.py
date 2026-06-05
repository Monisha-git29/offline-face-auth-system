"""
Face Alignment Module using OpenCV.

This module provides a production-grade, highly optimized similarity transformation
(rotation, scaling, and translation) to align, scale, and crop faces based on
eye coordinates. This is a critical preprocessing step for face recognition networks
like MobileFaceNet.

It features:
1. Input quality validation layers.
2. Alignment feasibility verification.
3. Configurable interpolation settings (optimized for mobile).
4. Structured dictionary outputs for downstream assessment and debugging.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union


class FaceAligner:
    """
    Production-ready face aligner using OpenCV similarity transform.
    Optimized for mobile-friendly pipelines and offline use.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (112, 112),
        eye_x_ratio: float = 0.35,
        eye_y_ratio: float = 0.35,
        interpolation: int = cv2.INTER_LINEAR,
        min_eye_distance: float = 20.0,
        min_face_size: int = 50
    ):
        """
        Initializes the FaceAligner.

        Args:
            target_size (Tuple[int, int]): Output (width, height) of the aligned face image.
                                          Default is (112, 112), standard for MobileFaceNet.
            eye_x_ratio (float): Desired horizontal position of the left eye as a ratio of width.
                                 The right eye will be at (1 - eye_x_ratio). Standard is 0.35.
            eye_y_ratio (float): Desired vertical position of both eyes as a ratio of height.
                                 Standard is 0.35.
            interpolation (int): OpenCV interpolation method. Default is cv2.INTER_LINEAR (Fast for mobile).
                                 Use cv2.INTER_CUBIC for high quality.
            min_eye_distance (float): Minimum physical distance (in pixels) between eyes.
            min_face_size (int): Minimum width/height of the face bounding box.
        """
        # Validate initialization parameters
        if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(f"Invalid target_size {target_size}. Must be positive integers (width, height).")
        
        if not (0.0 < eye_x_ratio < 0.5):
            raise ValueError(f"Invalid eye_x_ratio {eye_x_ratio}. Must be between 0.0 and 0.5.")
            
        if not (0.0 < eye_y_ratio < 1.0):
            raise ValueError(f"Invalid eye_y_ratio {eye_y_ratio}. Must be between 0.0 and 1.0.")

        self.target_width, self.target_height = target_size
        self.eye_x_ratio = eye_x_ratio
        self.eye_y_ratio = eye_y_ratio
        self.interpolation = interpolation
        self.min_eye_distance = min_eye_distance
        self.min_face_size = min_face_size

    def align(
        self,
        image: np.ndarray,
        left_eye: Tuple[float, float],
        right_eye: Tuple[float, float],
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Validates the face input, assesses alignment feasibility, and returns
        the aligned face crop inside a structured dictionary.

        Args:
            image (np.ndarray): Input face image (BGR, RGB, or Grayscale).
            left_eye (Tuple[float, float]): (x, y) coordinates of the left eye.
            right_eye (Tuple[float, float]): (x, y) coordinates of the right eye.
            face_bbox (Optional[Tuple[int, int, int, int]]): Optional face bounding box as (x, y, w, h).

        Returns:
            Dict[str, Any]: Structured output.
                If validation or feasibility check fails:
                    {"success": False, "error": "INVALID_FACE_INPUT" or "FACE_POORLY_POSITIONED"}
                If success:
                    {
                        "success": True,
                        "aligned_face": np.ndarray (size target_size),
                        "angle": float (tilt angle in degrees),
                        "scale": float (scale factor applied),
                        "eye_distance": float (original eye distance in pixels)
                    }
        """
        # ==========================================
        # 1. Input Quality Validation Layer
        # ==========================================
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {"success": False, "error": "INVALID_FACE_INPUT"}

        # Check dimensions
        img_h, img_w = image.shape[:2]
        if img_h <= 0 or img_w <= 0:
            return {"success": False, "error": "INVALID_FACE_INPUT"}

        # Validate eye coordinates format
        if len(left_eye) != 2 or len(right_eye) != 2:
            return {"success": False, "error": "INVALID_FACE_INPUT"}

        lx, ly = left_eye
        rx, ry = right_eye

        # Check boundaries
        if not (0 <= lx < img_w) or not (0 <= ly < img_h):
            return {"success": False, "error": "INVALID_FACE_INPUT"}
        if not (0 <= rx < img_w) or not (0 <= ry < img_h):
            return {"success": False, "error": "INVALID_FACE_INPUT"}

        # Check for identical eye coordinates
        if lx == rx and ly == ry:
            return {"success": False, "error": "INVALID_FACE_INPUT"}

        # Validate face bounding box size if provided
        if face_bbox is not None:
            if len(face_bbox) != 4:
                return {"success": False, "error": "INVALID_FACE_INPUT"}
            bx, by, bw, bh = face_bbox
            if bw < self.min_face_size or bh < self.min_face_size:
                return {"success": False, "error": "INVALID_FACE_INPUT"}

        # ==========================================
        # 2. Alignment Feasibility Checks
        # ==========================================
        dx = rx - lx
        dy = ry - ly
        eye_distance = np.sqrt(dx**2 + dy**2)

        # Reject if eye distance is below minimum threshold
        if eye_distance < self.min_eye_distance:
            return {"success": False, "error": "FACE_POORLY_POSITIONED"}

        # Calculate face rotation angle
        angle = np.degrees(np.arctan2(dy, dx))

        # Reject extreme head poses (> 45 degrees tilt)
        if abs(angle) > 45.0:
            return {"success": False, "error": "FACE_POORLY_POSITIONED"}

        # ==========================================
        # 3. Affine Transformation & Warping
        # ==========================================
        # Calculate target distance between eyes in the aligned image
        d_target = self.target_width * (1.0 - 2.0 * self.eye_x_ratio)
        scale = d_target / eye_distance

        # Midpoint of eyes in the input image
        midpoint_input = (float(lx + rx) / 2.0, float(ly + ry) / 2.0)
        # Target midpoint of eyes in the aligned image
        midpoint_target = (self.target_width / 2.0, self.target_height * self.eye_y_ratio)

        # Compute initial rotation & scaling matrix R around input midpoint
        R = cv2.getRotationMatrix2D(midpoint_input, angle, scale)

        # Adjust translation components so the input midpoint maps exactly to the target midpoint
        tx = midpoint_target[0] - (R[0, 0] * midpoint_input[0] + R[0, 1] * midpoint_input[1])
        ty = midpoint_target[1] - (R[1, 0] * midpoint_input[0] + R[1, 1] * midpoint_input[1])
        
        R[0, 2] = tx
        R[1, 2] = ty

        # Apply warpAffine to get the aligned, cropped, and scaled face
        aligned_face = cv2.warpAffine(
            image,
            R,
            (self.target_width, self.target_height),
            flags=self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return {
            "success": True,
            "aligned_face": aligned_face,
            "angle": float(angle),
            "scale": float(scale),
            "eye_distance": float(eye_distance)
        }
