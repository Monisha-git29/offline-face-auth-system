"""
Face Quality Assessment (FQA) Engine Module.

This module provides a unified orchestration layer that compiles spatial,
lighting, and texture details from FaceAligner, CLAHEEnhancer, and BlurDetector.
It computes a weighted quality score (0 to 100) and returns clear, actionable
recommendations (EXCELLENT, PROCEED, WARNING, RECAPTURE) designed to reject
poor captures early in resource-constrained offline environments.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

from .alignment import FaceAligner
from .enhancement import CLAHEEnhancer
from .blur import BlurDetector


class FaceQualityEngine:
    """
    Unified Face Quality Assessment Engine compiling pipeline sub-scores into
    a single high-level liveness preprocessing decision.
    """

    def __init__(
        self,
        w_brightness: float = 0.25,
        w_blur: float = 0.35,
        w_pose: float = 0.20,
        w_face_size: float = 0.20,
        target_size: Tuple[int, int] = (112, 112),
        blur_target: float = 300.0,
        min_face_size_target: float = 150.0,
        critical_score_threshold: float = 20.0
    ):
        """
        Initializes the FaceQualityEngine.

        Args:
            w_brightness (float): Score weight for lighting (0.0 to 1.0).
            w_blur (float): Score weight for camera focus (0.0 to 1.0).
            w_pose (float): Score weight for face rotation (0.0 to 1.0).
            w_face_size (float): Score weight for face bounding box scale (0.0 to 1.0).
            target_size (Tuple[int, int]): Processing crop size.
            blur_target (float): Variance score above which blur score is 100.
            min_face_size_target (float): Bbox dimension above which size score is 100.
            critical_score_threshold (float): Sub-score limit below which recapture is forced.
        """
        # Validate weights sum
        total_weight = w_brightness + w_blur + w_pose + w_face_size
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0 (currently sum to {total_weight}).")

        self.w_brightness = w_brightness
        self.w_blur = w_blur
        self.w_pose = w_pose
        self.w_face_size = w_face_size
        self.target_width, self.target_height = target_size
        self.blur_target = blur_target
        self.min_face_size_target = min_face_size_target
        self.critical_score_threshold = critical_score_threshold

        # Pre-allocate default sub-modules for self-contained use
        self._default_aligner = FaceAligner(target_size=target_size)
        self._default_enhancer = CLAHEEnhancer()
        self._default_detector = BlurDetector()

    def _score_brightness(self, mean: float) -> int:
        """
        Scores the mean brightness piecewise linearly (ideal: 100 to 180).
        """
        if mean < 100.0:
            # Linear decay from 100 down to 30
            score = ((mean - 30.0) / 70.0) * 100.0
            return max(0, min(100, int(score)))
        elif mean > 180.0:
            # Linear decay from 180 up to 240
            score = ((240.0 - mean) / 60.0) * 100.0
            return max(0, min(100, int(score)))
        return 100

    def _score_blur(self, blur_score: float) -> int:
        """
        Scores the focus sharpness linearly against the blur target.
        """
        score = (blur_score / self.blur_target) * 100.0
        return max(0, min(100, int(score)))

    def _score_pose(self, angle: float) -> int:
        """
        Scores the roll tilt angle linearly (0 deg = 100 score, 45 deg = 0 score).
        """
        score = 100.0 - (abs(angle) / 45.0) * 100.0
        return max(0, min(100, int(score)))

    def _score_face_size(self, size: float) -> int:
        """
        Scores the face scale against target size (ideal >= 150px, minimum 50px).
        """
        if size >= self.min_face_size_target:
            return 100
        elif size < 50.0:
            return 0
        score = ((size - 50.0) / (self.min_face_size_target - 50.0)) * 100.0
        return max(0, min(100, int(score)))

    def assess_raw_face(
        self,
        image: np.ndarray,
        left_eye: Tuple[float, float],
        right_eye: Tuple[float, float],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        aligner: Optional[FaceAligner] = None,
        enhancer: Optional[CLAHEEnhancer] = None,
        detector: Optional[BlurDetector] = None
    ) -> Dict[str, Any]:
        """
        Runs the complete integrated preprocessing pipeline (Align -> CLAHE -> Blur)
        and assess the compiled face metrics, returning scores and recommendation.

        Args:
            image (np.ndarray): Original BGR camera frame.
            left_eye (Tuple[float, float]): (x, y) coordinates of the left eye.
            right_eye (Tuple[float, float]): (x, y) coordinates of the right eye.
            face_bbox (Optional[Tuple[int, int, int, int]]): Optional bounding box (x, y, w, h).
            aligner (Optional[FaceAligner]): Custom aligner override.
            enhancer (Optional[CLAHEEnhancer]): Custom enhancer override.
            detector (Optional[BlurDetector]): Custom detector override.

        Returns:
            Dict[str, Any]: Structured quality assessment report.
        """
        # Resolve sub-modules
        val_aligner = aligner if aligner is not None else self._default_aligner
        val_enhancer = enhancer if enhancer is not None else self._default_enhancer
        val_detector = detector if detector is not None else self._default_detector

        # ==========================================
        # 1. Pipeline Execution
        # ==========================================
        # Stage A: Alignment & Crop
        align_res = val_aligner.align(image, left_eye, right_eye, face_bbox)
        if not align_res["success"]:
            # If alignment fails early, force recapture due to extreme tilt or out-of-bounds coords
            return {
                "success": False,
                "error": align_res["error"],
                "recommendation": "RECAPTURE",
                "reason_details": [f"Face alignment failed: {align_res['error']}"]
            }

        aligned_face = align_res["aligned_face"]
        tilt_angle = align_res["angle"]
        orig_eye_distance = align_res["eye_distance"]

        # Stage B: CLAHE Contrast Enhancement
        enhance_res = val_enhancer.enhance(aligned_face, color_mode=False, compute_entropy=False)
        if not enhance_res["success"]:
            return {
                "success": False,
                "error": enhance_res["error"],
                "recommendation": "RECAPTURE",
                "reason_details": ["Contrast enhancement failed."]
            }

        enhanced_face = enhance_res["enhanced_image"]
        mean_brightness = enhance_res["metrics"]["before"]["mean"]
        std_brightness = enhance_res["metrics"]["before"]["std"]

        # Stage C: Blur Detection
        blur_res = val_detector.detect(enhanced_face)
        if not blur_res["success"]:
            return {
                "success": False,
                "error": blur_res["error"],
                "recommendation": "RECAPTURE",
                "reason_details": ["Blur detection failed."]
            }

        blur_score_raw = blur_res["blur_score"]

        # ==========================================
        # 2. Sub-Score Normalization Calculations
        # ==========================================
        s_brightness = self._score_brightness(mean_brightness)
        s_blur = self._score_blur(blur_score_raw)
        s_pose = self._score_pose(tilt_angle)

        # Resolve face size (use bbox width/height, fallback to scaled eye distance if bbox is omitted)
        if face_bbox is not None:
            face_size = float(min(face_bbox[2], face_bbox[3]))
        else:
            # Mathematical fallback: inter-pupillary distance spans roughly 40% of standard face width
            face_size = float(orig_eye_distance * 2.5)

        s_face_size = self._score_face_size(face_size)

        # ==========================================
        # 3. Overall Weighted Compilation & Overrides
        # ==========================================
        overall_score = int(
            self.w_brightness * s_brightness +
            self.w_blur * s_blur +
            self.w_pose * s_pose +
            self.w_face_size * s_face_size
        )

        # Identify critical failures (sub-scores below safe threshold)
        reason_details = []
        if s_brightness < self.critical_score_threshold:
            reason_details.append("Image is extremely under-exposed or over-exposed.")
        if s_blur < self.critical_score_threshold:
            reason_details.append("Image is heavily blurred.")
        if s_pose < self.critical_score_threshold:
            reason_details.append("Face tilt angle is too high.")
        if s_face_size < self.critical_score_threshold:
            reason_details.append("Face is too small or far from the camera.")

        # Determine Recommendation
        if len(reason_details) > 0 or overall_score < 50:
            recommendation = "RECAPTURE"
            if overall_score >= 50:
                # Override triggered due to a critical check failing
                overall_score = min(49, overall_score)  # Clip to recapture range
        elif overall_score >= 85:
            recommendation = "EXCELLENT"
        elif overall_score >= 70:
            recommendation = "PROCEED"
        else:
            recommendation = "WARNING"
            reason_details.append("Face parameters are acceptable, but consider re-aligning.")

        return {
            "success": True,
            "brightness_score": s_brightness,
            "blur_score": s_blur,
            "pose_score": s_pose,
            "face_size_score": s_face_size,
            "overall_quality_score": overall_score,
            "recommendation": recommendation,
            "reason_details": reason_details,
            "aligned_face": aligned_face,
            "enhanced_face": enhanced_face
        }
