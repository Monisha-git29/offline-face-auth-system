"""
Smile Detection Module using MediaPipe Face Mesh.

Provides the SmileDetector class using mouth width and mouth opening normalized by
the 3D inter-eye distance to achieve scale invariance. Includes state transitions,
consecutive frame gates, and cooldown lockout.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional


class SmileDetector:
    """
    Production-grade offline Smile Detector based on normalized mouth geometric scale.
    """

    # MediaPipe Face Mesh landmarks
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    UPPER_LIP_CENTER = 13
    LOWER_LIP_CENTER = 14

    # Eye landmarks for scaling normalization
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    def __init__(
        self,
        smile_threshold: float = 0.45,
        min_consecutive_frames: int = 3,
        cooldown_duration: float = 0.30,
        min_confidence: float = 0.50
    ):
        """
        Initializes the SmileDetector.

        Args:
            smile_threshold (float): Smile score above which user is smiling.
            min_consecutive_frames (int): Minimum frames smile must remain held to verify.
            cooldown_duration (float): Cooldown lock (seconds) after challenge completion.
            min_confidence (float): Face tracking confidence threshold.
        """
        if smile_threshold <= 0.0 or smile_threshold >= 1.5:
            raise ValueError("smile_threshold must be between 0.0 and 1.5.")
        if min_consecutive_frames <= 0:
            raise ValueError("min_consecutive_frames must be a positive integer.")
        if cooldown_duration < 0.0:
            raise ValueError("cooldown_duration must be non-negative.")

        self.smile_threshold = smile_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.cooldown_duration = cooldown_duration
        self.min_confidence = min_confidence

        self.reset()

    def reset(self) -> None:
        """
        Resets session state variables and frames counter.
        """
        self.state = "IDLE"  # IDLE, FACE_CENTERED, SMILE_DETECTED, SMILE_HELD, CHALLENGE_COMPLETE
        self.frames_held = 0
        self.t_last_complete = 0.0

    def _euclidean_distance(self, pt1: np.ndarray, pt2: np.ndarray) -> float:
        """
        Calculates L2 distance between two 2D points.
        """
        return float(np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2))

    def calculate_smile_score(
        self,
        landmarks: Union[np.ndarray, List[Tuple[float, float, float]]],
        width: int,
        height: int
    ) -> float:
        """
        Calculates the normalized smile score based on mouth aspect ratio.
        """
        lms = np.array(landmarks, dtype=np.float32)

        # Helper to convert normalized landmark coordinate to absolute pixel (x, y)
        def get_pixel_pt(idx):
            return np.array([lms[idx][0] * width, lms[idx][1] * height], dtype=np.float32)

        # 1. Mouth width (corner-to-corner)
        p_lc = get_pixel_pt(self.LEFT_MOUTH_CORNER)
        p_rc = get_pixel_pt(self.RIGHT_MOUTH_CORNER)
        mouth_w = self._euclidean_distance(p_lc, p_rc)

        # 2. Mouth opening (lip-to-lip)
        p_ul = get_pixel_pt(self.UPPER_LIP_CENTER)
        p_ll = get_pixel_pt(self.LOWER_LIP_CENTER)
        mouth_h = self._euclidean_distance(p_ul, p_ll)

        # 3. Inter-eye distance
        le_c1 = get_pixel_pt(self.LEFT_EYE_OUTER)
        le_c2 = get_pixel_pt(self.LEFT_EYE_INNER)
        left_eye_center = (le_c1 + le_c2) / 2.0

        re_c1 = get_pixel_pt(self.RIGHT_EYE_INNER)
        re_c2 = get_pixel_pt(self.RIGHT_EYE_OUTER)
        right_eye_center = (re_c1 + re_c2) / 2.0

        d_eyes = self._euclidean_distance(left_eye_center, right_eye_center)

        if d_eyes < 1e-5:
            return 0.0

        # Normalization
        w_norm = mouth_w / d_eyes
        h_norm = mouth_h / d_eyes

        # Weighted score
        return float(0.7 * w_norm + 0.3 * h_norm)

    def update(
        self,
        landmarks: Optional[Union[np.ndarray, List[Tuple[float, float, float]]]],
        width: int,
        height: int,
        tracking_confidence: float = 1.0,
        timestamp: Optional[float] = None,
        active_challenge: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Receives landmarks, computes smile score, and drives the state transitions.
        """
        if landmarks is None or len(landmarks) < 400:
            return {"success": False, "error": "FACE_NOT_DETECTED"}
        if width <= 0 or height <= 0:
            return {"success": False, "error": "FACE_NOT_DETECTED"}
        if tracking_confidence < self.min_confidence:
            return {"success": False, "error": "LANDMARKS_UNSTABLE"}

        if timestamp is None:
            timestamp = time.perf_counter()

        try:
            # 1. Compute smile score
            smile_score = self.calculate_smile_score(landmarks, width, height)

            # 2. Drive Smile State Machine
            challenge_verified = False

            # A. Cooldown Lock Release Check
            if self.state == "CHALLENGE_COMPLETE":
                if timestamp - self.t_last_complete >= self.cooldown_duration:
                    self.state = "IDLE"

            # B. State Transitions
            is_smiling = (smile_score >= self.smile_threshold)

            if self.state == "IDLE":
                self.state = "FACE_CENTERED"
                self.frames_held = 0

            elif self.state == "FACE_CENTERED":
                if active_challenge == "SMILE":
                    if is_smiling:
                        self.state = "SMILE_DETECTED"
                        self.frames_held = 1
                    else:
                        self.frames_held = 0
                else:
                    self.frames_held = 0

            elif self.state == "SMILE_DETECTED":
                if active_challenge != "SMILE":
                    self.state = "IDLE"
                elif is_smiling:
                    self.frames_held += 1
                    if self.frames_held >= self.min_consecutive_frames:
                        self.state = "SMILE_HELD"
                        self.t_last_complete = timestamp
                        challenge_verified = True
                else:
                    # Reset counter if smile drops below threshold
                    self.state = "FACE_CENTERED"
                    self.frames_held = 0

            elif self.state == "SMILE_HELD":
                # Hold verification state
                self.state = "CHALLENGE_COMPLETE"
                challenge_verified = True

            elif self.state == "CHALLENGE_COMPLETE":
                # Maintain complete state until cooldown finishes
                challenge_verified = True

            return {
                "success": True,
                "smiling": is_smiling,
                "smile_score": float(smile_score),
                "verification_state": self.state,
                "frames_held": self.frames_held,
                "challenge_verified": challenge_verified,
                "error": None
            }

        except Exception as e:
            return {"success": False, "error": f"SMILE_DETECTION_FAILURE: {str(e)}"}
