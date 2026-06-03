"""
Blink Detection Module using MediaPipe Face Mesh.

This module provides a production-grade, lightweight, and fully-offline
Blink Detection engine using the Eye Aspect Ratio (EAR) method.
It operates on normalized landmarks scaled to absolute pixel space to achieve
aspect-ratio invariance. It incorporates a consecutive-frame state machine,
tracking confidence validation, and a high-precision time-based gate (50ms to 500ms)
to ensure robust offline liveness challenge verification across varying device frame rates.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional


class BlinkDetector:
    """
    Production-ready offline Blink Detector based on Eye Aspect Ratio (EAR).
    Optimized for React Native deployments in secure challenge liveness modules.
    """

    # Standard MediaPipe Face Mesh (468/478 points) Indices
    # Left eye landmarks
    LEFT_EYE_CORNERS = (33, 133)
    LEFT_EYE_VERTICAL_1 = (160, 144)
    LEFT_EYE_VERTICAL_2 = (158, 145)

    # Right eye landmarks
    RIGHT_EYE_CORNERS = (362, 263)
    RIGHT_EYE_VERTICAL_1 = (385, 380)
    RIGHT_EYE_VERTICAL_2 = (387, 373)

    def __init__(
        self,
        ear_threshold: float = 0.22,
        min_consecutive_frames: int = 2,
        min_blink_duration: float = 0.05,  # 50 ms in seconds
        max_blink_duration: float = 0.50,  # 500 ms in seconds
        min_confidence: float = 0.50
    ):
        """
        Initializes the BlinkDetector.

        Args:
            ear_threshold (float): EAR value below which eyes are considered closed.
                                  Default is 0.22, calibrated for standard Face Mesh.
            min_consecutive_frames (int): Minimum frames eyes must remain closed to register.
                                          Default is 2, preventing noise spikes.
            min_blink_duration (float): Minimum duration (in seconds) for a valid human blink.
                                        Default is 0.05 (50ms).
            max_blink_duration (float): Maximum duration (in seconds) for a valid human blink.
                                        Default is 0.50 (500ms).
            min_confidence (float): Landmark tracking confidence below which frames are ignored.
        """
        if ear_threshold <= 0.0 or ear_threshold >= 1.0:
            raise ValueError(f"Invalid ear_threshold {ear_threshold}. Must be between 0.0 and 1.0.")
        if min_consecutive_frames <= 0:
            raise ValueError("min_consecutive_frames must be a positive integer.")
        if min_blink_duration <= 0.0 or max_blink_duration < min_blink_duration:
            raise ValueError("Blink durations must be positive and max_blink_duration >= min_blink_duration.")

        self.ear_threshold = ear_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration
        self.min_confidence = min_confidence

        # Session State Variables
        self.eye_state = "OPEN"
        self.blink_counter = 0
        self.total_blinks = 0
        self.t_start = 0.0

    def reset(self) -> None:
        """
        Resets the internal state machine counter and session totals.
        """
        self.eye_state = "OPEN"
        self.blink_counter = 0
        self.total_blinks = 0
        self.t_start = 0.0

    def _euclidean_distance(self, pt1: np.ndarray, pt2: np.ndarray) -> float:
        """
        Computes the L2 Euclidean distance between two 2D points.
        """
        return float(np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2))

    def calculate_ear(
        self,
        landmarks: Union[np.ndarray, List[Tuple[float, float, float]]],
        width: int,
        height: int
    ) -> float:
        """
        Scales normalized landmarks to pixel space and calculates average EAR.

        Args:
            landmarks (List/np.ndarray): 468/478 landmarks containing (x, y, z).
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.

        Returns:
            float: Average Eye Aspect Ratio.
        """
        # Convert landmarks to numpy array
        lms = np.array(landmarks, dtype=np.float32)

        # Helper to convert normalized coordinate to absolute pixel coordinate (x, y)
        def get_pixel_pt(idx):
            return np.array([lms[idx][0] * width, lms[idx][1] * height], dtype=np.float32)

        # --- Left Eye EAR ---
        le_c1 = get_pixel_pt(self.LEFT_EYE_CORNERS[0])
        le_c2 = get_pixel_pt(self.LEFT_EYE_CORNERS[1])
        le_v1_top = get_pixel_pt(self.LEFT_EYE_VERTICAL_1[0])
        le_v1_bot = get_pixel_pt(self.LEFT_EYE_VERTICAL_1[1])
        le_v2_top = get_pixel_pt(self.LEFT_EYE_VERTICAL_2[0])
        le_v2_bot = get_pixel_pt(self.LEFT_EYE_VERTICAL_2[1])

        h_left = self._euclidean_distance(le_c1, le_c2)
        v1_left = self._euclidean_distance(le_v1_top, le_v1_bot)
        v2_left = self._euclidean_distance(le_v2_top, le_v2_bot)

        if h_left < 1e-5:
            ear_left = 0.0
        else:
            ear_left = (v1_left + v2_left) / (2.0 * h_left)

        # --- Right Eye EAR ---
        re_c1 = get_pixel_pt(self.RIGHT_EYE_CORNERS[0])
        re_c2 = get_pixel_pt(self.RIGHT_EYE_CORNERS[1])
        re_v1_top = get_pixel_pt(self.RIGHT_EYE_VERTICAL_1[0])
        re_v1_bot = get_pixel_pt(self.RIGHT_EYE_VERTICAL_1[1])
        re_v2_top = get_pixel_pt(self.RIGHT_EYE_VERTICAL_2[0])
        re_v2_bot = get_pixel_pt(self.RIGHT_EYE_VERTICAL_2[1])

        h_right = self._euclidean_distance(re_c1, re_c2)
        v1_right = self._euclidean_distance(re_v1_top, re_v1_bot)
        v2_right = self._euclidean_distance(re_v2_top, re_v2_bot)

        if h_right < 1e-5:
            ear_right = 0.0
        else:
            ear_right = (v1_right + v2_right) / (2.0 * h_right)

        # Average EAR
        return (ear_left + ear_right) / 2.0

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
        Receives landmarks, computes EAR, and updates consecutive-frame state machine.

        Args:
            landmarks (List/np.ndarray): MediaPipe Face Mesh landmarks.
            width (int): Frame width.
            height (int): Frame height.
            tracking_confidence (float): Tracking confidence score from MediaPipe.
            timestamp (Optional[float]): Timestamp (in seconds) of the frame.
                                         Defaults to time.perf_counter() if None.
            active_challenge (Optional[str]): Active liveness challenge (e.g. "BLINK").

        Returns:
            Dict[str, Any]: Structured output.
                On failure:
                    {"success": False, "error": "FACE_NOT_DETECTED" | "LANDMARKS_UNSTABLE"}
                On success:
                    {
                        "success": True,
                        "ear": float,
                        "eye_state": "OPEN" | "CLOSED",
                        "blink_detected": bool,
                        "challenge_verified": bool,
                        "total_blinks": int
                    }
        """
        # ==========================================
        # 1. Face Presence & Stability Validation
        # ==========================================
        if landmarks is None or len(landmarks) < 400:
            return {"success": False, "error": "FACE_NOT_DETECTED"}
        if width <= 0 or height <= 0:
            return {"success": False, "error": "FACE_NOT_DETECTED"}

        # Ignored frames with low landmark tracking confidence
        if tracking_confidence < self.min_confidence:
            return {"success": False, "error": "LANDMARKS_UNSTABLE"}

        # Resolve timestamp
        if timestamp is None:
            timestamp = time.perf_counter()

        try:
            # ==========================================
            # 2. EAR Calculation
            # ==========================================
            ear = self.calculate_ear(landmarks, width, height)

            # ==========================================
            # 3. State Machine & Time Validation
            # ==========================================
            blink_detected = False
            challenge_verified = False

            if ear < self.ear_threshold:
                if self.blink_counter == 0:
                    # Record start timestamp when eyes first drop below threshold
                    self.t_start = timestamp
                
                self.blink_counter += 1
                if self.blink_counter >= self.min_consecutive_frames:
                    self.eye_state = "CLOSED"
            else:
                if self.eye_state == "CLOSED":
                    # Transition back to open: compute time duration
                    duration = timestamp - self.t_start
                    
                    # Validate against realistic human blink speed thresholds (50ms - 500ms)
                    if self.min_consecutive_frames <= self.blink_counter:
                        if self.min_blink_duration <= duration <= self.max_blink_duration:
                            blink_detected = True
                            self.total_blinks += 1
                            if active_challenge == "BLINK":
                                challenge_verified = True

                # Reset counters on opening
                self.blink_counter = 0
                self.t_start = 0.0
                self.eye_state = "OPEN"

            return {
                "success": True,
                "ear": ear,
                "eye_state": self.eye_state,
                "blink_detected": blink_detected,
                "challenge_verified": challenge_verified,
                "total_blinks": self.total_blinks
            }

        except Exception as e:
            return {"success": False, "error": f"BLINK_DETECTION_FAILURE: {str(e)}"}
