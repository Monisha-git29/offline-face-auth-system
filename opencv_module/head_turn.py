"""
Head Turn Detection Module using MediaPipe Face Mesh.

This module provides a production-grade, lightweight, and fully-offline
Head Turn Detection engine using geometric yaw estimation from landmarks.
It incorporates aspect-ratio pixel scaling, Exponential Moving Average (EMA)
temporal smoothing, consecutive-frame verification, and cooldown gates to
provide robust head turn challenge checking for liveness verification.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional


class HeadTurnDetector:
    """
    Production-ready offline Head Turn Detector based on geometric facial symmetry.
    Optimized for React Native challenge-based liveness verification.
    """

    # Stable landmark indices in MediaPipe Face Mesh (468/478 points)
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    def __init__(
        self,
        yaw_threshold: float = 18.0,       # Degrees past which head is TURNED (LEFT > 18, RIGHT < -18)
        center_threshold: float = 8.0,      # Degrees under which head is CENTERED (|yaw| < 8)
        min_turn_frames: int = 3,           # Consecutive frames to verify turn
        min_center_frames: int = 3,         # Consecutive frames to verify return to center
        cooldown_duration: float = 0.3,     # Cooldown lock (seconds) after challenge completion
        min_confidence: float = 0.50,       # Track confidence threshold
        yaw_calibration: float = 80.0,      # Multiplication coefficient for yaw degrees mapping
        ema_alpha: float = 0.35             # EMA smoothing alpha
    ):
        """
        Initializes the HeadTurnDetector.
        """
        if yaw_threshold <= center_threshold:
            raise ValueError("yaw_threshold must be strictly greater than center_threshold.")
        if center_threshold <= 0.0:
            raise ValueError("center_threshold must be positive.")
        if min_turn_frames <= 0 or min_center_frames <= 0:
            raise ValueError("Frame counters must be positive integers.")
        if cooldown_duration < 0.0:
            raise ValueError("cooldown_duration must be non-negative.")

        self.yaw_threshold = yaw_threshold
        self.center_threshold = center_threshold
        self.min_turn_frames = min_turn_frames
        self.min_center_frames = min_center_frames
        self.cooldown_duration = cooldown_duration
        self.min_confidence = min_confidence
        self.yaw_calibration = yaw_calibration
        self.ema_alpha = ema_alpha

        # Session State Variables
        self.state = "IDLE"  # IDLE, FACE_CENTERED, TURN_LEFT_REQUESTED, TURN_RIGHT_REQUESTED, LEFT_VERIFIED, RIGHT_VERIFIED, RETURN_TO_CENTER, CHALLENGE_COMPLETE
        self.smoothed_yaw = 0.0
        self.has_init_ema = False
        
        self.turn_frame_counter = 0
        self.center_frame_counter = 0
        self.t_last_complete = 0.0

    def reset(self) -> None:
        """
        Resets internal session counters, state machine, and EMA filters.
        """
        self.state = "IDLE"
        self.smoothed_yaw = 0.0
        self.has_init_ema = False
        self.turn_frame_counter = 0
        self.center_frame_counter = 0
        self.t_last_complete = 0.0

    def calculate_yaw(
        self,
        landmarks: Union[np.ndarray, List[Tuple[float, float, float]]],
        width: int,
        height: int
    ) -> float:
        """
        Calculates the horizontal geometric facial symmetry ratio and maps it to degrees.

        Args:
            landmarks (List/np.ndarray): 468/478 normalized coordinates.
            width (int): Frame width.
            height (int): Frame height.

        Returns:
            float: Yaw in degrees (positive is LEFT, negative is RIGHT).
        """
        lms = np.array(landmarks, dtype=np.float32)

        # Helper to convert normalized coordinate to absolute pixel coordinate x
        def get_pixel_x(idx):
            return lms[idx][0] * width

        # 1. Left Eye Center (midpoint of corners)
        x_le = (get_pixel_x(self.LEFT_EYE_OUTER) + get_pixel_x(self.LEFT_EYE_INNER)) / 2.0
        
        # 2. Right Eye Center (midpoint of corners)
        x_re = (get_pixel_x(self.RIGHT_EYE_INNER) + get_pixel_x(self.RIGHT_EYE_OUTER)) / 2.0
        
        # 3. Nose Tip
        x_n = get_pixel_x(self.NOSE_TIP)

        # 4. Spans (always positive horizontal spans)
        dx_l = abs(x_le - x_n)
        dx_r = abs(x_re - x_n)

        # 5. Symmetry Ratio Deviation (R)
        denom = dx_l + dx_r
        if denom < 1e-5:
            return 0.0

        R = (dx_l - dx_r) / denom

        # Convert to degrees
        return float(R * self.yaw_calibration)

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
        Processes frame landmarks, applies EMA filtering, and drives the state machine.

        Args:
            landmarks (List/np.ndarray): MediaPipe Face Mesh landmarks.
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.
            tracking_confidence (float): Landmark tracking confidence score (0.0 to 1.0).
            timestamp (Optional[float]): Precision timestamp in seconds.
            active_challenge (Optional[str]): Target pose challenge ("TURN_LEFT" or "TURN_RIGHT").

        Returns:
            Dict[str, Any]: Structured output.
        """
        # ==========================================
        # 1. Validation Gates
        # ==========================================
        if landmarks is None or len(landmarks) < 400:
            return {"success": False, "error": "FACE_NOT_DETECTED"}
        if width <= 0 or height <= 0:
            return {"success": False, "error": "FACE_NOT_DETECTED"}

        # Reject if landmark tracking confidence is low
        if tracking_confidence < self.min_confidence:
            return {"success": False, "error": "LANDMARKS_UNSTABLE"}

        # Resolve timestamp
        if timestamp is None:
            timestamp = time.perf_counter()

        try:
            # ==========================================
            # 2. Geometric Yaw Calculation & EMA Smoothing
            # ==========================================
            raw_yaw = self.calculate_yaw(landmarks, width, height)

            if not self.has_init_ema:
                self.smoothed_yaw = raw_yaw
                self.has_init_ema = True
            else:
                self.smoothed_yaw = (self.ema_alpha * raw_yaw) + ((1.0 - self.ema_alpha) * self.smoothed_yaw)

            # Classify head direction based on smoothed yaw
            if abs(self.smoothed_yaw) < self.center_threshold:
                direction = "CENTER"
            else:
                direction = "LEFT" if self.smoothed_yaw > 0 else "RIGHT"

            # ==========================================
            # 3. Liveness State Machine Gates
            # ==========================================
            challenge_verified = False

            # A. Cooldown Lock Release Check
            if self.state == "CHALLENGE_COMPLETE":
                if timestamp - self.t_last_complete >= self.cooldown_duration:
                    self.state = "IDLE"  # Release cooldown

            # B. State Transitions
            if self.state == "IDLE":
                if direction == "CENTER":
                    self.state = "FACE_CENTERED"
                    self.center_frame_counter = 0

            elif self.state == "FACE_CENTERED":
                if active_challenge == "TURN_LEFT":
                    self.state = "TURN_LEFT_REQUESTED"
                    self.turn_frame_counter = 0
                elif active_challenge == "TURN_RIGHT":
                    self.state = "TURN_RIGHT_REQUESTED"
                    self.turn_frame_counter = 0
                elif direction != "CENTER":
                    self.state = "IDLE"  # User turned away before prompt started

            elif self.state == "TURN_LEFT_REQUESTED":
                # Check target challenge changed
                if active_challenge != "TURN_LEFT":
                    self.state = "IDLE"
                elif self.smoothed_yaw >= self.yaw_threshold:
                    self.turn_frame_counter += 1
                    if self.turn_frame_counter >= self.min_turn_frames:
                        self.state = "LEFT_VERIFIED"
                else:
                    self.turn_frame_counter = 0

            elif self.state == "TURN_RIGHT_REQUESTED":
                if active_challenge != "TURN_RIGHT":
                    self.state = "IDLE"
                elif self.smoothed_yaw <= -self.yaw_threshold:
                    self.turn_frame_counter += 1
                    if self.turn_frame_counter >= self.min_turn_frames:
                        self.state = "RIGHT_VERIFIED"
                else:
                    self.turn_frame_counter = 0

            if self.state in ["LEFT_VERIFIED", "RIGHT_VERIFIED"]:
                # Transition to return lock
                self.state = "RETURN_TO_CENTER"
                self.center_frame_counter = 0

            elif self.state == "RETURN_TO_CENTER":
                # Enforce that user returns to center with consecutive frames
                if direction == "CENTER":
                    self.center_frame_counter += 1
                    if self.center_frame_counter >= self.min_center_frames:
                        self.state = "CHALLENGE_COMPLETE"
                        self.t_last_complete = timestamp
                        challenge_verified = True
                else:
                    self.center_frame_counter = 0

            elif self.state == "CHALLENGE_COMPLETE":
                # Keep returning success for the frame completing
                challenge_verified = True

            return {
                "success": True,
                "yaw": float(self.smoothed_yaw),
                "direction": direction,
                "challenge_verified": challenge_verified,
                "tracking_confidence": float(tracking_confidence),
                "timestamp": float(timestamp),
                "state": self.state
            }

        except Exception as e:
            return {"success": False, "error": f"HEAD_TURN_FAILURE: {str(e)}"}
