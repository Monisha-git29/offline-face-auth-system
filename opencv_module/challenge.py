"""
Unified Challenge Engine for Active Liveness Detection.

Coordinates active liveness challenges (such as blink detection and head turn detection)
into a cohesive, offline challenge-based liveness verification system.
Includes unique session IDs, result expiry support, challenge difficulty presets,
anti-repetition logic, frame freshness checks, explicit return-to-center verification,
expanded error codes, and session analytics.
"""

import uuid
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional

from .blink import BlinkDetector
from .head_turn import HeadTurnDetector
from .smile import SmileDetector


class ChallengeEngine:
    """
    Orchestrates sequential or randomized active liveness challenges.
    Optimized for fully offline mobile deployment.
    """

    # Difficulty Presets
    PRESETS = {
        "EASY": ["BLINK"],
        "MEDIUM": ["BLINK", "TURN_LEFT"],
        "HIGH": ["BLINK", "SMILE", "TURN_LEFT", "TURN_RIGHT"]
    }

    def __init__(
        self,
        preset: Optional[str] = None,
        challenges: Optional[List[str]] = None,
        randomize: bool = False,
        timeout_per_challenge: float = 5.0,
        session_timeout: Optional[float] = None,
        validity_period: float = 300.0,
        movement_threshold: float = 1e-5,
        min_stall_frames: int = 3,
        blink_config: Optional[Dict[str, Any]] = None,
        head_turn_config: Optional[Dict[str, Any]] = None,
        smile_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the ChallengeEngine.

        Args:
            preset (str, optional): EASY, MEDIUM, or HIGH. If set, ignores challenges arg.
            challenges (List[str], optional): Custom list of challenges, e.g. ["BLINK", "TURN_LEFT"].
            randomize (bool): If True, shuffles the challenge sequence.
            timeout_per_challenge (float): Max time (seconds) to complete an individual challenge.
            session_timeout (float, optional): Max time (seconds) for the entire session.
                                               Defaults to max(15.0, len(challenges) * timeout_per_challenge).
            validity_period (float): Number of seconds before the successful verification expires.
            movement_threshold (float): Minimum average landmark displacement to prevent FRAME_STALLED.
            min_stall_frames (int): Consecutive frames below movement_threshold to trigger FRAME_STALLED.
            blink_config (dict, optional): Custom config arguments to pass to BlinkDetector.
            head_turn_config (dict, optional): Custom config arguments to pass to HeadTurnDetector.
        """
        # Resolve challenges
        if preset is not None:
            preset_upper = preset.upper()
            if preset_upper not in self.PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(self.PRESETS.keys())}")
            base_challenges = list(self.PRESETS[preset_upper])
        elif challenges is not None:
            if not challenges:
                raise ValueError("Challenges list cannot be empty.")
            for c in challenges:
                if c not in ["BLINK", "TURN_LEFT", "TURN_RIGHT", "SMILE"]:
                    raise ValueError(f"Invalid challenge type '{c}'. Supported: 'BLINK', 'TURN_LEFT', 'TURN_RIGHT', 'SMILE'")
            base_challenges = list(challenges)
        else:
            # Default to MEDIUM if nothing is provided
            base_challenges = list(self.PRESETS["MEDIUM"])

        # Config validation
        if timeout_per_challenge <= 0.0:
            raise ValueError("timeout_per_challenge must be positive.")
        if validity_period <= 0.0:
            raise ValueError("validity_period must be positive.")
        if movement_threshold < 0.0:
            raise ValueError("movement_threshold must be non-negative.")
        if min_stall_frames <= 0:
            raise ValueError("min_stall_frames must be positive.")

        self.base_challenges = base_challenges
        self.randomize = randomize
        self.timeout_per_challenge = timeout_per_challenge
        self.validity_period = validity_period
        self.movement_threshold = movement_threshold
        self.min_stall_frames = min_stall_frames

        # Instantiation parameters for detectors
        b_conf = blink_config or {}
        h_conf = head_turn_config or {}
        s_conf = smile_config or {}

        self.blink_detector = BlinkDetector(**b_conf)
        self.head_turn_detector = HeadTurnDetector(**h_conf)
        self.smile_detector = SmileDetector(**s_conf)

        # Track the last generated sequence to enforce anti-repetition logic
        self._last_sequence: Optional[List[str]] = None

        # Resolve session timeout
        if session_timeout is not None:
            if session_timeout <= 0.0:
                raise ValueError("session_timeout must be positive.")
            self.session_timeout = session_timeout
        else:
            # Default session timeout logic
            self.session_timeout = max(15.0, len(self.base_challenges) * self.timeout_per_challenge)

        # Initialize/Reset session states
        self.reset()

    def reset(self) -> None:
        """
        Resets the liveness session, generates a new session_id,
        applies anti-repetition logic if randomized, and resets internal timers/states.
        """
        # Session Core
        self.session_id = uuid.uuid4().hex
        self.status = "PENDING"  # PENDING, IN_PROGRESS, SUCCESS, FAILED, TIMEOUT
        self.error_code: Optional[str] = None

        # Challenge sequence setup (with anti-repetition logic)
        self.challenges = list(self.base_challenges)
        if self.randomize:
            # Keep generating shuffles until it differs from the last run (if length > 1)
            attempts = 0
            while len(self.challenges) > 1 and attempts < 10:
                np.random.shuffle(self.challenges)
                if self.challenges != self._last_sequence:
                    break
                attempts += 1
        
        self._last_sequence = list(self.challenges)
        self.current_challenge_index = 0
        self.completed_challenges = [False] * len(self.challenges)

        # Timers
        self.session_start_time: Optional[float] = None
        self.challenge_start_time: Optional[float] = None
        self.verified_at: Optional[float] = None
        self.expires_at: Optional[float] = None

        # Frame Freshness & Replay Detection State
        self.last_landmarks: Optional[np.ndarray] = None
        self.last_timestamp: Optional[float] = None
        self.stall_frame_counter = 0

        # Analytics
        self.failed_attempts = 0
        self.total_frames_processed = 0
        self.running_confidence_sum = 0.0
        self.challenge_history: List[Dict[str, Any]] = []

        # Reset underlying detectors
        self.blink_detector.reset()
        self.head_turn_detector.reset()
        self.smile_detector.reset()

        # Internal trackers to avoid double-counting attempts
        self._wrong_turn_in_frame = False
        self._blink_overlong_in_frame = False

    def update(
        self,
        landmarks: Optional[Union[np.ndarray, List[Tuple[float, float, float]]]],
        width: int,
        height: int,
        tracking_confidence: float = 1.0,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a camera frame, validates frame freshness, routes landmarks to the active
        liveness detector, and updates challenge timers and state machines.

        Args:
            landmarks (List/np.ndarray): MediaPipe Face Mesh landmarks.
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.
            tracking_confidence (float): MediaPipe face tracking confidence (0.0 to 1.0).
            timestamp (Optional[float]): Precision frame timestamp in seconds.

        Returns:
            Dict[str, Any]: Structured liveness verification state output.
        """
        # Resolve timestamp
        if timestamp is None:
            timestamp = time.perf_counter()

        # ==========================================
        # 1. Terminal State Check
        # ==========================================
        if self.status in ["SUCCESS", "FAILED", "TIMEOUT"]:
            return self._generate_response(timestamp)

        # Initialize timers on first frame
        if self.session_start_time is None:
            self.session_start_time = timestamp
            self.challenge_start_time = timestamp
            self.status = "IN_PROGRESS"

        self.total_frames_processed += 1
        self.running_confidence_sum += tracking_confidence

        # ==========================================
        # 2. Frame Freshness & Replay Checks
        # ==========================================
        # A. Monotonicity validation
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            self.status = "FAILED"
            self.error_code = "FRAME_STALLED"
            return self._generate_response(timestamp)

        # B. Average landmark displacement check
        if landmarks is not None and len(landmarks) >= 400:
            current_lms = np.array(landmarks, dtype=np.float32)
            if self.last_landmarks is not None:
                # Compute L2 distances for 2D coords
                diffs = np.linalg.norm(current_lms[:, :2] - self.last_landmarks[:, :2], axis=1)
                avg_displacement = float(np.mean(diffs))

                if avg_displacement < self.movement_threshold:
                    self.stall_frame_counter += 1
                    if self.stall_frame_counter >= self.min_stall_frames:
                        self.status = "FAILED"
                        self.error_code = "FRAME_STALLED"
                        return self._generate_response(timestamp)
                else:
                    self.stall_frame_counter = 0
            
            self.last_landmarks = current_lms
        
        self.last_timestamp = timestamp

        # ==========================================
        # 3. Timeout Gates
        # ==========================================
        # A. Total session timeout check
        elapsed_session = timestamp - self.session_start_time
        if elapsed_session > self.session_timeout:
            self.status = "TIMEOUT"
            self.error_code = "TIMEOUT"
            return self._generate_response(timestamp)

        # B. Active challenge timeout check
        elapsed_challenge = timestamp - self.challenge_start_time
        if elapsed_challenge > self.timeout_per_challenge:
            self.status = "TIMEOUT"
            self.error_code = "TIMEOUT"
            return self._generate_response(timestamp)

        # ==========================================
        # 4. Face Presence & Stability Gates
        # ==========================================
        if landmarks is None or len(landmarks) < 400 or width <= 0 or height <= 0:
            return self._generate_response(timestamp, error="FACE_NOT_DETECTED")

        if tracking_confidence < 0.50:
            return self._generate_response(timestamp, error="LANDMARKS_UNSTABLE")

        # ==========================================
        # 5. Route Landmarks to Active Detector
        # ==========================================
        current_challenge = self.challenges[self.current_challenge_index]
        challenge_verified = False
        detector_meta = {}

        if current_challenge == "BLINK":
            res = self.blink_detector.update(
                landmarks=landmarks,
                width=width,
                height=height,
                tracking_confidence=tracking_confidence,
                timestamp=timestamp,
                active_challenge="BLINK"
            )
            detector_meta = res
            
            if res.get("success"):
                challenge_verified = res.get("challenge_verified", False)
                # Analytics failed attempt check: if eyes closed for too long
                if self.blink_detector.eye_state == "CLOSED":
                    duration = timestamp - self.blink_detector.t_start
                    if duration > self.blink_detector.max_blink_duration:
                        if not self._blink_overlong_in_frame:
                            self.failed_attempts += 1
                            self._blink_overlong_in_frame = True
                else:
                    self._blink_overlong_in_frame = False
            else:
                # Map detector errors
                return self._generate_response(timestamp, error=res.get("error"))

        elif current_challenge in ["TURN_LEFT", "TURN_RIGHT"]:
            res = self.head_turn_detector.update(
                landmarks=landmarks,
                width=width,
                height=height,
                tracking_confidence=tracking_confidence,
                timestamp=timestamp,
                active_challenge=current_challenge
            )
            detector_meta = res

            if res.get("success"):
                challenge_verified = res.get("challenge_verified", False)
                
                # Analytics failed attempt check: user turned in the wrong direction
                yaw = res.get("yaw", 0.0)
                is_wrong_direction = False
                if current_challenge == "TURN_LEFT" and yaw <= -self.head_turn_detector.yaw_threshold:
                    is_wrong_direction = True
                elif current_challenge == "TURN_RIGHT" and yaw >= self.head_turn_detector.yaw_threshold:
                    is_wrong_direction = True

                if is_wrong_direction:
                    if not self._wrong_turn_in_frame:
                        self.failed_attempts += 1
                        self._wrong_turn_in_frame = True
                else:
                    self._wrong_turn_in_frame = False
            else:
                # Map detector errors
                return self._generate_response(timestamp, error=res.get("error"))

        elif current_challenge == "SMILE":
            res = self.smile_detector.update(
                landmarks=landmarks,
                width=width,
                height=height,
                tracking_confidence=tracking_confidence,
                timestamp=timestamp,
                active_challenge="SMILE"
            )
            detector_meta = res

            if res.get("success"):
                challenge_verified = res.get("challenge_verified", False)
            else:
                return self._generate_response(timestamp, error=res.get("error"))

        # ==========================================
        # 6. Process Challenge Completion
        # ==========================================
        if challenge_verified:
            # 1. Record completed challenge analytics history
            duration_ms = (timestamp - self.challenge_start_time) * 1000.0
            self.challenge_history.append({
                "challenge": current_challenge,
                "started_at": float(self.challenge_start_time),
                "completed_at": float(timestamp),
                "duration_ms": float(duration_ms)
            })

            # 2. Mark complete
            self.completed_challenges[self.current_challenge_index] = True
            self.current_challenge_index += 1

            # 3. Transition or Reset
            if self.current_challenge_index >= len(self.challenges):
                # Entire sequence completed!
                self.status = "SUCCESS"
                self.verified_at = timestamp
                self.expires_at = timestamp + self.validity_period
            else:
                # Reset timers and state machines for the next challenge
                self.challenge_start_time = timestamp
                self.blink_detector.reset()
                self.head_turn_detector.reset()
                self.smile_detector.reset()
                self._wrong_turn_in_frame = False
                self._blink_overlong_in_frame = False

        return self._generate_response(timestamp, challenge_verified=challenge_verified, detector_meta=detector_meta)

    def _generate_response(
        self,
        timestamp: float,
        challenge_verified: bool = False,
        error: Optional[str] = None,
        detector_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assembles a standardized state dictionary response.
        """
        # Resolve current active challenge name
        current_challenge = None
        if self.status == "IN_PROGRESS" and self.current_challenge_index < len(self.challenges):
            current_challenge = self.challenges[self.current_challenge_index]

        # Calculate session elapsed duration
        duration_ms = 0.0
        if self.session_start_time is not None:
            duration_ms = (timestamp - self.session_start_time) * 1000.0

        # Calculate average tracking confidence
        avg_confidence = 1.0
        if self.total_frames_processed > 0:
            avg_confidence = self.running_confidence_sum / self.total_frames_processed

        # Resolve completed challenge names
        completed_names = [
            self.challenges[i] for i in range(len(self.challenges)) if self.completed_challenges[i]
        ]

        # Analytics structure
        analytics = {
            "total_duration_ms": float(duration_ms),
            "completed_challenges": completed_names,
            "failed_attempts": self.failed_attempts,
            "average_tracking_confidence": float(avg_confidence),
            "challenge_history": list(self.challenge_history)
        }

        # Success boolean in output refers to the whole active session status
        session_success = (self.status == "SUCCESS")

        # Handle mapped error codes
        err_out = self.error_code if self.error_code is not None else error
        if err_out is None and self.status in ["TIMEOUT", "FAILED"] and self.error_code is None:
            err_out = "CHALLENGE_NOT_COMPLETED"

        return {
            "success": session_success,
            "session_id": self.session_id,
            "session_status": self.status,
            "current_challenge": current_challenge,
            "challenge_progress": list(self.completed_challenges),
            "challenge_verified": challenge_verified,
            "verified_at": float(self.verified_at) if self.verified_at is not None else None,
            "expires_at": float(self.expires_at) if self.expires_at is not None else None,
            "analytics": analytics,
            "error": err_out,
            "detector_meta": detector_meta or {}
        }
