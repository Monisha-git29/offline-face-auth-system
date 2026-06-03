"""
Liveness SDK Coordinator - Production Integration Entry Point.

Wraps FaceQualityEngine, ChallengeEngine, PassiveAntiSpoofEngine, and LivenessDecisionEngine
into a single, clean API for React Native and Attendance punch-in integrations.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional

from .fqa import FaceQualityEngine
from .decision import LivenessDecisionEngine
from .config import PRESETS
from . import errors


class LivenessSDK:
    """
    Single entry point for the offline face liveness verification pipeline.
    """

    def __init__(self, preset: Union[str, Dict[str, Any]] = "MEDIUM"):
        """
        Initializes the LivenessSDK with a config dictionary or preset string name.

        Args:
            preset (str or dict): 'LOW_SECURITY', 'MEDIUM_SECURITY', 'HIGH_SECURITY',
                                  or custom config dict.
        """
        # Resolve config dict
        if isinstance(preset, str):
            preset_name = preset.upper()
            if preset_name not in PRESETS:
                # Fallback to check if name is 'LOW', 'MEDIUM', 'HIGH'
                if f"{preset_name}_SECURITY" in PRESETS:
                    preset_name = f"{preset_name}_SECURITY"
                else:
                    preset_name = "MEDIUM_SECURITY"
            self.config = PRESETS[preset_name]
        elif isinstance(preset, dict):
            self.config = preset
        else:
            self.config = PRESETS["MEDIUM_SECURITY"]

        # Parse config fields
        self.challenges = self.config.get("challenges", ["BLINK", "TURN_LEFT"])
        self.timeout_per_challenge = self.config.get("timeout_per_challenge", 5.0)
        self.session_timeout = self.config.get("session_timeout", 20.0)
        self.passive_history_size = self.config.get("passive_history_size", 12)
        self.passive_live_threshold = self.config.get("passive_live_threshold", 0.75)
        self.passive_spoof_threshold = self.config.get("passive_spoof_threshold", 0.40)
        self.min_face_size_target = self.config.get("min_face_size_target", 150.0)
        self.critical_score_threshold = self.config.get("critical_score_threshold", 20.0)
        self.min_confidence = self.config.get("min_confidence", 0.50)

        # 1. Instantiate FaceQualityEngine
        self.fqa_engine = FaceQualityEngine(
            min_face_size_target=self.min_face_size_target,
            critical_score_threshold=self.critical_score_threshold
        )

        # 2. Instantiate LivenessDecisionEngine
        self.decision_engine = LivenessDecisionEngine(
            preset=None,  # Set to None to respect custom challenges list
            challenges=self.challenges,
            randomize=False,
            timeout_per_challenge=self.timeout_per_challenge,
            session_timeout=self.session_timeout,
            passive_history_size=self.passive_history_size,
            passive_live_threshold=self.passive_live_threshold,
            passive_spoof_threshold=self.passive_spoof_threshold,
            min_reliability_size=int(self.min_face_size_target)
        )
        self.last_fqa_result = None
        self.mp_detector = None
        self.mp_model_path = self.config.get("mp_model_path", "face_recognition/face_landmarker.task")

    def reset(self) -> None:
        """
        Resets active challenge sequences, timers, and passive history buffers.
        """
        self.decision_engine.reset()
        self.last_fqa_result = None

    def close(self) -> None:
        """
        Releases background resources (e.g. MediaPipe FaceLandmarker).
        """
        if self.mp_detector is not None:
            self.mp_detector.close()
            self.mp_detector = None

    def process_frame(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[np.ndarray] = None,
        tracking_confidence: float = 1.0,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a single frame through FQA, Challenge verification, and Passive PAD.

        Args:
            frame (np.ndarray): Original BGR camera frame.
            bbox (Tuple[int, int, int, int]): Optional face bounding box (x, y, w, h).
            landmarks (np.ndarray): Optional MediaPipe landmarks array.
            tracking_confidence (float): Face tracking confidence (0.0 to 1.0).
            timestamp (float, optional): Precision epoch or frame timestamp.

        Returns:
            Dict[str, Any]: Standardized liveness decision response.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        # ==========================================
        # 1. Input Presence & Stability Gates
        # ==========================================
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return self._build_error_response(errors.FACE_NOT_DETECTED)

        # MediaPipe Face Mesh Auto-detection (default production path if missing)
        if landmarks is None or bbox is None:
            if self.mp_detector is None:
                from .landmarks import MediaPipeLandmarkDetector
                try:
                    self.mp_detector = MediaPipeLandmarkDetector(model_path=self.mp_model_path)
                except Exception:
                    return self._build_error_response(errors.FACE_NOT_DETECTED)
            
            res_det = self.mp_detector.detect_landmarks(frame)
            if res_det is None:
                return self._build_error_response(errors.FACE_NOT_DETECTED)
            bbox = res_det["face_bbox"]
            landmarks = res_det["landmarks"]

        if landmarks is None or len(landmarks) < 400:
            return self._build_error_response(errors.FACE_NOT_DETECTED)
        if bbox is None or len(bbox) != 4:
            return self._build_error_response(errors.FACE_NOT_DETECTED)
        
        bx, by, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            return self._build_error_response(errors.FACE_NOT_DETECTED)

        if tracking_confidence < self.min_confidence:
            return self._build_error_response(errors.LANDMARKS_UNSTABLE)

        # ==========================================
        # 2. Eye Coordinate Extraction for FQA
        # ==========================================
        img_h, img_w = frame.shape[:2]
        try:
            lms = np.array(landmarks, dtype=np.float32)
            # In the image space, 'left_eye' for the alignment/FQA engine must be the eye on the left side of the image (smaller x coordinate).
            # 'right_eye' must be the eye on the right side of the image (larger x coordinate).
            pt_a = (lms[33][:2] + lms[133][:2]) / 2.0
            pt_b = (lms[362][:2] + lms[263][:2]) / 2.0

            if pt_a[0] < pt_b[0]:
                left_img_pt = pt_a
                right_img_pt = pt_b
            else:
                left_img_pt = pt_b
                right_img_pt = pt_a

            left_eye = (float(left_img_pt[0] * img_w), float(left_img_pt[1] * img_h))
            right_eye = (float(right_img_pt[0] * img_w), float(right_img_pt[1] * img_h))
        except Exception:
            return self._build_error_response(errors.FACE_NOT_DETECTED)

        # ==========================================
        # 3. Face Quality Engine Execution
        # ==========================================
        fqa_res = self.fqa_engine.assess_raw_face(
            image=frame,
            left_eye=left_eye,
            right_eye=right_eye,
            face_bbox=bbox
        )
        self.last_fqa_result = fqa_res

        if not fqa_res.get("success"):
            err = fqa_res.get("error")
            if err == "INVALID_FACE_INPUT":
                return self._build_error_response(errors.FACE_NOT_DETECTED)
            elif err == "FACE_POORLY_POSITIONED":
                return self._build_error_response(errors.LANDMARKS_UNSTABLE)
            return self._build_error_response(errors.FACE_NOT_DETECTED)

        if fqa_res.get("recommendation") == "RECAPTURE":
            # Map quality checks failure to standardized error codes
            if fqa_res.get("face_size_score", 100) < self.critical_score_threshold:
                return self._build_error_response(errors.FACE_TOO_SMALL)
            if fqa_res.get("pose_score", 100) < self.critical_score_threshold:
                return self._build_error_response(errors.LANDMARKS_UNSTABLE)
            if fqa_res.get("blur_score", 100) < self.critical_score_threshold:
                return self._build_error_response(errors.LANDMARKS_UNSTABLE)
            return self._build_error_response(errors.CHALLENGE_NOT_COMPLETED)

        # ==========================================
        # 4. Fused Liveness Engine Execution
        # ==========================================
        decision_res = self.decision_engine.update(
            frame=frame,
            bbox=bbox,
            landmarks=landmarks,
            tracking_confidence=tracking_confidence,
            timestamp=timestamp
        )

        # Extract values
        success = decision_res.get("success", False)
        decision = decision_res.get("decision", "SPOOF")
        err = decision_res.get("error")

        # Map decision values
        if not success:
            if err is None:
                if decision == "SPOOF":
                    err = errors.PASSIVE_SPOOF_DETECTED
                else:
                    err = errors.CHALLENGE_NOT_COMPLETED
            elif err == "CHALLENGE_NOT_COMPLETED" and decision == "SPOOF":
                err = errors.PASSIVE_SPOOF_DETECTED

        # Map error to standardized output code
        if err is not None:
            valid_errors = [
                errors.FACE_NOT_DETECTED,
                errors.FACE_TOO_SMALL,
                errors.LANDMARKS_UNSTABLE,
                errors.FRAME_STALLED,
                errors.TIMEOUT,
                errors.CHALLENGE_NOT_COMPLETED,
                errors.PASSIVE_SPOOF_DETECTED
            ]
            if err not in valid_errors:
                if "SPOOF" in err:
                    err = errors.PASSIVE_SPOOF_DETECTED
                elif "TIMEOUT" in err:
                    err = errors.TIMEOUT
                elif "STALL" in err:
                    err = errors.FRAME_STALLED
                elif "SMALL" in err:
                    err = errors.FACE_TOO_SMALL
                elif "UNSTABLE" in err:
                    err = errors.LANDMARKS_UNSTABLE
                else:
                    err = errors.CHALLENGE_NOT_COMPLETED

        # Construct final unified response
        fused_score = float(decision_res.get("final_liveness_score", 0.0))
        
        current_challenge = None
        if self.decision_engine.challenge_engine.current_challenge_index < len(self.decision_engine.challenge_engine.challenges):
            current_challenge = self.decision_engine.challenge_engine.challenges[self.decision_engine.challenge_engine.current_challenge_index]

        return {
            "success": bool(success),
            "decision": decision,
            "trust_score": fused_score,
            "session_id": decision_res.get("session_id"),
            "current_challenge": current_challenge,
            "active_score": float(decision_res.get("active_score", 0.0)),
            "passive_score": float(decision_res.get("passive_score", 0.0)),
            "final_liveness_score": fused_score,
            "verified_at": decision_res.get("verified_at"),
            "expires_at": decision_res.get("expires_at"),
            "error": err
        }

    def _build_error_response(self, error_code: str) -> Dict[str, Any]:
        """
        Builds a standard SDK error response structure.
        """
        # Get active challenge name
        current_challenge = None
        if self.decision_engine.challenge_engine.current_challenge_index < len(self.decision_engine.challenge_engine.challenges):
            current_challenge = self.decision_engine.challenge_engine.challenges[self.decision_engine.challenge_engine.current_challenge_index]

        return {
            "success": False,
            "decision": "SPOOF",
            "trust_score": 0.0,
            "session_id": self.decision_engine.session_id,
            "current_challenge": current_challenge,
            "active_score": 0.0,
            "passive_score": 0.0,
            "final_liveness_score": 0.0,
            "verified_at": None,
            "expires_at": None,
            "error": error_code
        }
