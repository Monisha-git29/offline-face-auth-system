"""
Liveness Decision Engine fusing active and passive liveness checks.

Combines ChallengeEngine (active Blink and Head Turn challenges) and
PassiveAntiSpoofEngine (LBP texture, scale-invariant depth, motion, and temporal stability)
into a cohesive session-based, fully offline liveness decision maker.
Includes adaptive weight adjustments, explicit fusion equations, and hard security overrides.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional

from .challenge import ChallengeEngine
from .antispoof import PassiveAntiSpoofEngine


class LivenessDecisionEngine:
    """
    Stateful Active + Passive Liveness Fusion Decision Engine.
    Fuses active challenges with passive sliding-window signals.
    """

    def __init__(
        self,
        preset: str = "MEDIUM",
        challenges: Optional[List[str]] = None,
        randomize: bool = False,
        timeout_per_challenge: float = 5.0,
        session_timeout: Optional[float] = None,
        validity_period: float = 300.0,
        passive_history_size: int = 12,
        passive_live_threshold: float = 0.75,
        passive_spoof_threshold: float = 0.40,
        movement_threshold: float = 1e-5,
        min_stall_frames: int = 3,
        min_reliability_size: int = 50,
        alpha_f: float = 0.50,  # Fusion balance: active_score vs passive_score
        smile_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the LivenessDecisionEngine.
        """
        # 1. Instantiate sub-engines
        self.challenge_engine = ChallengeEngine(
            preset=preset,
            challenges=challenges,
            randomize=randomize,
            timeout_per_challenge=timeout_per_challenge,
            session_timeout=session_timeout,
            validity_period=validity_period,
            movement_threshold=movement_threshold,
            min_stall_frames=min_stall_frames,
            smile_config=smile_config
        )

        self.antispoof_engine = PassiveAntiSpoofEngine(
            history_size=passive_history_size,
            live_threshold=passive_live_threshold,
            spoof_threshold=passive_spoof_threshold,
            movement_threshold=movement_threshold,
            min_stall_frames=min_stall_frames,
            min_reliability_size=min_reliability_size
        )

        self.alpha_f = alpha_f
        self.reset()

    def reset(self) -> None:
        """
        Resets both active and passive sub-engines, and local variables.
        """
        self.challenge_engine.reset()
        self.antispoof_engine.reset()
        self.session_id = self.challenge_engine.session_id
        self.status = "PENDING"
        self.error_code: Optional[str] = None
        self.verified_at: Optional[float] = None
        self.expires_at: Optional[float] = None

    def update(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        landmarks: Optional[Union[np.ndarray, List[Tuple[float, float, float]]]],
        tracking_confidence: float = 1.0,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes frame pixels and face landmarks, drives active liveness challenges,
        extracts passive texture/depth/motion metrics, applies hard security overrides,
        and produces the final fused decision.

        Args:
            frame (np.ndarray): Input BGR frame.
            bbox (Tuple): (x, y, w, h) bounding box.
            landmarks (List/np.ndarray): MediaPipe landmarks.
            tracking_confidence (float): Face tracking confidence.
            timestamp (Optional[float]): Precision timestamp in seconds.

        Returns:
            Dict[str, Any]: Fused liveness decision response.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        # Update Session ID locally
        self.session_id = self.challenge_engine.session_id

        # ==========================================
        # 1. Sub-Engine Process Routing
        # ==========================================
        # Get width and height of frame
        h_frame, w_frame, _ = frame.shape

        # A. Process Active Challenges
        # Pass landmarks to challenge engine
        active_res = self.challenge_engine.update(
            landmarks=landmarks,
            width=w_frame,
            height=h_frame,
            tracking_confidence=tracking_confidence,
            timestamp=timestamp
        )

        # Get head yaw for passive adaptive weighting
        yaw = self.challenge_engine.head_turn_detector.smoothed_yaw

        # B. Process Passive Anti-Spoofing
        passive_res = self.antispoof_engine.update(
            frame=frame,
            bbox=bbox,
            landmarks=landmarks,
            tracking_confidence=tracking_confidence,
            timestamp=timestamp,
            yaw=yaw
        )

        # ==========================================
        # 2. Hard Security Override Rules
        # ==========================================
        # Rule 1: Monotonicity / Stall Failures
        if not active_res["success"] and active_res.get("error") == "FRAME_STALLED":
            self.status = "FAILED"
            self.error_code = "FRAME_STALLED"
            return self._generate_fused_response(active_res, passive_res, timestamp)

        if not passive_res["success"] and passive_res.get("error") == "FRAME_STALLED":
            self.status = "FAILED"
            self.error_code = "FRAME_STALLED"
            return self._generate_fused_response(active_res, passive_res, timestamp)

        # Rule 2: Active Session Timeout
        if active_res["session_status"] == "TIMEOUT":
            self.status = "TIMEOUT"
            self.error_code = "TIMEOUT"
            return self._generate_fused_response(active_res, passive_res, timestamp)

        # Rule 3: Critical Face Quality Gating
        if not passive_res["success"] and passive_res.get("error") == "FACE_TOO_SMALL":
            self.status = "FAILED"
            self.error_code = "FACE_TOO_SMALL"
            return self._generate_fused_response(active_res, passive_res, timestamp)

        if not active_res["success"] and active_res.get("error") == "FACE_NOT_DETECTED":
            return self._generate_fused_response(active_res, passive_res, timestamp, error="FACE_NOT_DETECTED")

        if not active_res["success"] and active_res.get("error") == "LANDMARKS_UNSTABLE":
            return self._generate_fused_response(active_res, passive_res, timestamp, error="LANDMARKS_UNSTABLE")

        # Get status from active engine
        self.status = active_res["session_status"]

        # Rule 4: Critical Passive Failure Override (Flat screen/print bypass)
        # Even if active challenges are completed, if the passive anti-spoofing score
        # drops below the critical spoof threshold (0.40), we issue a hard spoof fail!
        passive_score = 1.0 - passive_res["spoof_score"]
        if passive_res["analytics"]["buffer_ready"] and passive_score < self.antispoof_engine.spoof_threshold:
            self.status = "FAILED"
            self.error_code = "CHALLENGE_NOT_COMPLETED"  # General rejection code
            passive_res["classification"] = "SPOOF"

        # Record liveness success timestamps
        if self.status == "SUCCESS" and self.error_code is None:
            self.verified_at = active_res["verified_at"]
            self.expires_at = active_res["expires_at"]

        return self._generate_fused_response(active_res, passive_res, timestamp)

    def _generate_fused_response(
        self,
        active_res: Dict[str, Any],
        passive_res: Dict[str, Any],
        timestamp: float,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Applies explicit fusion equations and yields a cohesive decision response.
        """
        # ==========================================
        # 1. Explicit Fusion Equations
        # ==========================================
        # A. Active liveness progress scoring
        # If successfully completed, active_score = 1.0. Otherwise scales with progress.
        if active_res["session_status"] == "SUCCESS":
            s_active = 1.0
        else:
            prog = active_res["challenge_progress"]
            total = len(prog)
            completed = sum(1 for p in prog if p)
            s_active = float(completed / total) if total > 0 else 0.0

        # B. Passive anti-spoofing liveness score
        s_passive = float(1.0 - passive_res["spoof_score"])

        # C. Combined Fused Liveness Score
        # We perform weighted linear fusion of active and passive liveness channels
        final_liveness_score = self.alpha_f * s_active + (1.0 - self.alpha_f) * s_passive

        # Hard overrides on score in case of terminal failures
        err_out = self.error_code if self.error_code is not None else error
        if self.status in ["FAILED", "TIMEOUT"]:
            final_liveness_score = 0.0
            s_active = 0.0
            s_passive = 0.0
            decision = "SPOOF"
        else:
            # Classification Decision
            if passive_res["classification"] == "SPOOF":
                decision = "SPOOF"
            elif passive_res["classification"] == "SUSPECT":
                decision = "SUSPECT"
            else:
                decision = "LIVE"

        # If success, double check passive signals are solid
        success_final = (self.status == "SUCCESS" and decision == "LIVE" and err_out is None)

        # Assemble analytics
        analytics = {
            "active_challenges_completed": active_res["analytics"]["completed_challenges"],
            "passive_signals": passive_res["signals"],
            "session_spoof_accumulator": passive_res["analytics"]["session_spoof_accumulator"],
            "average_tracking_confidence": active_res["analytics"]["average_tracking_confidence"],
            "passive_reliability": passive_res["analytics"]["tracking_reliability"],
            "challenge_history": active_res["analytics"]["challenge_history"],
            "total_duration_ms": active_res["analytics"]["total_duration_ms"]
        }

        return {
            "success": success_final,
            "session_id": self.session_id,
            "session_status": self.status,
            "decision": decision,
            "active_score": float(s_active),
            "passive_score": float(s_passive),
            "final_liveness_score": float(final_liveness_score),
            "verified_at": self.verified_at,
            "expires_at": self.expires_at,
            "analytics": analytics,
            "error": err_out
        }
