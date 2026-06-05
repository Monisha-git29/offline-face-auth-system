"""
Passive Anti-Spoofing Engine for Offline Facial Authentication.

Implements Local Binary Pattern (LBP) texture analysis, temporal stability,
scale-invariant nose-to-eye depth ratios, landmark/bounding-box motion,
reliability gates, and session-level spoof accumulation.
"""

import time
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Union, Optional


class PassiveAntiSpoofEngine:
    """
    Stateful Multi-Frame Passive Anti-Spoofing Engine.
    Optimized for fully offline mobile deployment.
    """

    def __init__(
        self,
        history_size: int = 12,
        live_threshold: float = 0.75,
        spoof_threshold: float = 0.40,
        movement_threshold: float = 1e-5,
        min_stall_frames: int = 3,
        min_reliability_size: int = 50,
        beta: float = 0.85  # Session spoof accumulator decay factor
    ):
        """
        Initializes the PassiveAntiSpoofEngine.

        Args:
            history_size (int): Size of sliding rolling history buffer window (10-15).
            live_threshold (float): Minimum score to classify as LIVE.
            spoof_threshold (float): Maximum score below which is classified as SPOOF.
            movement_threshold (float): Variance threshold to check frozen frame stalls.
            min_stall_frames (int): Consecutive stalled frames before marking failed.
            min_reliability_size (int): Bounding box width/height below which FQA/reliability fails.
            beta (float): EMA coefficient for temporal session spoof accumulator.
        """
        if history_size < 3:
            raise ValueError("history_size must be at least 3 to perform temporal analysis.")
        if live_threshold <= spoof_threshold:
            raise ValueError("live_threshold must be strictly greater than spoof_threshold.")
        if min_reliability_size <= 0:
            raise ValueError("min_reliability_size must be positive.")

        self.history_size = history_size
        self.live_threshold = live_threshold
        self.spoof_threshold = spoof_threshold
        self.movement_threshold = movement_threshold
        self.min_stall_frames = min_stall_frames
        self.min_reliability_size = min_reliability_size
        self.beta = beta

        # Constants for Scoring
        self.H_TARGET = 6.0       # Target skin LBP entropy
        self.H_SIGMA = 0.8        # Sigma for skin entropy distribution
        self.R_TARGET = 0.30      # Target 3D nose-to-eye protrusion ratio
        self.R_SIGMA = 0.08       # Sigma for depth protrusion ratio
        self.STD_TARGET = 0.0035  # Target temporal histogram standard deviation

        self.reset()

    def reset(self) -> None:
        """
        Resets history buffers, session accumulators, and trackers.
        """
        # History buffers
        self.lbp_histograms: List[np.ndarray] = []
        self.entropies: List[float] = []
        self.depth_ratios: List[float] = []
        self.box_centers: List[Tuple[float, float]] = []
        self.landmark_centers: List[Tuple[float, float]] = []
        
        # Frame Freshness & Replay check trackers
        self.last_landmarks: Optional[np.ndarray] = None
        self.last_timestamp: Optional[float] = None
        self.stall_frame_counter = 0

        # Session accumulator liveness/spoof tracking
        self.session_spoof_accumulator = 0.0
        self.total_frames_processed = 0

    def compute_lbp_entropy(self, gray_crop: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculates high-performance, vectorized Local Binary Pattern (LBP) texture
        representation on a standardized 112x112 grayscale crop, and returns its Shannon entropy.
        """
        # Downsample to a standard 112x112 resolution for scale-invariant texture statistics
        img = cv2.resize(gray_crop, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        h, w = img.shape

        # Vectorized 3x3 LBP implementation
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        center = img[1:-1, 1:-1]

        # 8-neighborhood pixel shifts
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1),    (1, 1),   (1, 0),
            (1, -1),   (0, -1)
        ]

        for i, (dy, dx) in enumerate(offsets):
            neighbor = img[1 + dy: h - 1 + dy, 1 + dx: w - 1 + dx]
            lbp += ((neighbor >= center) * (1 << i)).astype(np.uint8)

        # Compute normalized histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-7)

        # Compute Shannon entropy
        p = hist_norm[hist_norm > 0]
        entropy = float(-np.sum(p * np.log2(p)))

        return entropy, hist_norm

    def get_adaptive_weights(self, bbox_width: int, yaw: float) -> Dict[str, float]:
        """
        Calculates dynamic weights based on face scale and rotation angle
        to adaptively privilege more reliable signal channels in extreme edge cases.
        """
        # Base weight distributions: equal high-profile channels
        weights = {"texture": 0.30, "depth": 0.30, "motion": 0.20, "temporal": 0.20}

        # 1. Bounding box size: if crop is low-resolution, texture detail degrades
        if bbox_width < 80:
            weights["texture"] = 0.10
            weights["temporal"] = 0.15
            weights["depth"] = 0.45
            weights["motion"] = 0.30

        # 2. Bounding box size: extreme low res
        elif bbox_width < 60:
            weights["texture"] = 0.05
            weights["temporal"] = 0.10
            weights["depth"] = 0.50
            weights["motion"] = 0.35

        # 3. Extreme Yaw: depth protrusion and motion can degrade
        if abs(yaw) > 15.0:
            weights["depth"] = 0.15
            weights["motion"] = 0.15
            weights["texture"] = 0.40
            weights["temporal"] = 0.30

        # Normalize weights so they sum to 1.0
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def update(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        landmarks: Optional[Union[np.ndarray, List[Tuple[float, float, float]]]],
        tracking_confidence: float = 1.0,
        timestamp: Optional[float] = None,
        yaw: float = 0.0
    ) -> Dict[str, Any]:
        """
        Updates sliding windows with current frame details, validates frame freshness,
        computes scale-invariant depth and texture signals, and runs the spoof accumulator.

        Args:
            frame (np.ndarray): Full BGR image frame.
            bbox (Tuple): (x, y, w, h) bounding box coordinates.
            landmarks (List/np.ndarray): MediaPipe 3D landmark points.
            tracking_confidence (float): Face tracking confidence from MediaPipe.
            timestamp (Optional[float]): Precision epoch/perf timestamp in seconds.
            yaw (float): Smoothed head yaw in degrees.

        Returns:
            Dict[str, Any]: Structured anti-spoofing verdict.
        """
        # Resolve timestamp
        if timestamp is None:
            timestamp = time.perf_counter()

        self.total_frames_processed += 1

        # ==========================================
        # 1. Bounding Box & Landmarks Validation
        # ==========================================
        if landmarks is None or len(landmarks) < 400:
            return self._generate_error_response("FACE_NOT_DETECTED", timestamp)

        x, y, w, h = bbox
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return self._generate_error_response("FACE_NOT_DETECTED", timestamp)

        # Face-size Reliability Gating
        if w < self.min_reliability_size or h < self.min_reliability_size:
            return self._generate_error_response("FACE_TOO_SMALL", timestamp)

        # Monotonicity Frame Freshness check
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return self._generate_error_response("FRAME_STALLED", timestamp)

        # ==========================================
        # 2. Replay Frozen Frame / Stall Verification
        # ==========================================
        current_lms = np.array(landmarks, dtype=np.float32)
        if self.last_landmarks is not None:
            # L2 Euclidean displacement of 2D coordinates across all landmarks
            diffs = np.linalg.norm(current_lms[:, :2] - self.last_landmarks[:, :2], axis=1)
            avg_displacement = float(np.mean(diffs))

            if avg_displacement < self.movement_threshold:
                self.stall_frame_counter += 1
                if self.stall_frame_counter >= self.min_stall_frames:
                    return self._generate_error_response("FRAME_STALLED", timestamp)
            else:
                self.stall_frame_counter = 0

        self.last_landmarks = current_lms
        self.last_timestamp = timestamp

        try:
            # ==========================================
            # 3. Calculate Immediate Frame Signals
            # ==========================================
            # A. Scale-Invariant Depth Ratio
            # Indices: Left Eye centers corners 33, 133; Right Eye corners 362, 263; Nose Tip 1
            left_eye_3d = (current_lms[33] + current_lms[133]) / 2.0
            right_eye_3d = (current_lms[362] + current_lms[263]) / 2.0
            eyes_mid_3d = (left_eye_3d + right_eye_3d) / 2.0
            nose_3d = current_lms[1]

            # 3D Euclidean distances
            d_eyes = float(np.linalg.norm(left_eye_3d - right_eye_3d))
            dz_nose = float(abs(nose_3d[2] - eyes_mid_3d[2]))

            r_depth = dz_nose / (d_eyes + 1e-7)

            # B. Crop Face & LBP Texture Extraction
            # Adjust crop boundary safely to match frame dims
            h_frame, w_frame, _ = frame.shape
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_frame, x + w), min(h_frame, y + h)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return self._generate_error_response("FACE_NOT_DETECTED", timestamp)

            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            entropy, hist_norm = self.compute_lbp_entropy(gray_crop)

            # C. Track Bounding Box and Landmarks center
            c_box = (float(x + w / 2.0), float(y + h / 2.0))
            c_lms = (float(np.mean(current_lms[:, 0]) * w_frame), float(np.mean(current_lms[:, 1]) * h_frame))

            # ==========================================
            # 4. Slide rolling window buffers
            # ==========================================
            self.lbp_histograms.append(hist_norm)
            self.entropies.append(entropy)
            self.depth_ratios.append(r_depth)
            self.box_centers.append(c_box)
            self.landmark_centers.append(c_lms)

            # Cap buffers to window size
            if len(self.lbp_histograms) > self.history_size:
                self.lbp_histograms.pop(0)
                self.entropies.pop(0)
                self.depth_ratios.pop(0)
                self.box_centers.pop(0)
                self.landmark_centers.pop(0)

            # Wait until buffer is full before issuing liveness assessments
            if len(self.lbp_histograms) < self.history_size:
                confidence = float(tracking_confidence * (1.0 - min(0.3, abs(yaw) / 90.0)) * (min(w, 200.0) / 200.0))
                return {
                    "success": True,
                    "spoof_score": 1.0,
                    "classification": "SUSPECT",
                    "signals": {"texture_score": 1.0, "depth_score": 1.0, "motion_score": 1.0, "temporal_score": 1.0},
                    "analytics": {
                        "session_spoof_accumulator": 0.0,
                        "buffer_ready": False,
                        "lbp_entropy": entropy,
                        "nose_depth_dz": dz_nose,
                        "scale_depth_ratio": r_depth,
                        "tracking_reliability": confidence
                    },
                    "error": None
                }

            # ==========================================
            # 5. Evaluate Windowed Signals
            # ==========================================
            M = len(self.lbp_histograms)

            # A. Texture Score (Average Window LBP Entropy)
            avg_entropy = float(np.mean(self.entropies))
            s_texture = float(np.exp(-((avg_entropy - self.H_TARGET) ** 2) / (2.0 * (self.H_SIGMA ** 2))))

            # B. Depth Score (Average Depth Ratio)
            avg_r_depth = float(np.mean(self.depth_ratios))
            s_depth = float(np.exp(-((avg_r_depth - self.R_TARGET) ** 2) / (2.0 * (self.R_SIGMA ** 2))))

            # C. Temporal Stability Score (Histogram variances over time)
            hist_matrix = np.array(self.lbp_histograms)  # (M, 256)
            bin_stds = np.std(hist_matrix, axis=0)      # std per bin
            mean_std = float(np.mean(bin_stds))          # average bin volatility

            # Penalize static freezes and screen refresh flares/fluctuations
            s_temporal = float(np.exp(-((np.log(mean_std + 1e-7) - np.log(self.STD_TARGET)) ** 2) / (2.0 * (1.2 ** 2))))

            # D. Motion Consistency (Landmark movement variances vs Box centers)
            box_movements = np.array(self.box_centers)
            lms_movements = np.array(self.landmark_centers)
            
            # Distance shifts in pixels
            box_shifts = np.linalg.norm(box_movements[1:] - box_movements[:-1], axis=1) if M > 1 else np.zeros(1)
            lms_shifts = np.linalg.norm(lms_movements[1:] - lms_movements[:-1], axis=1) if M > 1 else np.zeros(1)

            box_var = float(np.var(box_shifts)) if len(box_shifts) > 1 else 0.0
            lms_var = float(np.var(lms_shifts)) if len(lms_shifts) > 1 else 0.0

            # Static photo checks: if landmarks do not exhibit micro-variations
            total_lms_variance = float(np.sum(np.var(lms_movements, axis=0)))
            distances = np.linalg.norm(box_movements - lms_movements, axis=1)
            dist_var = float(np.var(distances))

            if total_lms_variance < 0.05:
                s_motion = 0.0  # Rigid/Frozen photo print
            elif dist_var < 0.05:
                s_motion = 0.0  # Flat photo translation spoof
            else:
                s_motion = 1.0

            # ==========================================
            # 6. Adaptive Score Fusion & Accumulation
            # ==========================================
            adaptive_w = self.get_adaptive_weights(w, yaw)
            
            # Current frame liveness confidence
            s_passive_frame = (
                adaptive_w["texture"] * s_texture +
                adaptive_w["depth"] * s_depth +
                adaptive_w["motion"] * s_motion +
                adaptive_w["temporal"] * s_temporal
            )

            # Cumulative Spoof Accumulator (EMA filter of deviations)
            suspect_threshold = 0.70
            e_spoof_frame = max(0.0, suspect_threshold - s_passive_frame)
            self.session_spoof_accumulator = (self.beta * self.session_spoof_accumulator) + ((1.0 - self.beta) * e_spoof_frame)

            # Final liveness score based on session accumulator
            final_passive_score = float(max(0.0, 1.0 - self.session_spoof_accumulator))

            # Classification decision logic
            if final_passive_score >= self.live_threshold:
                classification = "LIVE"
            elif final_passive_score >= self.spoof_threshold:
                classification = "SUSPECT"
            else:
                classification = "SPOOF"

            # Confidence Estimation
            confidence = float(tracking_confidence * (1.0 - min(0.3, abs(yaw) / 90.0)) * (min(w, 200.0) / 200.0))

            return {
                "success": True,
                "spoof_score": float(1.0 - final_passive_score),  # Inverse: 1.0 is highest spoof risk
                "classification": classification,
                "signals": {
                    "texture_score": s_texture,
                    "depth_score": s_depth,
                    "motion_score": s_motion,
                    "temporal_score": s_temporal
                },
                "analytics": {
                    "session_spoof_accumulator": float(self.session_spoof_accumulator),
                    "buffer_ready": True,
                    "lbp_entropy": float(avg_entropy),
                    "nose_depth_dz": float(dz_nose),
                    "scale_depth_ratio": float(avg_r_depth),
                    "temporal_stdev": float(mean_std),
                    "tracking_reliability": confidence
                },
                "error": None
            }

        except Exception as e:
            return self._generate_error_response(f"ANTISPOOF_FAILURE: {str(e)}", timestamp)

    def _generate_error_response(self, error_code: str, timestamp: float) -> Dict[str, Any]:
        """
        Creates a structured fail state dictionary response.
        """
        self.session_spoof_accumulator = 1.0  # Force maximum spoof state on failure
        return {
            "success": False,
            "spoof_score": 1.0,
            "classification": "SPOOF",
            "signals": {"texture_score": 0.0, "depth_score": 0.0, "motion_score": 0.0, "temporal_score": 0.0},
            "analytics": {
                "session_spoof_accumulator": 1.0,
                "buffer_ready": False,
                "lbp_entropy": 0.0,
                "nose_depth_dz": 0.0,
                "scale_depth_ratio": 0.0,
                "temporal_stdev": 0.0,
                "tracking_reliability": 0.0
            },
            "error": error_code
        }
