"""
MediaPipe Face Mesh Landmark Detector Module.

Integrates MediaPipe FaceLandmarker Tasks API to perform offline, local
landmark detection and eye center extraction from raw camera frames.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions


class MediaPipeLandmarkDetector:
    """
    Production-grade local landmark detector using MediaPipe FaceLandmarker.
    """

    def __init__(self, model_path: str = "face_recognition/face_landmarker.task"):
        """
        Initializes the MediaPipe FaceLandmarker.

        Args:
            model_path (str): Path to the face_landmarker.task model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe FaceLandmarker task file not found at: {model_path}")

        self.model_path = model_path
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detects a single primary face and returns coordinates and landmarks.

        Args:
            frame (np.ndarray): Input BGR camera image.

        Returns:
            Dict[str, Any] containing:
                "left_eye": Tuple[float, float] (x, y) coordinates
                "right_eye": Tuple[float, float] (x, y) coordinates
                "face_bbox": Tuple[int, int, int, int] (x, y, w, h) bounding box
                "landmarks": np.ndarray (shape (468, 3)) of normalized coordinates
            Or None if no face is detected.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self.landmarker.detect(mp_image)
        if not results.face_landmarks:
            return None

        # Take the first face detected
        landmarks = results.face_landmarks[0]

        # Left eye center: midpoint of landmarks 33 and 133 (image left side)
        left_eye = (
            float((landmarks[33].x + landmarks[133].x) / 2.0 * w),
            float((landmarks[33].y + landmarks[133].y) / 2.0 * h)
        )

        # Right eye center: midpoint of landmarks 362 and 263 (image right side)
        right_eye = (
            float((landmarks[362].x + landmarks[263].x) / 2.0 * w),
            float((landmarks[362].y + landmarks[263].y) / 2.0 * h)
        )

        # Calculate bounding box from landmark bounds
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        face_bbox = (
            int(max(0, min_x)),
            int(max(0, min_y)),
            int(min(w - min_x, max_x - min_x)),
            int(min(h - min_y, max_y - min_y))
        )

        # Convert face landmarks list to (468, 3) numpy array for compatibility
        lms_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "face_bbox": face_bbox,
            "landmarks": lms_array
        }

    def close(self):
        """Releases the FaceLandmarker resources."""
        try:
            self.landmarker.close()
        except Exception:
            pass
