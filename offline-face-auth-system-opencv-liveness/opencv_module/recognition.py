import os
import sqlite3
import threading
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union, List

from .sdk import LivenessSDK
from . import errors

# =====================================================================
# 1. TFLITE INTERPRETER LOADING
# =====================================================================
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None


class FaceRecognizer:
    """
    MobileFaceNet face embedding extractor using TFLite.
    Exposes a clean API to extract a 128D normalized embedding vector
    from a 112x112 aligned face crop.
    """

    def __init__(self, model_path: str = "face_recognition/mobilefacenet.tflite"):
        """
        Initializes the FaceRecognizer and allocates TFLite tensors.

        Args:
            model_path (str): Path to the mobilefacenet.tflite model file.
        """
        if tflite is None:
            raise ImportError(
                "TFLite libraries not found. Please install tflite-runtime or tensorflow."
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MobileFaceNet model not found at: {model_path}")

        self.model_path = model_path
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Verify input shapes
        self.expected_shape = self.input_details[0]['shape'] # [1, 112, 112, 3]
        _, self.img_height, self.img_width, _ = self.expected_shape

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extracts a L2-normalized 128D face embedding from a BGR face crop.

        Args:
            aligned_face (np.ndarray): Aligned face crop (BGR, size 112x112).

        Returns:
            np.ndarray: L2-normalized 128D float32 embedding vector.
        """
        if aligned_face is None or not isinstance(aligned_face, np.ndarray) or aligned_face.size == 0:
            raise ValueError("Invalid aligned_face image.")

        # 1. Transform color space: BGR to RGB
        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # 2. Resize if necessary (should already be 112x112)
        h, w = rgb_face.shape[:2]
        if h != self.img_height or w != self.img_width:
            rgb_face = cv2.resize(rgb_face, (self.img_width, self.img_height))

        # 3. Cast to float32 and apply scaling: (pixel - 127.5) / 128.0
        face_float = rgb_face.astype(np.float32)
        normalized = (face_float - 127.5) / 128.0
        input_tensor = np.expand_dims(normalized, axis=0)

        # 4. Invoke TFLite model
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # 5. Extract output tensor and flatten
        embedding = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))

        # 6. Apply L2-normalization for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 1e-5:
            embedding = embedding / norm

        return embedding


# =====================================================================
# 2. SQLITE FACE REGISTRY WITH VECTOR CACHE
# =====================================================================
class SQLiteFaceRegistry:
    """
    SQLite-backed Face Registry with an in-memory vector cache for high-speed
    offline face verification and matching.
    """

    def __init__(self, db_path: str = "face_recognition/face_db.sqlite"):
        """
        Initializes the registry database and loads vectors into cache.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Cache variables
        self.cache_users = []
        self.cache_embeddings = np.empty((0, 128), dtype=np.float32)

        self._init_db()
        self._load_cache()

    def _init_db(self):
        """Creates the required tables and registers metadata."""
        with self.lock:
            dir_name = os.path.dirname(self.db_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            conn = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                cursor = conn.cursor()
                # User face registry table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_registry (
                        user_id TEXT PRIMARY KEY,
                        embedding BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Metadata version table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS registry_metadata (
                        meta_key TEXT PRIMARY KEY,
                        meta_value TEXT
                    )
                """)
                
                # Check model compatibility metadata
                cursor.execute("SELECT meta_value FROM registry_metadata WHERE meta_key = 'model_name'")
                row = cursor.fetchone()
                if row is None:
                    cursor.execute("INSERT INTO registry_metadata (meta_key, meta_value) VALUES ('model_name', 'mobilefacenet')")
                    cursor.execute("INSERT INTO registry_metadata (meta_key, meta_value) VALUES ('embedding_dim', '128')")
                else:
                    if row[0] != 'mobilefacenet':
                        raise ValueError(f"Database model mismatch: Expected 'mobilefacenet', found '{row[0]}'")
                
                conn.commit()
            finally:
                conn.close()

    def _load_cache(self):
        """Loads all database records into memory for high-performance vector lookup."""
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT user_id, embedding FROM face_registry")
                rows = cursor.fetchall()
                
                users_list = []
                embs_list = []
                for user_id, emb_blob in rows:
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                    if len(emb) == 128:
                        users_list.append(user_id)
                        embs_list.append(emb)
                
                self.cache_users = users_list
                if embs_list:
                    self.cache_embeddings = np.stack(embs_list, axis=0)
                else:
                    self.cache_embeddings = np.empty((0, 128), dtype=np.float32)
            finally:
                conn.close()

    def enroll(self, user_id: str, embedding: np.ndarray):
        """
        Registers a user's L2-normalized embedding vector. Updates SQLite and cache.

        Args:
            user_id (str): Unique employee ID or name.
            embedding (np.ndarray): 128D embedding vector.
        """
        if not isinstance(embedding, np.ndarray) or embedding.shape != (128,):
            raise ValueError("Embedding must be a 128-dimensional float32 array.")

        # L2 normalize just to guarantee unit norm
        norm = np.linalg.norm(embedding)
        if norm > 1e-5:
            embedding = embedding / norm

        emb_blob = embedding.tobytes()

        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO face_registry (user_id, embedding) VALUES (?, ?)",
                    (user_id, sqlite3.Binary(emb_blob))
                )
                conn.commit()
            finally:
                conn.close()

            # Thread-safe in-memory cache update
            if user_id in self.cache_users:
                idx = self.cache_users.index(user_id)
                self.cache_embeddings[idx] = embedding
            else:
                self.cache_users.append(user_id)
                if self.cache_embeddings.shape[0] == 0:
                    self.cache_embeddings = np.expand_dims(embedding, axis=0)
                else:
                    self.cache_embeddings = np.vstack([self.cache_embeddings, embedding])

    def verify(self, embedding: np.ndarray, threshold: float = 0.60) -> Tuple[Optional[str], float]:
        """
        Matches a query embedding against the cached templates using cosine similarity.

        Args:
            embedding (np.ndarray): Query face embedding (128D).
            threshold (float): Minimum similarity to match (default 0.60).

        Returns:
            Tuple[str or None, float]: Matched user_id (None if not matched) and similarity score.
        """
        if not isinstance(embedding, np.ndarray) or embedding.shape != (128,):
            raise ValueError("Embedding must be a 128-dimensional float32 array.")

        with self.lock:
            if self.cache_embeddings.shape[0] == 0:
                return None, 0.0

            # Execute fast vectorized matrix dot-product
            similarities = np.dot(self.cache_embeddings, embedding)
            max_idx = np.argmax(similarities)
            max_score = float(similarities[max_idx])

            if max_score >= threshold:
                return self.cache_users[max_idx], max_score
            return None, max_score

    def remove(self, user_id: str) -> bool:
        """
        Removes a user's record from SQLite and the vector cache.

        Args:
            user_id (str): User ID to delete.

        Returns:
            bool: True if the user was found and deleted, False otherwise.
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM face_registry WHERE user_id = ?", (user_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
            finally:
                conn.close()

            if deleted and user_id in self.cache_users:
                idx = self.cache_users.index(user_id)
                self.cache_users.pop(idx)
                if len(self.cache_users) == 0:
                    self.cache_embeddings = np.empty((0, 128), dtype=np.float32)
                else:
                    self.cache_embeddings = np.delete(self.cache_embeddings, idx, axis=0)
            return deleted


# =====================================================================
# 3. UNIFIED FACE AUTHENTICATION COORDINATOR (COMPOSITION WRAPPER)
# =====================================================================
class FaceAuthSDK:
    """
    Unified Face Authentication SDK.
    Integrates offline face quality assessment, liveness verification,
    and face recognition identity matching using composition.
    """

    def __init__(
        self,
        preset: Union[str, Dict[str, Any]] = "MEDIUM",
        model_path: str = "face_recognition/mobilefacenet.tflite",
        db_path: str = "face_recognition/face_db.sqlite",
        similarity_threshold: float = 0.60
    ):
        """
        Initializes the FaceAuthSDK wrapper.

        Args:
            preset (str or dict): Liveness preset config.
            model_path (str): Path to mobilefacenet.tflite.
            db_path (str): Path to SQLite face registry.
            similarity_threshold (float): Recognition match score threshold.
        """
        self.liveness_sdk = LivenessSDK(preset=preset)
        self.recognizer = FaceRecognizer(model_path=model_path)
        self.registry = SQLiteFaceRegistry(db_path=db_path)
        self.similarity_threshold = similarity_threshold

    def reset(self) -> None:
        """Resets the liveness challenges state and session variables."""
        self.liveness_sdk.reset()

    def close(self) -> None:
        """Releases underlying MediaPipe and model resources."""
        self.liveness_sdk.close()

    def enroll_user(
        self,
        user_id: str,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Enrolls a new user by running MediaPipe Face Mesh landmark extraction (or using provided ones),
        performing alignment and Face Quality Assessment (FQA), extracting
        the MobileFaceNet embedding, and registering it in SQLite.

        Args:
            user_id (str): Unique identifier for the user.
            frame (np.ndarray): Original BGR camera frame.
            bbox (Tuple[int, int, int, int], optional): Face bounding box.
            landmarks (np.ndarray, optional): MediaPipe landmarks array.

        Returns:
            Dict[str, Any]: Enrollment status dictionary.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {"success": False, "error": errors.FACE_NOT_DETECTED}

        img_h, img_w = frame.shape[:2]

        if bbox is None or landmarks is None:
            if self.liveness_sdk.mp_detector is None:
                from .landmarks import MediaPipeLandmarkDetector
                try:
                    self.liveness_sdk.mp_detector = MediaPipeLandmarkDetector(model_path=self.liveness_sdk.mp_model_path)
                except Exception:
                    return {"success": False, "error": errors.FACE_NOT_DETECTED}
                    
            res_det = self.liveness_sdk.mp_detector.detect_landmarks(frame)
            if res_det is None:
                return {"success": False, "error": errors.FACE_NOT_DETECTED}
                
            bbox = res_det["face_bbox"]
            landmarks = res_det["landmarks"]
            left_eye = res_det["left_eye"]
            right_eye = res_det["right_eye"]
        else:
            # Extract eye positions from external landmarks
            try:
                lms = np.array(landmarks, dtype=np.float32)
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
                return {"success": False, "error": errors.FACE_NOT_DETECTED}
            
        fqa_res = self.liveness_sdk.fqa_engine.assess_raw_face(
            image=frame,
            left_eye=left_eye,
            right_eye=right_eye,
            face_bbox=bbox
        )
        if not fqa_res.get("success"):
            return {"success": False, "error": errors.LANDMARKS_UNSTABLE}
            
        aligned_face = fqa_res["aligned_face"]
        if aligned_face is None or not isinstance(aligned_face, np.ndarray) or aligned_face.size == 0:
            return {"success": False, "error": errors.RECOGNITION_ERROR}
            
        try:
            embedding = self.recognizer.extract_embedding(aligned_face)
            self.registry.enroll(user_id, embedding)
            return {"success": True, "user_id": user_id}
        except Exception:
            return {"success": False, "error": errors.RECOGNITION_ERROR}

    def authenticate(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Authenticates a user from a raw camera frame. Performs liveness checks,
        alignment, FQA, embedding extraction, and database matching.

        Args:
            frame (np.ndarray): Original BGR camera frame.
            bbox (Tuple[int, int, int, int], optional): Face bounding box.
            landmarks (np.ndarray, optional): MediaPipe landmarks array.

        Returns:
            Dict[str, Any]: Authentication results dictionary.
        """
        return self.process_frame(frame=frame, bbox=bbox, landmarks=landmarks)

    def process_frame(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[np.ndarray] = None,
        tracking_confidence: float = 1.0,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Performs full liveness validation and matches identity on success.

        Args:
            frame (np.ndarray): Original BGR frame.
            bbox (Tuple[int, int, int, int], optional): Face bounding box (x, y, w, h).
            landmarks (np.ndarray, optional): MediaPipe landmarks array.
            tracking_confidence (float): Landmark tracking confidence.
            timestamp (float, optional): Clock timestamp in seconds.

        Returns:
            Dict[str, Any]: Combined liveness and face recognition response dictionary.
        """
        # 1. Run liveness using composition-held LivenessSDK
        res = self.liveness_sdk.process_frame(
            frame=frame,
            bbox=bbox,
            landmarks=landmarks,
            tracking_confidence=tracking_confidence,
            timestamp=timestamp
        )

        # Extend schema with face recognition output properties
        res["identity"] = None
        res["similarity_score"] = 0.0

        # 2. Check if liveness succeeded and the decision is LIVE
        if res.get("success") and res.get("decision") == "LIVE":
            # Retrieve the aligned face crop stored inside LivenessSDK instance
            last_fqa = getattr(self.liveness_sdk, "last_fqa_result", None)
            if not last_fqa or "aligned_face" not in last_fqa:
                res["success"] = False
                res["error"] = errors.RECOGNITION_ERROR
                return res

            aligned_face = last_fqa["aligned_face"]
            if aligned_face is None or not isinstance(aligned_face, np.ndarray) or aligned_face.size == 0:
                res["success"] = False
                res["error"] = errors.RECOGNITION_ERROR
                return res

            # 3. Match face crop against SQLite registry
            try:
                embedding = self.recognizer.extract_embedding(aligned_face)
                user_id, similarity = self.registry.verify(embedding, self.similarity_threshold)
                
                res["similarity_score"] = similarity
                if user_id is not None:
                    res["identity"] = user_id
                else:
                    # Face is LIVE, but doesn't match any enrolled user
                    res["success"] = False
                    res["error"] = errors.FACE_NOT_MATCHED
            except Exception:
                # Capture database or interpreter exceptions
                res["success"] = False
                res["error"] = errors.RECOGNITION_ERROR

        return res
