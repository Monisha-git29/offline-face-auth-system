import os
import sys
import time
import shutil
import threading
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.recognition import FaceRecognizer, SQLiteFaceRegistry, FaceAuthSDK
from opencv_module import errors

# =====================================================================
# 1. HELPER GENERATORS
# =====================================================================
def create_mock_crop(color: Tuple[int, int, int]) -> np.ndarray:
    """Generates a color 112x112 BGR face crop."""
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    crop[:, :] = color
    return crop

def create_synthetic_landmarks() -> np.ndarray:
    """Generates a minimal 468x3 array representing MediaPipe landmarks."""
    landmarks = np.ones((468, 3), dtype=np.float32) * 0.5
    # Standard eye corners to avoid eye coordinate extraction crash
    landmarks[33] = [0.62, 0.40, -0.02]
    landmarks[133] = [0.58, 0.40, -0.02]
    landmarks[362] = [0.42, 0.40, -0.02]
    landmarks[263] = [0.38, 0.40, -0.02]
    return landmarks

# =====================================================================
# 2. VERIFICATION RUNNER
# =====================================================================
def run_tests():
    print("======================================================================")
    print("       STARTING INTEGRATION VERIFICATION: FACE RECOGNITION            ")
    print("======================================================================\n")

    # Determine TFLite backend
    try:
        import tflite_runtime.interpreter as tflite
        runtime_used = "tflite-runtime"
    except ImportError:
        try:
            import tensorflow.lite as tflite
            runtime_used = "tensorflow"
        except ImportError:
            runtime_used = "None"
            
    print(f"Verified TFLite backend runtime used: {runtime_used}\n")

    db_path = "face_recognition/test_face_db.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    # -----------------------------------------------------------------
    # Test A: SQLite Registry Initialization & Metadata Verification
    # -----------------------------------------------------------------
    print("Test A: Initializing SQLite Face Registry...")
    registry = SQLiteFaceRegistry(db_path=db_path)
    
    # Check that database file was created
    assert os.path.exists(db_path), "Failed to create database file."
    
    # Check metadata properties
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT meta_value FROM registry_metadata WHERE meta_key = 'model_name'")
    model_name = cursor.fetchone()[0]
    cursor.execute("SELECT meta_value FROM registry_metadata WHERE meta_key = 'embedding_dim'")
    dim = int(cursor.fetchone()[0])
    conn.close()
    
    assert model_name == "mobilefacenet", f"Expected model_name 'mobilefacenet', got '{model_name}'"
    assert dim == 128, f"Expected embedding_dim 128, got {dim}"
    print("[PASS] SQLite registry initialized with correct metadata tables.\n")

    # -----------------------------------------------------------------
    # Test B: Model Inference Execution & Registry Enrollment
    # -----------------------------------------------------------------
    print("Test B: Testing TFLite Model Execution & Enrollment...")
    crop_alice = create_mock_crop((255, 0, 0))   # Red
    crop_bob = create_mock_crop((0, 128, 0))     # Green
    
    # 1. Instantiate REAL FaceRecognizer and run inference on crops
    # This verifies that the TFLite model is parsed and executed successfully on the device CPU
    recognizer = FaceRecognizer()
    real_emb_alice = recognizer.extract_embedding(crop_alice)
    real_emb_bob = recognizer.extract_embedding(crop_bob)
    
    assert real_emb_alice.shape == (128,), f"Expected shape (128,), got {real_emb_alice.shape}"
    assert np.isclose(np.linalg.norm(real_emb_alice), 1.0), "Expected output embedding to be L2-normalized."
    print("  [SUCCESS] MobileFaceNet TFLite inference completed successfully on BGR face crops.")
    
    # 2. Define orthogonal unit embeddings for Alice, Bob, and Charlie
    # This isolates registry matching logic from flat-image embedding collisions in deep learning space
    emb_alice = np.zeros(128, dtype=np.float32)
    emb_alice[0] = 1.0
    
    emb_bob = np.zeros(128, dtype=np.float32)
    emb_bob[1] = 1.0
    
    emb_charlie = np.zeros(128, dtype=np.float32)
    emb_charlie[2] = 1.0
    
    # Enroll in registry database
    registry.enroll("alice_101", emb_alice)
    registry.enroll("bob_202", emb_bob)
    
    assert len(registry.cache_users) == 2, f"Expected 2 users in cache, got {len(registry.cache_users)}"
    assert registry.cache_embeddings.shape == (2, 128), f"Expected cache shape (2, 128), got {registry.cache_embeddings.shape}"
    assert "alice_101" in registry.cache_users
    assert "bob_202" in registry.cache_users
    print("[PASS] TFLite inference run, and distinct embeddings enrolled in database.\n")

    # -----------------------------------------------------------------
    # Test C: Cosine Similarity Vector Verification
    # -----------------------------------------------------------------
    print("Test C: Testing Cosine Similarity Verification...")
    # Exact match for Alice
    matched_id, score = registry.verify(emb_alice, threshold=0.60)
    assert matched_id == "alice_101", f"Expected 'alice_101', got '{matched_id}'"
    assert np.isclose(score, 1.0), f"Expected exact similarity score 1.0, got {score}"

    # Verify Bob
    matched_id, score = registry.verify(emb_bob, threshold=0.60)
    assert matched_id == "bob_202"

    # Verify Charlie (unenrolled)
    matched_id, score = registry.verify(emb_charlie, threshold=0.60)
    assert matched_id is None, f"Expected Charlie to mismatch (None), got '{matched_id}'"
    print(f"[PASS] Cosine matching verify function behaves correctly. Charlie score against registry: {score:.4f}\n")

    # -----------------------------------------------------------------
    # Test D: Thread-Safe Concurrency Test
    # -----------------------------------------------------------------
    print("Test D: Running Concurrent Enrollment & Match Tests...")
    errors_list = []
    
    def worker_thread(thread_idx):
        try:
            # Use orthogonal unit vector for each thread to verify index mapping
            uid = f"worker_{thread_idx}"
            emb = np.zeros(128, dtype=np.float32)
            emb[thread_idx + 10] = 1.0
            
            # Write to database (concurrent)
            registry.enroll(uid, emb)
            
            # Immediately verify (concurrent)
            matched, score = registry.verify(emb, threshold=0.60)
            if matched != uid:
                errors_list.append(f"Thread {thread_idx}: verification failed. Expected {uid}, got {matched}")
        except Exception as e:
            errors_list.append(f"Thread {thread_idx} crashed: {str(e)}")

    threads = []
    for t in range(10):
        th = threading.Thread(target=worker_thread, args=(t,))
        threads.append(th)
        th.start()
        
    for th in threads:
        th.join()
        
    assert len(errors_list) == 0, f"Concurrency errors occurred:\n" + "\n".join(errors_list)
    assert len(registry.cache_users) == 12, f"Expected 12 total users in cache, got {len(registry.cache_users)}"
    print("[PASS] SQLite database and cache operations are thread-safe under parallel load.\n")

    # -----------------------------------------------------------------
    # Test E: FaceAuthSDK Composition & Integration Flow
    # -----------------------------------------------------------------
    print("Test E: Testing FaceAuthSDK Wrapper Flow...")
    auth_sdk = FaceAuthSDK(db_path=db_path, similarity_threshold=0.60)
    
    # Enroll Alice
    auth_sdk.registry.enroll("alice_101", emb_alice)
    
    # Mock FaceRecognizer in SDK to return Alice's embedding for testing SDK wrapper flow
    auth_sdk.recognizer.extract_embedding = lambda crop: emb_alice
    
    frame = np.ones((400, 400, 3), dtype=np.uint8) * 128
    bbox = (100, 80, 200, 240)
    lms = create_synthetic_landmarks()
    
    # 1. Run frame process: Liveness fails (SPOOF)
    res = auth_sdk.process_frame(frame, bbox, lms, tracking_confidence=0.96)
    assert res["decision"] == "SPOOF", f"Expected liveness SPOOF, got {res['decision']}"
    assert res["identity"] is None, f"Expected identity None on spoof, got {res['identity']}"
    assert res["similarity_score"] == 0.0
    
    # 2. Simulate Liveness Success (LIVE)
    def mock_liveness_process(frame, bbox, landmarks, tracking_confidence=1.0, timestamp=None):
        auth_sdk.liveness_sdk.last_fqa_result = {
            "success": True,
            "overall_quality_score": 90,
            "aligned_face": crop_alice
        }
        return {
            "success": True,
            "decision": "LIVE",
            "trust_score": 0.95,
            "session_id": "mock_session",
            "current_challenge": None,
            "active_score": 1.0,
            "passive_score": 0.90,
            "final_liveness_score": 0.95,
            "error": None
        }
        
    auth_sdk.liveness_sdk.process_frame = mock_liveness_process
    
    # Run process: Should match Alice
    res = auth_sdk.process_frame(frame, bbox, lms, tracking_confidence=0.96)
    assert res["success"] is True, f"Expected auth success True, got {res['success']}, error={res['error']}"
    assert res["identity"] == "alice_101", f"Expected matched identity 'alice_101', got '{res['identity']}'"
    assert np.isclose(res["similarity_score"], 1.0)

    # 3. Run process with an un-enrolled face crop (Bob's crop, which matches Bob)
    auth_sdk.registry.remove("bob_202")
    auth_sdk.recognizer.extract_embedding = lambda crop: emb_bob
    
    def mock_liveness_bob_crop(frame, bbox, landmarks, tracking_confidence=1.0, timestamp=None):
        auth_sdk.liveness_sdk.last_fqa_result = {
            "success": True,
            "overall_quality_score": 90,
            "aligned_face": crop_bob
        }
        return {
            "success": True,
            "decision": "LIVE",
            "trust_score": 0.95,
            "error": None
        }
        
    auth_sdk.liveness_sdk.process_frame = mock_liveness_bob_crop
    
    # Run process: Should reject with FACE_NOT_MATCHED
    res = auth_sdk.process_frame(frame, bbox, lms, tracking_confidence=0.96)
    assert res["success"] is False
    assert res["error"] == errors.FACE_NOT_MATCHED, f"Expected FACE_NOT_MATCHED error, got {res['error']}"
    assert res["identity"] is None
    
    print("[PASS] FaceAuthSDK correctly routes frames, processes liveness, and validates match identities.\n")

    # Clean up test DB file
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print("=======================================================")
    print("     ALL INTEGRATION VERIFICATION TESTS PASSED!        ")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
