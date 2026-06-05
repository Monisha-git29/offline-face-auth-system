"""
Engineering Validation & Performance Evaluation Script.

Performs verification of face recognition pipeline correctness, embedding stability
under physical/visual transformations, system latency (embedding extraction and database search),
memory foot-print (RSS delta), SQLite database registry lookup scalability up to 10,000 users,
concurrency lock contention effects, cold start timings, and database robustness.
"""

import os
import sys
import time
import csv
import sqlite3
import threading
import psutil
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.recognition import FaceRecognizer, SQLiteFaceRegistry, FaceAuthSDK
from opencv_module.alignment import FaceAligner


# =====================================================================
# 0. HELPER FUNCTIONS
# =====================================================================
def get_sample_eye_coords(width: int, height: int, angle: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Reconstructs eye coordinates for synthetic sample images.
    """
    cx, cy = width // 2, height // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    def rotate_point(pt: Tuple[float, float]) -> Tuple[float, float]:
        px, py = pt
        rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        return (float(rx), float(ry))
        
    left_eye = rotate_point((cx - 35, cy - 25))
    right_eye = rotate_point((cx + 35, cy - 25))
    return left_eye, right_eye


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculates cosine similarity between two L2-normalized embeddings.
    """
    return float(np.dot(emb1, emb2))


def populate_database_fast(db_path: str, num_users: int) -> None:
    """
    Populates SQLite database registry with random normalized 128D embeddings.
    Uses bulk transaction for high speed.
    """
    dir_name = os.path.dirname(db_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_registry (
            user_id TEXT PRIMARY KEY,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS registry_metadata (
            meta_key TEXT PRIMARY KEY,
            meta_value TEXT
        )
    """)
    cursor.execute("INSERT OR REPLACE INTO registry_metadata (meta_key, meta_value) VALUES ('model_name', 'mobilefacenet')")
    cursor.execute("INSERT OR REPLACE INTO registry_metadata (meta_key, meta_value) VALUES ('embedding_dim', '128')")
    
    # Generate random normalized embeddings
    embeddings = np.random.randn(num_users, 128).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)
    
    data = []
    for i in range(num_users):
        user_id = f"user_{i}"
        emb_blob = embeddings[i].tobytes()
        data.append((user_id, emb_blob))
        
    with conn:
        cursor.executemany("INSERT OR REPLACE INTO face_registry (user_id, embedding) VALUES (?, ?)", data)
    conn.close()


# =====================================================================
# MAIN RUNNER
# =====================================================================
def main():
    print("======================================================================")
    print("       NHAI OFFLINE FACE AUTHENTICATION SYSTEM - VALIDATION RUNNER     ")
    print("======================================================================\n")

    # Define paths
    sample_normal_path = "examples/sample_images/sample_normal.jpg"
    csv_out_path = "evaluation/similarity_distribution.csv"
    os.makedirs("evaluation", exist_ok=True)

    # Verify baseline image exists
    if not os.path.exists(sample_normal_path):
        print(f"[ERROR] Baseline image not found at '{sample_normal_path}'.")
        print("        Please run python examples/align_face_demo.py first to generate samples.")
        sys.exit(1)

    # Get baseline memory info
    process = psutil.Process()
    rss_start = process.memory_info().rss

    # =====================================================================
    # SECTION 8: COLD START BENCHMARK (Part 1 - TFLite Init)
    # =====================================================================
    print("--- SECTION 8: COLD START BENCHMARK ---")
    t0 = time.perf_counter()
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
        
    t_import = (time.perf_counter() - t0) * 1000.0
    print(f"TFLite Import Latency:        {t_import:.2f} ms")

    # Model Load
    t0 = time.perf_counter()
    interpreter = tflite.Interpreter(model_path="face_recognition/mobilefacenet.tflite")
    t_model_load = (time.perf_counter() - t0) * 1000.0
    print(f"Model File Load Latency:      {t_model_load:.2f} ms")

    # Interpreter Allocation
    t0 = time.perf_counter()
    interpreter.allocate_tensors()
    t_allocate = (time.perf_counter() - t0) * 1000.0
    print(f"Tensor Allocation Latency:    {t_allocate:.2f} ms")

    # SQLite Fresh DB Init
    fresh_db_path = "face_recognition/fresh_test.sqlite"
    if os.path.exists(fresh_db_path):
        os.remove(fresh_db_path)
    t0 = time.perf_counter()
    reg_fresh = SQLiteFaceRegistry(db_path=fresh_db_path)
    t_db_init = (time.perf_counter() - t0) * 1000.0
    print(f"SQLite DB fresh init Latency: {t_db_init:.2f} ms")
    if os.path.exists(fresh_db_path):
        os.remove(fresh_db_path)

    # SQLite populated Cache Load
    pop_db_path = "face_recognition/populated_test.sqlite"
    populate_database_fast(pop_db_path, 100)
    t0 = time.perf_counter()
    reg_pop = SQLiteFaceRegistry(db_path=pop_db_path)
    t_cache_load = (time.perf_counter() - t0) * 1000.0
    print(f"Cache Load (100 users) Lat:   {t_cache_load:.2f} ms")
    if os.path.exists(pop_db_path):
        os.remove(pop_db_path)

    # SDK startup time
    t0 = time.perf_counter()
    sdk = FaceAuthSDK(model_path="face_recognition/mobilefacenet.tflite", db_path="face_recognition/sdk_startup_test.sqlite")
    t_sdk_startup = (time.perf_counter() - t0) * 1000.0
    print(f"FaceAuthSDK Startup Latency:  {t_sdk_startup:.2f} ms")
    sdk_db_path = "face_recognition/sdk_startup_test.sqlite"
    if os.path.exists(sdk_db_path):
        os.remove(sdk_db_path)

    print("\n------------------------------------------------------\n")

    # =====================================================================
    # SECTION 4: MEMORY USAGE BENCHMARK (Part 1 - Init)
    # =====================================================================
    # Track memory milestones
    rss_milestones = {}
    rss_milestones["before_init"] = rss_start

    # Initialize FaceRecognizer
    recognizer = FaceRecognizer(model_path="face_recognition/mobilefacenet.tflite")
    rss_milestones["after_init"] = process.memory_info().rss

    # =====================================================================
    # SETUP ALIGNED BASELINE CROP
    # =====================================================================
    image_normal = cv2.imread(sample_normal_path)
    aligner = FaceAligner(target_size=(112, 112))
    left_eye, right_eye = get_sample_eye_coords(400, 400, 0.0)
    align_res = aligner.align(image_normal, left_eye, right_eye)
    if not align_res["success"]:
        print("[ERROR] Failed to align baseline crop.")
        sys.exit(1)
        
    aligned_baseline = align_res["aligned_face"]

    # First inference
    ref_emb = recognizer.extract_embedding(aligned_baseline)
    rss_milestones["after_first_inf"] = process.memory_info().rss

    # Run 100 inferences for memory evaluation
    for _ in range(100):
        recognizer.extract_embedding(aligned_baseline)
    rss_milestones["after_100_inf"] = process.memory_info().rss

    # Calculate memory deltas
    peak_wset = getattr(process.memory_info(), 'peak_wset', process.memory_info().rss)
    print("--- SECTION 4: MEMORY USAGE BENCHMARK ---")
    print(f"Memory RSS Before Init:   {rss_milestones['before_init'] / (1024*1024):.2f} MB")
    print(f"Memory RSS After Init:    {rss_milestones['after_init'] / (1024*1024):.2f} MB")
    print(f"Memory RSS First Inf:     {rss_milestones['after_first_inf'] / (1024*1024):.2f} MB")
    print(f"Memory RSS 100 Inf:       {rss_milestones['after_100_inf'] / (1024*1024):.2f} MB")
    print(f"Model Load RSS Delta:     {(rss_milestones['after_init'] - rss_milestones['before_init']) / (1024*1024):.2f} MB")
    print(f"First Inf RSS Delta:      {(rss_milestones['after_first_inf'] - rss_milestones['after_init']) / (1024*1024):.2f} MB")
    print(f"100 Inf RSS Delta:        {(rss_milestones['after_100_inf'] - rss_milestones['after_first_inf']) / (1024*1024):.2f} MB")
    print(f"Peak Observed Memory:     {peak_wset / (1024*1024):.2f} MB")
    print("\n------------------------------------------------------\n")

    # =====================================================================
    # SECTION 3: MOBILEFACENET LATENCY BENCHMARK
    # =====================================================================
    print("--- SECTION 3: MOBILEFACENET LATENCY BENCHMARK ---")
    latencies = []
    for _ in range(100):
        t_inf0 = time.perf_counter()
        recognizer.extract_embedding(aligned_baseline)
        latencies.append((time.perf_counter() - t_inf0) * 1000.0)

    latencies = np.array(latencies)
    print(f"Inference Count: 100")
    print(f"  - Mean Latency:   {np.mean(latencies):.2f} ms")
    print(f"  - Median Latency: {np.median(latencies):.2f} ms")
    print(f"  - P95 Latency:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  - P99 Latency:    {np.percentile(latencies, 99):.2f} ms")
    print(f"  - Min Latency:    {np.min(latencies):.2f} ms")
    print(f"  - Max Latency:    {np.max(latencies):.2f} ms")
    print("\n------------------------------------------------------\n")

    # =====================================================================
    # SECTION 1: EMBEDDING STABILITY BENCHMARK
    # =====================================================================
    print("--- SECTION 1: EMBEDDING STABILITY BENCHMARK ---")
    stability_records = []

    # 1. Brightness
    brightness_factors = [0.50, 0.75, 1.25, 1.50]
    for factor in brightness_factors:
        transformed = np.clip(aligned_baseline.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        emb = recognizer.extract_embedding(transformed)
        sim = cosine_similarity(ref_emb, emb)
        stability_records.append(("Brightness", f"factor_{factor:.2f}", sim, transformed))
        print(f"Brightness Factor {factor:.2f} Similarity: {sim:.6f}")

    # 2. Blur
    blur_kernels = [3, 5, 7]
    for k in blur_kernels:
        transformed = cv2.GaussianBlur(aligned_baseline, (k, k), 0)
        emb = recognizer.extract_embedding(transformed)
        sim = cosine_similarity(ref_emb, emb)
        stability_records.append(("Blur", f"Gaussian_{k}x{k}", sim, transformed))
        print(f"Blur Gaussian {k}x{k} Similarity:   {sim:.6f}")

    # 3. Resize
    resize_dims = [56, 84]
    for dim in resize_dims:
        temp = cv2.resize(aligned_baseline, (dim, dim), interpolation=cv2.INTER_LINEAR)
        transformed = cv2.resize(temp, (112, 112), interpolation=cv2.INTER_LINEAR)
        emb = recognizer.extract_embedding(transformed)
        sim = cosine_similarity(ref_emb, emb)
        stability_records.append(("Resize", f"{dim}x{dim}->112x112", sim, transformed))
        print(f"Resize {dim}x{dim}->112x112 Similarity: {sim:.6f}")

    # 4. Rotation
    rotation_angles = [-15, -10, -5, 5, 10, 15]
    for angle in rotation_angles:
        M = cv2.getRotationMatrix2D((56, 56), angle, 1.0)
        transformed = cv2.warpAffine(aligned_baseline, M, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        emb = recognizer.extract_embedding(transformed)
        sim = cosine_similarity(ref_emb, emb)
        stability_records.append(("Rotation", f"angle_{angle:+d}deg", sim, transformed))
        print(f"Rotation {angle:+d} deg Similarity:      {sim:.6f}")

    # 5. Noise
    noise_sigmas = [5, 10, 20]
    for sigma in noise_sigmas:
        noise = np.random.normal(0, sigma, aligned_baseline.shape).astype(np.float32)
        transformed = np.clip(aligned_baseline.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        emb = recognizer.extract_embedding(transformed)
        sim = cosine_similarity(ref_emb, emb)
        stability_records.append(("Noise", f"Gaussian_std_{sigma}", sim, transformed))
        print(f"Noise Gaussian std {sigma} Similarity: {sim:.6f}")

    print("\n------------------------------------------------------\n")

    # =====================================================================
    # SECTION 2: SIMILARITY DISTRIBUTION COLLECTION
    # =====================================================================
    print("--- SECTION 2: SIMILARITY DISTRIBUTION COLLECTION ---")
    csv_rows = []

    # A. Identical image
    sim_self = cosine_similarity(ref_emb, ref_emb)
    csv_rows.append({
        "source_image": "sample_normal.jpg",
        "target_image": "sample_normal.jpg",
        "comparison_type": "Identical",
        "transformation": "None",
        "parameter": "None",
        "similarity": f"{sim_self:.6f}"
    })
    print(f"Identical Comparison Score: {sim_self:.6f}")

    # B. Transformed images (Section 1 results)
    for transform_type, param_val, sim, _ in stability_records:
        csv_rows.append({
            "source_image": "sample_normal.jpg",
            "target_image": "sample_normal.jpg",
            "comparison_type": "Transformed",
            "transformation": transform_type,
            "parameter": param_val,
            "similarity": f"{sim:.6f}"
        })

    # C. Different images comparison
    different_images = [
        {"name": "sample_mild_tilt.jpg", "angle": -15.0, "size": (400, 400)},
        {"name": "sample_extreme_tilt.jpg", "angle": 55.0, "size": (400, 400)},
        {"name": "sample_close_up.jpg", "angle": 10.0, "size": (600, 600)},
        {"name": "sample_small_eyes.jpg", "angle": 5.0, "size": (150, 150)}
    ]

    for item in different_images:
        path = os.path.join("examples/sample_images", item["name"])
        if not os.path.exists(path):
            print(f"Gracefully skipping comparison file (does not exist): {item['name']}")
            continue
            
        img = cv2.imread(path)
        lx, rx = get_sample_eye_coords(item["size"][0], item["size"][1], item["angle"])
        
        # Execute alignment
        res_align = aligner.align(img, lx, rx)
        if res_align["success"]:
            aligned_diff = res_align["aligned_face"]
            emb_diff = recognizer.extract_embedding(aligned_diff)
            sim = cosine_similarity(ref_emb, emb_diff)
            csv_rows.append({
                "source_image": "sample_normal.jpg",
                "target_image": item["name"],
                "comparison_type": "Different",
                "transformation": "None",
                "parameter": f"pose_angle_{item['angle']:.1f}",
                "similarity": f"{sim:.6f}"
            })
            print(f"Compare against {item['name']} Similarity: {sim:.6f}")
        else:
            print(f"Face Alignment failed for different image: {item['name']} (Error: {res_align.get('error')})")

    # Write to CSV
    headers = ["source_image", "target_image", "comparison_type", "transformation", "parameter", "similarity"]
    with open(csv_out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
            
    print(f"Similarity scores successfully written to: {csv_out_path}")
    print("\n------------------------------------------------------\n")

    # =====================================================================
    # SECTION 5: SQLITE REGISTRY SCALABILITY BENCHMARK
    # =====================================================================
    print("--- SECTION 5: SQLITE REGISTRY SCALABILITY BENCHMARK ---")
    sizes = [10, 100, 1000, 10000]
    db_scalability_results = {}

    for size in sizes:
        temp_db = f"face_recognition/temp_scale_{size}.sqlite"
        if os.path.exists(temp_db):
            os.remove(temp_db)
            
        populate_database_fast(temp_db, size)
        registry = SQLiteFaceRegistry(db_path=temp_db)
        
        # Perform 100 lookups
        lookup_latencies = []
        for i in range(100):
            query_emb = np.random.randn(128).astype(np.float32)
            query_emb = query_emb / np.maximum(np.linalg.norm(query_emb), 1e-8)
            
            t0 = time.perf_counter()
            # Call verify (matches cache vector lookups)
            registry.verify(query_emb, threshold=0.60)
            lookup_latencies.append((time.perf_counter() - t0) * 1000.0)
            
        lookup_latencies = np.array(lookup_latencies)
        db_scalability_results[size] = {
            "mean": np.mean(lookup_latencies),
            "median": np.median(lookup_latencies),
            "p95": np.percentile(lookup_latencies, 95),
            "p99": np.percentile(lookup_latencies, 99),
            "min": np.min(lookup_latencies),
            "max": np.max(lookup_latencies)
        }
        
        print(f"DB Size: {size:5d} users lookup latencies:")
        print(f"  - Mean:   {db_scalability_results[size]['mean']:.4f} ms")
        print(f"  - Median: {db_scalability_results[size]['median']:.4f} ms")
        print(f"  - P95:    {db_scalability_results[size]['p95']:.4f} ms")
        print(f"  - P99:    {db_scalability_results[size]['p99']:.4f} ms")
        print(f"  - Min/Max:{db_scalability_results[size]['min']:.4f} / {db_scalability_results[size]['max']:.4f} ms")
        
        # Clean up database file
        if os.path.exists(temp_db):
            os.remove(temp_db)

    print("\n------------------------------------------------------\n")

    # =====================================================================
    # SECTION 6: ENROLLMENT PERFORMANCE BENCHMARK
    # =====================================================================
    print("--- SECTION 6: ENROLLMENT PERFORMANCE BENCHMARK ---")
    enroll_counts = [100, 1000]
    enroll_results = {}

    for count in enroll_counts:
        temp_db = f"face_recognition/temp_enroll_{count}.sqlite"
        if os.path.exists(temp_db):
            os.remove(temp_db)
            
        registry = SQLiteFaceRegistry(db_path=temp_db)
        
        enroll_latencies = []
        for i in range(count):
            user_id = f"user_{i}"
            emb = np.random.randn(128).astype(np.float32)
            emb = emb / np.maximum(np.linalg.norm(emb), 1e-8)
            
            t0 = time.perf_counter()
            registry.enroll(user_id, emb)
            enroll_latencies.append((time.perf_counter() - t0) * 1000.0)
            
        enroll_latencies = np.array(enroll_latencies)
        enroll_results[count] = {
            "mean": np.mean(enroll_latencies),
            "median": np.median(enroll_latencies),
            "p95": np.percentile(enroll_latencies, 95),
            "p99": np.percentile(enroll_latencies, 99)
        }
        
        print(f"Enrollment of {count:4d} users latency stats:")
        print(f"  - Mean Latency:   {enroll_results[count]['mean']:.2f} ms")
        print(f"  - Median Latency: {enroll_results[count]['median']:.2f} ms")
        print(f"  - P95 Latency:    {enroll_results[count]['p95']:.2f} ms")
        print(f"  - P99 Latency:    {enroll_results[count]['p99']:.2f} ms")
        
        if os.path.exists(temp_db):
            os.remove(temp_db)

    print("\n------------------------------------------------------\n")

    # =====================================================================
    # SECTION 7: CONCURRENT VERIFICATION BENCHMARK
    # =====================================================================
    print("--- SECTION 7: CONCURRENT VERIFICATION BENCHMARK ---")
    
    # Establish a stable benchmark database of 1000 users
    bench_db = "face_recognition/bench_concurrent.sqlite"
    if os.path.exists(bench_db):
        os.remove(bench_db)
    populate_database_fast(bench_db, 1000)
    
    registry = SQLiteFaceRegistry(db_path=bench_db)
    thread_configs = [1, 5, 10, 20]
    concurrency_results = {}
    
    for tc in thread_configs:
        query_emb = np.random.randn(128).astype(np.float32)
        query_emb = query_emb / np.maximum(np.linalg.norm(query_emb), 1e-8)
        
        latencies_list = []
        threads = []
        barrier = threading.Barrier(tc)
        
        def worker():
            barrier.wait() # Synced startup
            local_lats = []
            for _ in range(1000):
                t0 = time.perf_counter()
                registry.verify(query_emb)
                local_lats.append((time.perf_counter() - t0) * 1000.0)
            latencies_list.extend(local_lats)
            
        t_start = time.perf_counter()
        for _ in range(tc):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        t_duration = time.perf_counter() - t_start
        total_calls = tc * 1000
        throughput = total_calls / t_duration
        
        lats_arr = np.array(latencies_list)
        concurrency_results[tc] = {
            "duration": t_duration * 1000.0,
            "throughput": throughput,
            "mean": np.mean(lats_arr),
            "max": np.max(lats_arr)
        }
        
        print(f"Threads: {tc:2d} | Duration: {t_duration*1000:.1f} ms | Throughput: {throughput:8.1f} ops/sec | Avg Lat: {np.mean(lats_arr):.4f} ms | Max Lat: {np.max(lats_arr):.1f} ms")

    if os.path.exists(bench_db):
        os.remove(bench_db)

    print("\n------------------------------------------------------\n")

    # =====================================================================
    # SECTION 9: EDGE CASE VALIDATION
    # =====================================================================
    print("--- SECTION 9: EDGE CASE VALIDATION ---")
    
    # A. Empty registry
    empty_db = "face_recognition/temp_empty.sqlite"
    if os.path.exists(empty_db):
        os.remove(empty_db)
        
    reg_empty = SQLiteFaceRegistry(db_path=empty_db)
    query_emb = np.random.randn(128).astype(np.float32)
    query_emb = query_emb / np.maximum(np.linalg.norm(query_emb), 1e-8)
    
    matched_id, score = reg_empty.verify(query_emb, threshold=0.60)
    print("Empty Registry Validation:")
    print(f"  - verify() output for empty db: ({matched_id}, {score})")
    assert matched_id is None and score == 0.0, "Expected empty verify to return (None, 0.0)"
    print("  [PASS] Empty registry correctly returns (None, 0.0)")
    
    if os.path.exists(empty_db):
        os.remove(empty_db)
        
    # B. Corrupted embedding records
    corrupt_db = "face_recognition/temp_corrupt.sqlite"
    if os.path.exists(corrupt_db):
        os.remove(corrupt_db)
        
    # Create tables
    conn = sqlite3.connect(corrupt_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_registry (
            user_id TEXT PRIMARY KEY,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS registry_metadata (
            meta_key TEXT PRIMARY KEY,
            meta_value TEXT
        )
    """)
    cursor.execute("INSERT OR REPLACE INTO registry_metadata (meta_key, meta_value) VALUES ('model_name', 'mobilefacenet')")
    cursor.execute("INSERT OR REPLACE INTO registry_metadata (meta_key, meta_value) VALUES ('embedding_dim', '128')")
    
    # Insert valid embedding
    valid_emb = np.random.randn(128).astype(np.float32)
    valid_emb = valid_emb / np.linalg.norm(valid_emb)
    cursor.execute("INSERT INTO face_registry (user_id, embedding) VALUES (?, ?)", ("valid_1", sqlite3.Binary(valid_emb.tobytes())))
    
    # Insert corrupted blobs (< 512 bytes, e.g. 100 bytes; > 512 bytes, e.g. 1000 bytes)
    corrupt_small = np.random.randn(25).astype(np.float32).tobytes() # 100 bytes
    corrupt_large = np.random.randn(250).astype(np.float32).tobytes() # 1000 bytes
    
    cursor.execute("INSERT INTO face_registry (user_id, embedding) VALUES (?, ?)", ("corrupt_small", sqlite3.Binary(corrupt_small)))
    cursor.execute("INSERT INTO face_registry (user_id, embedding) VALUES (?, ?)", ("corrupt_large", sqlite3.Binary(corrupt_large)))
    conn.commit()
    conn.close()
    
    # Initialize SQLiteFaceRegistry
    reg_corrupt = SQLiteFaceRegistry(db_path=corrupt_db)
    
    print("\nCorrupted Registry Validation:")
    print(f"  - Loaded cache users: {reg_corrupt.cache_users}")
    print(f"  - Cache embeddings shape: {reg_corrupt.cache_embeddings.shape}")
    
    # Assert corrupt ones are skipped and only valid_1 is loaded
    assert len(reg_corrupt.cache_users) == 1, "Expected only 1 user loaded"
    assert "valid_1" in reg_corrupt.cache_users, "Expected 'valid_1' to be loaded"
    assert "corrupt_small" not in reg_corrupt.cache_users, "Expected 'corrupt_small' to be skipped"
    assert "corrupt_large" not in reg_corrupt.cache_users, "Expected 'corrupt_large' to be skipped"
    print("  [PASS] Corrupted records correctly filtered. Cache remains valid.")
    
    # Verify lookup still works
    matched_id, score = reg_corrupt.verify(valid_emb, threshold=0.60)
    print(f"  - verify() lookup of valid_1: ({matched_id}, {score:.4f})")
    assert matched_id == "valid_1", "Expected lookup of valid_1 to succeed"
    print("  [PASS] Verification against registry containing corrupted records works correctly.")
    
    if os.path.exists(corrupt_db):
        os.remove(corrupt_db)
        
    print("\n=======================================================")
    print("      ALL ENGINEERING VALIDATION SECTIONS PASSED!      ")
    print("=======================================================")


if __name__ == "__main__":
    main()
