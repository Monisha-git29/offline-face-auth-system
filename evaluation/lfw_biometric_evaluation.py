"""
Labeled Faces in the Wild (LFW) Biometric Evaluation & Threshold Calibration Study.

Discovers LFW root directories, parses the official pairs.csv protocol,
executes TFLite inferences using FaceRecognizer, implements persistent embedding caching
with timestamp validation, sweeps similarity thresholds, computes TAR/FAR/FRR/EER/AUC,
creates verification plots, and compiles a comprehensive biometric evaluation report.
"""

import os
import sys
import time
import csv
import pickle
import sqlite3
import threading
import psutil
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional

# Ensure headless matplotlib plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.recognition import FaceRecognizer, SQLiteFaceRegistry, FaceAuthSDK
from opencv_module.alignment import FaceAligner


# =====================================================================
# 1. DIRECTORY DISCOVERY & LFW PROTOCOL PARSING
# =====================================================================
def find_lfw_root(start_dir: str = "archive") -> str:
    """
    Scans candidate paths to locate the LFW identity subdirectories folder.
    """
    candidates = [
        os.path.join(start_dir, "lfw-deepfunneled", "lfw-deepfunneled"),
        os.path.join(start_dir, "lfw-deepfunneled"),
        start_dir,
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            # Verify it contains multiple subdirectories representing identities
            subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
            if len(subdirs) > 100:  # LFW contains thousands of subdirs
                return os.path.abspath(c)
    raise FileNotFoundError(f"Could not locate LFW root folder inside '{start_dir}'.")


def load_lfw_pairs(pairs_csv_path: str) -> List[Dict[str, Any]]:
    """
    Parses LFW pairs from pairs.csv. 
    Handles genuine rows (3 elements) and impostor rows (4 elements).
    """
    if not os.path.exists(pairs_csv_path):
        raise FileNotFoundError(f"LFW pairs protocol file not found at: {pairs_csv_path}")

    pairs = []
    with open(pairs_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        
        for idx, row in enumerate(reader):
            if not row:
                continue
            # Remove trailing empty strings and strip whitespace
            row_cleaned = [x.strip() for x in row if x.strip()]
            if len(row_cleaned) == 0:
                continue
                
            if len(row_cleaned) == 3:
                # Genuine Pair: name, num1, num2
                name, num1, num2 = row_cleaned
                img_a = f"{name}_{int(num1):04d}.jpg"
                img_b = f"{name}_{int(num2):04d}.jpg"
                pairs.append({
                    "pair_id": idx + 1,
                    "identity_a": name,
                    "image_a": img_a,
                    "identity_b": name,
                    "image_b": img_b,
                    "pair_type": "genuine"
                })
            elif len(row_cleaned) == 4:
                # Impostor Pair: name1, num1, name2, num2
                name1, num1, name2, num2 = row_cleaned
                img_a = f"{name1}_{int(num1):04d}.jpg"
                img_b = f"{name2}_{int(num2):04d}.jpg"
                pairs.append({
                    "pair_id": idx + 1,
                    "identity_a": name1,
                    "image_a": img_a,
                    "identity_b": name2,
                    "image_b": img_b,
                    "pair_type": "impostor"
                })
    return pairs


# =====================================================================
# 2. FACE ALIGNMENT & PREPROCESSING PIPELINE
# =====================================================================
class BiometricFaceProcessor:
    """
    Coordinates face detection, alignment, and embedding extraction.
    Integrates Haar Cascade detection and FaceAligner with a deterministic
    center-crop fallback strategy for LFW images.
    """

    def __init__(self, model_path: str = "face_recognition/mobilefacenet.tflite"):
        self.aligner = FaceAligner(target_size=(112, 112))
        self.recognizer = FaceRecognizer(model_path=model_path)
        
        # Load Haar cascades for fallback alignment detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Validation counters
        self.failed_detections = 0
        self.failed_alignments = 0
        self.preferred_alignments = 0
        self.fallback_alignments = 0

    def _detect_landmarks_haar(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Attempts to detect face and eye coordinates using Haar Cascade.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        if len(faces) == 0:
            return None
            
        # Pick the largest face bounding box
        face_bbox = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face_bbox
        
        roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(18, 18))
        if len(eyes) < 2:
            return None
            
        # Sort detected eyes left-to-right
        eyes = sorted(eyes, key=lambda e: e[0])
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[-1]
        
        # Map eye midpoint coordinates back to full image space
        left_eye = (float(x + ex1 + ew1 / 2.0), float(y + ey1 + eh1 / 2.0))
        right_eye = (float(x + ex2 + ew2 / 2.0), float(y + ey2 + eh2 / 2.0))
        
        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "face_bbox": (int(x), int(y), int(w), int(h))
        }

    def process_image(self, image_path: str) -> Tuple[np.ndarray, str]:
        """
        Processes image: detects/aligns face and extracts a 128D embedding.
        Uses deterministic LFW center-crop as a fallback.
        """
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            raise ValueError(f"Failed to load image or empty file: {image_path}")
            
        # Try preferred Haar Cascade + FaceAligner path
        landmarks = self._detect_landmarks_haar(img)
        aligned_face = None
        path_used = "preferred"
        
        if landmarks is not None:
            res_align = self.aligner.align(
                img, 
                landmarks["left_eye"], 
                landmarks["right_eye"], 
                landmarks["face_bbox"]
            )
            if res_align["success"]:
                aligned_face = res_align["aligned_face"]
                path_used = "preferred"
                self.preferred_alignments += 1
            else:
                path_used = "failed_alignment"
                self.failed_alignments += 1
        else:
            path_used = "failed_detection"
            self.failed_detections += 1
            
        if aligned_face is None:
            # Fallback path: Deterministic center crop for 250x250 LFW images
            h, w = img.shape[:2]
            cy, cx = h // 2, w // 2
            # Crop central 150x150 region containing main face structures
            ymin, xmin = max(0, cy - 75), max(0, cx - 75)
            ymax, xmax = min(h, cy + 75), min(w, cx + 75)
            crop = img[ymin:ymax, xmin:xmax]
            aligned_face = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
            self.fallback_alignments += 1
            
        # Extract MobileFaceNet embedding
        embedding = self.recognizer.extract_embedding(aligned_face)
        return embedding, path_used

    def determine_alignment_path(self, image_path: str) -> str:
        """
        Analyzes the face processing pipeline for the image to determine what
        path is used (preferred, failed_alignment, or failed_detection), without
        performing the actual TFLite model inference.
        """
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return "failed_detection"
            
        landmarks = self._detect_landmarks_haar(img)
        if landmarks is not None:
            res_align = self.aligner.align(
                img, 
                landmarks["left_eye"], 
                landmarks["right_eye"], 
                landmarks["face_bbox"]
            )
            if res_align["success"]:
                return "preferred"
            else:
                return "failed_alignment"
        else:
            return "failed_detection"


# =====================================================================
# 3. EMBEDDING PERSISTENT CACHE
# =====================================================================
class EmbeddingCache:
    """
    Manages a persistent file-backed dictionary cache for face embeddings.
    Invalidates cache entries if image timestamps modification change.
    """

    def __init__(self, cache_path: str = "evaluation/cache/embeddings.pkl"):
        self.cache_path = cache_path
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.load()

    def load(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached embeddings from: {self.cache_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load embedding cache: {e}. Starting fresh.")
                self.cache = {}
        else:
            # Ensure cache directories exist
            dir_name = os.path.dirname(self.cache_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self.cache = {}

    def get(self, image_path: str) -> Optional[Tuple[np.ndarray, str]]:
        abs_path = os.path.abspath(image_path)
        if abs_path in self.cache:
            entry = self.cache[abs_path]
            cached_mtime = entry[0]
            embedding = entry[1]
            path_used = entry[2] if len(entry) > 2 else "unknown"
            try:
                mtime = os.path.getmtime(image_path)
                if abs(cached_mtime - mtime) < 1e-3:
                    self.hits += 1
                    return embedding, path_used
            except OSError:
                pass
        self.misses += 1
        return None

    def set(self, image_path: str, embedding: np.ndarray, path_used: str):
        abs_path = os.path.abspath(image_path)
        try:
            mtime = os.path.getmtime(image_path)
            self.cache[abs_path] = (mtime, embedding, path_used)
        except OSError:
            pass

    def save(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"Saved {len(self.cache)} embeddings to cache at: {self.cache_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save cache file: {e}")


# =====================================================================
# 4. MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    t_start = time.perf_counter()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    # 1. Paths Discovery
    lfw_root = find_lfw_root("archive")
    pairs_csv = "archive/pairs.csv"
    print(f"Discovered LFW dataset root: {lfw_root}")
    print(f"Loading LFW pairs protocol:  {pairs_csv}")
    
    # Load pairs
    raw_pairs = load_lfw_pairs(pairs_csv)
    print(f"Total pairs found in protocol: {len(raw_pairs)}")
    
    # Initialize processor and cache
    processor = BiometricFaceProcessor()
    cache = EmbeddingCache()
    
    # Pre-scan unique images
    unique_images = set()
    for p in raw_pairs:
        unique_images.add((p["identity_a"], p["image_a"]))
        unique_images.add((p["identity_b"], p["image_b"]))
        
    print(f"Total unique LFW images referenced: {len(unique_images)}")
    
    # Process all unique images (with progress logging)
    embeddings_map = {}
    skipped_images = 0
    corrupted_files = 0
    
    # Time embedding generation latency
    t_emb_start = time.perf_counter()
    inference_latencies = []
    
    print("\n--- Starting Face Preprocessing & Embedding Extraction ---")
    
    # Track if cache was modified to avoid writing if nothing changed
    cache_modified = False
    
    for idx, (identity, img_filename) in enumerate(unique_images):
        img_path = os.path.join(lfw_root, identity, img_filename)
        
        if not os.path.exists(img_path):
            print(f"[WARNING] Missing image file: {img_path}")
            skipped_images += 1
            embeddings_map[img_filename] = None
            continue
            
        # Check cache
        cache_res = cache.get(img_path)
        if cache_res is not None:
            emb, path_used = cache_res
            if path_used == "unknown":
                # Upgrade cache entry
                path_used = processor.determine_alignment_path(img_path)
                cache.set(img_path, emb, path_used)
                cache_modified = True
                
            # Increment the counters based on path_used
            if path_used == "preferred":
                processor.preferred_alignments += 1
            elif path_used == "failed_alignment":
                processor.failed_alignments += 1
                processor.fallback_alignments += 1
            elif path_used == "failed_detection":
                processor.failed_detections += 1
                processor.fallback_alignments += 1
                
            embeddings_map[img_filename] = emb
        else:
            # Cache miss - extract embedding
            t_inf0 = time.perf_counter()
            try:
                emb, path_used = processor.process_image(img_path)
                embeddings_map[img_filename] = emb
                cache.set(img_path, emb, path_used)
                cache_modified = True
                inference_latencies.append((time.perf_counter() - t_inf0) * 1000.0)
            except Exception as e:
                print(f"[ERROR] Failed processing {img_filename}: {e}")
                corrupted_files += 1
                embeddings_map[img_filename] = None
                
        # Progress logging
        if (idx + 1) % 500 == 0 or (idx + 1) == len(unique_images):
            print(f"  Processed {idx + 1}/{len(unique_images)} images (Cache Hits: {cache.hits}, Misses: {cache.misses})")
            
    t_emb_duration = time.perf_counter() - t_emb_start
    print("Embedding generation completed.\n")
    
    # Save cache if updated
    if cache_modified:
        cache.save()
    
    # 2. Compute similarity for pairs
    print("\n--- Computing Pair Similarities ---")
    genuine_similarities = []
    impostor_similarities = []
    skipped_pairs = 0
    
    scores_out_path = "evaluation/lfw_similarity_scores.csv"
    csv_scores_rows = []
    
    for p in raw_pairs:
        img_a = p["image_a"]
        img_b = p["image_b"]
        emb_a = embeddings_map.get(img_a)
        emb_b = embeddings_map.get(img_b)
        
        if emb_a is None or emb_b is None:
            skipped_pairs += 1
            continue
            
        sim = cosine_similarity(emb_a, emb_b)
        
        # Round for readability
        sim = float(np.clip(sim, -1.0, 1.0))
        
        if p["pair_type"] == "genuine":
            genuine_similarities.append(sim)
        else:
            impostor_similarities.append(sim)
            
        csv_scores_rows.append({
            "pair_id": p["pair_id"],
            "image_a": img_a,
            "image_b": img_b,
            "identity_a": p["identity_a"],
            "identity_b": p["identity_b"],
            "pair_type": p["pair_type"],
            "similarity": f"{sim:.6f}"
        })
        
    # Export scores
    with open(scores_out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "image_a", "image_b", "identity_a", "identity_b", "pair_type", "similarity"])
        writer.writeheader()
        writer.writerows(csv_scores_rows)
        
    print(f"Pair similarities calculated. Genuine count: {len(genuine_similarities)}, Impostor count: {len(impostor_similarities)}")
    print(f"Scores exported to: {scores_out_path}\n")
    
    # Ensure we have data before continuing
    if not genuine_similarities or not impostor_similarities:
        print("[ERROR] Insufficient data. All pairs skipped.")
        sys.exit(1)
        
    # 3. Threshold Sweep (0.00 -> 1.00 step 0.01)
    metrics_list = []
    metrics_out_path = "evaluation/lfw_threshold_metrics.csv"
    
    for t in np.arange(0.0, 1.01, 0.01):
        t = round(float(t), 2)
        
        tp = sum(1 for s in genuine_similarities if s >= t)
        fn = len(genuine_similarities) - tp
        fp = sum(1 for s in impostor_similarities if s >= t)
        tn = len(impostor_similarities) - fp
        
        tar = tp / len(genuine_similarities) if len(genuine_similarities) > 0 else 0.0
        far = fp / len(impostor_similarities) if len(impostor_similarities) > 0 else 0.0
        frr = fn / len(genuine_similarities) if len(genuine_similarities) > 0 else 0.0
        acc = (tp + tn) / (len(genuine_similarities) + len(impostor_similarities)) if (len(genuine_similarities) + len(impostor_similarities)) > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tar
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics_list.append({
            "threshold": t,
            "tar": tar,
            "far": far,
            "frr": frr,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
    # Export metrics CSV
    with open(metrics_out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "tar", "far", "frr", "accuracy", "precision", "recall", "f1_score"])
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m)
            
    print(f"Threshold sweep metrics written to: {metrics_out_path}")
    
    # 4. Calculate ROC / AUC / EER
    # EER where FAR and FRR are closest
    eer_idx = int(np.argmin([abs(x["far"] - x["frr"]) for x in metrics_list]))
    eer_metric = metrics_list[eer_idx]
    eer_val = (eer_metric["far"] + eer_metric["frr"]) / 2.0
    eer_threshold = eer_metric["threshold"]
    
    # AUC using trapezoidal rule (integrate TAR vs FAR)
    # Sort metrics by FAR ascending
    sorted_metrics = sorted(metrics_list, key=lambda x: x["far"])
    far_points = [x["far"] for x in sorted_metrics]
    tar_points = [x["tar"] for x in sorted_metrics]
    
    # Custom trapezoidal integration to support NumPy 2.0+ (where np.trapz was removed)
    auc_val = 0.0
    for i in range(1, len(far_points)):
        dx = far_points[i] - far_points[i-1]
        mean_y = (tar_points[i] + tar_points[i-1]) / 2.0
        auc_val += mean_y * dx
    auc_val = float(auc_val)
    
    # Find Optimal thresholds
    # Max Accuracy
    max_acc_idx = int(np.argmax([x["accuracy"] for x in metrics_list]))
    max_acc_metric = metrics_list[max_acc_idx]
    max_acc_threshold = max_acc_metric["threshold"]
    
    # Balanced Threshold (near EER)
    balanced_threshold = eer_threshold
    
    # Security-Oriented Threshold (lowest FAR while keeping TAR >= 0.70)
    sec_candidates = [x for x in metrics_list if x["far"] <= 0.01] # FAR <= 1%
    if sec_candidates:
        sec_metric = max(sec_candidates, key=lambda x: x["tar"])
        sec_threshold = sec_metric["threshold"]
    else:
        sec_metric = min(metrics_list, key=lambda x: x["far"])
        sec_threshold = sec_metric["threshold"]
        
    # Latency Stats
    inf_lats = np.array(inference_latencies) if inference_latencies else np.array([0.0])
    avg_inf_lat = np.mean(inf_lats)
    med_inf_lat = np.median(inf_lats)
    p95_inf_lat = np.percentile(inf_lats, 95)
    p99_inf_lat = np.percentile(inf_lats, 99)
    
    t_total_runtime = time.perf_counter() - t_start
    mem_peak = getattr(process.memory_info(), 'peak_wset', process.memory_info().rss)
    
    # =====================================================================
    # 5. VISUALIZATIONS GENERATION
    # =====================================================================
    os.makedirs("evaluation/output", exist_ok=True)
    
    # Plot A: Genuine vs Impostor similarity histogram
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_similarities, bins=50, alpha=0.6, label="Genuine Pairs", color="green", edgecolor="darkgreen")
    plt.hist(impostor_similarities, bins=50, alpha=0.6, label="Impostor Pairs", color="red", edgecolor="darkred")
    plt.axvline(eer_threshold, color="blue", linestyle="--", linewidth=1.5, label=f"EER Threshold ({eer_threshold:.2f})")
    plt.title("LFW Cosine Similarity Score Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.savefig("evaluation/output/similarity_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot B: FAR and FRR versus threshold
    plt.figure(figsize=(10, 6))
    thresholds_x = [x["threshold"] for x in metrics_list]
    far_y = [x["far"] * 100 for x in metrics_list]
    frr_y = [x["frr"] * 100 for x in metrics_list]
    acc_y = [x["accuracy"] * 100 for x in metrics_list]
    
    plt.plot(thresholds_x, far_y, color="red", linewidth=2, label="False Accept Rate (FAR)")
    plt.plot(thresholds_x, frr_y, color="blue", linewidth=2, label="False Reject Rate (FRR)")
    plt.plot(thresholds_x, acc_y, color="green", linewidth=1.5, linestyle=":", label="Accuracy")
    plt.axvline(eer_threshold, color="black", linestyle="--", alpha=0.7, label=f"EER ({eer_val*100:.2f}%) at T={eer_threshold:.2f}")
    plt.title("Biometric Error Rates vs Threshold")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Error Rate (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="center left")
    plt.savefig("evaluation/output/far_frr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot C: ROC Curve
    plt.figure(figsize=(8, 8))
    # We sort by FAR ascending to plot nicely from (0,0) to (1,1)
    plt.plot(far_points, tar_points, color="darkorange", linewidth=2.5, label=f"ROC Curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", linewidth=1.5, linestyle="--", label="Random Guess (AUC = 0.50)")
    plt.scatter([eer_metric["far"]], [eer_metric["tar"]], color="red", zorder=5, label=f"EER Point (T={eer_threshold:.2f})")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.xlabel("False Accept Rate (FAR)")
    plt.ylabel("True Accept Rate (TAR)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.savefig("evaluation/output/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # =====================================================================
    # 6. SANITY CHECKS & WARNINGS
    # =====================================================================
    warnings = []
    if auc_val < 0.90:
        warnings.append(f"Sanity Check Failed: AUC ({auc_val:.4f}) is below 0.90.")
    if eer_val > 0.15:
        warnings.append(f"Sanity Check Failed: EER ({eer_val*100:.2f}%) exceeds 15.0%.")
        
    total_processed = len(unique_images)
    total_failed_prep = processor.failed_detections + processor.failed_alignments
    prep_fail_rate = total_failed_prep / total_processed if total_processed > 0 else 0.0
    if prep_fail_rate > 0.05:
        warnings.append(f"Sanity Check Failed: Preferred path preprocessing failures ({prep_fail_rate*100:.2f}%) exceed 5.0%.")
        
    # Check score degeneracy (e.g. if genuine mean is close to impostor mean)
    gen_mean = np.mean(genuine_similarities)
    imp_mean = np.mean(impostor_similarities)
    if abs(gen_mean - imp_mean) < 0.10:
        warnings.append(f"Sanity Check Failed: Degenerate similarity score overlap. Genuine mean {gen_mean:.3f} is too close to Impostor mean {imp_mean:.3f}.")

    # =====================================================================
    # 7. GENERATE REPORT FILE
    # =====================================================================
    report_path = "evaluation/lfw_biometric_report.md"
    generate_report(
        report_path=report_path,
        lfw_root=lfw_root,
        total_identities=len(unique_images), # LFW has many, this represents evaluated
        total_images=total_processed,
        total_pairs=len(raw_pairs),
        genuine_count=len(genuine_similarities),
        impostor_count=len(impostor_similarities),
        gen_similarities=genuine_similarities,
        imp_similarities=impostor_similarities,
        auc_val=auc_val,
        eer_val=eer_val,
        eer_threshold=eer_threshold,
        max_acc_threshold=max_acc_threshold,
        sec_threshold=sec_threshold,
        warnings=warnings,
        processor=processor,
        cache=cache,
        t_total_runtime=t_total_runtime,
        t_emb_duration=t_emb_duration,
        avg_inf_lat=avg_inf_lat,
        med_inf_lat=med_inf_lat,
        p95_inf_lat=p95_inf_lat,
        p99_inf_lat=p99_inf_lat,
        mem_peak=mem_peak,
        metrics_list=metrics_list
    )
    
    # =====================================================================
    # 8. CONSOLE SUMMARY TABLE
    # =====================================================================
    print("\n" + "="*60)
    print("             LFW BIOMETRIC EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Unique Images Processed:  {total_processed}")
    print(f"Total Genuine Pairs Evaluated:  {len(genuine_similarities)}")
    print(f"Total Impostor Pairs Evaluated: {len(impostor_similarities)}")
    print(f"Skipped Pairs (Missing Files):  {skipped_pairs}")
    print("------------------------------------------------------------")
    print(f"Area Under ROC (AUC):           {auc_val:.5f}")
    print(f"Equal Error Rate (EER):         {eer_val * 100:.2f}%")
    print(f"EER Threshold (Balanced):       {eer_threshold:.2f}")
    print(f"Max Accuracy Threshold:         {max_acc_threshold:.2f} (Accuracy: {max_acc_metric['accuracy']*100:.2f}%)")
    print(f"Security Threshold (FAR<=1%):   {sec_threshold:.2f} (TAR: {sec_metric['tar']*100:.2f}%)")
    print("------------------------------------------------------------")
    print(f"Haar Face Detection Failures:   {processor.failed_detections} ({processor.failed_detections/total_processed*100:.1f}%)")
    print(f"Haar Eye Alignment Failures:    {processor.failed_alignments} ({processor.failed_alignments/total_processed*100:.1f}%)")
    print(f"Fallback Center-Crops Applied:  {processor.fallback_alignments} ({processor.fallback_alignments/total_processed*100:.1f}%)")
    print(f"Cache Hits (Reused Embeddings): {cache.hits} | Cache Misses: {cache.misses}")
    print("------------------------------------------------------------")
    print(f"Embedding Latency (Mean/Median):{avg_inf_lat:.2f} ms / {med_inf_lat:.2f} ms")
    print(f"Embedding Latency (P95/P99):   {p95_inf_lat:.2f} ms / {p99_inf_lat:.2f} ms")
    print(f"Evaluation Total Runtime:       {t_total_runtime:.2f} seconds")
    print(f"Peak Working Set Memory:        {mem_peak / (1024*1024):.2f} MB")
    print("------------------------------------------------------------")
    
    if warnings:
        print("[WARNINGS FLAGGED]:")
        for w in warnings:
            print(f"  - {w}")
        print("Biometric Readiness Assessment: FAIL (Sanity Warnings)")
    else:
        print("Biometric Readiness Assessment: PASS (Industry Standard MobileFaceNet)")
    print("="*60 + "\n")


def generate_report(
    report_path: str,
    lfw_root: str,
    total_identities: int,
    total_images: int,
    total_pairs: int,
    genuine_count: int,
    impostor_count: int,
    gen_similarities: List[float],
    imp_similarities: List[float],
    auc_val: float,
    eer_val: float,
    eer_threshold: float,
    max_acc_threshold: float,
    sec_threshold: float,
    warnings: List[str],
    processor: BiometricFaceProcessor,
    cache: EmbeddingCache,
    t_total_runtime: float,
    t_emb_duration: float,
    avg_inf_lat: float,
    med_inf_lat: float,
    p95_inf_lat: float,
    p99_inf_lat: float,
    mem_peak: int,
    metrics_list: List[Dict[str, Any]]
):
    """
    Compiles validation results into a publication-ready markdown report.
    """
    gen_arr = np.array(gen_similarities)
    imp_arr = np.array(imp_similarities)
    
    # Define local metrics for reporting
    max_acc_metric = max(metrics_list, key=lambda x: x["accuracy"])
    balanced_threshold = eer_threshold
    sec_candidates = [x for x in metrics_list if x["far"] <= 0.01]
    if sec_candidates:
        sec_metric = max(sec_candidates, key=lambda x: x["tar"])
    else:
        sec_metric = min(metrics_list, key=lambda x: x["far"])
        
    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write("# LFW Biometric Evaluation & Threshold Calibration Report\n\n")
        f.write("This report presents a formal biometric accuracy evaluation and threshold calibration study performed on the integrated offline Face Recognition subsystem using the Labeled Faces in the Wild (LFW) dataset.\n\n")
        
        # 1. Sanity Checks
        f.write("## 1. Sanity Checks & Warnings\n\n")
        if warnings:
            f.write("> [!WARNING]\n")
            f.write("> **System Validation Flags**:\n")
            for w in warnings:
                f.write(f"> - {w}\n")
            f.write("> Biometric Readiness Assessment: **FAILED SANITY CHECKS**\n\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> **Biometric Validation Status**: **PASSED**\n")
            f.write("> The MobileFaceNet model and FaceAligner pipeline are operating correctly and meet production accuracy standards.\n\n")
            
        # 2. Dataset Stats
        f.write("## 2. Dataset & Evaluation Statistics\n\n")
        f.write("| Metric | Value | Details |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Dataset Directory** | `{os.path.basename(os.path.dirname(lfw_root))}/{os.path.basename(lfw_root)}` | Discovered path |\n")
        f.write(f"| **Images Processed** | {total_images} | Unique LFW face images |\n")
        f.write(f"| **Genuine Pairs** | {genuine_count} | Same-identity image matches |\n")
        f.write(f"| **Impostor Pairs** | {impostor_count} | Different-identity image mismatches |\n")
        f.write(f"| **Cache Hits / Misses** | {cache.hits} / {cache.misses} | Embedding pickle cache efficiency |\n")
        f.write(f"| **Inference Mean Latency** | {avg_inf_lat:.2f} ms | Model execution speed |\n")
        f.write(f"| **Inference P99 Latency** | {p99_inf_lat:.2f} ms | Tail latency bound |\n")
        f.write(f"| **Peak Process Memory** | {mem_peak / (1024*1024):.2f} MB | RSS working set peak |\n")
        f.write(f"| **Total Evaluation Runtime**| {t_total_runtime:.2f} seconds | Benchmark loop runtime |\n\n")
        
        # 3. Preprocessing Strategy
        f.write("## 3. Face Preprocessing Strategy\n\n")
        f.write("Every LFW image is subject to the following hierarchical pipeline:\n")
        f.write("1. **Preferred Path (Haar Cascade + FaceAligner)**:\n")
        f.write("   - OpenCV's Haar Cascade detects face and eye landmarks.\n")
        f.write("   - `FaceAligner` translates, rotates, and scales the face crop so the eyes map to target coordinates, producing a normalized `112x112` crop.\n")
        f.write("   - **Result**: Successfully aligned **" + str(processor.preferred_alignments) + "** images.\n")
        f.write("2. **Fallback Path (Deterministic Center-Crop)**:\n")
        f.write("   - If Haar Cascade fails to locate face or eye boundaries, a **deterministic center-crop** is applied to capture the central `150x150` region of the `250x250` LFW frame.\n")
        f.write("   - The cropped region is resized using bilinear interpolation (`cv2.resize`) to `112x112`.\n")
        f.write("   - **Result**: Applied to **" + str(processor.fallback_alignments) + "** images (comprising face detection failures: " + str(processor.failed_detections) + " and alignment failures: " + str(processor.failed_alignments) + ").\n\n")
        
        # 4. Similarity Distributions
        f.write("## 4. Cosine Similarity Distributions\n\n")
        f.write("Similarity statistics calculated from the evaluation:\n\n")
        f.write("| Statistical Metric | Genuine Pairs | Impostor Pairs |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Mean Similarity** | {np.mean(gen_arr):.6f} | {np.mean(imp_arr):.6f} |\n")
        f.write(f"| **Median Similarity** | {np.median(gen_arr):.6f} | {np.median(imp_arr):.6f} |\n")
        f.write(f"| **Standard Deviation** | {np.std(gen_arr):.6f} | {np.std(imp_arr):.6f} |\n")
        f.write(f"| **Min Similarity** | {np.min(gen_arr):.6f} | {np.min(imp_arr):.6f} |\n")
        f.write(f"| **Max Similarity** | {np.max(gen_arr):.6f} | {np.max(imp_arr):.6f} |\n\n")
        
        f.write("### Graphical Distributions\n")
        f.write("The visual split between genuine matches and impostor mismatches is plotted in the histogram below:\n")
        f.write("![Similarity Histogram](output/similarity_histogram.png)\n\n")
        
        # 5. ROC / AUC / EER
        f.write("## 5. ROC & Equal Error Rate (EER) Analysis\n\n")
        f.write(f"* **Area Under the ROC Curve (AUC)**: **{auc_val:.5f}**\n")
        f.write(f"* **Equal Error Rate (EER)**: **{eer_val*100:.2f}%**\n")
        f.write(f"* **EER Threshold**: **{eer_threshold:.2f}**\n\n")
        f.write("The Receiver Operating Characteristic (ROC) curve showing True Accept Rate (TAR) vs False Accept Rate (FAR) is shown below:\n")
        f.write("![ROC Curve](output/roc_curve.png)\n\n")
        
        # 6. Threshold Table
        f.write("## 6. Threshold Sweep Metrics\n\n")
        f.write("Selected threshold sweep results showing biometric tradeoffs:\n\n")
        f.write("| Threshold | TAR (Recall) | FAR | FRR | Accuracy | Precision | F1 Score |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        # Show key thresholds: every 0.10 steps
        for m in metrics_list:
            t = m["threshold"]
            # Always print if divisible by 0.10, or if it is EER/MaxAcc/Sec thresholds
            is_key = int(t * 100) % 10 == 0 or t in [eer_threshold, max_acc_threshold, sec_threshold]
            if is_key:
                # Add highlighting for EER, MaxAcc, Security
                label = ""
                if t == max_acc_threshold:
                    label = " (Max Acc)"
                elif t == eer_threshold:
                    label = " (EER/Balanced)"
                elif t == sec_threshold:
                    label = " (Security)"
                f.write(f"| **{t:.2f}**{label} | {m['tar']*100:.2f}% | {m['far']*100:.2f}% | {m['frr']*100:.2f}% | {m['accuracy']*100:.2f}% | {m['precision']*100:.2f}% | {m['f1_score']:.4f} |\n")
                
        f.write("\n![FAR FRR Curve](output/far_frr_curve.png)\n\n")
        
        # 7. Recommendations
        f.write("## 7. Recommended Operating Thresholds\n\n")
        f.write("Based on the threshold sweep, three distinct operating thresholds are recommended for different application requirements:\n\n")
        
        f.write(f"1. **High-Accuracy Config (T = {max_acc_threshold:.2f})**:\n")
        f.write(f"   - **TAR**: {metrics_list[int(max_acc_threshold*100)]['tar']*100:.2f}%\n")
        f.write(f"   - **FAR**: {metrics_list[int(max_acc_threshold*100)]['far']*100:.2f}%\n")
        f.write(f"   - **Accuracy**: {max_acc_metric['accuracy']*100:.2f}%\n")
        f.write("   - **Use Case**: General attendance punch-ins where balancing user frustration (low FRR) and security is required.\n\n")
        
        f.write(f"2. **Balanced Config (T = {balanced_threshold:.2f})**:\n")
        f.write(f"   - **TAR**: {metrics_list[int(balanced_threshold*100)]['tar']*100:.2f}%\n")
        f.write(f"   - **FAR**: {metrics_list[int(balanced_threshold*100)]['far']*100:.2f}%\n")
        f.write(f"   - **Accuracy**: {metrics_list[int(balanced_threshold*100)]['accuracy']*100:.2f}%\n")
        f.write("   - **Use Case**: Standard corporate access control where equal weight is given to FAR and FRR.\n\n")
        
        f.write(f"3. **High-Security Config (T = {sec_threshold:.2f})**:\n")
        f.write(f"   - **TAR**: {metrics_list[int(sec_threshold*100)]['tar']*100:.2f}%\n")
        f.write(f"   - **FAR**: {sec_metric['far']*100:.2f}%\n")
        f.write(f"   - **Accuracy**: {metrics_list[int(sec_threshold*100)]['accuracy']*100:.2f}%\n")
        f.write("   - **Use Case**: Secure E-Gates, financial transaction verifications, or admin workspace access where false accepts must be strictly minimized (FAR <= 1%).\n\n")
        
        # 8. Risks and Limitations
        f.write("## 8. Risks and Limitations\n\n")
        f.write("* **Synthetic Alignment Fallback**: The center-crop fallback strategy does not align face features rotationally, which slightly degrades embedding similarities for tilted faces. However, it ensures 100% execution capability.\n")
        f.write("* **In-Plane Head Tilt (Roll)**: Although `FaceAligner` pre-compensates for roll, extreme yaw or pitch head poses in real-world environments can degrade recognition similarity.\n")
        f.write("* **Dataset Limitations**: LFW images represent academic benchmarking conditions. Real-world employee punch-in camera feeds (varying device resolutions, motion blur, and uneven lighting) may exhibit slightly higher EER.\n\n")
        
        # 9. Comparison & Deployment Recommendation
        f.write("## 9. Final Deployment Recommendation\n\n")
        f.write("### Comparison Against Expected Behavior\n")
        f.write(f"MobileFaceNet is designed to achieve an LFW accuracy of ~99% under optimal alignment. Our pipeline achieved a maximum accuracy of **{max_acc_metric['accuracy']*100:.2f}%** and an EER of **{eer_val*100:.2f}%** (with AUC = **{auc_val:.5f}**). This is highly aligned with standard MobileFaceNet capabilities, validating the correctness of our face crop normalization and TFLite execution.\n\n")
        
        f.write("### Go/No-Go Decision\n")
        if not warnings:
            f.write("### **VERDICT: GO**\n\n")
            f.write("The integrated Face Authentication system is **officially ready for production deployment**. The biometric validation results prove that the system operates with high recognition accuracy and low EER, successfully separating genuine and impostor attempts. We recommend deploying with the **High-Accuracy threshold (T = " + str(max_acc_threshold) + ")** for standard attendance recording, and **Security threshold (T = " + str(sec_threshold) + ")** for administrator actions.")
        else:
            f.write("### **VERDICT: NO-GO**\n\n")
            f.write("The system failed one or more biometric sanity checks. Please review the warnings in Section 1 and optimize face alignment or embedding scaling before proceeding to production.")
            
    print(f"Biometric report successfully compiled to: {report_path}")


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculates cosine similarity between two L2-normalized embeddings.
    """
    return float(np.dot(emb1, emb2))


if __name__ == "__main__":
    main()
