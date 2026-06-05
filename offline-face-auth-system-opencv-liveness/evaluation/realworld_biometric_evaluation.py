"""
Real-world Biometric Evaluation & Threshold Calibration.

Loads similarity scores, sweeps verification thresholds, computes FAR/FRR/TAR/Accuracy/EER/AUC,
identifies calibration targets, and saves metrics and plots.
"""

import os
import sys
import csv
import numpy as np

# Ensure headless matplotlib plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print("=" * 70)
    print("        REAL-WORLD PILOT BIOMETRIC EVALUATION & CALIBRATION")
    print("=" * 70)

    scores_csv_path = "evaluation/realworld_similarity_scores.csv"
    metrics_out_path = "evaluation/realworld_threshold_metrics.csv"
    output_dir = "evaluation/output"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(scores_csv_path):
        print(f"[ERROR] Similarity scores file not found at: {scores_csv_path}")
        print("        Please run python evaluation/build_realworld_dataset.py first.")
        sys.exit(1)

    genuine_scores = []
    impostor_scores = []

    # Read scores
    with open(scores_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["SimilarityScore"])
            attempt_type = row["AttemptType"].strip().lower()
            if attempt_type == "genuine":
                genuine_scores.append(score)
            elif attempt_type == "impostor":
                impostor_scores.append(score)

    if not genuine_scores or not impostor_scores:
        print("[ERROR] Insufficient data. Need both genuine and impostor attempts to run biometric evaluation.")
        print(f"        Genuine count: {len(genuine_scores)}, Impostor count: {len(impostor_scores)}")
        sys.exit(1)

    # Threshold sweep from 0.00 to 1.00 step 0.01
    metrics_list = []
    for t in np.arange(0.0, 1.01, 0.01):
        t = round(float(t), 2)
        
        tp = sum(1 for s in genuine_scores if s >= t)
        fn = len(genuine_scores) - tp
        fp = sum(1 for s in impostor_scores if s >= t)
        tn = len(impostor_scores) - fp
        
        tar = tp / len(genuine_scores) if len(genuine_scores) > 0 else 0.0
        far = fp / len(impostor_scores) if len(impostor_scores) > 0 else 0.0
        frr = fn / len(genuine_scores) if len(genuine_scores) > 0 else 0.0
        acc = (tp + tn) / (len(genuine_scores) + len(impostor_scores)) if (len(genuine_scores) + len(impostor_scores)) > 0 else 0.0
        
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

    # Save metrics to CSV
    with open(metrics_out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "tar", "far", "frr", "accuracy", "precision", "recall", "f1_score"])
        writer.writeheader()
        writer.writerows(metrics_list)
    print(f"[SUCCESS] Threshold metrics saved to: {metrics_out_path}")

    # Compute EER
    eer_idx = int(np.argmin([abs(x["far"] - x["frr"]) for x in metrics_list]))
    eer_metric = metrics_list[eer_idx]
    eer_val = (eer_metric["far"] + eer_metric["frr"]) / 2.0
    eer_threshold = eer_metric["threshold"]

    # Compute AUC (trapezoidal integration)
    sorted_metrics = sorted(metrics_list, key=lambda x: x["far"])
    far_points = [x["far"] for x in sorted_metrics]
    tar_points = [x["tar"] for x in sorted_metrics]
    auc_val = 0.0
    for i in range(1, len(far_points)):
        dx = far_points[i] - far_points[i-1]
        mean_y = (tar_points[i] + tar_points[i-1]) / 2.0
        auc_val += mean_y * dx
    auc_val = float(auc_val)

    # Key threshold targets
    # 1. Max Accuracy
    max_acc_idx = int(np.argmax([x["accuracy"] for x in metrics_list]))
    max_acc_metric = metrics_list[max_acc_idx]
    max_acc_val = max_acc_metric["accuracy"]
    max_acc_threshold = max_acc_metric["threshold"]

    # 2. Balanced (EER)
    balanced_threshold = eer_threshold
    balanced_acc = eer_metric["accuracy"]

    # 3. High Security (lowest FAR where TAR >= 0.50, target FAR <= 1%)
    sec_candidates = [x for x in metrics_list if x["far"] <= 0.01]
    if sec_candidates:
        sec_metric = max(sec_candidates, key=lambda x: x["tar"])
        sec_threshold = sec_metric["threshold"]
    else:
        sec_metric = min(metrics_list, key=lambda x: x["far"])
        sec_threshold = sec_metric["threshold"]
    sec_acc = sec_metric["accuracy"]

    # ==========================================
    # GENERATE PLOTS
    # ==========================================
    # 1. ROC Curve
    plt.figure(figsize=(7, 7))
    plt.plot(far_points, tar_points, color="purple", linewidth=2.5, label=f"Real-World Pilot (AUC = {auc_val:.4f}, EER = {eer_val*100:.2f}%)")
    plt.plot([0, 1], [0, 1], color="navy", linewidth=1.5, linestyle="--", label="Random Guess (AUC = 0.50)")
    plt.title("Real-World Pilot Validation - ROC Curve", fontsize=12, fontweight='bold')
    plt.xlabel("False Accept Rate (FAR)", fontsize=10)
    plt.ylabel("True Accept Rate (TAR)", fontsize=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    roc_path = os.path.join(output_dir, "realworld_roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 2. FAR/FRR vs Threshold Curve
    plt.figure(figsize=(9, 5))
    thresholds = [x["threshold"] for x in metrics_list]
    far_vals = [x["far"] for x in metrics_list]
    frr_vals = [x["frr"] for x in metrics_list]
    
    plt.plot(thresholds, far_vals, color="red", linewidth=2, label="False Accept Rate (FAR)")
    plt.plot(thresholds, frr_vals, color="blue", linewidth=2, label="False Reject Rate (FRR)")
    plt.axvline(eer_threshold, color="black", linestyle="--", label=f"EER Threshold ({eer_threshold:.2f})")
    plt.title("FAR and FRR Curves vs Verification Threshold", fontsize=12, fontweight='bold')
    plt.xlabel("Verification Threshold", fontsize=10)
    plt.ylabel("Error Rate", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper center", fontsize=10)
    far_frr_path = os.path.join(output_dir, "realworld_far_frr_curve.png")
    plt.savefig(far_frr_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Similarity Score Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(genuine_scores, bins=25, alpha=0.6, label="Genuine Scores", color="green", edgecolor="darkgreen")
    plt.hist(impostor_scores, bins=25, alpha=0.6, label="Impostor Scores", color="red", edgecolor="darkred")
    plt.axvline(max_acc_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Optimum Accuracy T ({max_acc_threshold:.2f})")
    plt.title("Real-World Cosine Similarity Score Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Cosine Similarity Score", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    hist_path = os.path.join(output_dir, "realworld_similarity_histogram.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Biometric plots generated in: {output_dir}")

    # Console Summary
    print("\n" + "=" * 60)
    print("            BIOMETRIC ANALYSIS EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Area Under ROC (AUC)    : {auc_val:.5f}")
    print(f"Equal Error Rate (EER)  : {eer_val*100:.2f}% (Threshold={eer_threshold:.2f})")
    print(f"Max Biometric Accuracy  : {max_acc_val*100:.2f}% (Threshold={max_acc_threshold:.2f})")
    print(f"High Security Threshold : {sec_threshold:.2f} (FAR={sec_metric['far']*100:.2f}%, TAR={sec_metric['tar']*100:.2f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
