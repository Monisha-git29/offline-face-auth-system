"""
Real-world Pilot Dataset Builder.

Parses authentication logs, separates genuine and impostor attempts,
exports similarity scores to realworld_similarity_scores.csv, and prints
dataset summary statistics.
"""

import os
import sys
import csv
import numpy as np


def main():
    print("=" * 60)
    print("         REAL-WORLD PILOT BIOMETRIC DATASET BUILDER")
    print("=" * 60)

    auth_log_path = "evaluation/logs/authentication_log.csv"
    scores_out_path = "evaluation/realworld_similarity_scores.csv"

    if not os.path.exists(auth_log_path):
        print(f"[ERROR] Authentication log file not found at: {auth_log_path}")
        print("        Please run the pilot application in headless or interactive mode first.")
        sys.exit(1)

    genuine_scores = []
    impostor_scores = []
    records = []

    # Read logs
    with open(auth_log_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for idx, row in enumerate(reader):
            if not row:
                continue
            
            # Fields: Timestamp, ClaimedUserID, ActualUserID, SimilarityScore, Threshold, MatchResult, LivenessResult, Latency, FaceQuality, SessionID, AttemptType
            if len(row) >= 11:
                claimed = row[1]
                actual = row[2]
                score = float(row[3])
                attempt_type = row[10].strip().lower()

                records.append({
                    "AttemptType": attempt_type,
                    "ClaimedUserID": claimed,
                    "ActualUserID": actual,
                    "SimilarityScore": f"{score:.4f}"
                })

                if attempt_type == "genuine":
                    genuine_scores.append(score)
                elif attempt_type == "impostor":
                    impostor_scores.append(score)

    # Export scores
    with open(scores_out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["AttemptType", "ClaimedUserID", "ActualUserID", "SimilarityScore"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[SUCCESS] Dataset successfully built. Exported to: {scores_out_path}")

    # Compute Statistics
    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)
    
    print("\n------------------------------------------------------")
    print("               DATASET STATISTICS SUMMARY")
    print("------------------------------------------------------")
    print(f"Total Genuine Attempts  : {n_gen}")
    print(f"Total Impostor Attempts : {n_imp}")
    print(f"Total Total Attempts    : {n_gen + n_imp}")

    if n_gen > 0:
        gen_mean = np.mean(genuine_scores)
        gen_std = np.std(genuine_scores)
        print(f"Genuine Scores Mean     : {gen_mean:.4f}")
        print(f"Genuine Scores Std      : {gen_std:.4f}")
    else:
        gen_mean = 0.0
        gen_std = 0.0
        print("Genuine Scores Mean     : N/A (no data)")

    if n_imp > 0:
        imp_mean = np.mean(impostor_scores)
        imp_std = np.std(impostor_scores)
        print(f"Impostor Scores Mean    : {imp_mean:.4f}")
        print(f"Impostor Scores Std     : {imp_std:.4f}")
    else:
        imp_mean = 0.0
        imp_std = 0.0
        print("Impostor Scores Mean    : N/A (no data)")

    # Calculate Decidability Index (d')
    # d' = |μ1 - μ2| / sqrt(0.5 * (σ1^2 + σ2^2))
    if n_gen > 1 and n_imp > 1:
        numerator = abs(gen_mean - imp_mean)
        denominator = np.sqrt(0.5 * (gen_std**2 + imp_std**2))
        if denominator > 1e-6:
            d_prime = numerator / denominator
            print(f"Decidability Index (d') : {d_prime:.4f}")
        else:
            print("Decidability Index (d') : N/A (zero variance)")
    print("------------------------------------------------------\n")


if __name__ == "__main__":
    main()
