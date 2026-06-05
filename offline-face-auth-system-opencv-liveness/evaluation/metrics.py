"""
Metrics Calculation and Reporting System.

Provides MetricsCollector class to compute FAR, FRR, spoof detection rates,
and compile liveness assessment runs into structured CSV reports.
"""

import os
import csv
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class MetricsCollector:
    """
    Stateful registry that aggregates session logs and computes performance metrics.
    """

    def __init__(self):
        self.sessions: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """
        Clears all recorded sessions.
        """
        self.sessions = []

    def add_session(
        self,
        session_id: str,
        true_label: str,  # "LIVE" or "SPOOF"
        predicted_decision: str,  # "LIVE", "SUSPECT", "SPOOF"
        success: bool,  # Final verdict boolean
        active_score: float,
        passive_score: float,
        final_score: float,
        duration_ms: float,
        avg_fps: float,
        avg_confidence: float,
        error: Optional[str],
        fqa_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registers a completed evaluation session.
        """
        fqa = fqa_metrics or {}
        session_data = {
            "session_id": session_id,
            "true_label": true_label.upper(),
            "predicted_decision": predicted_decision.upper(),
            "success": success,
            "active_score": active_score,
            "passive_score": passive_score,
            "final_score": final_score,
            "duration_ms": duration_ms,
            "avg_fps": avg_fps,
            "avg_confidence": avg_confidence,
            "error": error if error is not None else "",
            "fqa_brightness": fqa.get("brightness_score", -1),
            "fqa_blur": fqa.get("blur_score", -1),
            "fqa_size": fqa.get("face_size_score", -1),
            "fqa_pose": fqa.get("pose_score", -1),
            "overall_quality": fqa.get("overall_quality_score", -1)
        }
        self.sessions.append(session_data)

    def compute_rates(self) -> Dict[str, Any]:
        """
        Computes security metrics: FAR, FRR, SDR, and Active Success Rate.
        """
        live_sessions = [s for s in self.sessions if s["true_label"] == "LIVE"]
        spoof_sessions = [s for s in self.sessions if s["true_label"] == "SPOOF"]

        n_live = len(live_sessions)
        n_spoof = len(spoof_sessions)

        # 1. False Acceptance Rate (FAR): Spoof predicted as LIVE
        false_acceptances = 0
        for s in spoof_sessions:
            if s["predicted_decision"] == "LIVE" and s["success"]:
                false_acceptances += 1
        far = float(false_acceptances / n_spoof) if n_spoof > 0 else 0.0

        # 2. False Rejection Rate (FRR): Live predicted as SPOOF/SUSPECT or failed liveness
        false_rejections = 0
        for s in live_sessions:
            if s["predicted_decision"] != "LIVE" or not s["success"]:
                false_rejections += 1
        frr = float(false_rejections / n_live) if n_live > 0 else 0.0

        # 3. Spoof Detection Rate (SDR): Spoof correctly detected as SPOOF or SUSPECT
        correct_spoof_detections = 0
        for s in spoof_sessions:
            if s["predicted_decision"] in ["SPOOF", "SUSPECT"] or not s["success"]:
                correct_spoof_detections += 1
        sdr = float(correct_spoof_detections / n_spoof) if n_spoof > 0 else 0.0

        # 4. Active Challenge Success Rate in true LIVE runs
        active_completed = 0
        for s in live_sessions:
            # Active score is 1.0 if all active challenges in the queue were verified
            if s["active_score"] >= 0.99:
                active_completed += 1
        active_rate = float(active_completed / n_live) if n_live > 0 else 0.0

        return {
            "total_sessions": len(self.sessions),
            "n_live_sessions": n_live,
            "n_spoof_sessions": n_spoof,
            "false_accept_count": false_acceptances,
            "false_reject_count": false_rejections,
            "far": far,
            "frr": frr,
            "sdr": sdr,
            "active_challenge_success_rate": active_rate
        }

    def compute_latency_stats(self) -> Dict[str, Any]:
        """
        Computes summary latency metrics for the recorded sessions.
        """
        durations = [s["duration_ms"] for s in self.sessions]
        if not durations:
            return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p99_ms": 0.0}

        return {
            "mean_ms": float(np.mean(durations)),
            "min_ms": float(np.min(durations)),
            "max_ms": float(np.max(durations)),
            "p99_ms": float(np.percentile(durations, 99))
        }

    def save_to_csv(self, filepath: str = "evaluation_results.csv") -> str:
        """
        Writes all logged session records to a CSV file.
        """
        headers = [
            "session_id", "true_label", "predicted_decision", "success",
            "active_score", "passive_score", "final_score", "duration_ms",
            "avg_fps", "avg_confidence", "error", "fqa_brightness",
            "fqa_blur", "fqa_size", "fqa_pose", "overall_quality"
        ]

        # Make parent directories if necessary
        parent_dir = os.path.dirname(filepath)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for s in self.sessions:
                writer.writerow(s)

        return os.path.abspath(filepath)

    def print_summary(self) -> None:
        """
        Prints a formatted text summary of evaluation statistics.
        """
        rates = self.compute_rates()
        lat = self.compute_latency_stats()

        print("\n=======================================================")
        print("             LIVENESS EVALUATION REPORT SUMMARY        ")
        print("=======================================================")
        print(f"Total Sessions Processed: {rates['total_sessions']}")
        print(f"  - True LIVE runs:      {rates['n_live_sessions']}")
        print(f"  - True SPOOF runs:     {rates['n_spoof_sessions']}")
        print("-------------------------------------------------------")
        print("Security Metrics:")
        print(f"  - False Acceptance Rate (FAR):  {rates['far'] * 100:.2f}% ({rates['false_accept_count']}/{rates['n_spoof_sessions']})")
        print(f"  - False Rejection Rate (FRR):   {rates['frr'] * 100:.2f}% ({rates['false_reject_count']}/{rates['n_live_sessions']})")
        print(f"  - Spoof Detection Rate (SDR):   {rates['sdr'] * 100:.2f}%")
        print(f"  - Active Challenge Pass Rate:   {rates['active_challenge_success_rate'] * 100:.2f}%")
        print("-------------------------------------------------------")
        print("Latency Performance:")
        print(f"  - Mean Session Duration:        {lat['mean_ms']:.1f} ms")
        print(f"  - Min Session Duration:         {lat['min_ms']:.1f} ms")
        print(f"  - Max Session Duration:         {lat['max_ms']:.1f} ms")
        print(f"  - 99th Percentile Duration:     {lat['p99_ms']:.1f} ms")
        print("=======================================================\n")
