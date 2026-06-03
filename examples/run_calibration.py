"""
Liveness Calibration Execution Helper.

Allows executing the CalibrationAnalyzer class directly from the examples folder.
"""

import os
import sys

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.calibrate import CalibrationAnalyzer


def main():
    csv_path = "evaluation_results.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Evaluation CSV report file '{csv_path}' is missing.")
        print("        Please run the evaluation system first using:")
        print("        python examples/real_world_evaluation.py --simulate --runs 3")
        sys.exit(1)

    print(f"Reading evaluation records from: {os.path.abspath(csv_path)}")
    analyzer = CalibrationAnalyzer(csv_path)
    analyzer.print_publication_report()


if __name__ == "__main__":
    main()
