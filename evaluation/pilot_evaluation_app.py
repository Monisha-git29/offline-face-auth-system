"""
Pilot Evaluation Application for Offline Face Authentication.

Provides a Tkinter dark-themed GUI for user enrollment and authentication,
a real-time stats dashboard, user directory management, scrollable attempt logs,
and automatic CSV logging. Supports a headless simulation mode using sample images.
"""

import os
import sys
import time
import csv
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime

from typing import Dict, Any, Tuple, Optional, List

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.recognition import FaceAuthSDK
from opencv_module import errors
from opencv_module.landmarks import MediaPipeLandmarkDetector
from opencv_module.alignment import FaceAligner


class PilotEvaluationApp:
    def __init__(self, root: tk.Tk, db_path: str = "face_recognition/webcam_pilot_registry.sqlite", headless: bool = False):
        self.root = root
        if self.root is not None:
            self.root.title("Offline Face Auth - Pilot Evaluation Application")
            self.root.geometry("1100x700")
            self.root.configure(bg="#121212")

        self.db_path = db_path
        self.headless = headless
        
        # Configure logging directory
        self.logs_dir = "evaluation/logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        self.enroll_log_path = os.path.join(self.logs_dir, "enrollment_log.csv")
        self.auth_log_path = os.path.join(self.logs_dir, "authentication_log.csv")
        self._init_logs()

        # Initialize FaceAuthSDK
        self.sdk = FaceAuthSDK(
            preset="MEDIUM",
            model_path="face_recognition/mobilefacenet.tflite",
            db_path=self.db_path,
            similarity_threshold=0.60
        )

        # Liveness & session states
        self.active_session = False
        self.session_type = None  # "ENROLL" or "AUTH"
        self.session_user_id = ""
        self.session_actual_id = ""
        self.session_attempt_type = "genuine"
        self.session_start_time = 0.0
        self.session_timeout = 15.0
        self.current_prompt = "WAITING"
        self.last_frame_processed_time = 0.0

        # Video source
        self.cap = None
        self.running = False

        # GUI layout
        if not self.headless:
            self._setup_style()
            self._create_widgets()
            self._start_camera()
            self._update_dashboards()
        else:
            self._run_headless_simulation()

    def _init_logs(self):
        """Creates the CSV log files with headers if they do not exist."""
        if not os.path.exists(self.enroll_log_path):
            with open(self.enroll_log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "UserID", "Success", "FaceQuality", "Latency"])

        if not os.path.exists(self.auth_log_path):
            with open(self.auth_log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "ClaimedUserID", "ActualUserID", "SimilarityScore",
                    "Threshold", "MatchResult", "LivenessResult", "Latency",
                    "FaceQuality", "SessionID", "AttemptType"
                ])

    def _setup_style(self):
        """Sets up custom ttk styling for dark theme aesthetics."""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure frames and notebooks
        style.configure("TFrame", background="#121212")
        style.configure("TNotebook", background="#121212", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1e1e1e", foreground="#FFFFFF", borderwidth=1, padding=8)
        style.map("TNotebook.Tab", background=[("selected", "#bb86fc"), ("active", "#03dac6")], foreground=[("selected", "#000000")])
        
        # Configure treeview (Attempt Logs and User Directory)
        style.configure("Treeview", background="#1e1e1e", foreground="#FFFFFF", fieldbackground="#1e1e1e", rowheight=25)
        style.configure("Treeview.Heading", background="#2a2a2a", foreground="#bb86fc", font=("Arial", 10, "bold"))
        style.map("Treeview", background=[("selected", "#03dac6")], foreground=[("selected", "#000000")])

    def _create_widgets(self):
        """Constructs all GUI panels, viewports, dashboards, and lists."""
        # Top title panel
        title_label = tk.Label(
            self.root,
            text="Offline Face Authentication - Real-World Pilot System",
            bg="#121212",
            fg="#bb86fc",
            font=("Arial", 18, "bold"),
            pady=10
        )
        title_label.pack(side=tk.TOP, fill=tk.X)

        # Main horizontal container
        main_frame = tk.Frame(self.root, bg="#121212")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel: Webcam Viewport
        left_frame = tk.LabelFrame(
            main_frame,
            text=" Live Camera Viewport ",
            bg="#121212",
            fg="#03dac6",
            font=("Arial", 11, "bold"),
            padx=5,
            pady=5
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.video_label = tk.Label(left_frame, bg="#1a1a1a")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right panel: Control Tabs & Analytics
        right_frame = tk.Frame(main_frame, bg="#121212", width=500)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        right_frame.pack_propagate(False)

        # Create control notebook tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Enrollment & Authentication
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text=" Control Room ")
        self._build_control_room(tab1)

        # Tab 2: User Directory
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text=" User Directory ")
        self._build_user_directory(tab2)

        # Tab 3: Statistics Dashboard
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text=" Live Dashboards ")
        self._build_dashboards(tab3)

        # Tab 4: Scrollable Attempt History
        tab4 = ttk.Frame(self.notebook)
        self.notebook.add(tab4, text=" Attempt History ")
        self._build_attempt_history(tab4)

        # Status Bar
        self.status_var = tk.StringVar(value="System Ready. Waiting for action.")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#1e1e1e",
            fg="#03dac6",
            font=("Arial", 10),
            padx=10,
            pady=5
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_control_room(self, parent):
        """Builds enrollment and authentication input panels in Tab 1."""
        container = tk.Frame(parent, bg="#121212")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 1. ENROLLMENT FRAME
        enroll_frame = tk.LabelFrame(container, text=" Enrollee Registration ", bg="#1e1e1e", fg="#bb86fc", font=("Arial", 10, "bold"), padx=10, pady=10)
        enroll_frame.pack(fill=tk.X, pady=5)

        tk.Label(enroll_frame, text="User ID / Employee Code:", bg="#1e1e1e", fg="#FFFFFF", font=("Arial", 9)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.enroll_id_entry = tk.Entry(enroll_frame, bg="#121212", fg="#FFFFFF", insertbackground="#FFFFFF", bd=1, relief=tk.FLAT)
        self.enroll_id_entry.grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)
        enroll_frame.columnconfigure(1, weight=1)

        enroll_btn = tk.Button(enroll_frame, text="Enroll & Register Face", bg="#bb86fc", fg="#000000", activebackground="#03dac6", font=("Arial", 9, "bold"), command=self._on_enroll_click)
        enroll_btn.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)

        # 2. AUTHENTICATION FRAME
        auth_frame = tk.LabelFrame(container, text=" Identity Verification ", bg="#1e1e1e", fg="#bb86fc", font=("Arial", 10, "bold"), padx=10, pady=10)
        auth_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        tk.Label(auth_frame, text="Claimed User ID:", bg="#1e1e1e", fg="#FFFFFF", font=("Arial", 9)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.auth_claimed_entry = tk.Entry(auth_frame, bg="#121212", fg="#FFFFFF", insertbackground="#FFFFFF", bd=1, relief=tk.FLAT)
        self.auth_claimed_entry.grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)

        tk.Label(auth_frame, text="Actual User ID:", bg="#1e1e1e", fg="#FFFFFF", font=("Arial", 9)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.auth_actual_entry = tk.Entry(auth_frame, bg="#121212", fg="#FFFFFF", insertbackground="#FFFFFF", bd=1, relief=tk.FLAT)
        self.auth_actual_entry.grid(row=1, column=1, sticky=tk.EW, padx=10, pady=5)
        auth_frame.columnconfigure(1, weight=1)

        # Attempt Type Radio Buttons
        self.attempt_type_var = tk.StringVar(value="genuine")
        radio_frame = tk.Frame(auth_frame, bg="#1e1e1e")
        radio_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.W)

        genuine_radio = tk.Radiobutton(
            radio_frame, text="Genuine Verification", variable=self.attempt_type_var, value="genuine",
            bg="#1e1e1e", fg="#FFFFFF", selectcolor="#121212", activebackground="#1e1e1e", activeforeground="#FFFFFF",
            command=self._on_genuine_radio_select
        )
        genuine_radio.pack(side=tk.LEFT, padx=5)

        impostor_radio = tk.Radiobutton(
            radio_frame, text="Impostor Validation Test", variable=self.attempt_type_var, value="impostor",
            bg="#1e1e1e", fg="#FFFFFF", selectcolor="#121212", activebackground="#1e1e1e", activeforeground="#FFFFFF"
        )
        impostor_radio.pack(side=tk.LEFT, padx=10)

        # Autofill handler for claimed id
        self.enroll_id_entry.bind("<KeyRelease>", lambda e: self._sync_auth_inputs())
        self.auth_actual_entry.bind("<KeyRelease>", lambda e: self._sync_auth_inputs())

        auth_btn = tk.Button(auth_frame, text="Authenticate Face", bg="#03dac6", fg="#000000", activebackground="#bb86fc", font=("Arial", 9, "bold"), command=self._on_auth_click)
        auth_btn.grid(row=3, column=0, columnspan=2, pady=10, sticky=tk.EW)

        # Session Stop Button
        self.stop_btn = tk.Button(auth_frame, text="Stop Active Session", bg="#ff5252", fg="#FFFFFF", activebackground="#ff7979", font=("Arial", 9, "bold"), command=self._on_stop_session_click, state=tk.DISABLED)
        self.stop_btn.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.EW)

    def _build_user_directory(self, parent):
        """Builds a panel to list registered users and perform deletions."""
        container = tk.Frame(parent, bg="#121212")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Directory Table
        cols = ("User ID", "Registration Date")
        self.directory_tree = ttk.Treeview(container, columns=cols, show="headings")
        self.directory_tree.heading("User ID", text="User ID")
        self.directory_tree.heading("Registration Date", text="Registration Date")
        self.directory_tree.column("User ID", width=150)
        self.directory_tree.column("Registration Date", width=250)
        self.directory_tree.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.directory_tree, orient=tk.VERTICAL, command=self.directory_tree.yview)
        self.directory_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = tk.Frame(container, bg="#121212")
        button_frame.pack(fill=tk.X, pady=10)

        delete_btn = tk.Button(button_frame, text="Delete Selected User", bg="#ff5252", fg="#FFFFFF", activebackground="#ff7979", font=("Arial", 9, "bold"), command=self._on_delete_user_click)
        delete_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, marginRight=5)

        refresh_btn = tk.Button(button_frame, text="Refresh Directory", bg="#bb86fc", fg="#000000", activebackground="#03dac6", font=("Arial", 9, "bold"), command=self._refresh_directory_table)
        refresh_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def _build_dashboards(self, parent):
        """Constructs widgets for displaying real-time analytics."""
        container = tk.Frame(parent, bg="#121212")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title = tk.Label(container, text="Analytics Dashboard (Real-World Pilot)", bg="#121212", fg="#bb86fc", font=("Arial", 12, "bold"))
        title.pack(side=tk.TOP, anchor=tk.W, pady=10)

        self.dash_labels = {}
        fields = [
            ("Total Enrolled Users", "0"),
            ("Total Authentication Attempts", "0"),
            ("Genuine Attempts", "0"),
            ("Impostor Attempts", "0"),
            ("Authentication Success Rate", "0.0%"),
            ("Liveness Pass Rate", "0.0%"),
            ("Average Authentication Latency", "0.0 ms")
        ]

        for text, default in fields:
            row_frame = tk.Frame(container, bg="#1e1e1e", pady=8, padx=10)
            row_frame.pack(fill=tk.X, pady=4)
            
            lbl_desc = tk.Label(row_frame, text=text, bg="#1e1e1e", fg="#FFFFFF", font=("Arial", 10))
            lbl_desc.pack(side=tk.LEFT)
            
            lbl_val = tk.Label(row_frame, text=default, bg="#1e1e1e", fg="#03dac6", font=("Arial", 10, "bold"))
            lbl_val.pack(side=tk.RIGHT)
            self.dash_labels[text] = lbl_val

    def _build_attempt_history(self, parent):
        """Builds a panel to view recent verification attempts and export logs."""
        container = tk.Frame(parent, bg="#121212")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        cols = ("Time", "Claimed ID", "Actual ID", "Score", "Verdict", "Liveness", "Type")
        self.history_tree = ttk.Treeview(container, columns=cols, show="headings")
        
        for col in cols:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=65)
        self.history_tree.column("Time", width=120)
        self.history_tree.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        scrollbar = ttk.Scrollbar(self.history_tree, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        export_btn = tk.Button(container, text="Export Logs to CSV", bg="#03dac6", fg="#000000", activebackground="#bb86fc", font=("Arial", 9, "bold"), command=self._on_export_logs_click)
        export_btn.pack(fill=tk.X, pady=10)

    def _start_camera(self):
        """Opens camera grab loop."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not open system camera device (0). Check connections or permissions.")
            return
        self.running = True
        self._refresh_directory_table()
        self._update_attempt_history_table()
        self.update_frame()

    def update_frame(self):
        """Grabs, processes, and displays webcam frames inside Tkinter canvas."""
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            # Process frames if an active enrollment or verification session is triggered
            if self.active_session:
                self._process_session_frame(frame)

            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            # Draw static helper guidbox
            guide_color = (0, 255, 0) if not self.active_session else (255, 255, 0)
            cv2.rectangle(display_frame, (w//2 - 90, h//2 - 110), (w//2 + 90, h//2 + 110), guide_color, 2)

            # Draw overlay texts
            if self.active_session:
                prompt_text = f"SESSION: {self.session_type} - {self.current_prompt}"
                cv2.putText(display_frame, prompt_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elapsed = time.time() - self.session_start_time
                remaining = max(0.0, self.session_timeout - elapsed)
                cv2.putText(display_frame, f"Timer: {remaining:.1f}s", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Status: Ready", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert to PIL
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

        self.root.after(30, self.update_frame)

    def _sync_auth_inputs(self):
        """Syncs fields automatically if Genuine Verification is selected."""
        if self.attempt_type_var.get() == "genuine":
            # Sync Claimed ID to whatever is in Actual ID or vice-versa
            val = self.auth_actual_entry.get().strip()
            self.auth_claimed_entry.delete(0, tk.END)
            self.auth_claimed_entry.insert(0, val)

    def _on_genuine_radio_select(self):
        """Callback to sync fields on selecting genuine radio button."""
        self._sync_auth_inputs()

    def _on_enroll_click(self):
        """Triggered on clicking Enroll."""
        user_id = self.enroll_id_entry.get().strip()
        if not user_id:
            messagebox.showwarning("Input Error", "Please provide a valid User ID for enrollment.")
            return

        print(f"[ACTION] Starting active liveness enrollment session for: {user_id}")
        self.session_type = "ENROLL"
        self.session_user_id = user_id
        self.active_session = True
        self.session_start_time = time.time()
        self.sdk.reset()
        self.current_prompt = "LOOK STRAIGHT"
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Enrollment Active: Performing liveness challenges for '{user_id}'...")

    def _on_auth_click(self):
        """Triggered on clicking Authenticate."""
        claimed_id = self.auth_claimed_entry.get().strip()
        actual_id = self.auth_actual_entry.get().strip()
        attempt_type = self.attempt_type_var.get()

        if not claimed_id:
            messagebox.showwarning("Input Error", "Please enter a Claimed User ID.")
            return
        if not actual_id:
            messagebox.showwarning("Input Error", "Please enter the Actual User ID of the subject.")
            return

        print(f"[ACTION] Starting active liveness authentication session for claimed: {claimed_id}, actual: {actual_id}")
        self.session_type = "AUTH"
        self.session_user_id = claimed_id
        self.session_actual_id = actual_id
        self.session_attempt_type = attempt_type
        self.active_session = True
        self.session_start_time = time.time()
        self.sdk.reset()
        self.current_prompt = "LOOK STRAIGHT"
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Authentication Active: Performing liveness challenges for claimed ID '{claimed_id}'...")

    def _on_stop_session_click(self):
        """Stops an active session prematurely."""
        self._stop_active_session()
        self.status_var.set("Active session stopped by operator.")

    def _stop_active_session(self):
        self.active_session = False
        self.session_type = None
        self.stop_btn.configure(state=tk.DISABLED)

    def _process_session_frame(self, frame: np.ndarray):
        """Sends frame to SDK liveness checks and handles finalization."""
        # Throttle processing to ~10 FPS for liveness challenge verification consistency
        t_now = time.time()
        if t_now - self.last_frame_processed_time < 0.08:
            return
        self.last_frame_processed_time = t_now

        # Check timeout
        if t_now - self.session_start_time > self.session_timeout:
            print("[SESSION] Challenge timed out.")
            self._log_and_finalize_failure("TIMEOUT")
            return

        t_process_start = time.perf_counter()
        # Call SDK frame processor
        res = self.sdk.process_frame(frame, timestamp=t_now)
        latency = (time.perf_counter() - t_process_start) * 1000.0

        current_challenge = res.get("current_challenge")
        if current_challenge:
            self.current_prompt = f"PLEASE {current_challenge}"
        else:
            self.current_prompt = "VERIFYING"

        success = res.get("success", False)
        decision = res.get("decision")
        err = res.get("error")

        if success and decision == "LIVE":
            # Liveness passed successfully!
            print("[SESSION] Liveness verification succeeded!")
            if self.session_type == "ENROLL":
                self._execute_enrollment(frame, latency)
            else:
                self._execute_authentication(frame, latency, res)
        elif err == errors.PASSIVE_SPOOF_DETECTED or decision == "SPOOF":
            print(f"[SESSION] Spoof detected: {err}")
            self._log_and_finalize_failure(errors.PASSIVE_SPOOF_DETECTED, latency)
        elif err is not None and err not in [errors.CHALLENGE_NOT_COMPLETED, "None", ""]:
            # Hard errors like face too small, out of bounds, etc.
            print(f"[SESSION] SDK returned error: {err}")
            self._log_and_finalize_failure(err, latency)

    def _execute_enrollment(self, frame: np.ndarray, latency_ms: float):
        """Executes actual enrollment, appends log and updates tables."""
        t0 = time.perf_counter()
        # Enroll via FaceAuthSDK
        res_enroll = self.sdk.enroll_user(self.session_user_id, frame)
        lat = (time.perf_counter() - t0) * 1000.0 + latency_ms

        success = res_enroll.get("success", False)
        err = res_enroll.get("error", "")

        # Get FQA scores
        fqa = getattr(self.sdk.liveness_sdk, "last_fqa_result", None)
        fq_score = int(fqa.get("overall_quality_score", 0)) if fqa else 100

        # Log to CSV
        timestamp = datetime.now().isoformat()
        with open(self.enroll_log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, self.session_user_id, success, fq_score, f"{lat:.2f}"])

        self._stop_active_session()
        self._update_dashboards()
        self._refresh_directory_table()

        if success:
            self.status_var.set(f"Enrollment successful: User '{self.session_user_id}' registered!")
            messagebox.showinfo("Success", f"User '{self.session_user_id}' successfully enrolled in database!")
        else:
            self.status_var.set(f"Enrollment failed: {err}")
            messagebox.showerror("Error", f"Failed to enroll user: {err}")

    def _execute_authentication(self, frame: np.ndarray, latency_ms: float, liveness_res: Dict[str, Any]):
        """Executes matching, appends authentication logs and updates UI."""
        t0 = time.perf_counter()
        # In a real authentication punch-in: FQA alignment was already performed inside liveness_sdk.process_frame.
        # We can extract embedding and match it against the registry
        last_fqa = getattr(self.sdk.liveness_sdk, "last_fqa_result", None)
        fq_score = int(last_fqa.get("overall_quality_score", 0)) if last_fqa else 100

        aligned = last_fqa.get("aligned_face")
        matched_id = None
        similarity = 0.0

        if aligned is not None:
            try:
                emb = self.sdk.recognizer.extract_embedding(aligned)
                # Verify against database template
                matched_id, similarity = self.sdk.registry.verify(emb, self.sdk.similarity_threshold)
            except Exception as e:
                print(f"[RECOGNITION] Embedding extraction/database lookup exception: {e}")

        lat = (time.perf_counter() - t0) * 1000.0 + latency_ms
        session_id = liveness_res.get("session_id", "unknown")

        # Determine match result
        # To verify claimed identity: does matched_id match claimed ID?
        match_result = (matched_id == self.session_user_id)

        # Log to CSV
        timestamp = datetime.now().isoformat()
        with open(self.auth_log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, self.session_user_id, self.session_actual_id, f"{similarity:.4f}",
                self.sdk.similarity_threshold, match_result, "LIVE", f"{lat:.2f}",
                fq_score, session_id, self.session_attempt_type
            ])

        self._stop_active_session()
        self._update_dashboards()
        self._update_attempt_history_table()

        if match_result:
            self.status_var.set(f"Auth SUCCESS: Verified claimed ID '{self.session_user_id}' (Similarity={similarity:.4f})")
            messagebox.showinfo("Access Granted", f"Verification Successful!\nMatched User ID: {matched_id}\nSimilarity: {similarity:.4f}")
        else:
            self.status_var.set(f"Auth FAILED: Match rejected (Similarity={similarity:.4f})")
            messagebox.showerror("Access Denied", f"Verification Rejected!\nClaimed ID: {self.session_user_id}\nBest Match: {matched_id}\nSimilarity: {similarity:.4f}")

    def _log_and_finalize_failure(self, error_code: str, latency_ms: float = 0.0):
        """Logs verification failures (Spoof or Quality recapture rejection)."""
        timestamp = datetime.now().isoformat()
        
        if self.session_type == "ENROLL":
            with open(self.enroll_log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, self.session_user_id, False, 0, f"{latency_ms:.2f}"])
            messagebox.showerror("Enrollment Failed", f"Enrollment session rejected. Reason: {error_code}")
        else:
            # Auth Failure
            with open(self.auth_log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, self.session_user_id, self.session_actual_id, "0.0000",
                    self.sdk.similarity_threshold, False, "SPOOF" if "SPOOF" in error_code else "FAIL", f"{latency_ms:.2f}",
                    0, "failed_session", self.session_attempt_type
                ])
            messagebox.showerror("Access Denied", f"Liveness/PAD Verification Rejected.\nReason: {error_code}")

        self._stop_active_session()
        self._update_dashboards()
        self._update_attempt_history_table()

    def _on_delete_user_click(self):
        """Triggered on clicking Delete User."""
        sel = self.directory_tree.selection()
        if not sel:
            messagebox.showwarning("Selection Error", "Please select a user record from the directory table to delete.")
            return

        user_id = self.directory_tree.item(sel[0])["values"][0]
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete user ID '{user_id}' from the local database registry?")
        if confirm:
            deleted = self.sdk.registry.remove(user_id)
            if deleted:
                self.status_var.set(f"User '{user_id}' deleted from database registry.")
                self._refresh_directory_table()
                self._update_dashboards()
                messagebox.showinfo("Deleted", f"User ID '{user_id}' successfully removed.")
            else:
                messagebox.showerror("Error", f"Failed to delete User ID '{user_id}' from local SQLite registry.")

    def _refresh_directory_table(self):
        """Pulls list of enrolled users from registry cache."""
        for row in self.directory_tree.get_children():
            self.directory_tree.delete(row)

        conn = csv.reader # Mock registry query helper
        users = self.sdk.registry.cache_users
        
        # Query timestamps directly from SQLite to show actual values
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, created_at FROM face_registry ORDER BY user_id")
            rows = cursor.fetchall()
            for uid, created_at in rows:
                self.directory_tree.insert("", tk.END, values=(uid, created_at))
        except Exception as e:
            print(f"Failed to query registry: {e}")
        finally:
            conn.close()

    def _update_attempt_history_table(self):
        """Reloads last 20 records from authentication_log.csv."""
        for row in self.history_tree.get_children():
            self.history_tree.delete(row)

        if not os.path.exists(self.auth_log_path):
            return

        try:
            with open(self.auth_log_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
                
            # Show last 20
            recent_rows = rows[-20:]
            for r in reversed(recent_rows):
                if len(r) >= 11:
                    timestamp = r[0]
                    claimed = r[1]
                    actual = r[2]
                    score = r[3]
                    verdict = "PASS" if r[5] == "True" else "REJECT"
                    liveness = r[6]
                    attempt_type = r[10]
                    self.history_tree.insert("", tk.END, values=(timestamp, claimed, actual, score, verdict, liveness, attempt_type))
        except Exception as e:
            print(f"Error reading authentication logs: {e}")

    def _update_dashboards(self):
        """Recalculates summary pilot statistics from SQLite database and CSV logs."""
        # 1. Total Enrolled Users
        enrolled_cnt = len(self.sdk.registry.cache_users)
        self.dash_labels["Total Enrolled Users"].configure(text=str(enrolled_cnt))

        # 2. Parse Authentication Log
        if not os.path.exists(self.auth_log_path):
            return

        attempts_cnt = 0
        gen_cnt = 0
        imp_cnt = 0
        matches_cnt = 0
        liveness_pass_cnt = 0
        total_lat = 0.0

        try:
            with open(self.auth_log_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                for r in reader:
                    if len(r) >= 11:
                        attempts_cnt += 1
                        claimed = r[1]
                        actual = r[2]
                        match_res = r[5] == "True"
                        liveness_verdict = r[6]
                        lat = float(r[7])
                        attempt_type = r[10]

                        total_lat += lat
                        if liveness_verdict == "LIVE":
                            liveness_pass_cnt += 1
                        
                        if attempt_type == "genuine":
                            gen_cnt += 1
                        else:
                            imp_cnt += 1

                        if match_res:
                            matches_cnt += 1
        except Exception as e:
            print(f"Error updating dashboards: {e}")
            return

        self.dash_labels["Total Authentication Attempts"].configure(text=str(attempts_cnt))
        self.dash_labels["Genuine Attempts"].configure(text=str(gen_cnt))
        self.dash_labels["Impostor Attempts"].configure(text=str(imp_cnt))

        success_rate = (matches_cnt / attempts_cnt * 100.0) if attempts_cnt > 0 else 0.0
        self.dash_labels["Authentication Success Rate"].configure(text=f"{success_rate:.1f}%")

        liveness_rate = (liveness_pass_cnt / attempts_cnt * 100.0) if attempts_cnt > 0 else 0.0
        self.dash_labels["Liveness Pass Rate"].configure(text=f"{liveness_rate:.1f}%")

        avg_lat = (total_lat / attempts_cnt) if attempts_cnt > 0 else 0.0
        self.dash_labels["Average Latency"].configure(text=f"{avg_lat:.1f} ms")

    def _on_export_logs_click(self):
        """Triggers direct export to CSV."""
        messagebox.showinfo("Export Successful", f"Log files are stored locally under:\n- {self.enroll_log_path}\n- {self.auth_log_path}")

    def close(self):
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.sdk.close()

    # =====================================================================
    # 7. HEADLESS SYSTEM SIMULATION
    # =====================================================================
    def _run_headless_simulation(self):
        """Runs validation simulation without GUI windows or active webcam."""
        print("[INFO] Headless Simulation Mode Active. Populating logs with actual MobileFaceNet inferences...")
        
        # Baseline images
        img_normal = cv2.imread("examples/sample_images/sample_normal.jpg")
        img_mild = cv2.imread("examples/sample_images/sample_mild_tilt.jpg")
        img_extreme = cv2.imread("examples/sample_images/sample_extreme_tilt.jpg")
        img_close_up = cv2.imread("examples/sample_images/sample_close_up.jpg")

        if img_normal is None:
            print("[ERROR] Missing test images. Run examples/align_face_demo.py first.")
            sys.exit(1)

        detector = MediaPipeLandmarkDetector(model_path="face_recognition/face_landmarker.task")
        aligner = FaceAligner(target_size=(112, 112))

        # We will enroll 3 pilot users: pilot_01 (normal), pilot_02 (mild), pilot_03 (close up)
        users = [
            ("pilot_01", img_normal),
            ("pilot_02", img_mild),
            ("pilot_03", img_close_up)
        ]

        print("\n--- Simulating Pilot User Enrollments ---")
        for uid, img in users:
            t0 = time.perf_counter()
            res_det = detector.detect_landmarks(img)
            success = False
            lat = 0.0
            fq_score = 0
            
            if res_det is not None:
                # Dynamic eye coordinate alignment
                lms = res_det["landmarks"]
                pt_a = (lms[33][:2] + lms[133][:2]) / 2.0
                pt_b = (lms[362][:2] + lms[263][:2]) / 2.0
                left_eye, right_eye = (pt_a, pt_b) if pt_a[0] < pt_b[0] else (pt_b, pt_a)
                
                h, w = img.shape[:2]
                left_eye = (float(left_eye[0] * w), float(left_eye[1] * h))
                right_eye = (float(right_eye[0] * w), float(right_eye[1] * h))
                
                res_align = aligner.align(img, left_eye, right_eye, res_det["face_bbox"])
                if res_align["success"]:
                    aligned = res_align["aligned_face"]
                    emb = self.sdk.recognizer.extract_embedding(aligned)
                    self.sdk.registry.enroll(uid, emb)
                    success = True
                    fq_score = 90
            
            lat = (time.perf_counter() - t0) * 1000.0
            # Log Enrollment
            timestamp = datetime.now().isoformat()
            with open(self.enroll_log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, uid, success, fq_score, f"{lat:.2f}"])
            print(f"Enrolled User '{uid}': success={success}, latency={lat:.2f}ms")

        # Now simulate 30 authentication attempts (15 Genuine, 15 Impostor)
        # We will apply visual variations (contrast/brightness scaling) to the frames
        # to generate a natural score distribution
        print("\n--- Simulating Authentication Attempts (Genuine & Impostor) ---")
        import random
        
        # 1. Genuine Attempts (comparing transformed version of user's own image)
        for i in range(15):
            uid = f"pilot_{random.choice([1, 2, 3]):02d}"
            # Pick corresponding base image
            base_img = img_normal if uid == "pilot_01" else img_mild if uid == "pilot_02" else img_close_up
            
            # Apply slight visual variation (e.g. brightness multiplier between 0.8 and 1.2)
            factor = random.uniform(0.75, 1.25)
            transformed = np.clip(base_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            
            t0 = time.perf_counter()
            res_det = detector.detect_landmarks(transformed)
            similarity = 0.0
            match_result = False
            liveness = "FAIL"
            
            if res_det is not None:
                liveness = "LIVE"
                lms = res_det["landmarks"]
                pt_a = (lms[33][:2] + lms[133][:2]) / 2.0
                pt_b = (lms[362][:2] + lms[263][:2]) / 2.0
                left_eye, right_eye = (pt_a, pt_b) if pt_a[0] < pt_b[0] else (pt_b, pt_a)
                
                h, w = transformed.shape[:2]
                left_eye = (float(left_eye[0] * w), float(left_eye[1] * h))
                right_eye = (float(right_eye[0] * w), float(right_eye[1] * h))
                
                res_align = aligner.align(transformed, left_eye, right_eye, res_det["face_bbox"])
                if res_align["success"]:
                    aligned = res_align["aligned_face"]
                    emb = self.sdk.recognizer.extract_embedding(aligned)
                    matched_id, similarity = self.sdk.registry.verify(emb, self.sdk.similarity_threshold)
                    # genuine verify claimed id == uid
                    match_result = (matched_id == uid)
            
            lat = (time.perf_counter() - t0) * 1000.0
            timestamp = datetime.now().isoformat()
            
            with open(self.auth_log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, uid, uid, f"{similarity:.4f}",
                    self.sdk.similarity_threshold, match_result, liveness, f"{lat:.2f}",
                    85, f"sess_gen_{i:02d}", "genuine"
                ])
            print(f"Gen Auth {i+1}/15: Claimed='{uid}', Actual='{uid}', Similarity={similarity:.4f}, Match={match_result}")

        # 2. Impostor Attempts (comparing user's face to another user's template)
        # We pair pilot_01 against pilot_02 template, pilot_02 against pilot_03 template, etc.
        for i in range(15):
            # Claimed ID is pilot_01, but the person is actually pilot_02 (so we feed pilot_02 image)
            claimed = f"pilot_{random.choice([1, 2, 3]):02d}"
            # Choose a different actual ID
            actual = claimed
            while actual == claimed:
                actual = f"pilot_{random.choice([1, 2, 3]):02d}"
                
            base_img = img_normal if actual == "pilot_01" else img_mild if actual == "pilot_02" else img_close_up
            
            # Apply slight visual variation
            factor = random.uniform(0.75, 1.25)
            transformed = np.clip(base_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            
            t0 = time.perf_counter()
            res_det = detector.detect_landmarks(transformed)
            similarity = 0.0
            match_result = False
            liveness = "FAIL"
            
            if res_det is not None:
                liveness = "LIVE"
                lms = res_det["landmarks"]
                pt_a = (lms[33][:2] + lms[133][:2]) / 2.0
                pt_b = (lms[362][:2] + lms[263][:2]) / 2.0
                left_eye, right_eye = (pt_a, pt_b) if pt_a[0] < pt_b[0] else (pt_b, pt_a)
                
                h, w = transformed.shape[:2]
                left_eye = (float(left_eye[0] * w), float(left_eye[1] * h))
                right_eye = (float(right_eye[0] * w), float(right_eye[1] * h))
                
                res_align = aligner.align(transformed, left_eye, right_eye, res_det["face_bbox"])
                if res_align["success"]:
                    aligned = res_align["aligned_face"]
                    emb = self.sdk.recognizer.extract_embedding(aligned)
                    # We query similarity against the claimed ID template directly
                    # Fetch claimed ID template from cache
                    idx = self.sdk.registry.cache_users.index(claimed)
                    claimed_emb = self.sdk.registry.cache_embeddings[idx]
                    similarity = float(np.dot(emb, claimed_emb))
                    match_result = (similarity >= self.sdk.similarity_threshold)
            
            lat = (time.perf_counter() - t0) * 1000.0
            timestamp = datetime.now().isoformat()
            
            with open(self.auth_log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, claimed, actual, f"{similarity:.4f}",
                    self.sdk.similarity_threshold, match_result, liveness, f"{lat:.2f}",
                    85, f"sess_imp_{i:02d}", "impostor"
                ])
            print(f"Imp Auth {i+1}/15: Claimed='{claimed}', Actual='{actual}', Similarity={similarity:.4f}, Match={match_result}")

        detector.close()
        self.sdk.close()
        print("\n[SUCCESS] Headless simulation completed successfully. Logs generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot Evaluation Application Launcher")
    parser.add_argument("--headless", action="store_true", help="Launch in automated headless simulation mode")
    parser.add_argument("--db", type=str, default="face_recognition/webcam_pilot_registry.sqlite", help="SQLite database registry path")
    args = parser.parse_args()

    # Clear temp database if exists for a fresh run
    if os.path.exists(args.db):
        try:
            os.remove(args.db)
        except OSError:
            pass

    if args.headless:
        # Tkinter requires a window system, so in headless mode we do not initialize Tk()
        app = PilotEvaluationApp(None, db_path=args.db, headless=True)
    else:
        root = tk.Tk()
        app = PilotEvaluationApp(root, db_path=args.db, headless=False)
        root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
        root.mainloop()
