"""
Final Authentication Demo Application for Offline Face Authentication.

Provides a Tkinter dark-themed GUI for user enrollment, authentication,
real-time matching results with color-coded status panels, user management,
and CSV logging. Supports a headless simulation mode using sample images.
"""

import os
import sys
import time
import csv
import argparse
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime

# Add parent directory to path to enable package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opencv_module.recognition import FaceAuthSDK
from opencv_module import errors


class FinalAuthenticationDemoApp:
    def __init__(self, root: tk.Tk, db_path: str = "face_recognition/demo_registry.sqlite", headless: bool = False):
        self.root = root
        self.db_path = db_path
        self.headless = headless
        
        # Configure logging directory
        self.logs_dir = "evaluation/logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        self.log_path = os.path.join(self.logs_dir, "demo_authentication_log.csv")
        self._init_log()

        # Initialize FaceAuthSDK with Blink challenge preset
        self.sdk = FaceAuthSDK(
            preset={"challenges": ["BLINK"]},
            model_path="face_recognition/mobilefacenet.tflite",
            db_path=self.db_path,
            similarity_threshold=0.75
        )

        # Video source and threading state
        self.cap = None
        self.running = False
        self.last_frame = None

        # Lively Mode State
        self.auth_mode = None
        self.active_scan = False
        self.scan_start_time = 0
        self.best_fqa_score = -1
        self.best_frame_for_auth = None
        self.best_fqa_res = None
        self.scan_status_msg = "Waiting"
        self.scan_session_id = None

        # GUI layout
        if not self.headless:
            self.root.title("Offline Face Auth - Final Demo Application")
            self.root.geometry("1100x700")
            self.root.configure(bg="#121212")
            self._setup_style()
            self._create_widgets()
            self._start_camera()
        else:
            self._run_headless_simulation()

    def _init_log(self):
        """Creates the CSV log file with headers if it does not exist."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "SessionID", "ChallengeType", "ChallengeStatus",
                    "UserID", "SimilarityScore", "FaceQualityScore", "Threshold",
                    "MatchResult", "LivenessResult", "AuthenticationTimeMs"
                ])

    def _setup_style(self):
        """Sets up custom ttk styling for dark theme aesthetics."""
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure("TFrame", background="#121212")
        style.configure("Treeview", background="#1e1e1e", foreground="#FFFFFF", fieldbackground="#1e1e1e", rowheight=25)
        style.configure("Treeview.Heading", background="#2a2a2a", foreground="#bb86fc", font=("Arial", 10, "bold"))
        style.map("Treeview", background=[("selected", "#03dac6")], foreground=[("selected", "#000000")])

    def _create_widgets(self):
        """Constructs all GUI panels, viewport, results display, and lists."""
        # Top title panel
        title_label = tk.Label(
            self.root,
            text="Offline Face Authentication - Production Demo",
            bg="#121212",
            fg="#03dac6",
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
            text=" Live Webcam Feed ",
            bg="#121212",
            fg="#bb86fc",
            font=("Arial", 11, "bold"),
            padx=5,
            pady=5
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.video_label = tk.Label(left_frame, bg="#1a1a1a")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right panel: Controls and Results
        right_frame = tk.Frame(main_frame, bg="#121212", width=480)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        right_frame.pack_propagate(False)

        # 1. ENROLLMENT CARD
        enroll_frame = tk.LabelFrame(
            right_frame, text=" User Registration ", bg="#1e1e1e", fg="#03dac6",
            font=("Arial", 10, "bold"), padx=10, pady=10
        )
        enroll_frame.pack(fill=tk.X, pady=5)

        tk.Label(enroll_frame, text="Enter User ID:", bg="#1e1e1e", fg="#FFFFFF", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.enroll_id_entry = tk.Entry(enroll_frame, bg="#121212", fg="#FFFFFF", insertbackground="#FFFFFF", bd=1, relief=tk.FLAT, font=("Arial", 10))
        self.enroll_id_entry.grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)
        enroll_frame.columnconfigure(1, weight=1)

        enroll_btn = tk.Button(
            enroll_frame, text="Enroll User", bg="#03dac6", fg="#000000",
            activebackground="#bb86fc", font=("Arial", 10, "bold"), command=self._on_enroll_click
        )
        enroll_btn.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)

        # 2. AUTHENTICATION CONTROLS
        auth_control_frame = tk.Frame(right_frame, bg="#121212")
        auth_control_frame.pack(fill=tk.X, pady=5)

        self.auth_mode = tk.StringVar(value="STATIC")
        tk.Radiobutton(auth_control_frame, text="Static Mode", variable=self.auth_mode, value="STATIC", bg="#121212", fg="#bb86fc", selectcolor="#1e1e1e", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(auth_control_frame, text="Lively Mode (Blink)", variable=self.auth_mode, value="LIVELY", bg="#121212", fg="#bb86fc", selectcolor="#1e1e1e", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)

        btn_frame = tk.Frame(right_frame, bg="#121212")
        btn_frame.pack(fill=tk.X, pady=5)

        self.auth_btn = tk.Button(
            btn_frame, text="Authenticate (Static)", bg="#bb86fc", fg="#000000",
            activebackground="#03dac6", font=("Arial", 9, "bold"), command=self._on_auth_click
        )
        self.auth_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.start_scan_btn = tk.Button(
            btn_frame, text="Start Liveness Scan", bg="#03dac6", fg="#000000",
            activebackground="#bb86fc", font=("Arial", 9, "bold"), command=self._on_start_scan_click
        )
        self.start_scan_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.cancel_scan_btn = tk.Button(
            btn_frame, text="Cancel Scan", bg="#ff5252", fg="#FFFFFF",
            activebackground="#ff7979", font=("Arial", 9, "bold"), command=self._on_cancel_scan_click, state=tk.DISABLED
        )
        self.cancel_scan_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 3. DISPLAY RESULTS CARD (THE COLOR PANEL)
        self.result_panel = tk.Frame(right_frame, bg="#1e1e1e", bd=2, relief=tk.RIDGE, padx=15, pady=15)
        self.result_panel.pack(fill=tk.X, pady=10)

        self.result_labels = {}
        fields = [
            ("User", "N/A"),
            ("Similarity Score", "0.0000"),
            ("Threshold", "0.75"),
            ("Result", "IDLE"),
            ("Liveness", "N/A"),
            ("Authentication Time", "N/A ms")
        ]

        for idx, (name, val) in enumerate(fields):
            row_frame = tk.Frame(self.result_panel, bg="#1e1e1e")
            row_frame.pack(fill=tk.X, pady=4)
            
            lbl_name = tk.Label(row_frame, text=f"{name}:", bg="#1e1e1e", fg="#aaaaaa", font=("Courier", 12, "bold"))
            lbl_name.pack(side=tk.LEFT)
            
            lbl_val = tk.Label(row_frame, text=val, bg="#1e1e1e", fg="#FFFFFF", font=("Courier", 14, "bold"))
            lbl_val.pack(side=tk.RIGHT)
            self.result_labels[name] = (lbl_val, row_frame)

        # 4. USER DIRECTORY CARD
        dir_frame = tk.LabelFrame(
            right_frame, text=" Registered Users ", bg="#1e1e1e", fg="#bb86fc",
            font=("Arial", 10, "bold"), padx=10, pady=10
        )
        dir_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        cols = ("User ID", "Enrolled Date")
        self.directory_tree = ttk.Treeview(dir_frame, columns=cols, show="headings", height=5)
        self.directory_tree.heading("User ID", text="User ID")
        self.directory_tree.heading("Enrolled Date", text="Enrolled Date")
        self.directory_tree.column("User ID", width=120)
        self.directory_tree.column("Enrolled Date", width=220)
        self.directory_tree.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        scrollbar = ttk.Scrollbar(self.directory_tree, orient=tk.VERTICAL, command=self.directory_tree.yview)
        self.directory_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = tk.Frame(dir_frame, bg="#1e1e1e")
        button_frame.pack(fill=tk.X, pady=10)

        delete_btn = tk.Button(
            button_frame, text="Delete User", bg="#ff5252", fg="#FFFFFF",
            activebackground="#ff7979", font=("Arial", 9, "bold"), command=self._on_delete_user_click
        )
        delete_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        refresh_btn = tk.Button(
            button_frame, text="Refresh Directory", bg="#bb86fc", fg="#000000",
            activebackground="#03dac6", font=("Arial", 9, "bold"), command=self._refresh_directory_table
        )
        refresh_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

    def _start_camera(self):
        """Opens camera grab loop."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not open system camera device (0). Check connections.")
            return
        self.running = True
        self._refresh_directory_table()
        self.update_frame()

    def update_frame(self):
        """Grabs, processes, and displays webcam preview inside Tkinter label."""
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.last_frame = frame.copy()
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            # Detect face in preview and draw a bounding box overlay
            border_color = None
            
            # --- LIVELY MODE SCANNING LOGIC ---
            if self.active_scan:
                elapsed = time.time() - self.scan_start_time
                remaining = max(0.0, 10.0 - elapsed)
                
                # Draw purple border for active scan
                border_color = (252, 134, 187) # BGR for #bb86fc (Purple)
                
                if remaining <= 0:
                    self._on_cancel_scan_click()
                    self.scan_status_msg = "TIMEOUT"
                    self._update_ui_result("N/A", 0.0, "TIMEOUT", "FAIL", 0.0, bg_color="#b71c1c")
                else:
                    liveness_res = self.sdk.liveness_sdk.process_frame(frame)
                    err = liveness_res.get("error")
                    
                    if err == "FACE_NOT_DETECTED":
                        self.scan_status_msg = "Waiting for Face"
                    elif err == "CHALLENGE_NOT_COMPLETED":
                        self.scan_status_msg = "Please Blink"
                    elif liveness_res.get("success"):
                        self.scan_status_msg = "Blink Detected! Authenticating..."
                    else:
                        self.scan_status_msg = f"Status: {err}"

                    fqa_res = self.sdk.liveness_sdk.last_fqa_result
                    if fqa_res and fqa_res.get("success"):
                        score = fqa_res.get("face_size_score", 0) + fqa_res.get("pose_score", 0) + fqa_res.get("blur_score", 0)
                        if score > self.best_fqa_score:
                            self.best_fqa_score = score
                            self.best_frame_for_auth = frame.copy()
                            self.best_fqa_res = fqa_res
                    
                    cv2.putText(display_frame, f"Time: {remaining:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (252, 134, 187), 2)
                    cv2.putText(display_frame, self.scan_status_msg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    if liveness_res.get("success") and liveness_res.get("decision") == "LIVE":
                        self._finish_lively_auth()
            
            try:
                if not self.active_scan:
                    res_det = self.sdk.liveness_sdk.mp_detector.detect_landmarks(frame)
                    if res_det is not None:
                        bx, by, bw, bh = res_det["face_bbox"]
                        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (198, 218, 3), 2)
                        
                        self.sdk.liveness_sdk.decision_engine.antispoof_engine.update(
                            frame=frame,
                            bbox=res_det["face_bbox"],
                            landmarks=res_det["landmarks"],
                            timestamp=time.time()
                        )
                        # Optional: Draw Blue border if face is waiting in STATIC mode
                        border_color = (255, 0, 0) # Blue
            except Exception:
                pass

            if border_color:
                cv2.rectangle(display_frame, (0, 0), (w-1, h-1), border_color, 10)

            # Convert OpenCV BGR to Tkinter PhotoImage
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

        self.root.after(30, self.update_frame)

    def _on_enroll_click(self):
        """Triggers user enrollment from current frame."""
        user_id = self.enroll_id_entry.get().strip()
        if not user_id:
            messagebox.showwarning("Input Error", "Please provide a valid User ID for enrollment.")
            return

        if self.last_frame is None:
            messagebox.showwarning("Camera Error", "No frame captured from webcam yet.")
            return

        frame = self.last_frame.copy()
        
        t0 = time.perf_counter()
        # Perform enrollment using production SDK
        enroll_res = self.sdk.enroll_user(user_id, frame)
        lat = (time.perf_counter() - t0) * 1000.0

        if enroll_res.get("success"):
            self._refresh_directory_table()
            msg = f"Enrollment Successful\nUser: {user_id}\nEnrollment Time: {lat:.2f} ms"
            messagebox.showinfo("Success", msg)
            self.enroll_id_entry.delete(0, tk.END)
        else:
            err = enroll_res.get("error", "Unknown error")
            messagebox.showerror("Error", f"Enrollment Failed: {err}")

    def _on_auth_click(self):
        """Triggers identity verification and displays exact output style."""
        if self.last_frame is None:
            messagebox.showwarning("Camera Error", "No frame captured from webcam yet.")
            return

        frame = self.last_frame.copy()
        
        t0 = time.perf_counter()
        
        # 1. Run face detection and landmark check
        res_det = self.sdk.liveness_sdk.mp_detector.detect_landmarks(frame)
        if res_det is None:
            self._update_ui_result("N/A", 0.0, "NO MATCH", "FAIL", 0.0)
            messagebox.showerror("Auth Error", "Face not detected.")
            return

        # 2. Run quality check and alignment
        fqa_res = self.sdk.liveness_sdk.fqa_engine.assess_raw_face(
            image=frame,
            left_eye=res_det["left_eye"],
            right_eye=res_det["right_eye"],
            face_bbox=res_det["face_bbox"]
        )
        if not fqa_res.get("success"):
            self._update_ui_result("N/A", 0.0, "NO MATCH", "FAIL", 0.0)
            messagebox.showerror("Auth Error", f"Quality check failed: {fqa_res.get('error')}")
            return

        # 3. Run Liveness Detection (using passive classification from buffered history)
        passive_res = self.sdk.liveness_sdk.decision_engine.antispoof_engine.update(
            frame=frame,
            bbox=res_det["face_bbox"],
            landmarks=res_det["landmarks"],
            timestamp=time.time()
        )
        classification = passive_res.get("classification", "SPOOF")
        liveness_pass = (classification in ["LIVE", "SUSPECT"])
        liveness_str = "PASS" if liveness_pass else "FAIL"

        # 4. Run Face Recognition embedding lookup
        aligned = fqa_res["aligned_face"]
        embedding = self.sdk.recognizer.extract_embedding(aligned)
        matched_id, similarity = self.sdk.registry.verify(embedding, self.sdk.similarity_threshold)
        
        lat = (time.perf_counter() - t0) * 1000.0

        # Match criteria: matched user exists and liveness passed
        is_match = (matched_id is not None and liveness_pass)
        match_str = "MATCH" if is_match else "NO MATCH"
        user_str = matched_id if is_match else "N/A"

        # Update Results Dashboard
        self._update_ui_result(user_str, similarity, match_str, liveness_str, lat)

        fqa_score_sum = fqa_res.get("face_size_score", 0) + fqa_res.get("pose_score", 0) + fqa_res.get("blur_score", 0)

        # Log authentication attempt
        timestamp = datetime.now().isoformat()
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, "STATIC_SESSION", "NONE", "N/A",
                user_str, f"{similarity:.4f}", f"{fqa_score_sum:.1f}", "0.75",
                is_match, liveness_str, f"{lat:.2f}"
            ])

    def _on_start_scan_click(self):
        """Starts the Lively Mode Blink Challenge."""
        if not self.auth_mode or self.auth_mode.get() != "LIVELY":
            messagebox.showinfo("Mode Error", "Please select 'Lively Mode' to start a scan.")
            return

        self.sdk.liveness_sdk.reset()
        self.scan_session_id = self.sdk.liveness_sdk.decision_engine.session_id
        self.active_scan = True
        self.scan_start_time = time.time()
        self.best_fqa_score = -1
        self.best_frame_for_auth = None
        self.best_fqa_res = None
        self.scan_status_msg = "Waiting for Face"
        
        self.start_scan_btn.configure(state=tk.DISABLED)
        self.auth_btn.configure(state=tk.DISABLED)
        self.cancel_scan_btn.configure(state=tk.NORMAL)
        self._update_ui_result("Scanning...", 0.0, "PENDING", "PENDING", 0.0, bg_color="#1e1e1e")

    def _on_cancel_scan_click(self):
        """Cancels an active scan."""
        self.active_scan = False
        self.start_scan_btn.configure(state=tk.NORMAL)
        self.auth_btn.configure(state=tk.NORMAL)
        self.cancel_scan_btn.configure(state=tk.DISABLED)
        self.scan_status_msg = "Cancelled"
        
    def _finish_lively_auth(self):
        """Completes authentication using the best frame from the active scan."""
        self.active_scan = False
        self.start_scan_btn.configure(state=tk.NORMAL)
        self.auth_btn.configure(state=tk.NORMAL)
        self.cancel_scan_btn.configure(state=tk.DISABLED)
        
        if self.best_fqa_res is None or "aligned_face" not in self.best_fqa_res:
            self._update_ui_result("N/A", 0.0, "NO MATCH", "FAIL", 0.0, bg_color="#b71c1c")
            return
            
        t0 = time.perf_counter()
        
        aligned = self.best_fqa_res["aligned_face"]
        embedding = self.sdk.recognizer.extract_embedding(aligned)
        matched_id, similarity = self.sdk.registry.verify(embedding, self.sdk.similarity_threshold)
        
        lat = (time.perf_counter() - t0) * 1000.0
        
        is_match = (matched_id is not None)
        match_str = "MATCH" if is_match else "NO MATCH"
        user_str = matched_id if is_match else "N/A"
        
        self._update_ui_result(user_str, similarity, match_str, "PASS", lat)
        
        timestamp = datetime.now().isoformat()
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, self.scan_session_id, "BLINK", "SUCCESS",
                user_str, f"{similarity:.4f}", f"{self.best_fqa_score:.1f}", "0.75",
                is_match, "PASS", f"{lat:.2f}"
            ])

    def _update_ui_result(self, user: str, score: float, result: str, liveness: str, latency: float, bg_color: str = None):
        """Updates the results labels and changes background panel color."""
        self.result_labels["User"][0].configure(text=user)
        self.result_labels["Similarity Score"][0].configure(text=f"{score:.4f}")
        self.result_labels["Result"][0].configure(text=result)
        self.result_labels["Liveness"][0].configure(text=liveness)
        self.result_labels["Authentication Time"][0].configure(text=f"{latency:.2f} ms")

        # Color-coded panel logic
        if bg_color:
            color = bg_color
        elif liveness == "FAIL":
            # Yellow panel for liveness failure (spoof)
            color = "#f57f17" # dark yellow/orange
        elif result == "MATCH":
            # Green panel for successful match
            color = "#1b5e20" # dark green
        else:
            # Red panel for mismatch/failure
            color = "#b71c1c" # dark red

        self.result_panel.configure(bg=color)
        for name in self.result_labels:
            lbl_val, row = self.result_labels[name]
            row.configure(bg=color)
            lbl_val.configure(bg=color)
            # Find name label and set bg
            for widget in row.winfo_children():
                widget.configure(bg=color)

    def _on_delete_user_click(self):
        """Removes selected user from directory."""
        sel = self.directory_tree.selection()
        if not sel:
            messagebox.showwarning("Selection Error", "Please select a user from the list to delete.")
            return

        user_id = self.directory_tree.item(sel[0])["values"][0]
        confirm = messagebox.askyesno("Confirm Delete", f"Delete user '{user_id}' from the database?")
        if confirm:
            deleted = self.sdk.registry.remove(user_id)
            if deleted:
                self._refresh_directory_table()
                messagebox.showinfo("Deleted", f"User '{user_id}' removed.")
            else:
                messagebox.showerror("Error", f"Failed to delete '{user_id}' from registry.")

    def _refresh_directory_table(self):
        """Reloads enrollees list from registry database."""
        for row in self.directory_tree.get_children():
            self.directory_tree.delete(row)

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

    def close(self):
        """Releases SDK and camera resources."""
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.sdk.close()

    # =====================================================================
    # 7. HEADLESS DEMO SIMULATION
    # =====================================================================
    def _run_headless_simulation(self):
        """Performs automated validation of the demo flow without GUI."""
        print("[INFO] Headless Simulation Mode Active. Validating Final Authentication Demo flow...")
        
        # Load sample images
        img_normal = cv2.imread("examples/sample_images/sample_normal.jpg")
        img_mild = cv2.imread("examples/sample_images/sample_mild_tilt.jpg")

        if img_normal is None:
            print("[ERROR] Sample test images are missing. Headless run aborted.")
            sys.exit(1)

        test_user = "demo_pilot_user"

        # 1. Verify User Enrollment
        print("\n--- Phase 1: User Enrollment ---")
        t0 = time.perf_counter()
        enroll_res = self.sdk.enroll_user(test_user, img_normal)
        lat_enroll = (time.perf_counter() - t0) * 1000.0
        
        if enroll_res.get("success"):
            print("Enrollment Successful")
            print(f"User: {test_user}")
            print(f"Enrollment Time: {lat_enroll:.2f} ms")
        else:
            print(f"[ERROR] Enrollment failed: {enroll_res.get('error')}")
            sys.exit(1)

        # 2. Verify Persistence after re-initialization (Restart simulation)
        print("\n--- Phase 2: Database Persistence Validation (Simulator Restart) ---")
        self.sdk.close()
        
        # Re-initialize SDK and read registry database
        self.sdk = FaceAuthSDK(
            preset={"challenges": ["BLINK"]},
            model_path="face_recognition/mobilefacenet.tflite",
            db_path=self.db_path,
            similarity_threshold=0.75
        )
        
        # Initialize detector for headless single-shot analysis
        from opencv_module.landmarks import MediaPipeLandmarkDetector
        self.sdk.liveness_sdk.mp_detector = MediaPipeLandmarkDetector(model_path=self.sdk.liveness_sdk.mp_model_path)
        
        # Verify user is still in database registry
        enrolled_users = self.sdk.registry.cache_users
        print(f"Registered enrollees in database: {enrolled_users}")
        if test_user not in enrolled_users:
            print("[ERROR] Database persistence validation failed: user not found after restart.")
            sys.exit(1)
        print("[SUCCESS] Database persistence check passed.")

        # 3. Verify Authentication Flow (Genuine MATCH)
        print("\n--- Phase 3: Authentication Flow (Genuine Subject) ---")
        t0 = time.perf_counter()
        
        # Detect landmarks
        res_det = self.sdk.liveness_sdk.mp_detector.detect_landmarks(img_normal)
        if res_det is None:
            print("[ERROR] Landmark detection failed on normal image.")
            sys.exit(1)
            
        # Run FQA and alignment
        fqa_res = self.sdk.liveness_sdk.fqa_engine.assess_raw_face(
            image=img_normal,
            left_eye=res_det["left_eye"],
            right_eye=res_det["right_eye"],
            face_bbox=res_det["face_bbox"]
        )
        if not fqa_res.get("success"):
            print("[ERROR] FQA quality checks failed.")
            sys.exit(1)

        # Run passive liveness check (since buffer is not full, returns SUSPECT/PASS)
        passive_res = self.sdk.liveness_sdk.decision_engine.antispoof_engine.update(
            frame=img_normal,
            bbox=res_det["face_bbox"],
            landmarks=res_det["landmarks"],
            timestamp=time.time()
        )
        classification = passive_res.get("classification")
        liveness_pass = (classification in ["LIVE", "SUSPECT"])
        liveness_str = "PASS" if liveness_pass else "FAIL"

        # Verify embedding
        aligned = fqa_res["aligned_face"]
        embedding = self.sdk.recognizer.extract_embedding(aligned)
        matched_id, similarity = self.sdk.registry.verify(embedding, self.sdk.similarity_threshold)
        
        lat_auth = (time.perf_counter() - t0) * 1000.0
        
        is_match = (matched_id == test_user and liveness_pass)
        match_str = "MATCH" if is_match else "NO MATCH"

        fqa_score_sum = fqa_res.get("face_size_score", 0) + fqa_res.get("pose_score", 0) + fqa_res.get("blur_score", 0)

        # Display EXACTLY the required style
        print(f"\nUser: {test_user}")
        print(f"Similarity Score: {similarity:.4f}")
        print("Threshold: 0.75")
        print(f"\nResult: {match_str}")
        print(f"Liveness: {liveness_str}")
        print(f"\nAuthentication Time: {lat_auth:.2f} ms")

        # Log attempt
        timestamp = datetime.now().isoformat()
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, "HEADLESS_TEST", "NONE", "N/A",
                test_user, f"{similarity:.4f}", f"{fqa_score_sum:.1f}", "0.75",
                is_match, liveness_str, f"{lat_auth:.2f}"
            ])

        # Verify output matches expectation
        if not is_match:
            print("[ERROR] MATCH validation failed.")
            sys.exit(1)

        # 4. Verify Authentication Flow (Impostor NO MATCH)
        print("\n--- Phase 4: Authentication Flow (Impostor/Mismatched Subject) ---")
        img_impostor = cv2.imread("archive/lfw-deepfunneled/lfw-deepfunneled/AJ_Cook/AJ_Cook_0001.jpg")
        if img_impostor is None:
            print("[ERROR] Impostor test image is missing.")
            sys.exit(1)

        t0 = time.perf_counter()
        
        # Detect landmarks
        res_det_imp = self.sdk.liveness_sdk.mp_detector.detect_landmarks(img_impostor)
        if res_det_imp is None:
            print("[ERROR] Landmark detection failed on impostor image.")
            sys.exit(1)
            
        # Run FQA and alignment
        fqa_res_imp = self.sdk.liveness_sdk.fqa_engine.assess_raw_face(
            image=img_impostor,
            left_eye=res_det_imp["left_eye"],
            right_eye=res_det_imp["right_eye"],
            face_bbox=res_det_imp["face_bbox"]
        )
        if not fqa_res_imp.get("success"):
            print("[ERROR] FQA quality checks failed on impostor image.")
            sys.exit(1)

        # Run passive liveness check
        passive_res_imp = self.sdk.liveness_sdk.decision_engine.antispoof_engine.update(
            frame=img_impostor,
            bbox=res_det_imp["face_bbox"],
            landmarks=res_det_imp["landmarks"],
            timestamp=time.time()
        )
        classification_imp = passive_res_imp.get("classification")
        liveness_pass_imp = (classification_imp in ["LIVE", "SUSPECT"])
        liveness_str_imp = "PASS" if liveness_pass_imp else "FAIL"

        # Verify embedding
        aligned_imp = fqa_res_imp["aligned_face"]
        embedding_imp = self.sdk.recognizer.extract_embedding(aligned_imp)
        matched_id_imp, similarity_imp = self.sdk.registry.verify(embedding_imp, self.sdk.similarity_threshold)
        
        lat_auth_imp = (time.perf_counter() - t0) * 1000.0
        
        is_match_imp = (matched_id_imp is not None and liveness_pass_imp)
        match_str_imp = "MATCH" if is_match_imp else "NO MATCH"
        user_str_imp = matched_id_imp if is_match_imp else "N/A"

        fqa_score_sum_imp = fqa_res_imp.get("face_size_score", 0) + fqa_res_imp.get("pose_score", 0) + fqa_res_imp.get("blur_score", 0)

        # Display EXACTLY the required style
        print(f"\nUser: {user_str_imp}")
        print(f"Similarity Score: {similarity_imp:.4f}")
        print("Threshold: 0.75")
        print(f"\nResult: {match_str_imp}")
        print(f"Liveness: {liveness_str_imp}")
        print(f"\nAuthentication Time: {lat_auth_imp:.2f} ms")

        # Log attempt
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, "HEADLESS_TEST", "NONE", "N/A",
                user_str_imp, f"{similarity_imp:.4f}", f"{fqa_score_sum_imp:.1f}", "0.75",
                is_match_imp, liveness_str_imp, f"{lat_auth_imp:.2f}"
            ])

        if is_match_imp:
            print("[ERROR] NO MATCH validation failed (impostor matched user).")
            sys.exit(1)

        # 5. Verify User Deletion
        print("\n--- Phase 5: User Deletion ---")
        deleted = self.sdk.registry.remove(test_user)
        print(f"Delete Success: {deleted}")
        if not deleted:
            print("[ERROR] Deletion validation failed.")
            sys.exit(1)
            
        # Verify user is gone
        if test_user in self.sdk.registry.cache_users:
            print("[ERROR] Deletion failed to remove user from cache registry.")
            sys.exit(1)
        print("[SUCCESS] Deletion validation passed.")

        # Cleanup SDK
        self.sdk.close()
        print("\n[SUCCESS] Headless validation completed successfully. All verification tests PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Authentication Demo Application Launcher")
    parser.add_argument("--headless", action="store_true", help="Launch in automated headless simulation mode")
    parser.add_argument("--db", type=str, default="face_recognition/demo_registry.sqlite", help="SQLite database registry path")
    args = parser.parse_args()

    # Clear registry database if exists for a fresh run
    if os.path.exists(args.db):
        try:
            os.remove(args.db)
        except OSError:
            pass

    if args.headless:
        # Headless mode does not initialize Tkinter to run safely on displayless environments
        app = FinalAuthenticationDemoApp(None, db_path=args.db, headless=True)
    else:
        root = tk.Tk()
        app = FinalAuthenticationDemoApp(root, db_path=args.db, headless=False)
        root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
        root.mainloop()
