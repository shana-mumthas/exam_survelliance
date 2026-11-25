import cv2
import threading
import queue
import time
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from app.core.temporal_logic import BehavioralEngine
from app.core.forensics import ForensicsManager

class ExamIntegrityEngine:
    def __init__(self, model_path="yolo11n.pt", device="CPU"):
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Initialize AI Models
        print(f"Initializing YOLOv11n on {device}...")
        self.detector = YOLO(model_path)
        # Export to OpenVINO if not already done (for Ryzen iGPU acceleration)
        if not model_path.endswith(".xml"):
            try:
                print("Exporting to OpenVINO for acceleration...")
                # Note: Using dynamic=False as it is more stable for conversion
                self.detector.export(format="openvino", dynamic=False, device=device)
                ov_path = model_path.replace(".pt", "_openvino_model")
                if os.path.exists(ov_path):
                    self.detector = YOLO(ov_path)
                    print("✓ OpenVINO model loaded successfully")
                else:
                    print("⚠️ OpenVINO export path not found, falling back to PyTorch")
            except Exception as e:
                print(f"⚠️ OpenVINO export failed: {e}")
                print("Falling back to standard PyTorch inference...")
            
        # 2. MediaPipe Face Mesh (Tasks API)
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        self.behavioral_engine = BehavioralEngine()
        self.forensics = ForensicsManager()
        
        self.current_frame = None
        self.fps = 0

    def start(self):
        self.running = True
        threading.Thread(target=self._thread_capture, daemon=True).start()
        threading.Thread(target=self._thread_inference, daemon=True).start()
        threading.Thread(target=self._thread_logic, daemon=True).start()
        print("✓ All engines started (Capture, Inference, Logic)")

    def stop(self):
        self.running = False

    def _thread_capture(self):
        """Thread A: Frame Capture (Low Latency Ingestion)"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            
            # Keep most recent frame
            if self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except queue.Empty: pass
            
            self.frame_queue.put(frame)
            self.current_frame = frame
        cap.release()

    def _thread_inference(self):
        """Thread B: AI Inference (Object Detection + Face Mesh)"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                start_time = time.time()
                
                # 1. YOLO Detection (Phones, Earbuds)
                results = self.detector(frame, verbose=False, conf=0.4)
                detections = []
                for r in results:
                    for box in r.boxes:
                        detections.append({
                            "label": self.detector.names[int(box.cls[0])],
                            "conf": float(box.conf[0]),
                            "box": box.xyxy[0].tolist()
                        })
                
                # 2. MediaPipe Face Mesh (New Tasks API)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                mesh_results = self.face_landmarker.detect(mp_image)
                
                pose_data = None
                if mesh_results.face_landmarks:
                    # Tasks API returns NormalizedLandmark objects in a list
                    pose_data = self.behavioral_engine.estimate_pose(
                        mesh_results.face_landmarks[0], 
                        frame.shape
                    )

                inference_time = time.time() - start_time
                self.fps = 1.0 / inference_time if inference_time > 0 else 0
                
                # Update Forensics Buffer
                self.forensics.update_buffer(frame)
                
                self.result_queue.put({
                    "frame": frame,
                    "detections": detections,
                    "pose": pose_data,
                    "timestamp": time.time()
                })
                
            except queue.Empty:
                continue

    def _thread_logic(self):
        """Thread C: Temporal Alerting Logic & Forensics"""
        while self.running:
            try:
                data = self.result_queue.get(timeout=1)
                
                # Update Behavioral Engine with latest pose
                violation = self.behavioral_engine.update(data["pose"], data["detections"])
                
                if violation:
                    print(f"⚠️ [ALERT] {violation['type']} detected (Confidence: {violation['conf']})")
                    # Trigger Forensics
                    self.forensics.record_violation(data["frame"], violation)
                    
            except queue.Empty:
                continue
