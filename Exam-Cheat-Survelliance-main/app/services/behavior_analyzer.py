import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import defaultdict
import time
import os
import urllib.request

class BehaviorAnalyzer:
    def __init__(self):
        print("Loading MediaPipe Pose & Hand Tasks for behavior analysis...")
        
        # Paths for modern .task files
        pose_model_path = "pose_landmarker.task"
        hand_model_path = "hand_landmarker.task"
        
        # Download models if missing
        self._ensure_model(pose_model_path, "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        self._ensure_model(hand_model_path, "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

        # Initialize Pose Landmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        
        # Initialize Hand Landmarker
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        
        # Temporal tracking for duration-based alerts
        self.hand_face_tracker = defaultdict(lambda: {"start_time": None})
        self.hand_pocket_tracker = defaultdict(lambda: {"start_time": None})
        self.hand_collar_tracker = defaultdict(lambda: {"start_time": None})  # NEW: collar/neck area
        self.material_exchange_tracker = {"last_alert": 0, "cooldown": 5}
        
        # Thresholds
        self.face_distance_threshold = 0.4
        self.pocket_distance_threshold = 0.35  # Increased for better pocket detection
        self.collar_distance_threshold = 0.25  # NEW: collar/neck area threshold
        self.face_duration_threshold = 3.0
        self.pocket_duration_threshold = 1.5   # Fast detection - 1.5 seconds
        self.collar_duration_threshold = 1.5   # Fast detection - 1.5 seconds
        self.hand_overlap_threshold = 50
        
        print("✓ Behavior Analyzer (Tasks API) loaded successfully.")

    def _ensure_model(self, path, url):
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            urllib.request.urlretrieve(url, path)

    def analyze_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        alerts = []
        pose_data = []
        hand_bboxes = []
        current_time = time.time()
        
        # 1. Run Pose detection
        pose_results = self.pose_landmarker.detect(mp_image)
        
        if pose_results.pose_landmarks:
            for pose_landmarks in pose_results.pose_landmarks:
                # Tasks API returns a list of NormalizedLandmark objects
                landmarks = pose_landmarks
                
                # key landmarks: 0 (nose), 15(lw), 16(rw), 23(lh), 24(rh), 11(ls), 12(rs)
                nose = landmarks[0]
                left_wrist = landmarks[15]
                right_wrist = landmarks[16]
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                
                # Calculate shoulder metrics for reference
                shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + (left_shoulder.y - right_shoulder.y)**2)
                if shoulder_width < 0.01: shoulder_width = 0.2
                shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                
                # COLLAR AREA: just below chin, at shoulder center level (use Y position just above shoulders)
                # This is where shirt collar is - between shoulders and below chin
                collar_x = shoulder_center_x
                collar_y = shoulder_center_y - (shoulder_width * 0.15)  # Just above shoulder line
                
                # POCKET AREA: When sitting, hips may not be visible
                # Use a fallback: pocket is roughly at waist level, below shoulders
                # If hips are visible, use them; otherwise estimate from shoulders
                hip_visible = (left_hip.visibility > 0.3 if hasattr(left_hip, 'visibility') else True) and \
                              (right_hip.visibility > 0.3 if hasattr(right_hip, 'visibility') else True)
                
                if hip_visible and left_hip.y > 0.1 and right_hip.y > 0.1:
                    pocket_x = (left_hip.x + right_hip.x) / 2
                    pocket_y = (left_hip.y + right_hip.y) / 2
                else:
                    # Estimate pocket area: roughly shoulder_width * 1.5 below shoulders
                    pocket_x = shoulder_center_x
                    pocket_y = shoulder_center_y + (shoulder_width * 1.2)  # Below shoulder line
                
                # Debug: Draw reference points
                # Collar reference point (cyan dot)
                cv2.circle(frame, (int(collar_x * w), int(collar_y * h)), 8, (255, 255, 0), -1)
                cv2.putText(frame, "collar", (int(collar_x * w) + 10, int(collar_y * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Pocket reference point (magenta dot)
                cv2.circle(frame, (int(pocket_x * w), int(pocket_y * h)), 8, (255, 0, 255), -1)
                cv2.putText(frame, "pocket", (int(pocket_x * w) + 10, int(pocket_y * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # Draw wrist positions
                cv2.circle(frame, (int(left_wrist.x * w), int(left_wrist.y * h)), 6, (0, 255, 0), -1)
                cv2.circle(frame, (int(right_wrist.x * w), int(right_wrist.y * h)), 6, (0, 255, 0), -1)

                # Store for frontend visualization
                pose_data.append({
                    "collar": {"x": collar_x, "y": collar_y},
                    "pocket": {"x": pocket_x, "y": pocket_y},
                    "shoulders": {"left": {"x": left_shoulder.x, "y": left_shoulder.y}, "right": {"x": right_shoulder.x, "y": right_shoulder.y}}
                })

                # Hand-to-Face
                for hand_name, wrist in [("left", left_wrist), ("right", right_wrist)]:
                    dist = np.sqrt((wrist.x - nose.x)**2 + (wrist.y - nose.y)**2) / shoulder_width
                    if dist < self.face_distance_threshold:
                        if self.hand_face_tracker[hand_name]["start_time"] is None:
                            self.hand_face_tracker[hand_name]["start_time"] = current_time
                        duration = current_time - self.hand_face_tracker[hand_name]["start_time"]
                        if duration > self.face_duration_threshold:
                            alerts.append(f"⚠️ {hand_name.upper()} hand near FACE ({duration:.1f}s)")
                            cv2.circle(frame, (int(wrist.x * w), int(wrist.y * h)), 20, (0, 0, 255), 3)
                    else:
                        self.hand_face_tracker[hand_name]["start_time"] = None

                # Hand-to-Collar/Neck - PIXEL BASED for accuracy
                for hand_name, wrist in [("left", left_wrist), ("right", right_wrist)]:
                    # Calculate pixel distance
                    wrist_px = (wrist.x * w, wrist.y * h)
                    collar_px = (collar_x * w, collar_y * h)
                    pixel_dist = np.sqrt((wrist_px[0] - collar_px[0])**2 + (wrist_px[1] - collar_px[1])**2)
                    collar_threshold_px = shoulder_width * w * 0.5  # 50% of shoulder width in pixels
                    
                    if pixel_dist < collar_threshold_px:
                        if self.hand_collar_tracker[hand_name]["start_time"] is None:
                            self.hand_collar_tracker[hand_name]["start_time"] = current_time
                            print(f"[TRACKING] Hand near COLLAR started ({hand_name}, dist={pixel_dist:.0f}px, thresh={collar_threshold_px:.0f}px)")
                        duration = current_time - self.hand_collar_tracker[hand_name]["start_time"]
                        if duration > self.collar_duration_threshold:
                            alerts.append(f"👔 {hand_name.upper()} hand at COLLAR ({duration:.1f}s)")
                            print(f"[ALERT] Hand at COLLAR for {duration:.1f}s!")
                            cv2.circle(frame, (int(wrist.x * w), int(wrist.y * h)), 22, (0, 165, 255), 3)  # Orange
                    else:
                        self.hand_collar_tracker[hand_name]["start_time"] = None

                # Hand-to-Pocket - PIXEL BASED for accuracy
                for hand_name, wrist in [("left", left_wrist), ("right", right_wrist)]:
                    # Calculate pixel distance
                    wrist_px = (wrist.x * w, wrist.y * h)
                    pocket_px = (pocket_x * w, pocket_y * h)
                    pixel_dist = np.sqrt((wrist_px[0] - pocket_px[0])**2 + (wrist_px[1] - pocket_px[1])**2)
                    pocket_threshold_px = shoulder_width * w * 0.6  # 60% of shoulder width in pixels
                    
                    if pixel_dist < pocket_threshold_px:
                        if self.hand_pocket_tracker[hand_name]["start_time"] is None:
                            self.hand_pocket_tracker[hand_name]["start_time"] = current_time
                            print(f"[TRACKING] Hand near POCKET started ({hand_name}, dist={pixel_dist:.0f}px, thresh={pocket_threshold_px:.0f}px)")
                        duration = current_time - self.hand_pocket_tracker[hand_name]["start_time"]
                        if duration > self.pocket_duration_threshold:
                            alerts.append(f"📱 {hand_name.upper()} hand at POCKET ({duration:.1f}s)")
                            print(f"[ALERT] Hand at POCKET for {duration:.1f}s!")
                            cv2.circle(frame, (int(wrist.x * w), int(wrist.y * h)), 25, (255, 0, 255), 3)
                    else:
                        self.hand_pocket_tracker[hand_name]["start_time"] = None

        # 2. Run Hand detection
        hand_results = self.hand_landmarker.detect(mp_image)
        
        if hand_results.hand_landmarks:
            for hand_landmarks in hand_results.hand_landmarks:
                x_coords = [lm.x * w for lm in hand_landmarks]
                y_coords = [lm.y * h for lm in hand_landmarks]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                hand_bboxes.append({"bbox": (x_min, y_min, x_max, y_max), "center": ((x_min + x_max)/2, (y_min + y_max)/2)})
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            
            if len(hand_bboxes) >= 2:
                for i in range(len(hand_bboxes)):
                    for j in range(i+1, len(hand_bboxes)):
                        if self._check_bbox_overlap(hand_bboxes[i]["bbox"], hand_bboxes[j]["bbox"], self.hand_overlap_threshold):
                            if current_time - self.material_exchange_tracker["last_alert"] > self.material_exchange_tracker["cooldown"]:
                                alerts.append("🔴 HIGH PRIORITY: Possible MATERIAL EXCHANGE!")
                                self.material_exchange_tracker["last_alert"] = current_time
                                cx, cy = int((hand_bboxes[i]["center"][0] + hand_bboxes[j]["center"][0])/2), int((hand_bboxes[i]["center"][1] + hand_bboxes[j]["center"][1])/2)
                                cv2.circle(frame, (cx, cy), 40, (0, 0, 255), 4)

        return frame, alerts, pose_data

    def _check_bbox_overlap(self, b1, b2, t):
        return (b1[0] <= b2[2]+t and b1[2] >= b2[0]-t and b1[1] <= b2[3]+t and b1[3] >= b2[1]-t)

    def reset_trackers(self):
        self.hand_face_tracker.clear()
        self.hand_pocket_tracker.clear()
        self.hand_collar_tracker.clear()
