import torch
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

class FaceVerifier:
    def __init__(self):
        print("Loading FaceNet & MediaPipe models...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # FaceNet for Identification
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.known_faces = {} # name -> embedding

        # MediaPipe Face Landmarker (new API)
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading face landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # 3D model points for pose estimation (generic face)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)
        
        # Temporal tracking for gaze alerts (only alert after 3 seconds)
        import time
        self.gaze_tracker = {
            "left": {"start_time": None},
            "right": {"start_time": None},
            "down": {"start_time": None},
            "eye_left": {"start_time": None},
            "eye_right": {"start_time": None},
            "eye_down": {"start_time": None}
        }
        self.gaze_duration_threshold = 3.0  # seconds - flag after looking away for 3+ seconds

    def calculate_mar(self, face_landmarks, w, h):
        """
        Calculate Mouth Aspect Ratio (MAR) to detect talking.
        Uses inner lip landmarks for accurate measurement.
        MAR = vertical_dist / horizontal_dist
        Returns: float (0.0 = closed, 0.3+ = talking, 0.6+ = wide open)
        """
        try:
            # Inner lip landmarks (MediaPipe Face Mesh)
            # Top inner lip: 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82
            # Bottom inner lip: 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87
            # Simplified: use key points for vertical height
            
            # Top lip center (point 13)
            top_lip = np.array([face_landmarks[13].x * w, face_landmarks[13].y * h])
            # Bottom lip center (point 14)  
            bottom_lip = np.array([face_landmarks[14].x * w, face_landmarks[14].y * h])
            # Left corner (point 78)
            left_corner = np.array([face_landmarks[78].x * w, face_landmarks[78].y * h])
            # Right corner (point 308)
            right_corner = np.array([face_landmarks[308].x * w, face_landmarks[308].y * h])
            
            # Additional vertical points for accuracy
            top_lip_2 = np.array([face_landmarks[82].x * w, face_landmarks[82].y * h])
            bottom_lip_2 = np.array([face_landmarks[87].x * w, face_landmarks[87].y * h])
            top_lip_3 = np.array([face_landmarks[312].x * w, face_landmarks[312].y * h])
            bottom_lip_3 = np.array([face_landmarks[317].x * w, face_landmarks[317].y * h])
            
            # Calculate vertical distances
            vert_dist_1 = np.linalg.norm(top_lip - bottom_lip)
            vert_dist_2 = np.linalg.norm(top_lip_2 - bottom_lip_2)
            vert_dist_3 = np.linalg.norm(top_lip_3 - bottom_lip_3)
            avg_vert = (vert_dist_1 + vert_dist_2 + vert_dist_3) / 3.0
            
            # Calculate horizontal distance
            horiz_dist = np.linalg.norm(left_corner - right_corner)
            
            if horiz_dist < 1:  # Avoid division by zero
                return 0.0
            
            mar = avg_vert / horiz_dist
            return mar
        except Exception as e:
            print(f"Error calculating MAR: {e}")
            return 0.0

    def get_iris_gaze(self, face_landmarks, w, h):
        """
        Calculate iris position relative to eye corners for gaze estimation.
        Works even when head is still - detects eye-only movement.
        Returns: (h_gaze, v_gaze) where -1 = left/up, 0 = center, 1 = right/down
        """
        try:
            # Left eye landmarks (MediaPipe Face Mesh 478 landmarks include iris)
            # Left iris center: 468 (if available in 478 model)
            # Right iris center: 473
            # Fallback to pupil estimation from eye corners if iris not available
            
            # Left eye corners
            left_eye_inner = np.array([face_landmarks[133].x * w, face_landmarks[133].y * h])
            left_eye_outer = np.array([face_landmarks[33].x * w, face_landmarks[33].y * h])
            left_eye_top = np.array([face_landmarks[159].x * w, face_landmarks[159].y * h])
            left_eye_bottom = np.array([face_landmarks[145].x * w, face_landmarks[145].y * h])
            
            # Right eye corners
            right_eye_inner = np.array([face_landmarks[362].x * w, face_landmarks[362].y * h])
            right_eye_outer = np.array([face_landmarks[263].x * w, face_landmarks[263].y * h])
            right_eye_top = np.array([face_landmarks[386].x * w, face_landmarks[386].y * h])
            right_eye_bottom = np.array([face_landmarks[374].x * w, face_landmarks[374].y * h])
            
            # Try to get iris landmarks (indices 468-477 for left iris, 473 is center)
            # Note: Only available if face_landmarker model outputs refine_landmarks
            left_iris_center = None
            right_iris_center = None
            
            if len(face_landmarks) > 468:
                # Iris landmarks available
                left_iris_center = np.array([face_landmarks[468].x * w, face_landmarks[468].y * h])
                right_iris_center = np.array([face_landmarks[473].x * w, face_landmarks[473].y * h])
            else:
                # Estimate iris from eye center (less accurate but functional)
                left_iris_center = (left_eye_inner + left_eye_outer + left_eye_top + left_eye_bottom) / 4
                right_iris_center = (right_eye_inner + right_eye_outer + right_eye_top + right_eye_bottom) / 4
            
            # Calculate gaze ratios for left eye
            left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
            left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
            
            if left_eye_width < 1 or left_eye_height < 1:
                return 0.0, 0.0
            
            # Horizontal gaze: position of iris relative to eye corners
            left_h_ratio = (left_iris_center[0] - left_eye_outer[0]) / left_eye_width
            # Vertical gaze: position of iris relative to eye top/bottom
            left_v_ratio = (left_iris_center[1] - left_eye_top[1]) / left_eye_height
            
            # Same for right eye
            right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
            right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
            
            if right_eye_width < 1 or right_eye_height < 1:
                return 0.0, 0.0
            
            right_h_ratio = (right_iris_center[0] - right_eye_inner[0]) / right_eye_width
            right_v_ratio = (right_iris_center[1] - right_eye_top[1]) / right_eye_height
            
            # Average both eyes and normalize to [-1, 1]
            avg_h = (left_h_ratio + right_h_ratio) / 2
            avg_v = (left_v_ratio + right_v_ratio) / 2
            
            # Normalize: 0.5 = center, <0.35 = looking left, >0.65 = looking right
            h_gaze = (avg_h - 0.5) * 2  # Now -1 to 1
            v_gaze = (avg_v - 0.5) * 2  # Now -1 to 1
            
            return h_gaze, v_gaze
        except Exception as e:
            print(f"Error calculating iris gaze: {e}")
            return 0.0, 0.0
        
    def detect_faces(self, frame):
        """
        Detect faces and return boxes and landmarks.
        """
        # MTCNN expects RGB image. OpenCV uses BGR.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
        return boxes, probs, landmarks

    def check_peeking(self, frame):
        """
        Estimate gaze/head pose using MediaPipe Face Landmarker.
        Uses simple landmark ratios for reliable left/right/down detection.
        Returns: boxes (for drawing) and alerts.
        """
        try:
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Ensure frame is C-contiguous for MediaPipe
            rgb_frame = np.ascontiguousarray(rgb_frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.face_landmarker.detect(mp_image)
            
            alerts = []
            face_data = [] # List of {box, yaw, pitch, landmarks}
            import time
            current_time = time.time()

            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    try:
                        # === SIMPLE LANDMARK-BASED HEAD POSE DETECTION ===
                        # Use nose position relative to eye centers for yaw (left/right)
                        # Use nose-to-chin vs nose-to-forehead ratio for pitch (up/down)
                        
                        # Key landmarks
                        nose_tip = face_landmarks[1]      # Nose tip
                        left_eye = face_landmarks[33]     # Left eye outer corner
                        right_eye = face_landmarks[263]   # Right eye outer corner
                        chin = face_landmarks[152]        # Chin
                        forehead = face_landmarks[10]     # Forehead center
                        left_ear = face_landmarks[234]    # Left ear
                        right_ear = face_landmarks[454]   # Right ear
                        
                        # Calculate face width using ears or eyes
                        face_width = abs(right_eye.x - left_eye.x)
                        if face_width < 0.01:
                            face_width = 0.1  # Prevent division by zero
                        
                        # Calculate eye center
                        eye_center_x = (left_eye.x + right_eye.x) / 2
                        
                        # === YAW (Left/Right) Detection ===
                        # Ratio: How far nose is from eye center relative to face width
                        # Positive = looking right (nose shifts right of eye center)
                        # Negative = looking left (nose shifts left of eye center)
                        nose_offset = (nose_tip.x - eye_center_x) / face_width
                        
                        # Convert to approximate degrees (calibrated for typical webcam)
                        # nose_offset of 0.15 ≈ 15 degrees turn
                        yaw = nose_offset * 100  # Scale to degrees-like value
                        
                        # === PITCH (Up/Down) Detection ===
                        # Use vertical position of nose relative to eyes and chin
                        eye_center_y = (left_eye.y + right_eye.y) / 2
                        face_height = abs(chin.y - forehead.y)
                        if face_height < 0.01:
                            face_height = 0.1
                        
                        # Ratio: nose below eye center = looking down
                        nose_drop = (nose_tip.y - eye_center_y) / face_height
                        pitch = nose_drop * 100  # Scale to degrees-like value
                        
                        # Debug output - ALWAYS print to see values
                        print(f"[HEAD POSE] Yaw: {yaw:.1f} (nose_offset: {nose_offset:.3f}), Pitch: {pitch:.1f}")

                        # Calibration/Thresholds
                        status = "Normal"
                        status_details = []
                        
                        # === PEEKING DETECTION with 3-second threshold ===
                        # Thresholds calibrated for webcam (values are nose_offset * 100)
                        YAW_THRESHOLD = 12  # ~12% offset = looking sideways
                        PITCH_DOWN_THRESHOLD = 35  # Looking down
                        
                        # Looking LEFT (person's left = negative yaw in mirrored view)
                        # When you turn your head LEFT, nose moves LEFT of eye center = negative offset
                        if yaw < -YAW_THRESHOLD:
                            direction = "left"
                            if self.gaze_tracker[direction]["start_time"] is None:
                                self.gaze_tracker[direction]["start_time"] = current_time
                                print(f"[TRACKING] Started tracking LOOKING LEFT (yaw={yaw:.1f})")
                            duration = current_time - self.gaze_tracker[direction]["start_time"]
                            status_details.append(f"Looking LEFT ({duration:.1f}s)")
                            if duration >= self.gaze_duration_threshold:
                                alerts.append(f"⚠️ PEEKING LEFT for {duration:.1f}s!")
                                print(f"[ALERT!] PEEKING LEFT for {duration:.1f}s!")
                        else:
                            self.gaze_tracker["left"]["start_time"] = None
                        
                        # Looking RIGHT (person's right = positive yaw)
                        if yaw > YAW_THRESHOLD:
                            direction = "right"
                            if self.gaze_tracker[direction]["start_time"] is None:
                                self.gaze_tracker[direction]["start_time"] = current_time
                                print(f"[TRACKING] Started tracking LOOKING RIGHT (yaw={yaw:.1f})")
                            duration = current_time - self.gaze_tracker[direction]["start_time"]
                            status_details.append(f"Looking RIGHT ({duration:.1f}s)")
                            if duration >= self.gaze_duration_threshold:
                                alerts.append(f"⚠️ PEEKING RIGHT for {duration:.1f}s!")
                                print(f"[ALERT!] PEEKING RIGHT for {duration:.1f}s!")
                        else:
                            self.gaze_tracker["right"]["start_time"] = None
                        
                        # Looking DOWN
                        if pitch > PITCH_DOWN_THRESHOLD:
                            direction = "down"
                            if self.gaze_tracker[direction]["start_time"] is None:
                                self.gaze_tracker[direction]["start_time"] = current_time
                                print(f"[TRACKING] Started tracking LOOKING DOWN (pitch={pitch:.1f})")
                            duration = current_time - self.gaze_tracker[direction]["start_time"]
                            status_details.append(f"Looking DOWN ({duration:.1f}s)")
                            if duration >= self.gaze_duration_threshold:
                                alerts.append(f"⚠️ LOOKING DOWN for {duration:.1f}s!")
                                print(f"[ALERT!] LOOKING DOWN for {duration:.1f}s!")
                        else:
                            self.gaze_tracker["down"]["start_time"] = None

                        # === Mouth Aspect Ratio (MAR) Detection ===
                        mar = self.calculate_mar(face_landmarks, w, h)
                        is_talking = mar > 0.15 and mar < 0.50
                        
                        # Combined alert: talking while head is turned
                        if is_talking and (abs(yaw) > 8):
                            alerts.append("⚠️ WHISPERING DETECTED")
                            status_details.append("Whispering")
                        
                        # === Iris-based Gaze (eye movement without head turn) ===
                        h_gaze, v_gaze = self.get_iris_gaze(face_landmarks, w, h)
                        
                        # Eye-only detection when head is straight
                        if abs(yaw) < YAW_THRESHOLD:
                            # Eyes looking RIGHT
                            if h_gaze > 0.35:
                                if self.gaze_tracker["eye_right"]["start_time"] is None:
                                    self.gaze_tracker["eye_right"]["start_time"] = current_time
                                duration = current_time - self.gaze_tracker["eye_right"]["start_time"]
                                if duration >= self.gaze_duration_threshold:
                                    alerts.append(f"👀 Eyes looking RIGHT ({duration:.1f}s)")
                                    status_details.append("Eyes RIGHT")
                            else:
                                self.gaze_tracker["eye_right"]["start_time"] = None
                            
                            # Eyes looking LEFT
                            if h_gaze < -0.35:
                                if self.gaze_tracker["eye_left"]["start_time"] is None:
                                    self.gaze_tracker["eye_left"]["start_time"] = current_time
                                duration = current_time - self.gaze_tracker["eye_left"]["start_time"]
                                if duration >= self.gaze_duration_threshold:
                                    alerts.append(f"👀 Eyes looking LEFT ({duration:.1f}s)")
                                    status_details.append("Eyes LEFT")
                            else:
                                self.gaze_tracker["eye_left"]["start_time"] = None
                            
                            # Eyes looking DOWN
                            if v_gaze > 0.4:
                                if self.gaze_tracker["eye_down"]["start_time"] is None:
                                    self.gaze_tracker["eye_down"]["start_time"] = current_time
                                duration = current_time - self.gaze_tracker["eye_down"]["start_time"]
                                if duration >= self.gaze_duration_threshold:
                                    alerts.append(f"👀 Eyes looking DOWN ({duration:.1f}s)")
                                    status_details.append("Eyes DOWN")
                            else:
                                self.gaze_tracker["eye_down"]["start_time"] = None
                        else:
                            # Reset eye trackers if head is turned
                            self.gaze_tracker["eye_left"]["start_time"] = None
                            self.gaze_tracker["eye_right"]["start_time"] = None
                            self.gaze_tracker["eye_down"]["start_time"] = None

                        # Build final status
                        if status_details:
                            status = " | ".join(status_details)
                        else:
                            status = "Normal"

                        # Calculate bounding box from landmarks
                        x_coords = [lm.x * w for lm in face_landmarks]
                        y_coords = [lm.y * h for lm in face_landmarks]
                        box = [
                            max(0, min(x_coords)), 
                            max(0, min(y_coords)), 
                            min(w, max(x_coords)), 
                            min(h, max(y_coords))
                        ]
                        
                        face_data.append({
                            "box": box,
                            "yaw": yaw,
                            "pitch": pitch,
                            "roll": 0,
                            "mar": mar,
                            "h_gaze": h_gaze,
                            "v_gaze": v_gaze,
                            "status": status,
                            "landmarks": face_landmarks
                        })
                    except Exception as e:
                        print(f"Error processing single face: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            return face_data, alerts
        except Exception as e:
            print(f"Error in check_peeking: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    def get_embedding(self, face_frame):
        """
        Extract embedding for a single aligned face.
        """
        # MTCNN for alignment
        img = Image.fromarray(cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB))
        face_tensor = self.mtcnn(img) # Returns [1, 3, 160, 160]
        
        if face_tensor is None:
            return None
            
        # Get embedding
        with torch.no_grad():
            embedding = self.resnet(face_tensor.to(self.device))
        return embedding.cpu().numpy()

    def register_face(self, name, frame):
        """
        Find face in frame and save embedding.
        """
        boxes, probs, landmarks = self.detect_faces(frame)
        if boxes is None or len(boxes) == 0:
            return False, "No face detected"
            
        # Take the largest face
        box = boxes[0] 
        face_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        
        emb = self.get_embedding(face_img)
        if emb is not None:
            self.known_faces[name] = emb
            print(f"Registered face: {name}")
            return True, f"Registered {name}"
        return False, "Failed to extract embedding"

    def identify_face(self, face_img):
        """
        Identify face by comparing embedding.
        """
        if not self.known_faces:
            return "Unknown"
            
        emb = self.get_embedding(face_img)
        if emb is None:
            return "Unknown"
            
        best_name = "Unknown"
        min_dist = 1.0 # Standard FaceNet threshold
        
        for name, known_emb in self.known_faces.items():
            dist = np.linalg.norm(emb - known_emb)
            if dist < min_dist:
                min_dist = dist
                best_name = name
                
        return best_name
