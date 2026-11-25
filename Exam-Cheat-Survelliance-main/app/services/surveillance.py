import cv2
import base64
import time
from app.services.detector import CheatDetector
from app.services.face_verify import FaceVerifier
from app.services.behavior_analyzer import BehaviorAnalyzer

class SurveillanceSystem:
    def __init__(self):
        self.cheat_detector = CheatDetector()
        self.face_verifier = FaceVerifier()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.active = False
        self.frame_count = 0
        self.detection_history = {} # label -> list of recent frames (bool: detected or from last_detections)
        self.history_limit = 5 # buffer size

    def process_frame(self, frame, mirror=False):
        """
        Process a single frame:
        Returns annotated frame (base64) and alerts
        """
        self.frame_count += 1
        
        if mirror:
            frame = cv2.flip(frame, 1)

        annot_frame = frame.copy()
        alerts = []
        
        try:
            # 1. Object Detection - Run every 3 frames for better reactivity
            if self.frame_count % 3 == 0:
                self.last_detections = self.cheat_detector.detect(frame)
            
            if hasattr(self, 'last_detections'):
                current_frame_labels = set()
                # Draw last detections for visual continuity
                for det in self.last_detections:
                    box = [int(x) for x in det['box']]
                    label = det['label']
                    conf = det['confidence']
                    current_frame_labels.add(label)
                    
                    # Yellow for specialized detection
                    color = (0, 255, 255) if label in self.cheat_detector.classes else (0, 0, 255)
                    
                    cv2.rectangle(annot_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(annot_frame, f"{label} {conf:.2f}", (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Temporal Smoothing Logic
                # Setting to 1/5 for MAX responsiveness during debugging
                for label in self.cheat_detector.classes:
                    if label not in self.detection_history:
                        self.detection_history[label] = []
                    
                    # Update history
                    is_present = label in current_frame_labels
                    self.detection_history[label].append(is_present)
                    if len(self.detection_history[label]) > self.history_limit:
                        self.detection_history[label].pop(0)
                    
                    # Trigger alert immediately if seen in 1 of last 5 frames for debugging
                    if sum(self.detection_history[label]) >= 1:
                        # Find confidence from current detections if available
                        conf = next((d['confidence'] for d in self.last_detections if d['label'] == label), 0.0)
                        if conf > 0:
                            alerts.append(f"STABLE ALERT: {label} ({conf:.2f})")

            # 2. Face Tracking & Identification - Run every 2 frames for better gaze tracking
            if self.frame_count % 2 == 0:
                self.last_face_data, self.last_peeking_alerts = self.face_verifier.check_peeking(frame)
            
            # CRITICAL: Always extend alerts with peeking results
            if hasattr(self, 'last_peeking_alerts'):
                alerts.extend(self.last_peeking_alerts)
            face_data = getattr(self, 'last_face_data', [])
            peeking_alerts = getattr(self, 'last_peeking_alerts', [])
            
            for face in face_data:
                box = face["box"]
                # Identification
                face_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face_img.size > 0 and self.frame_count % 10 == 0:
                    self.last_name = self.face_verifier.identify_face(face_img)
                name = getattr(self, 'last_name', "Detecting...")
                
                # Draw box and info
                color = (255, 0, 0) if face["status"] == "Normal" else (0, 0, 255)
                cv2.rectangle(annot_frame, 
                             (int(box[0]), int(box[1])), 
                             (int(box[2]), int(box[3])), 
                             color, 2)
                
                # Display name and status
                label = f"{name} | {face['status']}"
                cv2.putText(annot_frame, label, (int(box[0]), int(box[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Display MAR and Gaze info if available
                if "mar" in face:
                    mar_text = f"MAR:{face['mar']:.2f}"
                    cv2.putText(annot_frame, mar_text, (int(box[0]), int(box[3]) + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Display Yaw/Pitch for debugging peeking detection
                if "yaw" in face and "pitch" in face:
                    yaw_pitch_text = f"Yaw:{face['yaw']:.1f} Pitch:{face['pitch']:.1f}"
                    cv2.putText(annot_frame, yaw_pitch_text, (int(box[0]), int(box[3]) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                if "h_gaze" in face and "v_gaze" in face:
                    gaze_text = f"Eye H:{face['h_gaze']:.2f} V:{face['v_gaze']:.2f}"
                    cv2.putText(annot_frame, gaze_text, (int(box[0]), int(box[3]) + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                if face["status"] != "Normal":
                    alerts.append(f"{name}: {face['status']}")
            
            # 3. Behavior Analysis - Hand-to-face/pocket detection & Material exchange
            # Run every 4 frames for more responsive behavior detection
            if self.frame_count % 4 == 0:
                try:
                    annot_frame, behavior_alerts, pose_data = self.behavior_analyzer.analyze_frame(annot_frame)
                    self.last_behavior_alerts = behavior_alerts
                    self.last_pose_data = pose_data
                    alerts.extend(behavior_alerts)
                except Exception as inner_e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[BEHAVIOR ERROR] analyze_frame failed: {inner_e}\n{tb}")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Error in process_frame: {e}\n{tb}")

        # Prepare raw data for frontend rendering to avoid quality loss
        raw_data = {
            "detections": getattr(self, "last_detections", []),
            "faces": [],
            "alerts": alerts,
            "pose_data": getattr(self, "last_pose_data", [])
        }
        
        # Sanitize face data (remove non-serializable landmarks)
        for face in face_data:
            sanitized_face = {k: v for k, v in face.items() if k != 'landmarks'}
            raw_data["faces"].append(sanitized_face)

        # Encode frame to JPEG then Base64 (medium quality for speed)
        # Frontend will use local video for smoothness, but we keep this for legacy/remote view
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
        _, buffer = cv2.imencode('.jpg', annot_frame, encode_params)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64, alerts, raw_data

    def register_student(self, name, frame):
        """
        Register a student name with the current frame's face.
        Returns: (success: bool, message: str)
        """
        success, msg = self.face_verifier.register_face(name, frame)
        return success, msg
