import time
import numpy as np
import cv2
from collections import deque

class BehavioralEngine:
    def __init__(self):
        # Sliding windows
        self.pose_history = deque(maxlen=90) # ~3s at 30fps
        self.violation_counts = deque(maxlen=1800) # 1 minute window
        
        self.is_looking_away = False
        self.look_away_start_time = None
        self.violation_threshold_duration = 3.0 # seconds
        self.violation_threshold_count = 4 # per minute
        
        # Directional gaze tracking (left, right, down)
        self.gaze_direction_tracker = {
            "left": {"start_time": None, "is_active": False},
            "right": {"start_time": None, "is_active": False},
            "down": {"start_time": None, "is_active": False}
        }
        self.gaze_direction_threshold = 3.0  # 3 seconds threshold for flagging
        
        # Thresholds for detecting gaze direction (in degrees)
        self.yaw_left_threshold = -25   # Looking left (negative yaw)
        self.yaw_right_threshold = 25   # Looking right (positive yaw)
        self.pitch_down_threshold = 20  # Looking down (positive pitch)

    def estimate_pose(self, landmarks, frame_size):
        """Calculate Yaw and Pitch from Face Mesh landmarks"""
        h, w, _ = frame_size
        
        # Reference points (indices from MediaPipe Face Mesh)
        # 1: Nose tip, 152: Chin, 33: Left Eye, 263: Right Eye, 61: Mouth Left, 291: Mouth Right
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
            (landmarks[152].x * w, landmarks[152].y * h), # Chin
            (landmarks[33].x * w, landmarks[33].y * h),   # Left eye left corner
            (landmarks[263].x * w, landmarks[263].y * h), # Right eye right corner
            (landmarks[61].x * w, landmarks[61].y * h),   # Left mouth corner
            (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth corner
        ], dtype="double")

        # Generic 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Camera internals
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Convert to Euler angles
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))
        pitch, yaw, roll = angles.flatten()

        # Iris Tracking Logic
        left_iris = self._get_iris_center(landmarks, [468, 469, 470, 471])
        right_iris = self._get_iris_center(landmarks, [473, 474, 475, 476])
        
        return {"yaw": yaw, "pitch": pitch, "roll": roll, "left_iris": left_iris, "right_iris": right_iris}

    def _get_iris_center(self, landmarks, indices):
        points = []
        for i in indices:
            p = landmarks.landmark[i]
            points.append([p.x, p.y])
        return np.mean(points, axis=0).tolist()

    def _track_gaze_direction(self, direction, is_looking, current_time):
        """
        Track gaze in a specific direction and return a violation if duration exceeds threshold.
        
        Args:
            direction: "left", "right", or "down"
            is_looking: Boolean indicating if currently looking in that direction
            current_time: Current timestamp
            
        Returns:
            Violation dict if threshold exceeded, None otherwise
        """
        tracker = self.gaze_direction_tracker[direction]
        
        if is_looking:
            if not tracker["is_active"]:
                # Start tracking this direction
                tracker["start_time"] = current_time
                tracker["is_active"] = True
            else:
                # Check duration
                duration = current_time - tracker["start_time"]
                if duration >= self.gaze_direction_threshold:
                    # Reset tracker to avoid repeated alerts
                    tracker["is_active"] = False
                    tracker["start_time"] = None
                    self.violation_counts.append(current_time)
                    return {
                        "type": f"PEEKING_{direction.upper()}_PROLONGED",
                        "conf": 0.92,
                        "duration": round(duration, 1),
                        "direction": direction
                    }
        else:
            # Reset this direction's tracker if not looking there
            if tracker["is_active"]:
                tracker["is_active"] = False
                tracker["start_time"] = None
        
        return None

    def update(self, pose, detections):
        """Main logic for filtering and flagging violations"""
        current_time = time.time()
        
        # 1. Immediate violation: Prohibited Object Detection
        for d in detections:
            if d['label'] in ['cell phone', 'earbud', 'earphone', 'book']:
                return {"type": f"PROHIBITED_OBJECT_{d['label'].upper()}", "conf": d['conf']}

        # 2. Temporal logic: Directional Gaze Tracking (Left, Right, Down)
        if pose:
            yaw = pose['yaw']
            pitch = pose['pitch']
            
            # Determine current gaze direction
            looking_left = yaw < self.yaw_left_threshold
            looking_right = yaw > self.yaw_right_threshold
            looking_down = pitch > self.pitch_down_threshold and not looking_left and not looking_right
            
            # Track LEFT gaze
            gaze_violation = self._track_gaze_direction("left", looking_left, current_time)
            if gaze_violation:
                return gaze_violation
            
            # Track RIGHT gaze
            gaze_violation = self._track_gaze_direction("right", looking_right, current_time)
            if gaze_violation:
                return gaze_violation
            
            # Track DOWN gaze
            gaze_violation = self._track_gaze_direction("down", looking_down, current_time)
            if gaze_violation:
                return gaze_violation
            
            # General looking away tracking (for frequency counting only)
            is_off_center = looking_left or looking_right or looking_down
            
            if is_off_center:
                if not self.is_looking_away:
                    self.is_looking_away = True
                    self.look_away_start_time = current_time
            else:
                if self.is_looking_away:
                    # Just looked back after a brief glance
                    duration = current_time - self.look_away_start_time
                    if duration > 0.5: # Filter micro-glances
                        self.violation_counts.append(current_time)
                    self.is_looking_away = False

        # 3. Frequency check: > 4 times in a minute
        recent_violations = [v for v in self.violation_counts if current_time - v < 60]
        if len(recent_violations) >= self.violation_threshold_count:
            self.violation_counts.clear() # Reset after flagging
            return {"type": "SUSPICIOUS_BEHAVIOR_FREQUENCY", "conf": 0.85}

        return None

    def get_gaze_status(self, pose):
        """
        Get current gaze direction and duration for UI display.
        Returns dict with direction info and time remaining before flag.
        """
        if not pose:
            return {"direction": "center", "duration": 0, "warning": False}
        
        yaw = pose['yaw']
        pitch = pose['pitch']
        current_time = time.time()
        
        # Determine direction
        if yaw < self.yaw_left_threshold:
            direction = "left"
        elif yaw > self.yaw_right_threshold:
            direction = "right"
        elif pitch > self.pitch_down_threshold:
            direction = "down"
        else:
            direction = "center"
        
        # Get duration if looking away
        duration = 0
        warning = False
        if direction != "center":
            tracker = self.gaze_direction_tracker[direction]
            if tracker["is_active"] and tracker["start_time"]:
                duration = current_time - tracker["start_time"]
                warning = duration >= (self.gaze_direction_threshold * 0.66)  # Warn at 2 seconds
        
        return {
            "direction": direction,
            "duration": round(duration, 1),
            "warning": warning,
            "threshold": self.gaze_direction_threshold
        }
