import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
from app.services.face_verify import FaceVerifier
import time

def main():
    print("Initializing FaceVerifier...")
    try:
        verifier = FaceVerifier()
    except Exception as e:
        print(f"Failed to init FaceVerifier: {e}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n=== HEAD POSE CALIBRATION ===")
    print("Press 'q' to quit.")
    print("Instructions:")
    print("1. Look straight at the camera. Note the ratios.")
    print("2. Look slowly LEFT. Note the H-Ratio.")
    print("3. Look slowly RIGHT. Note the H-Ratio.")
    print("4. Look DOWN (as if writing). Note the V-Ratio.")
    print("=============================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror for calibration ease
        frame = cv2.flip(frame, 1)

        boxes, probs, landmarks = verifier.detect_faces(frame)

        if landmarks is not None:
            for i, lm in enumerate(landmarks):
                # Landmarks: 0:L-Eye, 1:R-Eye, 2:Nose, 3:L-Mouth, 4:R-Mouth
                left_eye = lm[0]
                right_eye = lm[1]
                nose = lm[2]
                
                # --- Horizontal Ratio (Yaw) ---
                dist_l_n = nose[0] - left_eye[0]
                dist_n_r = right_eye[0] - nose[0]
                # Avoid div by zero
                if dist_n_r == 0: dist_n_r = 0.001
                if dist_l_n == 0: dist_l_n = 0.001
                
                h_ratio = dist_l_n / dist_n_r
                
                # --- Vertical Ratio (Pitch) ---
                # Eye width as scale reference
                eye_width = right_eye[0] - left_eye[0]
                if eye_width == 0: eye_width = 1
                
                # Vertical dist from eye-line to nose
                eye_y_avg = (left_eye[1] + right_eye[1]) / 2
                v_dist = nose[1] - eye_y_avg
                
                v_ratio = v_dist / eye_width

                # Draw on frame
                box = boxes[i]
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                
                # Display values
                info_text = f"H: {h_ratio:.2f} | V: {v_ratio:.2f}"
                cv2.putText(frame, info_text, (int(box[0]), int(box[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw landmarks
                for point in lm:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 255), -1)

                # Threshold Guidance (Matching face_verify.py)
                status = "Normal"
                if v_ratio > 0.65: 
                    status = "DOWN"
                elif h_ratio > 1.7:
                    status = "LEFT"
                elif h_ratio < 0.6:
                    status = "RIGHT"
                
                cv2.putText(frame, f"Status: {status}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('Calibration - Press Q to Quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
