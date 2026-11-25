import cv2
import numpy as np
from app.services.detector import CheatDetector
from ultralytics import YOLO

def diagnose():
    print("--- Detector Diagnosis ---")
    detector = CheatDetector()
    
    print(f"World Model Classes: {detector.world_classes}")
    print(f"World Model Indices: {detector.world_indices}")
    
    # Check if the model actually has these names
    print(f"Internal Model Names: {detector.world_model.names}")
    
    # Test with a very low threshold
    frame = np.zeros((480, 640, 3), dtype=np.uint8) # Black frame
    # Simulate a white dot where an earbud might be
    cv2.circle(frame, (320, 240), 10, (255, 255, 255), -1)
    
    print("\nRunning test detection (conf=0.01)...")
    results = detector.world_model(frame, conf=0.01, verbose=True)
    
    for r in results:
        print(f"Detected {len(r.boxes)} objects at 0.01 confidence.")
        for box in r.boxes:
            print(f"Label: {detector.world_model.names[int(box.cls[0])]}, Conf: {float(box.conf[0]):.4f}")

if __name__ == "__main__":
    diagnose()
