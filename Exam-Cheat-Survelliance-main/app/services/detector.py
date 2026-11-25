from ultralytics import YOLO
import cv2
import numpy as np
import os

class CheatDetector:
    def __init__(self):
        """
        Dual-model detector:
        1. YOLOv11n for general objects (phone, book, person, etc.)
        2. Custom-trained model for earbuds/earpods
        """
        print("Loading Dual Detection Models...")
        
        # Model 1: YOLOv11 for general object detection (COCO classes)
        self.yolo11_model = YOLO("yolo11n.pt")
        # Classes we care about from COCO (indices)
        # 67=cell phone, 73=book, 63=laptop, 66=keyboard, 64=mouse, 0=person
        self.yolo11_target_classes = {
            67: "cell phone",
            73: "book",
            63: "laptop",
            66: "keyboard",
            64: "mouse",
            74: "scissors",
            76: "scissors",
            39: "bottle"
        }

        # Labels we consider non-threatening and want to ignore (avoid false positives)
        self.ignored_labels = set(["scissors"])  # Add labels here to ignore them from alerts

        print("✓ YOLOv11n loaded for general object detection")
        
        # Model 2: Custom-trained model for earbuds/earpods
        self.earpods_model = YOLO("earpods_best.pt")
        print("✓ Custom earpods model loaded")
        
        # Combined classes for alert system (exclude ignored labels)
        self.classes = [l for l in self.yolo11_target_classes.values() if l not in self.ignored_labels] + ["earbud", "earpod", "airpod"]

    def detect(self, frame, conf_threshold=0.25):
        """
        Run both models and combine detections.
        """
        detections = []
        
        # 1. YOLOv11 for general objects (phone, book, etc.)
        yolo11_results = self.yolo11_model(frame, verbose=False, conf=conf_threshold, imgsz=640)
        
        for result in yolo11_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.yolo11_target_classes:
                    label = self.yolo11_target_classes[cls_id]
                    conf = float(box.conf[0])
                    # Skip ignored labels (not considered threats)
                    if label in self.ignored_labels:
                        # Optionally log ignored detections for debugging
                        print(f"YOLO11 IGNORED: {label} ({conf:.2f})")
                        continue
                    if conf > conf_threshold:
                        print(f"YOLO11 DETECTED: {label} ({conf:.2f})")
                        detections.append({
                            "label": label,
                            "confidence": conf,
                            "box": box.xyxy[0].tolist(),
                            "source": "yolo11"
                        })
        
        # 2. Custom earpods model (use lower threshold for small objects)
        earpods_results = self.earpods_model(frame, verbose=False, conf=0.15, imgsz=640)
        
        for result in earpods_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # Get class name from model
                label = result.names[cls_id] if cls_id in result.names else "earbud"
                if conf > 0.15:
                    print(f"EARPODS DETECTED: {label} ({conf:.2f})")
                    detections.append({
                        "label": label, 
                        "confidence": conf, 
                        "box": box.xyxy[0].tolist(),
                        "source": "earpods_custom"
                    })
        
        return detections

    def _draw(self, frame, box, label, conf, color):
        b = [int(x) for x in box]
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (b[0], b[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
