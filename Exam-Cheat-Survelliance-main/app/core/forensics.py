import cv2
import os
import time
import threading
from collections import deque

class ForensicsManager:
    def __init__(self, output_dir="forensics_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.frame_buffer = deque(maxlen=150) # 5 seconds at 30fps

    def update_buffer(self, frame):
        self.frame_buffer.append(frame.copy())

    def record_violation(self, current_frame, violation):
        """Asynchronous saving of evidence"""
        threading.Thread(target=self._save_task, args=(current_frame, violation), daemon=True).start()

    def _save_task(self, frame, violation):
        timestamp = int(time.time())
        v_type = violation['type']
        
        # 1. Save Screenshot
        img_path = os.path.join(self.output_dir, f"violation_{timestamp}_{v_type}.jpg")
        cv2.imwrite(img_path, frame)
        
        # 2. Save 5s Video Clip (from buffer)
        if len(self.frame_buffer) > 0:
            vid_path = os.path.join(self.output_dir, f"evidence_{timestamp}_{v_type}.mp4")
            h, w = self.frame_buffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_path, fourcc, 30, (w, h))
            
            for f in list(self.frame_buffer):
                out.write(f)
            out.release()
            print(f"✅ Evidence saved: {img_path} and {vid_path}")
