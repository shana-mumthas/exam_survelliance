
import unittest
import numpy as np
import sys
import os

# Ensure app is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.face_verify import FaceVerifier

class TestFaceVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Initializing FaceVerifier for tests...")
        cls.verifier = FaceVerifier()

    def test_initialization(self):
        """Test that models are loaded correctly."""
        self.assertIsNotNone(self.verifier.mtcnn)
        self.assertIsNotNone(self.verifier.resnet)

    def test_detect_faces_no_face(self):
        """Test detection on a blank black image."""
        # Create a black image (480x640x3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        boxes, probs, landmarks = self.verifier.detect_faces(frame)
        
        # Expecting empty or None depending on library behavior, 
        # but facenet-pytorch usually returns None or empty arrays if no faces.
        # Based on code: "if landmarks is None: return []" in check_peeking implies it can be None.
        
        # Just ensure it runs without error and returns consistent types
        if boxes is not None:
            self.assertEqual(len(boxes), 0)

    def test_check_peeking_straight(self):
        """Test peeking logic for straight face."""
        # Mock landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        # Only X coordinates matter for the current logic
        # Straight: Nose (50) exactly between Left Eye (10) and Right Eye (90)
        # l_n = 40, n_r = 40, ratio = 1.0
        
        landmarks = [[
            [10, 0], # Left Eye
            [90, 0], # Right Eye
            [50, 25], # Nose (Vertical neutral position)
            [20, 50],
            [80, 50]
        ]]
        
        alerts = self.verifier.check_peeking(landmarks, 100)
        self.assertEqual(len(alerts), 0, f"Should be straight, got: {alerts}")

    def test_check_peeking_looking_right(self):
        """Test peeking logic for looking right (subject's right)."""
        # Looking Right -> Nose moves Left (e.g. to 20)
        # Left Eye (10), Right Eye (90)
        # l_n = 20-10 = 10
        # n_r = 90-20 = 70
        # Ratio = 10/70 = 0.14 (< 0.5)
        
        landmarks = [[
            [10, 0], 
            [90, 0], 
            [20, 0], # Nose moved left
            [20, 50],
            [80, 50]
        ]]
        
        alerts = self.verifier.check_peeking(landmarks, 100)
        self.assertTrue(any("Looking Right" in a for a in alerts), f"Expected 'Looking Right', got: {alerts}")

    def test_check_peeking_looking_left(self):
        """Test peeking logic for looking left (subject's left)."""
        # Looking Left -> Nose moves Right (e.g. to 80)
        # Left Eye (10), Right Eye (90)
        # l_n = 80-10 = 70
        # n_r = 90-80 = 10
        # Ratio = 70/10 = 7.0 (> 2.0)
        
        landmarks = [[
            [10, 0], 
            [90, 0], 
            [80, 0], # Nose moved right
            [20, 50],
            [80, 50]
        ]]
        
        alerts = self.verifier.check_peeking(landmarks, 100)
        self.assertTrue(any("Looking Left" in a for a in alerts), f"Expected 'Looking Left', got: {alerts}")

if __name__ == '__main__':
    unittest.main()
