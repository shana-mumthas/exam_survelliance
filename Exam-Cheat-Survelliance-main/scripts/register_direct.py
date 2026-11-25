from app.services.surveillance import SurveillanceSystem
import cv2

s = SurveillanceSystem()
img = cv2.imread('earbud_dataset/images/train/synthetic_0.jpg')
print('Image loaded:', img is not None)
try:
    success, msg = s.register_student('LocalTest', img)
    print('Register result:', success, msg)
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Exception during register:', e)
