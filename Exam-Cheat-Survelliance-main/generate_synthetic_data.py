import cv2
import numpy as np
import os
import random
import glob

def create_synthetic_dataset(num_images=1000):
    # Paths to base images (using the ones generated earlier)
    # Note: These paths are in the artifacts directory
    artifact_dir = r"C:\Users\afzal\.gemini\antigravity\brain\f46b929f-f9ff-4645-a452-693c3f4f4803"
    ear_img_path = glob.glob(os.path.join(artifact_dir, "human_ear_side_view_*.png"))[0]
    wireless_img_path = glob.glob(os.path.join(artifact_dir, "wireless_earbud_single_*.png"))[0]
    wired_img_path = glob.glob(os.path.join(artifact_dir, "black_earphone_single_*.png"))[0]

    # Create dataset directories
    dataset_dir = r"c:\Users\afzal\Exam_survaillence\earbud_dataset"
    os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "train"), exist_ok=True)

    # Load base images
    ear_base = cv2.imread(ear_img_path)
    wireless_base = cv2.imread(wireless_img_path)
    wired_base = cv2.imread(wired_img_path)

    # Function to remove background (simple chroma keying since I asked for plain backgrounds)
    def remove_bg(img, bg_color='green'):
        if bg_color == 'green':
            # White wireless earbud on green background
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (35, 50, 50), (90, 255, 255))
        else:
            # Black earphone on blue background
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (100, 50, 50), (140, 255, 255))
        
        mask_inv = cv2.bitwise_not(mask)
        res = cv2.bitwise_and(img, img, mask=mask_inv)
        return res, mask_inv

    wireless_obj, wireless_mask = remove_bg(wireless_base, 'green')
    wired_obj, wired_mask = remove_bg(wired_base, 'blue')

    for i in range(num_images):
        bg = ear_base.copy()
        
        # Randomly choose object
        use_wireless = random.choice([True, False])
        obj = wireless_obj if use_wireless else wired_obj
        mask = wireless_mask if use_wireless else wired_mask
        label_id = 0 # Single class: earbud/earphone

        # Random scaling
        scale = random.uniform(0.15, 0.3)
        obj_resized = cv2.resize(obj, None, fx=scale, fy=scale)
        mask_resized = cv2.resize(mask, None, fx=scale, fy=scale)

        # Random rotation
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((obj_resized.shape[1]//2, obj_resized.shape[0]//2), angle, 1.0)
        obj_final = cv2.warpAffine(obj_resized, M, (obj_resized.shape[1], obj_resized.shape[0]))
        mask_final = cv2.warpAffine(mask_resized, M, (obj_resized.shape[1], obj_resized.shape[0]))

        # Random position in the ear canal area (roughly center for this specific base image)
        h, w = bg.shape[:2]
        oh, ow = obj_final.shape[:2]
        x = random.randint(int(w*0.3), int(w*0.5))
        y = random.randint(int(h*0.3), int(h*0.6))

        # Overlay
        roi = bg[y:y+oh, x:x+ow]
        mask_inv = cv2.bitwise_not(mask_final)
        bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg_part = cv2.bitwise_and(obj_final, obj_final, mask=mask_final)
        bg[y:y+oh, x:x+ow] = cv2.add(bg_part, fg_part)

        # YOLO label (center_x, center_y, width, height) normalized
        cx = (x + ow/2) / w
        cy = (y + oh/2) / h
        bw = ow / w
        bh = oh / h

        # Save image and label
        img_name = f"synthetic_{i}.jpg"
        cv2.imwrite(os.path.join(dataset_dir, "images", "train", img_name), bg)
        with open(os.path.join(dataset_dir, "labels", "train", f"synthetic_{i}.txt"), 'w') as f:
            f.write(f"{label_id} {cx} {cy} {bw} {bh}\n")

    print(f"Generated {num_images} synthetic images in {dataset_dir}")

if __name__ == "__main__":
    create_synthetic_dataset(500) # Start with 500 for speed
