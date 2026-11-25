from ultralytics import YOLO
import os

def train_model():
    dataset_dir = r"c:\Users\afzal\Exam_survaillence\earbud_dataset"
    
    # Create data.yaml
    yaml_content = f"""
path: {dataset_dir}
train: images/train
val: images/train  # Using train as val for this synthetic demo

names:
  0: earbud
"""
    with open(os.path.join(dataset_dir, "data.yaml"), 'w') as f:
        f.write(yaml_content)

    # Load a pretrained YOLOv11n model
    model = YOLO("yolo11n.pt")

    # Train the model
    results = model.train(
        data=os.path.join(dataset_dir, "data.yaml"),
        epochs=25,
        imgsz=640,
        batch=16,
        name="earbud_detector",
        device="cpu" # Default to CPU for safety, user can change to 0 if GPU available
    )

    # Export the model
    model.export(format="onnx")
    
    # Copy best model to main directory
    import shutil
    best_path = os.path.join("runs", "detect", "earbud_detector", "weights", "best.pt")
    if os.path.exists(best_path):
        shutil.copy(best_path, r"c:\Users\afzal\Exam_survaillence\earpods_best_v3.pt")
        print("Model trained and saved as earpods_best_v3.pt")

if __name__ == "__main__":
    train_model()
