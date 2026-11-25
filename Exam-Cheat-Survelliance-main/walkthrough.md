# Exam Integrity & Forensics System Walkthrough

The project has been transformed into a professional-grade, multi-threaded surveillance system optimized for **Edge AI** and **Forensics**.

## Key Features

- **Multi-Threaded Performance**: Optimized with **OpenVINO** for Ryzen 7840HS, targeting 30+ FPS.
- **Behavioral Intelligence**: Tracks Yaw/Pitch and Iris Tracking to detect "Looking Away".
- **Automatic Forensics**: Auto-saves a high-res screenshot and a 5s MP4 clip upon violation.

## How to Run

1. **Start the System**:
   ```bash
   python app/main.py
   ```
2. **Access the Dashboard**:
   Open `app/templates/dashboard.html` in your browser.
3. **Review Evidence**:
   Check the `forensics_data/` directory.

> [!TIP]
> On the first run, YOLOv11n will export to OpenVINO format. This may take 1-2 minutes.
