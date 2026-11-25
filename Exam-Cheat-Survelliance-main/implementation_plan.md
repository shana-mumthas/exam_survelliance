# AI-Driven Exam Integrity & Forensics System Plan

This plan transforms the current project into a high-performance, multi-threaded surveillance system optimized for Ryzen Edge AI (OpenVINO).

## Proposed Changes

### Environment & Dependencies
Optimized for Windows and Ryzen iGPU: `openvino`, `ultralytics`, `mediapipe`, `fastapi`, `moviepy`.

### Core Engine (Threaded)
Managing three primary threads: **Capture**, **Inference**, and **Logic**.

### Behavioral & Temporal Logic
- Implements `ViolationWindow` using `collections.deque`.
- Filters "micro-movements" (sneezing, stretching).
- Logic: `look_away_duration > 3s` OR `look_away_count > 4 per minute`.

### Forensics & Alerting
- `save_evidence()`: Captures 5s video buffer and high-res screenshot.

### API & WebSocket
- FastAPI entry point for real-time alerting via WebSockets.
