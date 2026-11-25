from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from app.services.surveillance import SurveillanceSystem
import asyncio
import json
import cv2
import numpy as np
import base64
import os
import time

app = FastAPI(title="Exam Integrity Forensics API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# engine = ExamIntegrityEngine()  # Disabled - using SurveillanceSystem instead
surveillance = SurveillanceSystem()

@app.on_event("startup")
async def startup_event():
    # engine.start()  # Disabled to prevent conflicts with SurveillanceSystem
    pass

@app.on_event("shutdown")
def shutdown_event():
    # engine.stop()  # Disabled to prevent conflicts with SurveillanceSystem
    pass

@app.get("/", response_class=HTMLResponse)
async def home():
    # Serve the simplified dashboard (vanilla JS, no React)
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "dashboard_simple.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading dashboard: {e}</h1><p>Path tried: {dashboard_path}</p>", status_code=500)


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """Legacy alerts websocket - kept for compatibility but not actively used"""
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(1)  # Keep connection alive
    except WebSocketDisconnect:
        print("Client disconnected from Alerts WebSocket")

@app.websocket("/ws/surveillance")
async def websocket_surveillance(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to Surveillance WebSocket")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "frame":
                try:
                    # Decode base64 frame
                    frame_data = base64.b64decode(data["frame"])
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Process frame
                        frame_base64, alerts, raw_data = surveillance.process_frame(frame, data.get("mirror", False))
                        
                        # Send back processed frame, alerts AND raw data for local rendering
                        await websocket.send_json({
                            "type": "frame_result",
                            "frame": frame_base64,
                            "alerts": alerts,
                            "data": raw_data
                        })
                except Exception as inner_e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[WEBSOCKET ERROR] frame processing error: {inner_e}\n{tb}")
                    try:
                        await websocket.send_json({"type": "error", "message": "frame_processing_error", "detail": str(inner_e)})
                    except Exception:
                        pass
            elif data["type"] == "register":
                try:
                    # Handle face registration
                    frame_data = base64.b64decode(data["frame"])
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        res = surveillance.register_student(data["name"], frame)
                        # Support both tuple (success, msg) and dict return types
                        if isinstance(res, dict):
                            await websocket.send_json({
                                "type": "registration_result",
                                "status": res.get("status", "error"),
                                "message": res.get("message", "")
                            })
                        else:
                            success, msg = res
                            await websocket.send_json({
                                "type": "registration_result",
                                "status": "success" if success else "error",
                                "message": msg
                            })
                except Exception as inner_e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[WEBSOCKET ERROR] registration error: {inner_e}\n{tb}")
                    try:
                        await websocket.send_json({"type": "error", "message": "registration_error", "detail": str(inner_e)})
                    except Exception:
                        pass
    except WebSocketDisconnect:
        print("Client disconnected from Surveillance WebSocket")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error in surveillance websocket: {e}\n{tb}")

@app.get("/health")
def health_check():
    return {"status": "online", "message": "Exam Surveillance System Ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
