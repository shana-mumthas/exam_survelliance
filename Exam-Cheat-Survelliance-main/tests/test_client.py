import asyncio
import websockets
import json
import base64
import cv2
import numpy as np

async def test_connection():
    uri = "ws://127.0.0.1:8000/ws/surveillance"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Create a black image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            print("Sending frame...")
            await websocket.send(img_b64)
            
            print("Waiting for response...")
            response = await websocket.recv()
            data = json.loads(response)
            
            if "alerts" in data and "frame" in data:
                print("Success: Received annotated frame and alerts.")
            else:
                print("Failure: Invalid response format.")
                
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
