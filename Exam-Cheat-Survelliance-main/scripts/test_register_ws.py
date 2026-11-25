import asyncio
import base64
import json
import websockets

async def test_register():
    uri = "ws://localhost:8000/ws/surveillance"
    async with websockets.connect(uri) as websocket:
        # Load a sample image
        with open("earbud_dataset/images/train/synthetic_0.jpg", "rb") as f:
            img_b = f.read()
        img_b64 = base64.b64encode(img_b).decode('utf-8')

        payload = {
            "type": "register",
            "name": "TestUser",
            "frame": img_b64
        }
        await websocket.send(json.dumps(payload))
        # Wait for response(s)
        while True:
            msg = await websocket.recv()
            print("RECV:", msg)
            break

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(test_register())
