import asyncio
import websockets
import json
import logging

# 🌐 WebSocket server configuration
WEBSOCKET_URI = "ws://localhost:8000/alerts"  # Change this if your dashboard runs elsewhere

async def send_alert(message: str):
    """
    Sends a WebSocket alert message to the connected dashboard client.
    """
    try:
        async with websockets.connect(WEBSOCKET_URI) as websocket:
            payload = json.dumps({"alert": message})
            await websocket.send(payload)
            logging.info(f"📡 WebSocket alert sent: {message}")
    except Exception as e:
        logging.error(f"❌ WebSocket alert failed: {e}")

def send_websocket_alert(message="SMART_CCTV started successfully"):
    """
    Thread-safe wrapper to run WebSocket alert asynchronously.
    """
    try:
        asyncio.run(send_alert(message))
    except Exception as e:
        logging.error(f"❌ Failed to send WebSocket alert: {e}")
