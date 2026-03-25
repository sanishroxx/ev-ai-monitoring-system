import asyncio
import websockets
import json
from datetime import datetime
from ml_model import save_reading, predict, train_model

async def handle_charger(websocket):
    print(f"Charger Connected!")
    
    readings_count = 0
    
    async for message in websocket:
        try:
            # Data receive
            data = json.loads(message)
            
            # AI se predict 
            ai_result = predict(data)
            
            if isinstance(ai_result, dict):
                data['ai_status'] = ai_result['status']
                data['ai_score'] = ai_result['score']
            else:
                data['ai_status'] = 'Unknown'
                data['ai_score'] = 0
            
            # Save 
            save_reading(data)
            readings_count += 1
            
            # Print status
            status_icon = "⚠️" if data['ai_status'] == 'ANOMALY' else "✅"
            print(f"{status_icon} {data['charger_id']} | "
                  f"V:{data['voltage']}V | "
                  f"T:{data['temperature']}°C | "
                  f"AI:{data['ai_status']}")
            
            # Har 20 readings pe model retrain 
            if readings_count % 20 == 0:
                print("Retraining AI model...")
                train_model()
                
        except Exception as e:
            print(f"Error: {e}")

async def main():
    print("OCPP Server Starting...")
    print("Waiting for chargers...")
    
    async with websockets.serve(
        handle_charger, 
        "localhost", 
        8765
    ):
        print("✅ Server running on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())