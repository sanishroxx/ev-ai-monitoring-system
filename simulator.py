import asyncio
import websockets
import json
import random
from datetime import datetime

async def simulate_charger(charger_id):
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as websocket:
        print(f"{charger_id} Connected!")
        
        while True:
            # Battery data generate 
            data = {
                "charger_id": charger_id,
                "timestamp": str(datetime.now()),
                "energy_kwh": round(random.uniform(10, 25), 2),
                "voltage": round(random.uniform(380, 420), 2),
                "current": round(random.uniform(28, 36), 2),
                "temperature": round(random.uniform(30, 45), 2),
                "soc_percent": round(random.uniform(20, 95), 2),
                "status": "Charging"
            }
            
            # Kabhi kabhi anomaly inject 
            if random.random() < 0.05:
                data["voltage"] = round(random.uniform(450, 500), 2)
                data["temperature"] = round(random.uniform(70, 90), 2)
                data["current"] = round(random.uniform(80, 100), 2)
                data["status"] = "ANOMALY"
            
            await websocket.send(json.dumps(data))
            print(f"Data sent: {data['charger_id']} | {data['voltage']}V | {data['temperature']}°C | {data['status']}")
            
            await asyncio.sleep(3)

async def main():
    chargers = ["OVN-EV-001", "OVN-EV-002", "OVN-EV-003"]
    tasks = [simulate_charger(cid) for cid in chargers]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    print("OCPP Simulator Starting...")
    print("3 Chargers: OVN-EV-001, OVN-EV-002, OVN-EV-003")
    asyncio.run(main())