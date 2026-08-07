import os
import sys
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.simulate import RealtimeSimulator

app = FastAPI(title="AI Congestion Early Warning System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dir = os.path.join(project_root, "web")
os.makedirs(web_dir, exist_ok=True)

# Mount frontend files
app.mount("/dashboard", StaticFiles(directory=web_dir, html=True), name="web")

simulator = RealtimeSimulator()

async def event_generator():
    # Chạy vòng lặp sinh dữ liệu với tốc độ 1 giây/phút mô phỏng
    for data in simulator.generate_stream(interval_sec=0):
        # Do interval_sec=0 trong simulate.py để không block event loop
        # Ta dùng asyncio.sleep trong generator để đảm bảo bất đồng bộ
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(1.0) 

@app.get("/stream")
async def stream():
    """Endpoint Server-Sent Events cho Real-time Dashboard"""
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Khởi động AI Dashboard Server...")
    print(f"👉 Mở trình duyệt tại: http://localhost:8000/dashboard")
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
