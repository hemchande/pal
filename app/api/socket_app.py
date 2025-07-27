import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
import aioredis
import os
import uuid
from data import process_fire_data_for_date

app = FastAPI()

# Redis config (adjust as needed)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_QUEUE = "fire_pipeline_jobs"

# WebSocket endpoint for streaming pipeline status/results
@app.websocket("/ws/pipeline")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Here, you could parse 'data' for parameters or commands
            # For demo, just echo back and simulate progress
            await websocket.send_text(json.dumps({"status": "received", "data": data}))
            # Simulate pipeline progress (replace with real pipeline calls)
            for i in range(1, 6):
                await asyncio.sleep(1)
                await websocket.send_text(json.dumps({"progress": i * 20, "message": f"Step {i}/5 complete"}))
            await websocket.send_text(json.dumps({"status": "done", "result": "Pipeline finished!"}))
    except WebSocketDisconnect:
        print("WebSocket disconnected")


@app.post("/pipeline/trigger")
async def trigger_pipeline(request: Request):
    data = await request.json()
    date_str = data.get("date")
    client_sid = data.get("sid")
    job_id = str(uuid.uuid4())

    if not date_str or not client_sid:
        return JSONResponse({"error": "Missing 'date' or 'sid'"}, status_code=400)

    job = queue.enqueue(process_fire_data_for_date, date_str, client_sid, job_id)
    return JSONResponse({"status": "queued", "job_id": job_id})


# POST endpoint to enqueue jobs to Redis for worker processing
@app.post("/pipeline/trigger")
async def trigger_pipeline(request: Request):
    data = await request.json()
    # Connect to Redis and enqueue the job
    redis = await aioredis.create_redis_pool(REDIS_URL)
    try:
        await redis.rpush(REDIS_QUEUE, json.dumps(data))
        # Optionally, you could return a job ID or status
        return JSONResponse({"status": "queued", "queue": REDIS_QUEUE})
    finally:
        redis.close()
        await redis.wait_closed()

# Note: A separate worker process should consume from the Redis queue and run full_fire_pipeline.py as needed.
# You can use RQ, Celery, or a custom async worker for this purpose. 