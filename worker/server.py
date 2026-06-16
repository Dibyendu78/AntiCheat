#!/usr/bin/env python3
"""Backend API Server for Proctoring System.

Flow: 
1. React -> POST /upload -> S3
2. React -> GET /result -> DynamoDB
No local ML inference.
"""

from __future__ import annotations

import base64
import os
import time
import logging
import traceback
import uuid
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Key
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Configuration ──
PORT = int(os.environ.get("SERVER_PORT", "8000"))
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
BUCKET = os.environ.get("S3_BUCKET_NAME", "exam-proctoring-frames") # Use the correct bucket name or update in setup
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", os.environ.get("DYNAMODB_TABLE_NAME", "proctoring_results"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("proctoring-server")

# AWS Clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
dynamodb_resource = boto3.resource("dynamodb", region_name=AWS_REGION)

app = FastAPI(
    title="ExamGuard Proctoring API",
    description="Backend API for scalable async cheating detection",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──

class UploadRequest(BaseModel):
    image: str          # Base64-encoded JPEG
    session_id: str = "unknown_session"
    student_id: str = "unknown_student"
    exam_id: str = "unknown_exam"

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float

_start_time = time.time()


# ── Endpoints ──

@app.post("/upload")
async def upload_frame(request: UploadRequest):
    """Receive frame from React and upload to S3."""
    start = time.time()
    
    try:
        image_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # Path: frames/{session_id}/{timestamp_uuid}.jpg
    timestamp = int(time.time() * 1000)
    unique_id = str(uuid.uuid4())[:8]
    file_key = f"frames/{request.session_id}/{timestamp}_{unique_id}.jpg"

    try:
        s3_client.put_object(
            Bucket=BUCKET,
            Key=file_key,
            Body=image_bytes,
            ContentType="image/jpeg",
            Metadata={
                "session_id": request.session_id,
                "student_id": request.student_id,
                "exam_id": request.exam_id,
            }
        )
    except Exception as e:
        logger.error(f"S3 Upload failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"S3 Upload error: {str(e)}")

    elapsed = time.time() - start
    logger.info(
        f"Uploaded {file_key} | student={request.student_id} | {round(elapsed * 1000)}ms"
    )

    return {"status": "success", "file_key": file_key}


@app.get("/result")
async def get_result(session_id: str = Query(..., description="The session ID to fetch results for")):
    """Poll DynamoDB for the latest result for this session."""
    table = dynamodb_resource.Table(DYNAMODB_TABLE)
    
    try:
        response = table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=False,  # Sort descending (latest first)
            Limit=1
        )
        
        items = response.get("Items", [])
        if not items:
            return {
                "cheating": False,
                "status": "pending",
                "message": "No analysis result yet. Waiting for processing.",
                "details": {},
            }
            
        # Return latest result
        latest_result = items[0]
        
        # Convert Decimals to native types if needed, though FastAPI handles basic ones
        # Just passing it along is usually fine
        return latest_result

    except Exception as e:
        logger.error(f"DynamoDB Query failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"DynamoDB Query error: {str(e)}")


@app.get("/health")
async def health_check():
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/")
async def root():
    return {
        "service": "ExamGuard Backend API",
        "version": "3.0.0",
        "endpoints": {
            "POST /upload": "Upload frame to S3",
            "GET /result": "Fetch latest result from DynamoDB",
            "GET /health": "Health check",
        },
    }


# ── Entry Point ──
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Proctoring API Server")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
