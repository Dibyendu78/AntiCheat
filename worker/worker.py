#!/usr/bin/env python3
"""EC2 Worker: SQS consumer that runs ML models on exam frames.

Polls SQS for S3 Event Notification messages, downloads frames from S3, 
runs the ModelEngine, and writes result JSONs to DynamoDB.

Usage:
    python worker/worker.py --threads 6
    python worker/worker.py --models-dir /opt/models
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote_plus
import decimal
import boto3
from botocore.exceptions import ClientError

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_engine import ModelEngine

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

BUCKET = os.environ.get("S3_BUCKET_NAME", "exam-proctoring-frames")
QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "")
REGION = os.environ.get("AWS_REGION", "ap-south-1")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", os.environ.get("DYNAMODB_TABLE_NAME", "proctoring_results-v2"))

MAX_MESSAGES = 10
POLL_WAIT_SECONDS = 10
HEALTH_PORT = 8080


class Worker:
    """Multi-threaded SQS consumer with ML model inference."""

    def __init__(self, engine: ModelEngine, num_threads: int = 6):
        self.engine = engine
        self.num_threads = num_threads
        self.running = True
        self.processed_count = 0
        self.error_count = 0
        self._lock = threading.Lock()

        self.s3 = boto3.client("s3", region_name=REGION)
        self.sqs = boto3.client("sqs", region_name=REGION)
        self.dynamodb = boto3.resource("dynamodb", region_name=REGION)
        self.table = self.dynamodb.Table(DYNAMODB_TABLE)

        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def _shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        print(f"\n[Worker] Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _process_message(self, message: dict) -> None:
        """Process a single SQS message."""
        try:
            # Handle possible SNS wrapping or raw S3 event
            body = json.loads(message["Body"])
            
            # Ignore S3 Test Events
            if body.get("Event") == "s3:TestEvent":
                print("[Worker] Received S3 Test Event, ignoring.")
                self.sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=message["ReceiptHandle"],
                )
                return

            # S3 Native Event format
            if "Records" in body:
                record = body["Records"][0]
                file_key = unquote_plus(record["s3"]["object"]["key"])
                message_bucket = record["s3"]["bucket"]["name"]
            else:
                # Fallback for custom formatted messages
                file_key = body.get("file_key")
                message_bucket = BUCKET
                
            if not file_key:
                # To prevent poisonous messages from looping forever, delete them or print warning
                print(f"[Worker] WARNING: Unknown message format: {body}")
                self.sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=message["ReceiptHandle"])
                return

            # Get metadata from S3 object
            try:
                head = self.s3.head_object(Bucket=message_bucket, Key=file_key)
                metadata = head.get("Metadata", {})
                session_id = metadata.get("session_id", "unknown_session")
                student_id = metadata.get("student_id", "unknown_student")
                exam_id = metadata.get("exam_id", "unknown_exam")

                # Download frame from S3
                response = self.s3.get_object(Bucket=message_bucket, Key=file_key)
                image_bytes = response["Body"].read()
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                # 403 or 404 mean the object is likely gone or we lack permissions.
                # Delete the message so it stops endlessly looping.
                if error_code in ["403", "404", "NoSuchKey", "AccessDenied"]:
                    print(f"[Worker] WARNING: S3 object inaccessible ({error_code}). Deleting poison message: {file_key}")
                    self.sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=message["ReceiptHandle"])
                    return
                else:
                    raise e

            # Run ML model
            result = self.engine.analyze_image(
                image_bytes=image_bytes,
                session_id=session_id,
                student_id=student_id,
                exam_id=exam_id,
            )

            # Write result to DynamoDB
            # Ensure float types are converted to Decimal for DynamoDB
            result_item = json.loads(json.dumps(result), parse_float=decimal.Decimal)
            
            self.table.put_item(Item=result_item)

            # Delete processed message from SQS
            self.sqs.delete_message(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=message["ReceiptHandle"],
            )

            with self._lock:
                self.processed_count += 1

            # Log result
            status = "[CHEATING]" if result["cheating"] else "[NORMAL]"
            thread_name = threading.current_thread().name
            print(
                f"[{thread_name}] {status} | "
                f"student={student_id} session={session_id} "
                f"type={result['type']} | "
                f"total_processed={self.processed_count}"
            )

        except Exception as e:
            with self._lock:
                self.error_count += 1
            print(f"[Worker] ERROR processing message: {e}")

    def _worker_loop(self) -> None:
        """Single worker thread: poll SQS and process messages."""
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] Started")

        while self.running:
            try:
                response = self.sqs.receive_message(
                    QueueUrl=QUEUE_URL,
                    MaxNumberOfMessages=MAX_MESSAGES,
                    WaitTimeSeconds=POLL_WAIT_SECONDS,
                    # visibility timeout can be configured on the queue level,
                    # but we can also set it here if needed. 
                )

                messages = response.get("Messages", [])

                if not messages:
                    continue

                for message in messages:
                    if not self.running:
                        break
                    self._process_message(message)

            except ClientError as e:
                print(f"[{thread_name}] AWS error: {e}")
                time.sleep(2)
            except Exception as e:
                print(f"[{thread_name}] Unexpected error: {e}")
                time.sleep(2)

        print(f"[{thread_name}] Stopped")

    def _health_check_server(self) -> None:
        """Simple HTTP health check on port 8080."""
        import http.server
        import socketserver

        class HealthHandler(http.server.BaseHTTPRequestHandler):
            worker_ref = self

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                status = {
                    "status": "healthy",
                    "running": self.worker_ref.running,
                    "processed": self.worker_ref.processed_count,
                    "errors": self.worker_ref.error_count,
                    "threads": self.worker_ref.num_threads,
                }
                self.wfile.write(json.dumps(status).encode())

            def log_message(self, format, *args):
                pass  # Suppress access logs

        try:
            with socketserver.TCPServer(("", HEALTH_PORT), HealthHandler) as httpd:
                print(f"[Worker] Health check running on port {HEALTH_PORT}")
                httpd.timeout = 1
                while self.running:
                    httpd.handle_request()
        except Exception as e:
            print(f"[Worker] Health check server error: {e}")

    def run(self) -> None:
        """Start worker threads and health check."""
        print(f"[Worker] Starting {self.num_threads} worker threads...")
        print(f"[Worker] Queue URL: {QUEUE_URL}")
        print(f"[Worker] S3 Bucket: {BUCKET}")
        print(f"[Worker] DynamoDB: {DYNAMODB_TABLE}")
        print(f"[Worker] Region: {REGION}")

        threads: list[threading.Thread] = []

        # Start health check
        health_thread = threading.Thread(
            target=self._health_check_server,
            name="health-check",
            daemon=True,
        )
        health_thread.start()

        # Start worker threads
        for i in range(self.num_threads):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"worker-{i}",
                daemon=True,
            )
            t.start()
            threads.append(t)

        # Wait for all threads
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False

        print("[Worker] Waiting for threads to finish...")
        for t in threads:
            t.join(timeout=10)

        print(f"[Worker] Shutdown complete. Processed: {self.processed_count}, Errors: {self.error_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Proctoring system EC2 worker")
    parser.add_argument("--threads", type=int, default=6, help="Number of worker threads (default: 6)")
    parser.add_argument("--models-dir", type=str, default=None, help="Directory containing model files")
    parser.add_argument("--queue-url", type=str, default=None, help="SQS queue URL (overrides env)")
    parser.add_argument("--bucket", type=str, default=None, help="S3 bucket name (overrides env)")
    args = parser.parse_args()

    global QUEUE_URL, BUCKET

    if args.queue_url:
        QUEUE_URL = args.queue_url
    if args.bucket:
        BUCKET = args.bucket

    if not QUEUE_URL:
        print("ERROR: SQS_QUEUE_URL not set. Use --queue-url or set SQS_QUEUE_URL env var.")
        sys.exit(1)
    # Determine model paths
    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        # Prefer explicit environment variable `MODELS_DIR` (set in Dockerfile/compose).
        env_models = os.environ.get("MODELS_DIR")
        if env_models:
            models_dir = Path(env_models)
        else:
            # Fallback to repository-level `models/` directory (project root)
            models_dir = Path(__file__).resolve().parent.parent / "models"

    # Normalize path (expand ~ and resolve symlinks). Keep going even if it doesn't exist;
    # ModelEngine will raise a clearer error when attempting to load missing files.
    try:
        models_dir = models_dir.expanduser().resolve()
    except Exception:
        models_dir = models_dir.resolve()

    if not models_dir.exists():
        print(f"WARNING: models directory {models_dir} does not exist.\n" \
              "Ensure --models-dir or MODELS_DIR env var points to the directory containing model files (best.pt, yolov8n.pt, face_landmarker.task).")

    object_model = str(models_dir / "best.pt")
    person_model = str(models_dir / "yolov8n.pt")
    face_landmarker = str(models_dir / "face_landmarker.task")

    # Initialize model engine (loads all models once)
    engine = ModelEngine(
        object_model_path=object_model,
        person_model_path=person_model,
        face_landmarker_path=face_landmarker,
    )

    # Start worker
    worker = Worker(engine=engine, num_threads=args.threads)
    worker.run()


if __name__ == "__main__":
    main()
