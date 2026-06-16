#!/usr/bin/env python3
"""Proctoring Model Engine — adapted from unified_system.py.

Uses the EXACT same detection logic as unified_system.py (which works perfectly),
adapted for headless server use (no GUI, no webcam, JPEG bytes input).

Detection pipeline:
    1. best.pt   → object detection (phone, book, etc.) — separate model
    2. yolov8n   → person count ONLY (classes=[0]) — separate model
    3. MediaPipe → gaze direction (eye tracking)
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from mediapipe.tasks.python import vision

logger = logging.getLogger("proctoring-server")

# ──────────────────────────────────────────────
# Constants — MATCHED to unified_system.py
# ──────────────────────────────────────────────
INFERENCE_WIDTH = 768

GAZE_DIRECTIONS = {"LEFT", "RIGHT", "UP", "DOWN"}

SUSPICIOUS_OBJECT_KEYWORDS = {
    "book", "cell phone", "phone", "mobile", "laptop",
    "tablet", "computer", "monitor", "keyboard", "mouse",
    "remote", "tv",
}

# Gaze thresholds — same as unified_system.py
H_GAZE_THRESHOLD = (0.40, 0.60)
V_GAZE_THRESHOLD = (0.40, 0.60)

# Alert thresholds
GAZE_AWAY_ALERT = 3     # consecutive gaze away → warning (user requested 3)
NO_FACE_ALERT = 4       # consecutive no-face frames before alert


def is_suspicious_object(name: str) -> bool:
    """Check if detected object name matches suspicious keywords.
    Exactly matches unified_system.py logic."""
    normalized = name.lower().replace("_", " ").replace("-", " ")
    return any(keyword in normalized for keyword in SUSPICIOUS_OBJECT_KEYWORDS)


def resize_for_inference(frame: np.ndarray, target_width: int) -> tuple:
    """Resize frame for YOLO inference — same as unified_system.py."""
    h, w = frame.shape[:2]
    if target_width <= 0 or w <= target_width:
        return frame, 1.0, 1.0
    target_height = int(h * (target_width / w))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, w / target_width, h / target_height
# new git push


# ──────────────────────────────────────────────
# Object Detector — EXACT copy from unified_system.py
# ──────────────────────────────────────────────

class ObjectDetector:
    """Runs the custom YOLO object detector from best.pt.
    Same as unified_system.py ObjectDetector."""

    def __init__(self, model_path: str, conf: float = 0.15):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Object model not found: {path}")
        self.model = YOLO(str(path))
        self.conf = conf
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection on frame, return list of {name, confidence, suspicious}."""
        inference_frame, scale_x, scale_y = resize_for_inference(frame, INFERENCE_WIDTH)
        results = self.model(
            inference_frame,
            conf=self.conf,
            iou=0.45,
            imgsz=INFERENCE_WIDTH,
            verbose=False,
        )
        detections = []

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = str(self.names.get(class_id, class_id))
            detections.append({
                "name": name,
                "confidence": round(confidence, 3),
                "suspicious": is_suspicious_object(name),
            })

        return detections


# ──────────────────────────────────────────────
# Face/Gaze Tracker — EXACT copy from unified_system.py
# ──────────────────────────────────────────────

class FaceGazeTracker:
    """Counts people and checks gaze direction.
    Same as unified_system.py FaceGazeTracker."""

    def __init__(self, person_model_path: str, face_landmarker_path: str):
        path = Path(person_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Person model not found: {path}")

        lm_path = Path(face_landmarker_path)
        if not lm_path.exists():
            raise FileNotFoundError(f"Face landmarker not found: {lm_path}")

        self.person_model = YOLO(str(path))

        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(lm_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _horizontal_ratio(landmarks, left_idx, right_idx, iris_idx, w, h):
        left = np.array([landmarks[left_idx].x * w, landmarks[left_idx].y * h])
        right = np.array([landmarks[right_idx].x * w, landmarks[right_idx].y * h])
        iris = np.array([landmarks[iris_idx].x * w, landmarks[iris_idx].y * h])
        eye_width = np.linalg.norm(right - left)
        if eye_width == 0:
            return 0.5
        return float(np.linalg.norm(iris - left) / eye_width)

    @staticmethod
    def _vertical_ratio(landmarks, top_idx, bot_idx, iris_idx, w, h):
        top = np.array([landmarks[top_idx].x * w, landmarks[top_idx].y * h])
        bot = np.array([landmarks[bot_idx].x * w, landmarks[bot_idx].y * h])
        iris = np.array([landmarks[iris_idx].x * w, landmarks[iris_idx].y * h])
        eye_height = np.linalg.norm(bot - top)
        if eye_height == 0:
            return 0.5
        return float(np.linalg.norm(iris - top) / eye_height)

    def _gaze_direction(self, frame: np.ndarray) -> str:
        """Determine gaze direction — same thresholds as unified_system.py."""
        inference_frame, _, _ = resize_for_inference(frame, INFERENCE_WIDTH)
        h, w = inference_frame.shape[:2]
        rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return "NO_FACE"

        lm = results.face_landmarks[0]
        left_h = self._horizontal_ratio(lm, 33, 133, 468, w, h)
        right_h = self._horizontal_ratio(lm, 362, 263, 473, w, h)
        left_v = self._vertical_ratio(lm, 159, 145, 468, w, h)
        right_v = self._vertical_ratio(lm, 386, 374, 473, w, h)

        avg_h = (left_h + right_h) / 2
        avg_v = (left_v + right_v) / 2

        if avg_v < V_GAZE_THRESHOLD[0]:
            return "UP"
        if avg_v > V_GAZE_THRESHOLD[1]:
            return "DOWN"
        if avg_h < H_GAZE_THRESHOLD[0]:
            return "RIGHT"
        if avg_h > H_GAZE_THRESHOLD[1]:
            return "LEFT"
        return "CENTER"

    def analyze(self, frame: np.ndarray) -> dict:
        """Count persons and determine gaze — same as unified_system.py.

        KEY: person detection uses classes=[0] (ONLY persons),
        conf=0.25, imgsz=640 — exactly like the working code.
        """
        inference_frame, _, _ = resize_for_inference(frame, INFERENCE_WIDTH)
        person_results = self.person_model(
            inference_frame,
            classes=[0],       # ← ONLY detect persons, not all classes
            conf=0.45,        # ← same as unified_system.py
            imgsz=INFERENCE_WIDTH,
            verbose=False,
        )
        boxes = person_results[0].boxes
        face_count = 0 if boxes is None else len(boxes)

        gaze_direction = self._gaze_direction(frame)

        return {
            "face_count": face_count,
            "gaze_direction": gaze_direction,
        }


# ──────────────────────────────────────────────
# State Tracker (simple in-memory, same logic as unified_system.py)
# ──────────────────────────────────────────────

class StateTracker:
    """Tracks gaze away and no-face counts per session.
    Same consecutive-counter logic as unified_system.py."""

    def __init__(self):
        self._gaze_away_count: dict[str, int] = defaultdict(int)
        self._no_face_count: dict[str, int] = defaultdict(int)

    def track_gaze(self, session_id: str, direction: str) -> int:
        """Same as unified_system.py: increment when gaze is away,
        reset to 0 when looking at CENTER."""
        if direction in GAZE_DIRECTIONS:
            self._gaze_away_count[session_id] += 1
        else:
            self._gaze_away_count[session_id] = 0
        return self._gaze_away_count[session_id]

    def track_face(self, session_id: str, face_count: int) -> int:
        """Track consecutive no-face frames."""
        if face_count == 0:
            self._no_face_count[session_id] += 1
        else:
            self._no_face_count[session_id] = 0
        return self._no_face_count[session_id]


# ──────────────────────────────────────────────
# Model Engine (Main Entry Point)
# ──────────────────────────────────────────────

class ModelEngine:
    """Headless proctoring engine — uses exact same detection logic
    as unified_system.py but accepts JPEG bytes instead of webcam frames.

    Models:
        - best.pt              → object detection (phone, book, etc.)
        - yolov8n.pt           → person count ONLY (classes=[0])
        - face_landmarker.task → gaze tracking (MediaPipe)
    """

    def __init__(
        self,
        object_model_path: str,
        person_model_path: str,
        face_landmarker_path: str,
    ):
        logger.info("Loading models...")
        start = time.time()

        self.object_detector = ObjectDetector(object_model_path)
        self.face_gaze_tracker = FaceGazeTracker(person_model_path, face_landmarker_path)
        self.state = StateTracker()

        elapsed = time.time() - start
        logger.info(f"Models loaded in {elapsed:.1f}s")

    def analyze_image(
        self,
        image_bytes: bytes,
        session_id: str = "",
        student_id: str = "",
        exam_id: str = "",
    ) -> dict:
        """Analyze a JPEG frame → return cheating detection result.
        Uses the same flow as unified_system.py analyze_frame()."""

        # Decode image here 
        frame = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            return self._result(
                cheating=False, ctype="error", message="Could not decode image",
                session_id=session_id, student_id=student_id, exam_id=exam_id,
            )

        # ── 1. Object detection (best.pt) — same as unified_system.py ──
        object_detections = self.object_detector.detect(frame)
        suspicious = [d for d in object_detections if d["suspicious"]]

        # ── 2. Person count + gaze (yolov8n + MediaPipe) ──
        face_gaze = self.face_gaze_tracker.analyze(frame)
        face_count = face_gaze["face_count"]
        gaze_direction = face_gaze["gaze_direction"]

        # ── 3. State tracking (same as unified_system.py) ──
        gaze_away_count = self.state.track_gaze(session_id, gaze_direction)
        no_face_count = self.state.track_face(session_id, face_count)

        # ── 4. Build alerts (same logic as unified_system.py) ──
        alerts = []

        # Multiple persons
        if face_count > 1:
            alerts.append({
                "type": "multiple_persons",
                "message": f"Multiple persons detected ({face_count})",
                "severity": "critical",
            })

        # Suspicious objects (from best.pt)
        for obj in suspicious:
            alerts.append({
                "type": "suspicious_object",
                "message": f"{obj['name']} detected ({obj['confidence']:.2f})",
                "severity": "critical",
            })

        # Gaze away (consecutive counter — same as unified_system.py)
        if gaze_away_count > GAZE_AWAY_ALERT:
            alerts.append({
                "type": "gaze_away",
                "message": f"Eyes looking {gaze_direction} ({gaze_away_count}× consecutive)",
                "severity": "warning",
            })

        # No face (consecutive)
        if no_face_count >= NO_FACE_ALERT:
            alerts.append({
                "type": "no_face",
                "message": f"No person visible for {no_face_count} frames",
                "severity": "warning",
            })

        # ── 5. Build result ──
        cheating = len(alerts) > 0
        primary = alerts[0] if alerts else None

        return self._result(
            cheating=cheating,
            ctype=primary["type"] if primary else "none",
            message=primary["message"] if primary else "Normal — no suspicious activity",
            session_id=session_id,
            student_id=student_id,
            exam_id=exam_id,
            details={
                "face_count": face_count,
                "gaze_direction": gaze_direction,
                "gaze_away_count": gaze_away_count,
                "objects_detected": [
                    {"name": d["name"], "confidence": d["confidence"]}
                    for d in object_detections
                ],
                "suspicious_objects": [
                    {"name": d["name"], "confidence": d["confidence"]}
                    for d in suspicious
                ],
            },
            alerts=alerts,
        )

    @staticmethod
    def _result(
        cheating: bool, ctype: str, message: str,
        session_id: str = "", student_id: str = "", exam_id: str = "",
        details: Optional[dict] = None, alerts: Optional[list] = None,
    ) -> dict:
        return {
            "cheating": cheating,
            "type": ctype,
            "message": message,
            "session_id": session_id,
            "student_id": student_id,
            "exam_id": exam_id,
            "timestamp": int(time.time()),
            "details": details or {},
            "alerts": alerts or [],
        }
# /this is a new line for git push
print("Model engine loaded and ready to analyze images.")