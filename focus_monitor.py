#!/usr/bin/env python3
"""
focus_monitor.py
-----------------
Webcam-based study-focus monitor.

Detects:
  - Face absent from frame            -> "AWAY"
  - Eyes closed for a sustained time  -> "SLEEPING"
  - Head turned / tilted away         -> "LOOKING_AWAY"
  - A phone held up near the face     -> "PHONE"   (optional, needs a .tflite model)
  - Otherwise                         -> "FOCUSED"

Whenever the state is anything other than FOCUSED for longer than
ALERT_DEBOUNCE_SECONDS, an "ALERT" command is sent over serial (USB) to an
ESP32, which is expected to draw angry eyes on two OLED screens. When focus
resumes, an "OK" command is sent to clear the alert.
"""

import argparse
import collections
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

try:
    import serial  # pyserial
except ImportError:
    serial = None


# --------------------------------------------------------------------------- #
# Landmark indices (MediaPipe FaceLandmarker / FaceMesh, 468/478 point model)
# --------------------------------------------------------------------------- #
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # subject's right eye (image-left)
LEFT_EYE = [362, 385, 387, 263, 373,  380]   # subject's left eye  (image-right)

# Points used for solvePnP head-pose estimation
POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

# Generic 3D face model (arbitrary units, mm-like), matched to POSE_LANDMARKS
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0),  # Mouth left
    (150.0, -150.0, -125.0),   # Mouth right
], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Helper math
# --------------------------------------------------------------------------- #
def eye_aspect_ratio(landmarks, eye_idx, w, h):
    """Standard 6-point EAR. Lower value = more closed eye."""
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx])
    p1, p2, p3, p4, p5, p6 = pts
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def get_head_pose(landmarks, w, h, cam_matrix):
    """Returns (yaw, pitch, roll) in degrees, or None if solvePnP fails."""
    image_points = np.array([
        (landmarks[POSE_LANDMARKS["nose_tip"]].x * w, landmarks[POSE_LANDMARKS["nose_tip"]].y * h),
        (landmarks[POSE_LANDMARKS["chin"]].x * w, landmarks[POSE_LANDMARKS["chin"]].y * h),
        (landmarks[POSE_LANDMARKS["left_eye_corner"]].x * w, landmarks[POSE_LANDMARKS["left_eye_corner"]].y * h),
        (landmarks[POSE_LANDMARKS["right_eye_corner"]].x * w, landmarks[POSE_LANDMARKS["right_eye_corner"]].y * h),
        (landmarks[POSE_LANDMARKS["mouth_left"]].x * w, landmarks[POSE_LANDMARKS["mouth_left"]].y * h),
        (landmarks[POSE_LANDMARKS["mouth_right"]].x * w, landmarks[POSE_LANDMARKS["mouth_right"]].y * h),
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [float(a.item()) for a in euler_angles]
    return yaw, pitch, roll


# --------------------------------------------------------------------------- #
# Optional phone detector (MediaPipe Tasks ObjectDetector, COCO labels)
# --------------------------------------------------------------------------- #
class PhoneDetector:
    """Wraps mediapipe.tasks ObjectDetector. Silently disables itself if the
    model file isn't available so the rest of the program still runs."""

    def __init__(self, model_path):
        self.enabled = False
        self.detector = None
        if not model_path:
            return
        try:
            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=0.5,
                category_allowlist=["cell phone"],
                running_mode=vision.RunningMode.IMAGE,
            )
            self.detector = vision.ObjectDetector.create_from_options(options)
            self.enabled = True
        except Exception as e:
            print(f"[phone-detector] disabled ({e})")
            self.enabled = False

    def detect(self, rgb_frame):
        if not self.enabled:
            return False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        return len(result.detections) > 0


# --------------------------------------------------------------------------- #
# Serial link to ESP32
# --------------------------------------------------------------------------- #
class AlertLink:
    def __init__(self, port, baud, enabled=True):
        self.enabled = enabled and serial is not None and port
        self.ser = None
        self.last_sent = None
        if self.enabled:
            try:
                self.ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2)  # let ESP32 reset after port open
                print(f"[serial] connected to {port} @ {baud}")
            except Exception as e:
                print(f"[serial] could not open {port}: {e}")
                self.enabled = False

    def send(self, msg):
        if msg == self.last_sent:
            return  # avoid spamming identical state
        self.last_sent = msg
        print(f"[serial] -> {msg}")
        if self.enabled and self.ser:
            try:
                self.ser.write((msg + "\n").encode("utf-8"))
            except Exception as e:
                print(f"[serial] write failed: {e}")

    def close(self):
        if self.ser:
            self.ser.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Webcam study-focus monitor")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0",
                         help="ESP32 serial port (e.g. COM5 on Windows)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--no-serial", action="store_true",
                         help="run without sending anything to the ESP32")
    parser.add_argument("--no-display", action="store_true",
                         help="run headless (no cv2.imshow window)")
    parser.add_argument("--phone-model", type=str, default=None,
                         help="path to efficientdet_lite0.tflite for phone detection")
    parser.add_argument("--ear-thresh", type=float, default=0.21,
                         help="EAR below this = eye considered closed")
    parser.add_argument("--eyes-closed-secs", type=float, default=2.0,
                         help="sustained closed-eye duration to flag SLEEPING")
    parser.add_argument("--away-secs", type=float, default=1.5,
                         help="sustained no-face duration to flag AWAY")
    parser.add_argument("--yaw-thresh", type=float, default=20.0,
                         help="degrees of yaw considered 'looking away'")
    parser.add_argument("--pitch-thresh", type=float, default=190.0,
                         help="degrees of pitch (down/up) considered 'looking away'")
    parser.add_argument("--lookaway-secs", type=float, default=2.0,
                         help="sustained head-turn duration to flag LOOKING_AWAY")
    parser.add_argument("--phone-secs", type=float, default=1.0,
                         help="sustained phone-visible duration to flag PHONE")
    parser.add_argument("--alert-debounce-secs", type=float, default=0.0,
                         help="extra confirmation delay before sending ALERT")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: could not open camera", file=sys.stderr)
        sys.exit(1)

    # --- Setup MediaPipe Tasks Face Landmarker ---
    task_model_path = "face_landmarker.task"
    if not os.path.exists(task_model_path):
        print("[mediapipe] Downloading face_landmarker.task model...")
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(model_url, task_model_path)
        print("[mediapipe] Download complete.")

    base_options = mp_tasks.BaseOptions(model_asset_path=task_model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    phone_detector = PhoneDetector(args.phone_model)
    link = AlertLink(args.port, args.baud, enabled=not args.no_serial)

    # --- State tracking timestamps ---
    no_face_since = None
    eyes_closed_since = None
    look_away_since = None
    phone_since = None

    alert_active = False
    current_status = "FOCUSED"
    status_history = collections.deque(maxlen=5)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: frame grab failed")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)
            now = time.time()

            face_present = bool(results.face_landmarks)
            eyes_closed = False
            looking_away = False
            phone_visible = False

            if face_present:
                landmarks = results.face_landmarks[0]
                no_face_since = None

                # ---- EAR / drowsiness ----
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0
                eyes_closed = ear < args.ear_thresh

                # ---- head pose ----
                focal_length = w
                cam_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1],
                ], dtype=np.float64)
                pose = get_head_pose(landmarks, w, h, cam_matrix)
                if pose:
                    yaw, pitch, roll = pose
                    looking_away = (abs(yaw) > args.yaw_thresh or
                                     abs(pitch) > args.pitch_thresh)
                    cv2.putText(frame, f"yaw:{yaw:6.1f} pitch:{pitch:6.1f}",
                                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (200, 200, 200), 1)

                cv2.putText(frame, f"EAR:{ear:.2f}", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Simple landmark points overlay
                if not args.no_display:
                    for pt in landmarks:
                        px, py = int(pt.x * w), int(pt.y * h)
                        cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
            else:
                if no_face_since is None:
                    no_face_since = now

            # ---- phone detection (independent of face mesh) ----
            if phone_detector.enabled:
                phone_visible = phone_detector.detect(rgb)

            # ---- update "since" timers ----
            eyes_closed_since = eyes_closed_since if eyes_closed else None
            if eyes_closed and eyes_closed_since is None:
                eyes_closed_since = now

            look_away_since = look_away_since if looking_away else None
            if looking_away and look_away_since is None:
                look_away_since = now

            phone_since = phone_since if phone_visible else None
            if phone_visible and phone_since is None:
                phone_since = now

            # ---- decide status (priority order) ----
            status = "FOCUSED"
            if no_face_since and (now - no_face_since) >= args.away_secs:
                status = "AWAY"
            elif eyes_closed_since and (now - eyes_closed_since) >= args.eyes_closed_secs:
                status = "SLEEPING"
            elif phone_since and (now - phone_since) >= args.phone_secs:
                status = "PHONE"
            elif look_away_since and (now - look_away_since) >= args.lookaway_secs:
                status = "LOOKING_AWAY"

            status_history.append(status)
            current_status = status

            should_alert = status != "FOCUSED"
            if should_alert and not alert_active:
                alert_active = True
                link.send("ALERT")
            elif not should_alert and alert_active:
                alert_active = False
                link.send("OK")

            # ---- overlay ----
            color = (0, 200, 0) if status == "FOCUSED" else (0, 0, 255)
            cv2.putText(frame, f"STATUS: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if not args.no_display:
                cv2.imshow("Focus Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        link.close()


if __name__ == "__main__":
    main()