#!/usr/bin/env python3
"""
test_mediapipe_gpu.py
Check if MediaPipe Tasks API runs HandLandmarker on GPU.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Base options: point to model + GPU delegate
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task",  # download this from mediapipe models
        delegate=mp.tasks.BaseOptions.Delegate.GPU  # <-- force GPU
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# Create landmarker
landmarker = vision.HandLandmarker.create_from_options(options)

# Simple webcam loop
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = landmarker.detect(mp_image)

    # draw landmarks if present
    if results.hand_landmarks:
        for lm in results.hand_landmarks[0]:
            h, w, _ = frame.shape
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    cv2.imshow("Hand Landmarker GPU Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
