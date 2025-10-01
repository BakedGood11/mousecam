#!/usr/bin/env python3
"""
Final Polished Threaded Absolute-Mapping AirMouse with Sensitivity Module
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
from threading import Thread

# -----------------------
# Config / Controls
# -----------------------
CAM_INDEX = 0
WINDOW_NAME = "AirMouse"
EMA_FRAMES = 6

# Gesture thresholds
PINCH_THRESH = 0.08
MIN_SCROLL_INTERVAL = 0.05
RIGHT_CLICK_COOLDOWN = 0.5
CURSOR_EASE = 1.05  # Nonlinear easing

# Sensitivity multipliers
SENSITIVITY = {
    "cursor": 1.0,       # cursor movement multiplier
    "left_click": 1.0,   # left click pinch sensitivity
    "right_click": 1.0,  # right click pinch sensitivity
    "scroll": 1.0         # scroll speed multiplier
}

SCROLL_GAIN = 400.0  # base scroll speed

# Finger IDs
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
PINKY_TIP = 20

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = True

# -----------------------
# Smoothing
# -----------------------
ema_buf = deque(maxlen=EMA_FRAMES)
def smooth(x, y):
    ema_buf.append((x, y))
    arr = np.array(ema_buf)
    return float(arr[:,0].mean()), float(arr[:,1].mean())

# -----------------------
# Gesture State
# -----------------------
mouse_down = False
last_right_click_time = 0.0
scroll_mode = False
prev_scroll_time = 0.0
scroll_anchor = 0.0

# -----------------------
# Threaded Video Capture
# -----------------------
class VideoCaptureThread:
    def __init__(self, src=0, width=320, height=240):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = cv2.flip(frame, 1)

    def read(self):
        return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

cap_thread = VideoCaptureThread(CAM_INDEX)

# -----------------------
# MediaPipe Hands Setup
# -----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8,
)
mp_draw = mp.solutions.drawing_utils

print("[info] starting loop; press ESC to quit")

try:
    while True:
        ret, frame = cap_thread.read()
        if not ret:
            continue

        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            wrist = lm[WRIST]
            thumb = lm[THUMB_TIP]
            index = lm[INDEX_TIP]
            middle = lm[MIDDLE_TIP]
            ring = lm[RING_TIP]
            pinky = lm[PINKY_TIP]

            # Hand size scaling
            hand_size = np.linalg.norm(
                np.array([wrist.x, wrist.y]) - np.array([lm[MIDDLE_MCP].x, lm[MIDDLE_MCP].y])
            )

            # --------- Absolute Cursor Mapping ---------
            wrist_x = np.clip(wrist.x, 0.0, 1.0)
            wrist_y = np.clip(wrist.y, 0.0, 1.0)
            cursor_x = (wrist_x ** CURSOR_EASE) * screen_w * SENSITIVITY["cursor"]
            cursor_y = (wrist_y ** CURSOR_EASE) * screen_h * SENSITIVITY["cursor"]
            cursor_x, cursor_y = smooth(cursor_x, cursor_y)
            pyautogui.moveTo(int(cursor_x), int(cursor_y), _pause=False)

            # --------- Dynamic Pinch Thresholds ---------
            dynamic_thresh_left = np.clip(PINCH_THRESH * (hand_size / 0.2) / SENSITIVITY["left_click"], 0.04, 0.12)
            dynamic_thresh_right = np.clip(PINCH_THRESH * (hand_size / 0.2) / SENSITIVITY["right_click"], 0.04, 0.12)
            dynamic_thresh_scroll = dynamic_thresh_left * 1.6  # scroll threshold scaling

            # --------- Left Click (thumb + index) ---------
            if np.linalg.norm([index.x - thumb.x, index.y - thumb.y]) < dynamic_thresh_left:
                if not mouse_down:
                    pyautogui.mouseDown(button="left", _pause=False)
                    mouse_down = True
            else:
                if mouse_down:
                    pyautogui.mouseUp(button="left", _pause=False)
                    mouse_down = False

            # --------- Right Click (thumb + middle) ---------
            if np.linalg.norm([middle.x - thumb.x, middle.y - thumb.y]) < dynamic_thresh_right:
                if time.time() - last_right_click_time > RIGHT_CLICK_COOLDOWN:
                    pyautogui.click(button="right", _pause=False)
                    last_right_click_time = time.time()

            # --------- Scroll ---------
            fingers_down = (ring.y < wrist.y) and (pinky.y < wrist.y)
            idx_mid_dist = np.linalg.norm([index.x - middle.x, index.y - middle.y])
            thumb_mid = np.linalg.norm([thumb.x - middle.x, thumb.y - middle.y])
            if idx_mid_dist < dynamic_thresh_left and thumb_mid < dynamic_thresh_scroll and fingers_down:
                if not scroll_mode:
                    scroll_mode = True
                    scroll_anchor = wrist.y
                    prev_scroll_time = time.time()
                else:
                    dy_norm = scroll_anchor - wrist.y
                    scroll_amount = int(np.clip(dy_norm * SCROLL_GAIN * SENSITIVITY["scroll"] * (screen_h / img_h), -500, 500))
                    now = time.time()
                    if scroll_amount != 0 and (now - prev_scroll_time) >= MIN_SCROLL_INTERVAL:
                        pyautogui.scroll(scroll_amount, _pause=False)
                        prev_scroll_time = now
                        scroll_anchor = wrist.y
            else:
                scroll_mode = False

            # --------- Debug Visuals ---------
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(0,200,0), thickness=1))
            for tip in [WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP]:
                px, py = int(lm[tip].x*img_w), int(lm[tip].y*img_h)
                cv2.circle(frame, (px, py), 4, (0,255,255), -1)
            cv2.putText(frame, f"Cursor: {int(cursor_x)},{int(cursor_y)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap_thread.release()
    cv2.destroyAllWindows()
    print("[info] clean shutdown")
