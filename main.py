#!/usr/bin/env python3
"""
wrist_touchpad_mouse.py
Wrist-relative AirMouse with robust clicks and scroll.

Features:
 - Cursor moves relative to wrist movements (touchpad-style)
 - Left click: index+thumb pinch
 - Right click: middle+thumb pinch
 - Scroll: index+middle+thumb close, ring+pinkie down, vertical drag
 - EMA smoothing
 - Deadzone for jitter
 - Debug overlay highlighting fingertips
"""

import time
from collections import deque
import cv2
import numpy as np
import pyautogui
import mediapipe as mp

# -----------------------
# Config
# -----------------------
CAM_INDEX = 0
EMA_FRAMES = 6
DEADZONE = 0.015          # normalized delta
BASE_SENS = 3.0
ACCEL_POWER = 1.5
MAX_SENS = 50.0
PINCH_THRESH = 0.07
SCROLL_GAIN = 400.0
MIN_SCROLL_INTERVAL = 0.05
LEFT_CLICK_HOLD = 0.15
RIGHT_CLICK_HOLD = 0.15

# -----------------------
# Finger indices
# -----------------------
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# -----------------------
# Screen
# -----------------------
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = True

# -----------------------
# EMA smoothing
# -----------------------
ema_buf = deque(maxlen=EMA_FRAMES)
def smooth(x, y):
    ema_buf.append((x, y))
    arr = np.array(ema_buf)
    return float(arr[:,0].mean()), float(arr[:,1].mean())

# -----------------------
# Acceleration
# -----------------------
def apply_acceleration(dx, dy, base_sens=BASE_SENS, power=ACCEL_POWER, max_sens=MAX_SENS):
    mag = np.linalg.norm([dx, dy])
    if mag == 0: return 0.0, 0.0
    scale = base_sens * (mag ** power)
    scale = min(scale, max_sens)
    factor = scale / mag
    return dx * factor, dy * factor

# -----------------------
# Gesture state
# -----------------------
cursor_x, cursor_y = screen_w/2, screen_h/2
prev_wrist = None
scroll_mode = False
scroll_anchor = 0.0
prev_scroll_time = 0.0
mouse_down = False
last_left_click_time = 0.0
last_right_click_time = 0.0

# -----------------------
# MediaPipe setup
# -----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.2)

print("[info] starting loop; press ESC to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            keypoints = [(p.x, p.y) for p in lm]

            wrist = keypoints[WRIST]
            thumb = keypoints[THUMB_TIP]
            index = keypoints[INDEX_TIP]
            middle = keypoints[MIDDLE_TIP]
            ring = keypoints[RING_TIP]
            pinky = keypoints[PINKY_TIP]

            # --------- Cursor movement (wrist relative) ---------
            if prev_wrist is not None:
                dx, dy = wrist[0]-prev_wrist[0], wrist[1]-prev_wrist[1]
                if abs(dx) < DEADZONE: dx = 0.0
                if abs(dy) < DEADZONE: dy = 0.0
                dx_px, dy_px = apply_acceleration(dx, dy)
                cursor_x += dx_px * screen_w
                cursor_y += dy_px * screen_h
                cursor_x = np.clip(cursor_x, 0, screen_w-1)
                cursor_y = np.clip(cursor_y, 0, screen_h-1)
                sm_x, sm_y = smooth(cursor_x, cursor_y)
                pyautogui.moveTo(int(sm_x), int(sm_y), _pause=False)
            prev_wrist = wrist

            current_time = time.time()
            # --------- Left click ---------
            if np.linalg.norm(np.array(index)-np.array(thumb)) < PINCH_THRESH:
                if not mouse_down:
                    if current_time - last_left_click_time > LEFT_CLICK_HOLD:
                        pyautogui.mouseDown(button="left", _pause=False)
                        mouse_down = True
                        last_left_click_time = current_time
            else:
                if mouse_down and np.linalg.norm(np.array(index)-np.array(thumb)) > PINCH_THRESH*1.2:
                    pyautogui.mouseUp(button="left", _pause=False)
                    mouse_down = False

            # --------- Right click ---------
            if np.linalg.norm(np.array(middle)-np.array(thumb)) < PINCH_THRESH*0.9:
                if current_time - last_right_click_time > RIGHT_CLICK_HOLD:
                    pyautogui.click(button="right", _pause=False)
                    last_right_click_time = current_time

            # --------- Scroll: thumb+index+middle + ring+pinkie down ---------
            idx_mid_dist = np.linalg.norm(np.array(index)-np.array(middle))
            thumb_mid_dist = np.linalg.norm(np.array(thumb)-np.array(middle))
            fingers_down = ring[1] > wrist[1] and pinky[1] > wrist[1]
            if idx_mid_dist < PINCH_THRESH*1.0 and thumb_mid_dist < PINCH_THRESH*1.6 and fingers_down:
                if not scroll_mode:
                    scroll_mode = True
                    scroll_anchor = wrist[1]
                    prev_scroll_time = current_time
                else:
                    dy_norm = scroll_anchor - wrist[1]
                    scroll_amount = int(np.clip(dy_norm * SCROLL_GAIN, -200, 200))
                    if scroll_amount != 0 and (current_time - prev_scroll_time) >= MIN_SCROLL_INTERVAL:
                        pyautogui.scroll(scroll_amount, _pause=False)
                        prev_scroll_time = current_time
                        scroll_anchor = wrist[1]
            else:
                scroll_mode = False

            # --------- Debug overlay ---------
            for i, (nx, ny) in enumerate(keypoints):
                px, py = int(nx*img_w), int(ny*img_h)
                color = (0,255,0) if i in (WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP) else (150,150,150)
                cv2.circle(frame, (px, py), 6 if i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP) else 3, color, -1)

            cv2.putText(frame, f"Cursor: {int(cursor_x)},{int(cursor_y)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)

        cv2.imshow("Wrist Relative AirMouse", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[info] clean shutdown")
