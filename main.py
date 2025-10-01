#!/usr/bin/env python3
"""
hand_mouse_full.py
Full AirMouse prototype using Ultralytics YOLO pose model with robust postprocessing.

Features:
 - Index-finger cursor control
 - Pinch (thumb + index) => left click (with drag support)
 - Thumb + middle => right click (cooldown)
 - Index+middle pinch + vertical drag => scroll (clamped + debounced)
 - Smoothing: Kalman (recommended) or EMA (moving average)
 - Dead zone to prevent OS "shake-to-find-cursor" from jitter
 - Fallback to bbox center if keypoints missing (gestures disabled in fallback)

Requirements:
  pip install ultralytics opencv-python pyautogui numpy filterpy
  + PyTorch with CUDA if you want GPU inference (Ultralytics uses torch under the hood).
"""

import time
import argparse
from collections import deque

import cv2
import numpy as np
import pyautogui
import torch
from ultralytics import YOLO

# Kalman from filterpy (if not installed: pip install filterpy)
from filterpy.kalman import KalmanFilter

# -----------------------
# CLI / Config
# -----------------------
parser = argparse.ArgumentParser(description="Robust YOLO-pose AirMouse")
parser.add_argument("--model", type=str,
                    default="/home/migi/computervision/runs/pose/train2/weights/best.pt",
                    help="path to YOLO pose/keypoint model (or Ultralytics model name)")
parser.add_argument("--cam", type=int, default=0, help="webcam index")
parser.add_argument("--method", type=str, default="kalman", choices=["kalman", "ema"],
                    help="smoothing method")
parser.add_argument("--ema-frames", type=int, default=6,
                    help="if method=ema: number of frames to average")
parser.add_argument("--pinch-thresh", type=float, default=0.06,
                    help="normalized distance threshold for pinch gestures")
parser.add_argument("--scroll-gain", type=float, default=1200.0,
                    help="scale factor for scroll (tweak)")
parser.add_argument("--deadzone", type=int, default=6,
                    help="pixel deadzone: ignore moves smaller than this")
parser.add_argument("--bias-prev", type=float, default=0.75,
                    help="when index+middle are very close, bias index toward previous (0..1)")
parser.add_argument("--show-debug", action="store_true", help="show debug overlay")
args = parser.parse_args()

# -----------------------
# Device / model
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] torch.cuda.is_available(): {torch.cuda.is_available()}, device={device}")
print("[info] loading model:", args.model)
model = YOLO(args.model)   # ultralytics YOLO object

# -----------------------
# Keypoint mapping (21 points)
# (common layout: wrist, thumb(4), index(4), middle(4), ring(4), pinky(4))
# -----------------------
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
# (others available if you want)

# -----------------------
# Screen / pyautogui
# -----------------------
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = True  # keep OS failsafe ON (so shake-to-find still works)
print(f"[info] screen: {screen_w}x{screen_h}")

def norm_to_screen(nx, ny):
    """Normalized (0..1) → screen pixel coords, clipped."""
    x = int(np.clip(nx, 0.0, 1.0) * screen_w)
    y = int(np.clip(ny, 0.0, 1.0) * screen_h)
    return x, y

def euclidean(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.linalg.norm(a - b))

# -----------------------
# Smoothing: EMA (moving average) and Kalman
# -----------------------
# EMA buffer stores screen coords (x,y)
ema_buf = deque(maxlen=max(1, args.ema_frames))

def smooth_ema_px(x_px, y_px):
    """Simple moving average over last N screen-pixel coords."""
    ema_buf.append((float(x_px), float(y_px)))
    arr = np.array(ema_buf)
    mx, my = float(arr[:,0].mean()), float(arr[:,1].mean())
    return mx, my

# Kalman filter: state [x, y, vx, vy] in pixel units
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1,0,1,0],
                 [0,1,0,1],
                 [0,0,1,0],
                 [0,0,0,1]], dtype=float)
kf.H = np.array([[1,0,0,0],
                 [0,1,0,0]], dtype=float)
kf.P *= 1000.0
kf.R = np.eye(2) * 25.0   # measurement noise (tune)
kf.Q = np.eye(4) * 1.0    # process noise (tune)
kalman_initialized = False

def smooth_kalman_px(x_px, y_px):
    """Kalman smoothing; initialize on first call."""
    global kalman_initialized
    z = np.array([float(x_px), float(y_px)])
    if not kalman_initialized:
        kf.x = np.array([z[0], z[1], 0.0, 0.0])
        kalman_initialized = True
        return float(z[0]), float(z[1])
    kf.predict()
    kf.update(z)
    return float(kf.x[0]), float(kf.x[1])

# Choose smoothing function pointer
if args.method == "kalman":
    smooth_func = smooth_kalman_px
else:
    smooth_func = smooth_ema_px

# -----------------------
# Gesture / state variables
# -----------------------
prev_mouse = None            # last screen coords we actually moved to
prev_index_norm = None       # last normalized index location (for biasing)
mouse_down = False
last_right_click = 0.0
right_click_cooldown = 0.5
scroll_mode = False
scroll_anchor_norm = None
last_scroll_time = 0.0
min_scroll_interval = 0.03   # seconds between scroll events
fallback_mode = False        # True when we are using bbox center fallback (no gestures)

# -----------------------
# Video capture
# -----------------------
cap = cv2.VideoCapture(args.cam)
# try to set stable size (helps consistent normalization)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.2)

print("[info] starting loop; press ESC in the window or Ctrl+C in terminal to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[error] cannot read frame from camera")
            break

        # mirror for intuitive movement
        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]

        # run inference (Ultralytics: pass ndarray)
        results = model(frame)   # model will use device internally
        res = results[0]

        keypoints_norm = None
        fallback_mode = False

        # try to extract normalized keypoints consistently
        if hasattr(res, "keypoints") and res.keypoints is not None:
            try:
                xyn = res.keypoints.xyn  # often a tensor or numpy array
                if isinstance(xyn, torch.Tensor):
                    xyn = xyn.cpu().numpy()
                # xyn expected shape: (N_detections, K_keypoints, 2 or 3)
                if isinstance(xyn, np.ndarray):
                    if xyn.ndim == 3 and xyn.shape[0] > 0:
                        kp = xyn[0]   # first detection (K, 2/3)
                        if kp.shape[0] >= 21 and kp.shape[1] >= 2:
                            keypoints_norm = kp[:, :2].astype(float)
                    elif xyn.ndim == 2:
                        # maybe single detection already: (K,2)
                        if xyn.shape[0] >= 21 and xyn.shape[1] >= 2:
                            keypoints_norm = xyn[:, :2].astype(float)
            except Exception:
                keypoints_norm = None

        # fallback: use bbox center if no keypoints found (we'll disable gestures in that case)
        if keypoints_norm is None and hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            box = res.boxes.xyxy[0]
            box = box.cpu().numpy() if isinstance(box, torch.Tensor) else np.array(box)
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            keypoints_norm = np.zeros((21, 2), dtype=float)
            keypoints_norm[INDEX_TIP] = (cx, cy)  # only index tip approx
            fallback_mode = True

        # If we have keypoints (or fallback), process them
        if keypoints_norm is not None:
            # guard: some models may give out-of-range values
            def valid_norm(pt):
                x, y = float(pt[0]), float(pt[1])
                return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

            # read raw normalized coords
            raw_index = tuple(map(float, keypoints_norm[INDEX_TIP]))
            raw_thumb = tuple(map(float, keypoints_norm[THUMB_TIP]))
            raw_middle = tuple(map(float, keypoints_norm[MIDDLE_TIP]))

            # If index and middle are extremely close (ambiguous), bias index toward previous position
            idx_mid_dist = euclidean(raw_index, raw_middle)
            if prev_index_norm is not None and idx_mid_dist < args.pinch_thresh * 0.45:
                # bias: keep most weight on previous index to avoid jump onto middle
                bx = args.bias_prev * prev_index_norm[0] + (1.0 - args.bias_prev) * raw_index[0]
                by = args.bias_prev * prev_index_norm[1] + (1.0 - args.bias_prev) * raw_index[1]
                index_norm = (float(bx), float(by))
            else:
                index_norm = raw_index

            # save prev_index_norm for next frame
            prev_index_norm = index_norm

            # Map to screen pixels BEFORE smoothing (we smooth in pixel space)
            screen_x, screen_y = norm_to_screen(index_norm[0], index_norm[1])

            # apply selected smoothing (returns floats)
            sm_x_f, sm_y_f = smooth_func(screen_x, screen_y)

            # apply deadzone (in pixels)
            moved = True
            if prev_mouse is not None:
                dx = abs(sm_x_f - prev_mouse[0])
                dy = abs(sm_y_f - prev_mouse[1])
                if dx < args.deadzone and dy < args.deadzone:
                    moved = False

            # actually move cursor only if movement exceeds deadzone
            if moved:
                px, py = int(sm_x_f), int(sm_y_f)
                try:
                    pyautogui.moveTo(px, py, _pause=False)
                except Exception:
                    # in odd cases pyautogui may throw (e.g., failsafe); ignore to keep running
                    pass
                prev_mouse = (sm_x_f, sm_y_f)

            # ---------- GESTURE LOGIC (disabled in fallback_mode) ----------
            if not fallback_mode:
                thumb = raw_thumb
                index = index_norm       # biased index (normalized)
                middle = raw_middle

                # LEFT CLICK: pinch thumb+index (normalized dist)
                dist_thumb_index = euclidean(thumb, index)
                if dist_thumb_index < args.pinch_thresh:
                    if not mouse_down:
                        pyautogui.mouseDown(button="left", _pause=False)
                        mouse_down = True
                else:
                    if mouse_down:
                        pyautogui.mouseUp(button="left", _pause=False)
                        mouse_down = False

                # RIGHT CLICK: thumb+middle with cooldown (single click)
                dist_thumb_middle = euclidean(thumb, middle)
                if dist_thumb_middle < args.pinch_thresh * 0.9:
                    if time.time() - last_right_click > right_click_cooldown:
                        pyautogui.click(button="right", _pause=False)
                        last_right_click = time.time()

                # SCROLL: index+middle close and vertical drag
                idx_mid_dist = euclidean(index, middle)
                scroll_activation_thresh = args.pinch_thresh * 1.6
                if idx_mid_dist < scroll_activation_thresh:
                    # enter or continue scroll mode
                    if not scroll_mode:
                        scroll_mode = True
                        scroll_anchor_norm = index[1]
                        last_scroll_time = time.time()
                    else:
                        # normalized vertical delta (positive = move up on screen → scroll down typically)
                        dy_norm = scroll_anchor_norm - index[1]
                        scroll_amount = int(np.clip(dy_norm * args.scroll_gain, -200, 200))
                        now = time.time()
                        if scroll_amount != 0 and (now - last_scroll_time) >= min_scroll_interval:
                            try:
                                pyautogui.scroll(scroll_amount, _pause=False)
                            except Exception:
                                pass
                            last_scroll_time = now
                            scroll_anchor_norm = index[1]
                else:
                    scroll_mode = False
                    scroll_anchor_norm = None

            # ---------- debug overlay ----------
                if keypoints_norm is not None:
                    for i, (kx, ky) in enumerate(keypoints_norm):
                        if not (0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0):
                            continue
                        px = int(kx * img_w); py = int(ky * img_h)
                        color = (0, 255, 0) if i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP) else (150, 150, 150)
                        cv2.circle(frame, (px, py), 4, color, -1)

                    # overlay pinch / index-middle distance debug text
                    try:
                        txt = f"pinch={euclidean(raw_thumb, index):.3f} idx_mid={euclidean(index, raw_middle):.3f}"
                        cv2.putText(frame, txt, (8, 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    except Exception:
                        pass
        # Show output window
        cv2.imshow("Hand AirMouse (ESC to quit)", frame)
        k = cv2.waitKey(1)
        if k == 27:  # ESC
            break

except KeyboardInterrupt:
    print("[info] interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[info] clean shutdown")
