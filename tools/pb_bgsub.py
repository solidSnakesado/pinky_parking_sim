"""P-B: background subtraction for robot detection on the ceiling image (no learning).

The fixed camera sees the empty lot as a background image. Anything that differs from it is the robot.
1. brightness match : per-channel gain/offset fitted from image to background (lighting changed since
                      the background was taken), refitted without the largest residuals (the robot)
2. foreground       : pixels whose colour differs more than DIFF_TH, minus shadows
                      (darker than background but same colour), largest blob = robot
3. wheels           : blue pixels inside the foreground (same colour rule as P-A), 2 largest blobs
   body             : foreground without the wheels
The pose itself is computed by parking_geom.pose_from_masks (shared with P-A / P-S).
"""
import cv2
import numpy as np

# thresholds tuned on the val split only
DIFF_TH = 40              # max channel difference after brightness match
SHADOW_RATIO = (0.0, 0.0)     # V_image / V_background for a shadow
SHADOW_CHROMA = 18        # max colour (a, b in Lab) change for a shadow
BLUE_LO, BLUE_HI = (95, 90, 50), (125, 255, 255)
MIN_ROBOT_PX = 300
MIN_BLOB_PX = 15
WHEEL_SEARCH_PX = 3


def match_brightness(img, bg):
    """Fit img ~ gain * bg + offset per channel; return bg mapped to the image's lighting."""
    out = np.empty_like(bg)
    for c in range(3):
        x, y = bg[::4, ::4, c].ravel(), img[::4, ::4, c].ravel()      # subsampled: fit only, faster
        keep = np.ones_like(x, bool)
        for _ in range(2):
            A = np.stack([x[keep], np.ones(keep.sum())], 1)
            g, o = np.linalg.lstsq(A, y[keep], rcond=None)[0]
            res = np.abs(g * x + o - y)
            keep = res < np.percentile(res, 90)
        out[:, :, c] = g * bg[:, :, c] + o
    return out


def detect(image_bgr, camera, background_bgr):
    """Returns (wheel_masks, body_mask) as boolean images. camera is unused (same signature as P-A)."""
    img = image_bgr.astype(np.float32)
    bg = match_brightness(img, background_bgr.astype(np.float32))
    diff = np.abs(img - bg).max(axis=2)
    fg = diff > DIFF_TH

    # shadows: darker than the background but the same colour
    lab_i = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    ratio = (lab_i[:, :, 0] + 1) / (lab_b[:, :, 0] + 1)
    chroma = np.hypot(lab_i[:, :, 1] - lab_b[:, :, 1], lab_i[:, :, 2] - lab_b[:, :, 2])
    shadow = (ratio > SHADOW_RATIO[0]) & (ratio < SHADOW_RATIO[1]) & (chroma < SHADOW_CHROMA)
    fg &= ~shadow

    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return [], None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < MIN_ROBOT_PX:
        return [], None
    robot = lab == i
    # wheels sit on the robot's edge, so they are searched in a slightly grown robot region
    near = cv2.dilate(robot.astype(np.uint8), np.ones((WHEEL_SEARCH_PX,) * 2, np.uint8)) > 0

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blue = (cv2.inRange(hsv, BLUE_LO, BLUE_HI) > 0) & near
    blue = cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(blue)
    order = sorted(range(1, n), key=lambda k: -st[k, cv2.CC_STAT_AREA])
    wheels = [lab == k for k in order[:2] if st[k, cv2.CC_STAT_AREA] >= MIN_BLOB_PX]
    body = robot.copy()
    for w in wheels:
        body &= ~w
    return wheels, body
