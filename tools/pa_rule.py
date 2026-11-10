"""P-A: rule-based robot detection on the ceiling image (baseline, no learning).

wheels : blue pixels (HSV range), split into connected blobs, the two largest kept
body   : dark pixels that are not tape (known tape map projected with the calibrated camera),
         the dark blob closest to the wheels kept
The pose itself is computed by parking_geom.pose_from_masks (shared with the learned methods).
"""
import cv2
import numpy as np

import parking_geom as G

# thresholds (OpenCV HSV: H 0-180); tuned on the val split only
BLUE_LO, BLUE_HI = (95, 90, 50), (125, 255, 255)
DARK_V_MAX = 70
GRAY_V = (70, 215)       # grey parts on the robot top (screen board, rear bar): low saturation, mid value
GRAY_S_MAX = 45
TAPE_DILATE_PX = 4
MIN_BLOB_PX = 15


def tape_mask(camera):
    m = np.zeros((G.HEIGHT, G.WIDTH), np.uint8)
    for poly in G.tape_polygons(camera):
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return cv2.dilate(m, np.ones((2 * TAPE_DILATE_PX + 1,) * 2, np.uint8)) > 0


def detect(image_bgr, camera):
    """Returns (wheel_masks, body_mask) as boolean images."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(blue)
    order = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])
    wheels = [lab == i for i in order[:2] if st[i, cv2.CC_STAT_AREA] >= MIN_BLOB_PX]

    dark = (hsv[:, :, 2] <= DARK_V_MAX) & ~tape_mask(camera)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    body = None
    if n > 1:
        if wheels:
            ref = np.mean([np.argwhere(w).mean(0)[::-1] for w in wheels], axis=0)   # (x, y)
        else:
            ref = None
        best = None
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA]
            if area < 150 or area > 12000:          # too small, or a wall
                continue
            d = np.linalg.norm(cen[i] - ref) if ref is not None else -area
            if best is None or d < best[0]:
                best = (d, i)
        if best is not None:
            body = lab == best[1]
    if body is not None:
        # add grey robot parts (screen board, rear bar) touching the dark body, within its neighbourhood
        v, sat = hsv[:, :, 2], hsv[:, :, 1]
        grey = (v >= GRAY_V[0]) & (v <= GRAY_V[1]) & (sat <= GRAY_S_MAX)
        near = cv2.dilate(body.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        cand = (grey & near) | body
        n, lab = cv2.connectedComponents(cand.astype(np.uint8))
        ids = np.unique(lab[body])
        body = np.isin(lab, ids[ids > 0])
    return wheels, body
