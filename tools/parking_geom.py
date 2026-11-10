"""Shared geometry for parking recognition (used by every recognition method: P-A rule, P-S segmentation, ...).

Coordinates
- world (sim) frame: metres, origin = robot start, +x towards the parking lot, z up (same as MuJoCo / ROS).
- floor frame: millimetres on the boards, u from the left edge, v from the top edge (tape drawings).
- image: distorted 640x480 pixels of the ceiling webcam (radial distortion k1).

Pose from masks
- wheel masks (one per visible wheel) and a body mask, all in image pixels.
- base_link (= midpoint of the wheel axle) position and heading are computed on the floor plane.
"""
import math

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
HFOV_DEG = 37.21
K1 = -0.2184
F = WIDTH / 2 / math.tan(math.radians(HFOV_DEG) / 2)       # focal length in pixels
CX, CY = WIDTH / 2, HEIGHT / 2

START_U, START_V = 1125.0, 1090.0                           # robot start in floor mm (= world origin)
TAPES = [
    (0, 0, 900, 19), (0, 0, 19, 600), (881, 0, 19, 600),
    (22.5, 0, 19, 220), (231.5, 0, 19, 220), (440.5, 0, 19, 220), (649.5, 0, 19, 220), (858.5, 0, 19, 220),
    (0, 581, 195, 19), (497, 581, 403, 19),
    (176, 581, 19, 419), (497, 581, 19, 99), (497, 661, 479, 19), (176, 981, 800, 19), (540.5, 680, 19, 301),
    (957, 100, 19, 580), (957, 100, 337, 19), (1275, 100, 19, 1100), (957, 981, 19, 219), (957, 1181, 337, 19),
]

# Pinky geometry (from the sim model): wheels at y = +-40.6 mm from base_link, axle height 30 mm
WHEEL_Y = 0.0406
WHEEL_Z = 0.030          # height used to put wheel pixels back on a plane
BODY_Z = 0.100           # height used to put body pixels back on a plane (top surfaces)
WHEEL_Y_VISIBLE = 0.047  # centre of the visible tyre part (outer side), measured on the val split
# simple 3D model of Pinky in the robot frame (x forward, y left, z up): boxes (x0, x1, y0, y1, z0, z1)
ROBOT_BOXES = [
    (-0.076, 0.042, -0.042, 0.042, 0.005, 0.100),     # chassis
    (-0.052, 0.018, -0.035, 0.035, 0.100, 0.138),     # lidar
    (-0.028, 0.028, 0.026, 0.055, 0.002, 0.056),      # left wheel
    (-0.028, 0.028, -0.055, -0.026, 0.002, 0.056),    # right wheel
]


def mm_to_world(u, v):
    return np.array([(START_V - v) / 1000, (START_U - u) / 1000])


def world_to_mm(x, y):
    return START_U - y * 1000, START_V - x * 1000


def quat_to_mat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Camera:
    """Ceiling webcam: MuJoCo camera pose (looks along -z, +y up) + pinhole + radial k1."""

    def __init__(self, pos, quat_wxyz):
        self.C = np.asarray(pos, float)
        self.R = quat_to_mat(*quat_wxyz)        # camera -> world

    @classmethod
    def from_row(cls, row):
        return cls([row['cam_x'], row['cam_y'], row['cam_z']],
                   [row['cam_qw'], row['cam_qx'], row['cam_qy'], row['cam_qz']])

    def project(self, points):
        """World points (N,3) -> distorted pixels (N,2)."""
        local = (np.asarray(points, float) - self.C) @ self.R
        xu = local[:, 0] / -local[:, 2]
        yu = -local[:, 1] / -local[:, 2]
        s = 1 + K1 * (xu * xu + yu * yu)
        return np.column_stack([CX + F * xu * s, CY + F * yu * s])

    def pixels_to_plane(self, pixels, z):
        """Distorted pixels (N,2) -> world points (N,2) on the horizontal plane at height z."""
        p = np.asarray(pixels, float).reshape(-1, 2)
        xd, yd = (p[:, 0] - CX) / F, (p[:, 1] - CY) / F
        xu, yu = xd.copy(), yd.copy()
        for _ in range(15):
            s = 1 + K1 * (xu * xu + yu * yu)
            xu, yu = xd / s, yd / s
        rays = np.column_stack([xu, -yu, -np.ones_like(xu)]) @ self.R.T
        t = (z - self.C[2]) / rays[:, 2]
        return self.C[:2] + rays[:, :2] * t[:, None]

    def ground_xy(self):
        return self.C[:2]


def tape_polygons(camera, step=10.0):
    """Image polygons (distorted) of every tape rectangle on the floor."""
    polys = []
    for u, v, w, h in TAPES:
        pts = []
        for (u0, v0), (u1, v1) in (((u, v), (u + w, v)), ((u + w, v), (u + w, v + h)),
                                   ((u + w, v + h), (u, v + h)), ((u, v + h), (u, v))):
            n = max(1, int(math.ceil(math.hypot(u1 - u0, v1 - v0) / step)))
            for t in np.arange(n) / n:
                xy = mm_to_world(u0 + (u1 - u0) * t, v0 + (v1 - v0) * t)
                pts.append([xy[0], xy[1], 0.002])
        polys.append(camera.project(np.array(pts)))
    return polys


def rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def silhouette(camera, x, y, yaw):
    """Boolean image of the projected 3D robot model at pose (x, y, yaw)."""
    m = np.zeros((HEIGHT, WIDTH), np.uint8)
    R = rot(yaw)
    for x0, x1, y0, y1, z0, z1 in ROBOT_BOXES:
        corners = np.array([[a, b, c] for a in (x0, x1) for b in (y0, y1) for c in (z0, z1)])
        xy = corners[:, :2] @ R.T + [x, y]
        pix = camera.project(np.column_stack([xy, corners[:, 2]]))
        hull = cv2.convexHull(np.round(pix).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 1)
    return m > 0


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(1, np.logical_or(a, b).sum())


def pose_from_masks(camera, wheel_masks, body_mask, min_wheel_px=15, min_body_px=150):
    """Robot pose (x, y, yaw) in world metres / radians from image masks, or None.

    wheel_masks: list of boolean images (one per detected wheel); body_mask: boolean image.
    Rules (shared by all recognition methods):
      - two wheels: base_link = midpoint of the wheel centres; heading is perpendicular to the axle.
        Front/back: the projected 3D robot model (both directions) is compared with the detected
        robot pixels and the better overlap is kept (handles the camera's oblique view / parallax).
      - one wheel: the wheel can be the left or the right one and the heading is unknown; candidate
        headings every 5 deg are scored the same way and the best one is kept.
      - no wheel: not enough information -> None.
    Returns (x, y, yaw, n_wheels).
    """
    wheels = [m for m in wheel_masks if m.sum() >= min_wheel_px]
    if body_mask is None or body_mask.sum() < min_body_px or not wheels:
        return None
    robot = body_mask.copy()
    for m in wheels:
        robot |= m
    centres = []
    for m in wheels[:2]:
        ys, xs = np.nonzero(m)
        centres.append(camera.pixels_to_plane(np.column_stack([xs, ys]), WHEEL_Z).mean(0))
    if len(centres) == 2:
        W1, W2 = centres
        M = (W1 + W2) / 2
        axle = W2 - W1
        yaw0 = math.atan2(axle[0], -axle[1])          # one of the two perpendicular headings
        best = max(((iou(silhouette(camera, M[0], M[1], yaw), robot), yaw) for yaw in (yaw0, yaw0 + math.pi)))
        return M[0], M[1], wrap(best[1]), 2
    W = centres[0]
    best = None
    for side in (+1, -1):
        for yaw in np.radians(np.arange(0, 360, 5)):
            M = W - rot(yaw) @ np.array([0.0, side * WHEEL_Y_VISIBLE])
            score = iou(silhouette(camera, M[0], M[1], yaw), robot)
            if best is None or score > best[0]:
                best = (score, M, yaw)
    _, M, yaw = best
    return M[0], M[1], wrap(yaw), 1
