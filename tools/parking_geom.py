"""주차 인식 공통 기하 계산 (모든 인식 방법이 사용: P-A 규칙, P-S 세그멘테이션 등).

좌표계
- 월드(시뮬) 좌표: 미터, 원점 = 로봇 출발점, +x = 주차장 쪽, z = 위 (MuJoCo / ROS와 같음).
- 바닥 좌표: 판 위의 밀리미터, u = 왼쪽 끝에서, v = 위쪽 끝에서 (테이프 도면 기준).
- 이미지: 천장 웹캠의 왜곡된 640x480 픽셀 (방사 왜곡 k1).

마스크로 자세 구하기
- 바퀴 마스크(보이는 바퀴마다 하나)와 몸통 마스크, 모두 이미지 픽셀 단위.
- base_link(= 바퀴 축의 중점)의 위치와 방향을 바닥 평면에서 계산한다.
"""
import math

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
HFOV_DEG = 37.21
K1 = -0.2184
F = WIDTH / 2 / math.tan(math.radians(HFOV_DEG) / 2)       # 픽셀 단위 초점 거리
CX, CY = WIDTH / 2, HEIGHT / 2

START_U, START_V = 1125.0, 1090.0                           # 로봇 출발점의 바닥 좌표 mm (= 월드 원점)
TAPES = [
    (0, 0, 900, 19), (0, 0, 19, 600), (881, 0, 19, 600),
    (22.5, 0, 19, 220), (231.5, 0, 19, 220), (440.5, 0, 19, 220), (649.5, 0, 19, 220), (858.5, 0, 19, 220),
    (0, 581, 195, 19), (497, 581, 403, 19),
    (176, 581, 19, 419), (497, 581, 19, 99), (497, 661, 479, 19), (176, 981, 800, 19), (540.5, 680, 19, 301),
    (957, 100, 19, 580), (957, 100, 337, 19), (1275, 100, 19, 1100), (957, 981, 19, 219), (957, 1181, 337, 19),
]

# 핑키 형상 (시뮬 모델 기준): 바퀴는 base_link에서 y = ±40.6 mm, 축 높이 30 mm
WHEEL_Y = 0.0406
WHEEL_Z = 0.030          # 바퀴 픽셀을 평면으로 되돌릴 때 쓰는 높이
BODY_Z = 0.100           # 몸통 픽셀을 평면으로 되돌릴 때 쓰는 높이 (윗면)
WHEEL_Y_VISIBLE = 0.047  # 보이는 타이어 부분(바깥쪽)의 중심, val 분할에서 측정
# 로봇 좌표계(x 앞, y 왼쪽, z 위)의 단순한 핑키 3D 모델: 상자 (x0, x1, y0, y1, z0, z1)
ROBOT_BOXES = [
    (-0.076, 0.042, -0.042, 0.042, 0.005, 0.100),     # 몸통
    (-0.052, 0.018, -0.035, 0.035, 0.100, 0.138),     # 라이다
    (-0.028, 0.028, 0.026, 0.055, 0.002, 0.056),      # 왼쪽 바퀴
    (-0.028, 0.028, -0.055, -0.026, 0.002, 0.056),    # 오른쪽 바퀴
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
    """천장 웹캠: MuJoCo 카메라 자세(-z 방향을 봄, +y가 위) + 핀홀 모델 + 방사 왜곡 k1."""

    def __init__(self, pos, quat_wxyz):
        self.C = np.asarray(pos, float)
        self.R = quat_to_mat(*quat_wxyz)        # 카메라 -> 월드

    @classmethod
    def from_row(cls, row):
        return cls([row['cam_x'], row['cam_y'], row['cam_z']],
                   [row['cam_qw'], row['cam_qx'], row['cam_qy'], row['cam_qz']])

    def project(self, points):
        """월드 점 (N,3) -> 왜곡된 픽셀 (N,2)."""
        local = (np.asarray(points, float) - self.C) @ self.R
        xu = local[:, 0] / -local[:, 2]
        yu = -local[:, 1] / -local[:, 2]
        s = 1 + K1 * (xu * xu + yu * yu)
        return np.column_stack([CX + F * xu * s, CY + F * yu * s])

    def pixels_to_plane(self, pixels, z):
        """왜곡된 픽셀 (N,2) -> 높이 z인 수평면 위의 월드 점 (N,2)."""
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
    """바닥의 모든 테이프 사각형을 이미지에 투영한 다각형 (왜곡 포함)."""
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
    """자세 (x, y, yaw)의 로봇 3D 모델을 투영한 bool 이미지."""
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
    """이미지 마스크로 구한 로봇 자세 (x, y, yaw; 월드 미터 / 라디안). 구할 수 없으면 None.

    wheel_masks: bool 이미지 목록 (검출된 바퀴마다 하나), body_mask: bool 이미지.
    규칙 (모든 인식 방법 공통):
      - 바퀴 2개: base_link = 두 바퀴 중심의 중점, 방향은 바퀴 축에 수직.
        앞/뒤: 두 방향 각각의 로봇 3D 모델 투영을 검출된 로봇 픽셀과 비교해 더 잘 겹치는 쪽을 씀
        (카메라가 비스듬히 보는 시점과 시차를 처리).
      - 바퀴 1개: 왼쪽 바퀴인지 오른쪽 바퀴인지, 방향이 어느 쪽인지 모름. 5°마다 후보 방향을
        같은 방식으로 점수 매겨 가장 좋은 것을 씀.
      - 바퀴 없음: 정보 부족 -> None.
    반환값: (x, y, yaw, 사용한 바퀴 수).
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
        yaw0 = math.atan2(axle[0], -axle[1])          # 축에 수직인 두 방향 중 하나
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
