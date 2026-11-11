"""P-B: 천장 영상에서 배경 차분으로 로봇을 검출한다(학습 없음).

고정 카메라가 본 빈 주차장을 배경 이미지로 둔다. 배경과 다른 부분이 로봇이다.
1. 밝기 맞춤 : 채널마다 이미지와 배경 사이의 배율·오프셋을 맞춤(배경을 찍은 뒤 조명이 바뀌었으므로).
               잔차가 가장 큰 부분(로봇)을 빼고 한 번 더 맞춤
2. 전경      : 색 차이가 DIFF_TH보다 큰 픽셀에서 그림자(배경보다 어둡지만 색은 같음)를 뺌.
               가장 큰 덩어리 = 로봇
3. 바퀴      : 전경 안의 파란색 픽셀(P-A와 같은 색 규칙), 가장 큰 덩어리 2개
   몸통      : 전경에서 바퀴를 뺀 부분
자세 계산은 parking_geom.pose_from_masks가 한다(P-A / P-S와 공통).
"""
import cv2
import numpy as np

# 임계값은 val 분할로만 튜닝함
DIFF_TH = 40              # 밝기 맞춤 후 채널 차이의 최댓값
SHADOW_RATIO = (0.0, 0.0)     # 그림자로 볼 V_이미지 / V_배경 범위
SHADOW_CHROMA = 18        # 그림자로 볼 최대 색 변화(Lab의 a, b)
BLUE_LO, BLUE_HI = (95, 90, 50), (125, 255, 255)
MIN_ROBOT_PX = 300
MIN_BLOB_PX = 15
WHEEL_SEARCH_PX = 3


def match_brightness(img, bg):
    """채널마다 img ~ 배율 * bg + 오프셋을 맞추고, 이미지 조명에 맞춘 배경을 반환."""
    out = np.empty_like(bg)
    for c in range(3):
        x, y = bg[::4, ::4, c].ravel(), img[::4, ::4, c].ravel()      # 맞춤에만 쓰므로 간격을 두고 뽑아 빠르게
        keep = np.ones_like(x, bool)
        for _ in range(2):
            A = np.stack([x[keep], np.ones(keep.sum())], 1)
            g, o = np.linalg.lstsq(A, y[keep], rcond=None)[0]
            res = np.abs(g * x + o - y)
            keep = res < np.percentile(res, 90)
        out[:, :, c] = g * bg[:, :, c] + o
    return out


def detect(image_bgr, camera, background_bgr):
    """(바퀴 마스크 목록, 몸통 마스크)를 bool 이미지로 반환. camera는 쓰지 않음(P-A와 같은 함수 형태)."""
    img = image_bgr.astype(np.float32)
    bg = match_brightness(img, background_bgr.astype(np.float32))
    diff = np.abs(img - bg).max(axis=2)
    fg = diff > DIFF_TH

    # 그림자: 배경보다 어둡지만 색은 같음
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
    # 바퀴는 로봇 가장자리에 있으므로, 로봇 영역을 조금 넓혀서 찾음
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
