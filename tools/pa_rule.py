"""P-A: 천장 영상에서 색 규칙으로 로봇을 검출한다(기준선, 학습 없음).

바퀴 : 파란색 픽셀(HSV 범위)을 연결된 덩어리로 나누고 가장 큰 두 개를 씀
몸통 : 테이프가 아닌 어두운 픽셀(알고 있는 테이프 지도를 보정된 카메라로 투영해 제외)
       중에서 바퀴에 가장 가까운 어두운 덩어리를 씀
자세 계산은 parking_geom.pose_from_masks가 한다(학습 방법들과 공통).
"""
import cv2
import numpy as np

import parking_geom as G

# 임계값 (OpenCV HSV: H 0-180), val 분할로만 튜닝함
BLUE_LO, BLUE_HI = (95, 90, 50), (125, 255, 255)
DARK_V_MAX = 70
GRAY_V = (70, 215)       # 로봇 윗면의 회색 부품(화면 보드, 뒤쪽 바): 채도 낮음, 밝기 중간
GRAY_S_MAX = 45
TAPE_DILATE_PX = 4
MIN_BLOB_PX = 15


def tape_mask(camera):
    m = np.zeros((G.HEIGHT, G.WIDTH), np.uint8)
    for poly in G.tape_polygons(camera):
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return cv2.dilate(m, np.ones((2 * TAPE_DILATE_PX + 1,) * 2, np.uint8)) > 0


def detect(image_bgr, camera):
    """(바퀴 마스크 목록, 몸통 마스크)를 bool 이미지로 반환."""
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
            if area < 150 or area > 12000:          # 너무 작거나 벽임
                continue
            d = np.linalg.norm(cen[i] - ref) if ref is not None else -area
            if best is None or d < best[0]:
                best = (d, i)
        if best is not None:
            body = lab == best[1]
    if body is not None:
        # 어두운 몸통에 맞닿은 회색 로봇 부품(화면 보드, 뒤쪽 바)을 주변 범위 안에서 몸통에 더함
        v, sat = hsv[:, :, 2], hsv[:, :, 1]
        grey = (v >= GRAY_V[0]) & (v <= GRAY_V[1]) & (sat <= GRAY_S_MAX)
        near = cv2.dilate(body.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        cand = (grey & near) | body
        n, lab = cv2.connectedComponents(cand.astype(np.uint8))
        ids = np.unique(lab[body])
        body = np.isin(lab, ids[ids > 0])
    return wheels, body
