#!/usr/bin/env python3
"""실물 천장 웹캠의 자세(위치 3 + 회전 3)를 바닥 테이프로 다시 보정한다.

방법 (챔퍼 정합)
- 카메라 영상에서 경계(Canny)를 찾고, 경계까지의 거리 지도(distance transform)를 만든다.
- 테이프 외곽선(parking_geom.TAPES, 바닥 위 2 mm)을 현재 추정 자세로 투영하고, 투영점이 놓인 곳의
  거리 값을 잔차로 둔다. 이 잔차가 가장 작아지도록 자세 6개 값을 최소제곱(soft-L1)으로 찾는다.
- 화각(HFOV)과 렌즈 왜곡(k1)은 이전 실물 보정값으로 고정한다 (렌즈는 그대로이므로).
- 시작값은 이전 보정값(시뮬 카메라와 같은 값). 카메라 가이드(camera_align_viewer.py)로 대략 맞춘 뒤 쓴다.
- 결과: config/ceiling_camera_calib.json (위치, 쿼터니언 wxyz, 잔차 px) + 확인용 겹친 이미지.

시뮬 카메라 값(gen_parking_dataset.CAM_POS / CAM_QUAT)은 바꾸지 않는다. 이 결과는 실물 평가에서
픽셀 -> 바닥 좌표 계산에만 쓴다.

사용법 (pinky_train 가상환경, scipy 필요):
  python tools/calib_from_tape.py --device /dev/video2             # 프레임 10장을 받아 중앙값 영상으로 보정
  python tools/calib_from_tape.py --image results/camera_align/raw_XXXX.png
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402

CAM_POS0 = [-0.2894, 0.5706, 1.815]                  # 이전 보정값 (= 시뮬 카메라)
CAM_QUAT0 = [0.69746, 0.15253, -0.16265, -0.68105]
DIST_CAP = 30.0                                      # 거리 지도 상한 px (먼 경계의 영향 제한)


def tape_world_points(step_mm=5.0):
    """모든 테이프 사각형 외곽선을 따라 step_mm 간격의 월드 점 (N, 3)."""
    pts = []
    for u, v, w, h in G.TAPES:
        for (u0, v0), (u1, v1) in (((u, v), (u + w, v)), ((u + w, v), (u + w, v + h)),
                                   ((u + w, v + h), (u, v + h)), ((u, v + h), (u, v))):
            n = max(1, int(math.ceil(math.hypot(u1 - u0, v1 - v0) / step_mm)))
            for t in np.arange(n) / n:
                xy = G.mm_to_world(u0 + (u1 - u0) * t, v0 + (v1 - v0) * t)
                pts.append([xy[0], xy[1], 0.002])
    return np.array(pts)


def mat_to_quat(R):
    """회전 행렬 -> 쿼터니언 (w, x, y, z), w >= 0."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def camera_from_params(p, cam0):
    """p = [dx, dy, dz (m), rx, ry, rz (rad, 카메라 좌표계의 작은 회전)] -> Camera."""
    cam = G.Camera(cam0.C + p[:3], [1, 0, 0, 0])
    cam.R = cam0.R @ cv2.Rodrigues(np.asarray(p[3:], float).reshape(3, 1))[0]
    return cam


def sample(dist, pix):
    """거리 지도에서 픽셀 위치 값 (쌍선형 보간). 화면 밖은 DIST_CAP."""
    x, y = pix[:, 0], pix[:, 1]
    out = np.full(len(pix), DIST_CAP)
    ok = (x >= 0) & (x < G.WIDTH - 1) & (y >= 0) & (y < G.HEIGHT - 1)
    x0, y0 = np.floor(x[ok]).astype(int), np.floor(y[ok]).astype(int)
    fx, fy = x[ok] - x0, y[ok] - y0
    out[ok] = (dist[y0, x0] * (1 - fx) * (1 - fy) + dist[y0, x0 + 1] * fx * (1 - fy)
               + dist[y0 + 1, x0] * (1 - fx) * fy + dist[y0 + 1, x0 + 1] * fx * fy)
    return out


def grab(device, n):
    dev = int(device) if str(device).isdigit() else device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, G.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, G.HEIGHT)
    frames = []
    for _ in range(n + 5):                           # 처음 몇 장은 자동 노출이 안정되기 전이라 버림
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, (G.WIDTH, G.HEIGHT)))
    cap.release()
    if len(frames) < n:
        raise SystemExit(f'cannot read frames from {device}')
    return np.median(np.stack(frames[-n:]), axis=0).astype(np.uint8)   # 중앙값 영상 (노이즈 감소)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--device', help='camera device (grab frames)')
    src.add_argument('--image', help='saved raw frame (640x480)')
    ap.add_argument('--frames', type=int, default=10, help='frames to median when using --device')
    ap.add_argument('--out', default='config/ceiling_camera_calib.json')
    ap.add_argument('--canny', type=int, nargs=2, default=[50, 150], help='Canny thresholds')
    args = ap.parse_args()

    img = grab(args.device, args.frames) if args.device else cv2.imread(args.image)
    if img is None:
        raise SystemExit(f'cannot read {args.image}')
    img = cv2.resize(img, (G.WIDTH, G.HEIGHT))
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), *args.canny)
    dist = np.minimum(cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5), DIST_CAP).astype(np.float64)

    cam0 = G.Camera(CAM_POS0, CAM_QUAT0)
    world = tape_world_points()
    pix0 = cam0.project(world)
    keep = (pix0[:, 0] > 10) & (pix0[:, 0] < G.WIDTH - 10) & (pix0[:, 1] > 10) & (pix0[:, 1] < G.HEIGHT - 10)
    world = world[keep]                              # 처음에 화면 안에 있는 점만 씀 (점 집합 고정)

    def residual(p):
        return sample(dist, camera_from_params(p, cam0).project(world))

    r0 = residual(np.zeros(6))
    p = np.zeros(6)
    for f_scale in (8.0, 3.0, 1.0):                  # 넓은 범위 -> 좁은 범위로 (지역 최소 회피)
        p = least_squares(residual, p, loss='soft_l1', f_scale=f_scale, x_scale=[0.01] * 3 + [0.005] * 3).x
    r1 = residual(p)
    cam = camera_from_params(p, cam0)
    q = mat_to_quat(cam.R)
    # 이전 보정값 대비 변화: 위치 (cm), 회전 각도 (°)
    dpos_cm = np.linalg.norm(p[:3]) * 100
    drot_deg = math.degrees(np.linalg.norm(p[3:]))
    print(f'points={len(world)}  residual px median {np.median(r0):.2f} -> {np.median(r1):.2f}  '
          f'p90 {np.percentile(r0, 90):.2f} -> {np.percentile(r1, 90):.2f}')
    print(f'change vs previous calib: position {dpos_cm:.2f} cm, rotation {drot_deg:.2f} deg '
          f'(training jitter range 3 cm / 3 deg)')
    print(f'pos = [{cam.C[0]:.4f}, {cam.C[1]:.4f}, {cam.C[2]:.4f}]  '
          f'quat_wxyz = [{q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f}]')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(pos=[round(float(v), 5) for v in cam.C], quat_wxyz=[round(float(v), 6) for v in q],
                   hfov_deg=G.HFOV_DEG, k1=G.K1, width=G.WIDTH, height=G.HEIGHT,
                   residual_px_median=round(float(np.median(r1)), 3),
                   residual_px_p90=round(float(np.percentile(r1, 90)), 3),
                   change_pos_cm=round(dpos_cm, 2), change_rot_deg=round(drot_deg, 2),
                   source=args.image or args.device, date=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(out, 'w'), indent=2)
    vis = img.copy()                                 # 확인용: 이전(빨강) / 새 보정(초록) 외곽선
    cv2.polylines(vis, [np.round(p_).astype(np.int32) for p_ in G.tape_polygons(cam0)], True, (0, 0, 255), 1)
    cv2.polylines(vis, [np.round(p_).astype(np.int32) for p_ in G.tape_polygons(cam)], True, (0, 255, 0), 1)
    cv2.imwrite(str(out.with_suffix('.png')), vis)
    print(f'saved {out} and {out.with_suffix(".png")} (red = previous, green = new)')


if __name__ == '__main__':
    main()
