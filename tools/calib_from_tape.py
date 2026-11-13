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
  python tools/calib_from_tape.py --dataset datasets/real_test     # 사진마다 다시 보정해 poses.csv 카메라 열 갱신
"""
import argparse
import csv
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


def calibrate(img, canny=(50, 150)):
    """영상 1장으로 카메라 자세 보정. 반환 (Camera, 시작 잔차, 최종 잔차, 변화 파라미터 p)."""
    img = cv2.resize(img, (G.WIDTH, G.HEIGHT))
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), *canny)
    dist = np.minimum(cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5), DIST_CAP).astype(np.float64)

    cam0 = G.Camera(CAM_POS0, CAM_QUAT0)
    world_all = tape_world_points()

    def visible(cam, margin):
        pix = cam.project(world_all)
        return world_all[(pix[:, 0] > margin) & (pix[:, 0] < G.WIDTH - margin)
                         & (pix[:, 1] > margin) & (pix[:, 1] < G.HEIGHT - margin)]

    # 1차: 시뮬 카메라 기준으로 화면 안의 점 / 2차: 1차 결과 카메라 기준으로 다시 고름
    # (카메라가 많이 움직이면 시뮬 기준 점 일부가 실제 화면 밖이라 잔차가 부풀고 맞춤이 흔들리기 때문)
    world = visible(cam0, 10)
    r0 = None
    p = np.zeros(6)
    for stage in range(2):
        def residual(q, world=world):
            return sample(dist, camera_from_params(q, cam0).project(world))
        if r0 is None:
            r0 = residual(np.zeros(6))
        for f_scale in (8.0, 3.0, 1.0):              # 넓은 범위 -> 좁은 범위로 (지역 최소 회피)
            p = least_squares(residual, p, loss='soft_l1', f_scale=f_scale, x_scale=[0.01] * 3 + [0.005] * 3).x
        world = visible(camera_from_params(p, cam0), 10)
    return camera_from_params(p, cam0), r0, sample(dist, camera_from_params(p, cam0).project(world)), p


def calibrate_dataset(data, canny):
    """데이터셋의 사진마다 테이프로 카메라 자세를 다시 구해 poses.csv 카메라 열을 바꾼다.
    실행할 때마다 지금의 poses.csv를 읽어 카메라 열만 바꾼다 (사진이 늘어난 뒤 다시 실행해도 됨).
    poses_calib_once.csv에는 바꾸기 전의 poses.csv를 남기고, 사진별 결과는 calib_per_image.csv에 기록한다.
    (촬영 중 카메라를 건드려도 평가가 맞게 되도록. 정답 자세는 테이프 기준 실측이라 그대로 둔다.)"""
    data = Path(data)
    src = data / 'poses.csv'
    backup = data / 'poses_calib_once.csv'
    backup.write_text(src.read_text())               # 바꾸기 전 상태 (사진 수가 늘어도 항상 최신 목록 기준)
    with open(src) as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    ref = G.Camera(CAM_POS0, CAM_QUAT0)
    log = []
    for r in rows:
        img = cv2.imread(str(data / 'images' / r['split'] / f"{r['name']}.jpg"))
        if img is None:
            raise SystemExit(f"cannot read image for {r['name']}")
        cam, r0, r1, p = calibrate(img, canny)
        q = mat_to_quat(cam.R)
        old = G.Camera([float(r[k]) for k in ('cam_x', 'cam_y', 'cam_z')],
                       [float(r[k]) for k in ('cam_qw', 'cam_qx', 'cam_qy', 'cam_qz')])
        # 바닥 기준 변화: 칸 중앙(A칸 u 136, v 120)을 새 카메라로 본 픽셀을 이전 카메라로 되돌린 차이 (mm)
        x, y = G.mm_to_world(136, 120)
        w = old.pixels_to_plane(cam.project([[x, y, 0.06]]), 0.06)[0]
        du, dv = np.subtract(G.world_to_mm(*w), (136, 120))
        r.update(cam_x=f'{cam.C[0]:.5f}', cam_y=f'{cam.C[1]:.5f}', cam_z=f'{cam.C[2]:.5f}',
                 cam_qw=f'{q[0]:.6f}', cam_qx=f'{q[1]:.6f}', cam_qy=f'{q[2]:.6f}', cam_qz=f'{q[3]:.6f}')
        log.append(dict(name=r['name'], residual_before_px=round(float(np.median(r0)), 3),
                        residual_after_px=round(float(np.median(r1)), 3),
                        shift_vs_once_du_mm=round(float(du), 1), shift_vs_once_dv_mm=round(float(dv), 1)))
        print(f"{r['name']:24s} residual px {np.median(r0):5.2f} -> {np.median(r1):4.2f}   "
              f"floor shift vs single calib (slot A) du={du:+5.1f} dv={dv:+5.1f} mm")
    with open(src, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    with open(data / 'calib_per_image.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(log[0].keys()))
        wr.writeheader()
        wr.writerows(log)
    worst = max(l['residual_after_px'] for l in log)
    print(f'updated {src} ({len(rows)} images, worst residual {worst:.2f} px); original kept in {backup.name}')


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--device', help='camera device (grab frames)')
    src.add_argument('--image', help='saved raw frame (640x480)')
    src.add_argument('--dataset', help='dataset folder: re-calibrate every image and rewrite camera columns of poses.csv')
    ap.add_argument('--frames', type=int, default=10, help='frames to median when using --device')
    ap.add_argument('--out', default='config/ceiling_camera_calib.json')
    ap.add_argument('--canny', type=int, nargs=2, default=[50, 150], help='Canny thresholds')
    args = ap.parse_args()

    if args.dataset:
        calibrate_dataset(os.path.expanduser(args.dataset), args.canny)
        return
    img = grab(args.device, args.frames) if args.device else cv2.imread(args.image)
    if img is None:
        raise SystemExit(f'cannot read {args.image}')
    img = cv2.resize(img, (G.WIDTH, G.HEIGHT))
    cam, r0, r1, p = calibrate(img, args.canny)
    cam0 = G.Camera(CAM_POS0, CAM_QUAT0)
    q = mat_to_quat(cam.R)
    # 이전 보정값 대비 변화: 위치 (cm), 회전 각도 (°)
    dpos_cm = np.linalg.norm(p[:3]) * 100
    drot_deg = math.degrees(np.linalg.norm(p[3:]))
    print(f'points={len(r1)}  residual px median {np.median(r0):.2f} -> {np.median(r1):.2f}  '
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
