#!/usr/bin/env python3
"""인식 방법(천장 영상 → 로봇 자세)을 poses.csv의 정답과 비교해 평가한다.

사용법:
  python tools/eval_pose.py --method pa --split val
  python tools/eval_pose.py --method pa --split test --out results/pa_test.csv
  python tools/eval_pose.py --method pb --split val     (tools/gen_background.py --split val 실행 후)
  python tools/eval_pose.py --method ps --weights runs/segment/runs/ps/yolo11n_seed0/weights/best.pt --split val
  python tools/eval_pose.py --method pk --weights runs/pose/runs/pk/yolo11n_pk_seed0/weights/best.pt --split val

모든 방법은 바퀴 마스크 + 몸통 마스크를 내고, 자세는 공통 함수 parking_geom.pose_from_masks가
계산한다. 따라서 방법 간 차이는 마스크를 찾는 방식뿐이다. 예외: P-K(pk)는 키포인트에서 자세를 바로 구한다
(pk_kpt.pose_from_keypoints). used_wheels 열에는 사용한 키포인트 수가 들어간다.
이미지마다 poses.csv에 기록된 그 이미지의 (보정된) 카메라 자세로 평가한다.
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402


def load_rows(data, split):
    with open(data / 'poses.csv') as fh:
        rows = [r for r in csv.DictReader(fh) if r['split'] == split]
    for r in rows:
        for k in r:
            if k not in ('split', 'name', 'kind'):
                r[k] = float(r[k])
    return rows


def get_method(name, data, split, weights=None):
    """detect(image, camera, row) -> (바퀴 마스크 목록, 몸통 마스크) 함수를 반환."""
    if name == 'pa':
        import pa_rule
        return lambda img, cam, row: pa_rule.detect(img, cam)
    if name == 'ps':
        if not weights:
            raise SystemExit('--weights is required for ps')
        import ps_seg
        ps_seg.load(os.path.expanduser(weights))
        return lambda img, cam, row: ps_seg.detect(img, cam)
    if name == 'pb':
        import pb_bgsub
        bg_dir = data / 'background' / split
        if not bg_dir.is_dir():
            raise SystemExit(f'no backgrounds in {bg_dir}: run tools/gen_background.py --split {split}')
        # 배경은 시간 측정 전에 불러 둠(실제 노드도 배경 하나를 메모리에 들고 있음)
        bgs = {p.stem: cv2.imread(str(p)) for p in bg_dir.glob('*.png')}
        def detect(img, cam, row):
            return pb_bgsub.detect(img, cam, bgs[row['name']])
        return detect
    if name == 'pk':
        if not weights:
            raise SystemExit('--weights is required for pk')
        import pk_kpt
        pk_kpt.load(os.path.expanduser(weights))
        return lambda img, cam, row: pk_kpt.detect(img, cam)
    raise SystemExit(f'unknown method {name}')


def summary(res, label):
    ok = [r for r in res if r['found']]
    n = len(res)
    if not ok:
        return f'{label}: n={n} found=0'
    pos = np.array([r['pos_err_mm'] for r in ok])
    yaw = np.array([r['yaw_err_deg'] for r in ok])
    fb = np.mean([r['yaw_err_deg'] < 90 for r in ok])
    return (f'{label:10s} n={n:4d} found={len(ok) / n:6.1%}  pos mm mean={pos.mean():6.1f} '
            f'median={np.median(pos):6.1f} p95={np.percentile(pos, 95):6.1f}  yaw deg mean={yaw.mean():5.1f} '
            f'median={np.median(yaw):5.1f}  front/back ok={fb:6.1%}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='~/dev_ws/pinky_parking_sim/datasets/parking_seg_v2')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--method', default='pa')
    ap.add_argument('--weights', default=None, help='model weights for learned methods (ps, pk)')
    ap.add_argument('--out', default=None, help='per-image result csv (default results/<method>_<split>.csv)')
    ap.add_argument('--debug', type=int, default=10, help='overlay images to write next to the csv')
    args = ap.parse_args()

    data = Path(os.path.expanduser(args.data)).resolve()
    out = Path(args.out or f'results/{args.method}_{args.split}.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    dbg = out.with_suffix('')
    if args.debug:
        dbg.mkdir(exist_ok=True)
    detect = get_method(args.method, data, args.split, args.weights)
    rows = load_rows(data, args.split)
    res = []
    t_total = 0.0
    for i, r in enumerate(rows):
        img = cv2.imread(str(data / 'images' / args.split / f"{r['name']}.jpg"))
        cam = G.Camera.from_row(r)
        t0 = time.perf_counter()
        kp = None
        if args.method == 'pk':                    # P-K: 마스크 없이 키포인트에서 자세를 바로 구함
            pose, kp = detect(img, cam, r)
            wheels, body = [], None
        else:
            wheels, body = detect(img, cam, r)
            pose = G.pose_from_masks(cam, wheels, body)
        t_total += time.perf_counter() - t0
        item = {'name': r['name'], 'kind': r['kind'], 'found': pose is not None, 'n_wheels': len(wheels),
                'gt_x': r['x_m'], 'gt_y': r['y_m'], 'gt_yaw': r['yaw_rad']}
        if pose is not None:
            x, y, yaw, nw = pose
            item.update(est_x=x, est_y=y, est_yaw=yaw, used_wheels=nw,
                        pos_err_mm=1000 * math.hypot(x - r['x_m'], y - r['y_m']),
                        yaw_err_deg=abs(math.degrees(G.wrap(yaw - r['yaw_rad']))))
        res.append(item)
        if i < args.debug:
            vis = img.copy()
            for w in wheels:
                vis[w] = (0.5 * vis[w] + (127, 60, 0)).astype(np.uint8)
            if body is not None:
                vis[body] = (0.5 * vis[body] + (0, 110, 0)).astype(np.uint8)
            if kp is not None:                     # P-K 점: 앞끝 빨강, 뒤끝 파랑, 왼바퀴 초록, 오른바퀴 노랑
                for (px, py, c), col in zip(kp, [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 255)]):
                    cv2.circle(vis, (int(round(px)), int(round(py))), 4 if c >= 0.5 else 2, col, -1)
            def arrow(x, y, yaw, color):
                p = cam.project(np.array([[x, y, 0.03], [x + 0.08 * math.cos(yaw), y + 0.08 * math.sin(yaw), 0.03]]))
                cv2.arrowedLine(vis, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), color, 2,
                                tipLength=0.3)
            arrow(r['x_m'], r['y_m'], r['yaw_rad'], (0, 200, 0))          # 정답: 초록
            if pose is not None:
                arrow(pose[0], pose[1], pose[2], (0, 0, 255))              # 추정: 빨강
            cv2.imwrite(str(dbg / f"{r['name']}.jpg"), vis)
    keys = ['name', 'kind', 'found', 'n_wheels', 'used_wheels', 'gt_x', 'gt_y', 'gt_yaw', 'est_x', 'est_y',
            'est_yaw', 'pos_err_mm', 'yaw_err_deg']
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for item in res:
            w.writerow({k: item.get(k, '') for k in keys})
    print(f'method={args.method} split={args.split} images={len(res)} '
          f'time/img={1000 * t_total / max(1, len(res)):.1f} ms  -> {out}')
    print(summary(res, 'all'))
    for kind in sorted({r['kind'] for r in res}):
        print(summary([r for r in res if r['kind'] == kind], kind))
    for nw in (() if args.method == 'pk' else (0, 1, 2)):     # P-K는 바퀴 마스크가 없으므로 생략
        sub = [r for r in res if r['n_wheels'] == nw]
        if sub:
            print(summary(sub, f'{nw} wheels'))


if __name__ == '__main__':
    main()
