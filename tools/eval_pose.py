#!/usr/bin/env python3
"""Evaluate a recognition method (robot pose from the ceiling image) against poses.csv ground truth.

usage:
  python tools/eval_pose.py --method pa --split val
  python tools/eval_pose.py --method pa --split test --out results/pa_test.csv

Every method returns wheel masks + a body mask; the pose is computed by the shared
parking_geom.pose_from_masks, so methods differ only in how the masks are found.
Each image is evaluated with its own (calibrated) camera pose from poses.csv.
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


def get_method(name):
    if name == 'pa':
        import pa_rule
        return pa_rule.detect
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
    ap.add_argument('--out', default=None, help='per-image result csv (default results/<method>_<split>.csv)')
    ap.add_argument('--debug', type=int, default=10, help='overlay images to write next to the csv')
    args = ap.parse_args()

    data = Path(os.path.expanduser(args.data)).resolve()
    out = Path(args.out or f'results/{args.method}_{args.split}.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    dbg = out.with_suffix('')
    if args.debug:
        dbg.mkdir(exist_ok=True)
    detect = get_method(args.method)
    rows = load_rows(data, args.split)
    res = []
    t_total = 0.0
    for i, r in enumerate(rows):
        img = cv2.imread(str(data / 'images' / args.split / f"{r['name']}.jpg"))
        cam = G.Camera.from_row(r)
        t0 = time.perf_counter()
        wheels, body = detect(img, cam)
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
            def arrow(x, y, yaw, color):
                p = cam.project(np.array([[x, y, 0.03], [x + 0.08 * math.cos(yaw), y + 0.08 * math.sin(yaw), 0.03]]))
                cv2.arrowedLine(vis, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), color, 2,
                                tipLength=0.3)
            arrow(r['x_m'], r['y_m'], r['yaw_rad'], (0, 200, 0))          # ground truth: green
            if pose is not None:
                arrow(pose[0], pose[1], pose[2], (0, 0, 255))              # estimate: red
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
    for nw in (0, 1, 2):
        sub = [r for r in res if r['n_wheels'] == nw]
        if sub:
            print(summary(sub, f'{nw} wheels'))


if __name__ == '__main__':
    main()
