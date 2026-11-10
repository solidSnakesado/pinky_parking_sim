#!/usr/bin/env python3
"""Tune P-B (background subtraction) thresholds on the val split only (test is never used here).

For every parameter combination: run P-B on all val images and print
position / heading median, front/back accuracy, and body-mask IoU against the ground-truth
silhouette (how well the robot region was found). Pick the best row, put its values into
tools/pb_bgsub.py, then evaluate test once with eval_pose.py.

usage (pinky_venv):  python tools/tune_pb.py
"""
import argparse
import csv
import itertools
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402
import pb_bgsub as P  # noqa: E402

GRID = {
    'DIFF_TH': [25, 40, 60],
    'SHADOW_RATIO': [(0.45, 0.92), (0.3, 0.95), (0.0, 0.0)],     # (0, 0) = no shadow removal
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='~/dev_ws/pinky_parking_sim/datasets/parking_seg_v2')
    args = ap.parse_args()
    data = Path(os.path.expanduser(args.data))
    with open(data / 'poses.csv') as fh:
        rows = [r for r in csv.DictReader(fh) if r['split'] == 'val']
    for r in rows:
        for k in r:
            if k not in ('split', 'name', 'kind'):
                r[k] = float(r[k])
    imgs = {r['name']: (cv2.imread(str(data / 'images' / 'val' / f"{r['name']}.jpg")),
                        cv2.imread(str(data / 'background' / 'val' / f"{r['name']}.png"))) for r in rows}
    cams = {r['name']: G.Camera.from_row(r) for r in rows}
    sils = {r['name']: G.silhouette(cams[r['name']], r['x_m'], r['y_m'], r['yaw_rad']) for r in rows}

    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        for k, v in zip(keys, values):
            setattr(P, k, v)
        pos, yaw, ious, w1 = [], [], [], []
        for r in rows:
            img, bg = imgs[r['name']]
            wheels, body = P.detect(img, cams[r['name']], bg)
            if body is not None:
                ious.append(G.iou(body, sils[r['name']]))
            pose = G.pose_from_masks(cams[r['name']], wheels, body)
            if pose is None:
                continue
            e = 1000 * math.hypot(pose[0] - r['x_m'], pose[1] - r['y_m'])
            pos.append(e)
            yaw.append(abs(math.degrees(G.wrap(pose[2] - r['yaw_rad']))))
            if len(wheels) == 1:
                w1.append(e)
        yaw = np.array(yaw)
        print(' '.join(f'{k}={v}' for k, v in zip(keys, values)).ljust(46),
              f'found={len(pos) / len(rows):6.1%} pos med={np.median(pos):5.1f} p95={np.percentile(pos, 95):5.1f}'
              f' 1wheel pos med={np.median(w1) if w1 else float("nan"):5.1f}'
              f' yaw med={np.median(yaw):4.1f} f/b={np.mean(yaw < 90):6.1%} bodyIoU={np.mean(ious):.3f}',
              flush=True)


if __name__ == '__main__':
    main()
