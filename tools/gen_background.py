#!/usr/bin/env python3
"""Render the empty parking lot (no robot) for every image of a split, for P-B background subtraction.

Real use: the webcam is fixed, so an empty-lot background is captured once (average of many frames).
Sim equivalent: for each dataset image, render the same scene with the same (jittered) camera pose
from poses.csv, robot moved out of view, nominal lighting, no image noise (= averaged frames).
Lighting / brightness of the dataset image therefore differs from the background, as it would in reality.

usage (pinky_venv, needs mujoco):
  python tools/gen_background.py --split val
  python tools/gen_background.py --split test
output: <data>/background/<split>/<name>.png
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_parking_dataset as D  # noqa: E402  (same scene build, camera and lens distortion as the dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/.cache/pinky_mujoco/scene.xml')
    ap.add_argument('--data', default='~/dev_ws/pinky_parking_sim/datasets/parking_seg_v2')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    args = ap.parse_args()

    data_dir = Path(os.path.expanduser(args.data))
    with open(data_dir / 'poses.csv') as fh:
        rows = [r for r in csv.DictReader(fh) if r['split'] == args.split]
    out = data_dir / 'background' / args.split
    out.mkdir(parents=True, exist_ok=True)

    model, f, rw, rh = D.build(os.path.expanduser(args.model), args.width, args.height)
    data = mujoco.MjData(model)
    cam_id = model.camera('ceiling_camera').id
    map_x, map_y = D.distortion_map(args.width, args.height, f, rw, rh)
    renderer = mujoco.Renderer(model, rh, rw)
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = [-5.0, -5.0, float(model.qpos0[2])]         # robot far outside the camera view

    for i, r in enumerate(rows):
        model.cam_pos[cam_id] = [float(r['cam_x']), float(r['cam_y']), float(r['cam_z'])]
        model.cam_quat[cam_id] = [float(r['cam_qw']), float(r['cam_qx']), float(r['cam_qy']), float(r['cam_qz'])]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        renderer.update_scene(data, camera='ceiling_camera')
        rgb = D.remap_bilinear(renderer.render(), map_x, map_y)
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out / f"{r['name']}.png")
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f'{i + 1}/{len(rows)}', flush=True)
    print(f'done: {out}')


if __name__ == '__main__':
    main()
