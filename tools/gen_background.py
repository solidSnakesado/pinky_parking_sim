#!/usr/bin/env python3
"""P-B 배경 차분용: 한 분할(split)의 모든 이미지에 대해 빈 주차장(로봇 없음)을 렌더링한다.

실물에서는 웹캠이 고정되어 있으므로 빈 주차장 배경을 한 번 찍어 둔다(여러 프레임의 평균).
시뮬에서는 이에 맞춰, 데이터셋 이미지마다 poses.csv의 같은(흔들림이 들어간) 카메라 자세로 같은 장면을
렌더링하되 로봇은 화면 밖으로 옮기고, 조명은 기본값, 이미지 노이즈는 없음(= 여러 프레임 평균)으로 한다.
따라서 데이터셋 이미지와 배경은 실물처럼 조명·밝기가 서로 다르다.

사용법 (pinky_venv, mujoco 필요):
  python tools/gen_background.py --split val
  python tools/gen_background.py --split test
출력: <data>/background/<split>/<name>.png
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
import gen_parking_dataset as D  # noqa: E402  (데이터셋과 같은 장면 구성, 카메라, 렌즈 왜곡)


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
    data.qpos[0:3] = [-5.0, -5.0, float(model.qpos0[2])]         # 로봇을 카메라 화면 밖 멀리 둠

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
