#!/usr/bin/env python3
"""P-K용 YOLO-pose 라벨을 기존 데이터셋에서 만든다 (새로 렌더링하지 않음).

- 키포인트: poses.csv의 정답 자세 + 이미지별 카메라 자세로 4점(pk_kpt.KPTS)을 투영.
  가시성은 화면 안이면 2, 화면 밖이면 0 (좌표도 0). 몸통에 가려진 바퀴 점도 위치는 정답으로 학습.
- 박스: 기존 세그멘테이션 라벨의 바퀴(0)+몸통(2) 폴리곤을 감싸는 사각형. 둘 다 없으면 빈 라벨.
- 이미지: 원본 파일로 가는 링크 (복사하지 않음). 기존 데이터는 건드리지 않는다.
- data yaml: kpt_shape [4, 3], flip_idx [0, 1, 3, 2], 클래스 1개(robot).

사용법 (cv2·numpy만 필요):
  python tools/gen_kpt_labels.py
  출력: datasets/parking_kp/{images,labels}/{train,val,test}, data.yaml, debug/ (오버레이 확인 이미지)
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402
import pk_kpt as K  # noqa: E402

COLORS = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 255)]   # 앞끝 빨강, 뒤끝 파랑, 왼바퀴 초록, 오른바퀴 노랑


def load_rows(src):
    with open(src / 'poses.csv') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in r:
            if k not in ('split', 'name', 'kind'):
                r[k] = float(r[k])
    return rows


def robot_box(label_path):
    """세그 라벨의 바퀴·몸통 폴리곤을 감싸는 정규화 박스 (x0, y0, x1, y1) 또는 None."""
    xs, ys = [], []
    if label_path.exists():
        for line in label_path.read_text().split('\n'):
            parts = line.split()
            if len(parts) < 7 or parts[0] not in ('0', '2'):
                continue
            v = np.array(parts[1:], float)
            xs.extend(v[0::2]); ys.extend(v[1::2])
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='~/dev_ws/pinky_parking_sim/datasets/parking_seg_v2')
    ap.add_argument('--out', default='~/dev_ws/pinky_parking_sim/datasets/parking_kp')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--debug', type=int, default=10, help='overlay images per split')
    args = ap.parse_args()

    src = Path(os.path.expanduser(args.src)).resolve()
    out = Path(os.path.expanduser(args.out)).resolve()
    rows = load_rows(src)
    (out / 'debug').mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
        n = n_empty = n_hidden = 0
        for r in [r for r in rows if r['split'] == split]:
            name = r['name']
            img_src = src / 'images' / split / f'{name}.jpg'
            link = out / 'images' / split / f'{name}.jpg'
            if not link.exists():
                link.symlink_to(img_src)
            cam = G.Camera.from_row(r)
            xy = K.KPTS[:, :2] @ G.rot(r['yaw_rad']).T + [r['x_m'], r['y_m']]
            pix = cam.project(np.column_stack([xy, K.KPTS[:, 2]]))
            box = robot_box(src / 'labels' / split / f'{name}.txt')
            lines = []
            if box is not None:
                x0, y0, x1, y1 = box
                vals = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
                for px, py in pix:
                    inside = 0 <= px < G.WIDTH and 0 <= py < G.HEIGHT
                    n_hidden += not inside
                    vals += [px / G.WIDTH, py / G.HEIGHT, 2] if inside else [0, 0, 0]
                lines.append('0 ' + ' '.join(f'{v:.6f}' if not isinstance(v, int) else str(v) for v in vals))
            else:
                n_empty += 1
            (out / 'labels' / split / f'{name}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
            if n < args.debug:
                vis = cv2.imread(str(img_src))
                if box is not None:
                    cv2.rectangle(vis, (int(x0 * G.WIDTH), int(y0 * G.HEIGHT)),
                                  (int(x1 * G.WIDTH), int(y1 * G.HEIGHT)), (255, 255, 255), 1)
                for (px, py), col in zip(pix, COLORS):
                    cv2.circle(vis, (int(round(px)), int(round(py))), 4, col, -1)
                cv2.imwrite(str(out / 'debug' / f'{split}_{name}.jpg'), vis)
            n += 1
        print(f'{split}: images={n} empty_labels={n_empty} keypoints_outside_image={n_hidden}')
    (out / 'data.yaml').write_text(
        f'path: {out}\n'
        + ''.join(f'{s}: images/{s}\n' for s in args.splits)
        + 'kpt_shape: [4, 3]\nflip_idx: [0, 1, 3, 2]\nnames:\n  0: robot\n')
    print(f'done: {out}/data.yaml')


if __name__ == '__main__':
    main()
