#!/usr/bin/env python3
"""+F 실험: 프레임마다 인식만 하는 경우 vs. 인식 + 시간 필터 / 오도메트리 융합.

연속 주행(tools/gen_drive_sequences.py)의 모든 프레임에 대해:
  raw      : 이미지 한 장만으로 구한 자세 (방법 pa / pb / ps, eval_pose.py와 같음)
  filtered : 오도메트리가 직전 프레임 이후의 움직임을 예측하고, 카메라 자세로 보정
             - 앞/뒤 확인: 예측과 반대 방향(90° 넘게 차이)인 측정은 180° 돌려서 씀.
               이런 측정이 3번 연속이면 상태 쪽을 뒤집음
             - 이상값 차단: 예측에서 너무 먼 측정은 건너뜀. 5번 연속 건너뛰면 상태를 측정값으로 다시 시작
             - 섞기: 상태 += 이득 * (측정 - 상태)
  odom     : 오도메트리만, 첫 측정에서 시작 (참고용: 오도메트리가 얼마나 빨리 틀어지는가)
세 가지 모두 정답과의 오차를 요약한다. 필터 이득은 미리 정해 둔 값이다(이 주행들로 튜닝하지 않음).

사용법:
  python tools/eval_sequence.py --method pa
  python tools/eval_sequence.py --method pb
  python tools/eval_sequence.py --method ps --weights runs/segment/runs/ps/yolo11n_seed0/weights/best.pt
  python tools/eval_sequence.py --method pa --reuse      # 필터만 다시, 저장된 raw 추정값 재사용
출력: results/seq_<method>.csv (프레임별)
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

GAIN_POS, GAIN_YAW = 0.4, 0.3
GATE_POS_M, GATE_YAW_DEG = 0.04, 30.0
FLIP_CONFIRM, RESET_AFTER = 3, 5


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def detector(method, weights):
    if method == 'pa':
        import pa_rule
        return lambda img, cam, bg: pa_rule.detect(img, cam)
    if method == 'ps':
        import ps_seg
        ps_seg.load(os.path.expanduser(weights))
        return lambda img, cam, bg: ps_seg.detect(img, cam)
    if method == 'pb':
        import pb_bgsub
        return lambda img, cam, bg: pb_bgsub.detect(img, cam, bg)
    raise SystemExit(f'unknown method {method}')


def run_raw(data, runs, method, weights):
    detect = detector(method, weights)
    rows, t_total, n = [], 0.0, 0
    for run in runs:
        rd = data / f"run_{int(run['run']):03d}"
        cam = G.Camera.from_row(run)
        bg = cv2.imread(str(rd / 'background.png'))
        for fr in read_csv(rd / 'frames.csv'):
            img = cv2.imread(str(rd / 'images' / f"{int(fr['frame']):05d}.jpg"))
            t0 = time.perf_counter()
            wheels, body = detect(img, cam, bg)
            pose = G.pose_from_masks(cam, wheels, body)
            t_total += time.perf_counter() - t0
            n += 1
            item = dict(run=run['run'], **fr)
            if pose is not None:
                item.update(raw_x=pose[0], raw_y=pose[1], raw_yaw=pose[2])
            rows.append(item)
        print(f"run {run['run']}: {len(rows)} frames so far", flush=True)
    print(f'recognition time/frame = {1000 * t_total / max(1, n):.1f} ms')
    return rows


def body_delta(a, b):
    """오도메트리 자세 a에서 b까지의 움직임을 a의 로봇 좌표로 (dx, dy, dyaw)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = math.cos(a[2]), math.sin(a[2])
    return c * dx + s * dy, -s * dx + c * dy, G.wrap(b[2] - a[2])


def apply_delta(p, d):
    c, s = math.cos(p[2]), math.sin(p[2])
    return [p[0] + c * d[0] - s * d[1], p[1] + s * d[0] + c * d[1], G.wrap(p[2] + d[2])]


def run_filter(rows):
    by_run = {}
    for r in rows:
        by_run.setdefault(r['run'], []).append(r)
    for frames in by_run.values():
        state = odo = None
        prev_odom = None
        flips = skips = 0
        for r in frames:
            odom = (float(r['odom_x']), float(r['odom_y']), float(r['odom_yaw']))
            meas = None
            if r.get('raw_x') not in (None, ''):
                meas = [float(r['raw_x']), float(r['raw_y']), float(r['raw_yaw'])]
            if state is not None:                                       # 오도메트리로 예측
                d = body_delta(prev_odom, odom)
                state = apply_delta(state, d)
                odo = apply_delta(odo, d)
            prev_odom = odom
            if meas is not None:
                if state is None:                                       # 첫 측정에서 시작
                    state, odo = list(meas), list(meas)
                else:
                    if abs(G.wrap(meas[2] - state[2])) > math.pi / 2:   # 앞/뒤가 서로 다름
                        flips += 1
                        if flips >= FLIP_CONFIRM:                       # 상태 쪽이 반대로 되어 있었음
                            state[2] = G.wrap(state[2] + math.pi)
                            flips = 0
                        else:
                            meas[2] = G.wrap(meas[2] + math.pi)
                    else:
                        flips = 0
                    dpos = math.hypot(meas[0] - state[0], meas[1] - state[1])
                    dyaw = abs(math.degrees(G.wrap(meas[2] - state[2])))
                    if dpos > GATE_POS_M or dyaw > GATE_YAW_DEG:        # 이상값
                        skips += 1
                        if skips >= RESET_AFTER:
                            state, skips = list(meas), 0
                    else:
                        skips = 0
                        state[0] += GAIN_POS * (meas[0] - state[0])
                        state[1] += GAIN_POS * (meas[1] - state[1])
                        state[2] = G.wrap(state[2] + GAIN_YAW * G.wrap(meas[2] - state[2]))
            if state is not None:
                r.update(fil_x=state[0], fil_y=state[1], fil_yaw=state[2],
                         odo_x=odo[0], odo_y=odo[1], odo_yaw=odo[2])
    return rows


def errors(rows, prefix):
    pos, yaw = [], []
    for r in rows:
        if r.get(f'{prefix}_x') in (None, ''):
            continue
        gx, gy, gyaw = float(r['gt_x']), float(r['gt_y']), float(r['gt_yaw'])
        pos.append(1000 * math.hypot(float(r[f'{prefix}_x']) - gx, float(r[f'{prefix}_y']) - gy))
        yaw.append(abs(math.degrees(G.wrap(float(r[f'{prefix}_yaw']) - gyaw))))
    return np.array(pos), np.array(yaw)


def summary(rows, label):
    lines = []
    for prefix, name in (('raw', 'raw (image only)'), ('fil', 'filtered (+F)'), ('odo', 'odometry only')):
        pos, yaw = errors(rows, prefix)
        if len(pos) == 0:
            lines.append(f'  {name:18s} no estimates')
            continue
        lines.append(f'  {name:18s} avail={len(pos) / len(rows):6.1%}  pos mm median={np.median(pos):5.1f} '
                     f'p95={np.percentile(pos, 95):5.1f} max={pos.max():6.1f}  yaw deg median={np.median(yaw):4.1f} '
                     f'p95={np.percentile(yaw, 95):5.1f}  front/back ok={np.mean(yaw < 90):6.1%}')
    return f'{label} (frames={len(rows)})\n' + '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='~/dev_ws/pinky_parking_sim/datasets/drive_seq')
    ap.add_argument('--method', default='pa', choices=['pa', 'pb', 'ps'])
    ap.add_argument('--weights', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--reuse', action='store_true', help='reuse raw estimates from --out, run the filter only')
    args = ap.parse_args()
    if args.method == 'ps' and not args.weights and not args.reuse:
        raise SystemExit('--weights is required for ps')

    data = Path(os.path.expanduser(args.data))
    out = Path(args.out or f'results/seq_{args.method}.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = read_csv(data / 'runs.csv')
    for run in runs:
        for k in ('cam_x', 'cam_y', 'cam_z', 'cam_qw', 'cam_qx', 'cam_qy', 'cam_qz'):
            run[k] = float(run[k])
    if args.reuse:
        rows = read_csv(out)
    else:
        rows = run_raw(data, runs, args.method, args.weights)
    rows = run_filter(rows)

    keys = ['run', 'frame', 't', 'phase', 'gt_x', 'gt_y', 'gt_yaw', 'odom_x', 'odom_y', 'odom_yaw',
            'raw_x', 'raw_y', 'raw_yaw', 'fil_x', 'fil_y', 'fil_yaw', 'odo_x', 'odo_y', 'odo_yaw']
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})

    print(f'method={args.method} runs={len(runs)} -> {out}')
    print(summary(rows, 'all frames'))
    print(summary([r for r in rows if r['phase'] == 'hold'], 'parked (hold phase)'))
    last = {}
    for r in rows:
        last[r['run']] = r
    print(summary(list(last.values()), 'final frame per run'))


if __name__ == '__main__':
    main()
