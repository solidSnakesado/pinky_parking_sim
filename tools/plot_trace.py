#!/usr/bin/env python3
"""주차 판 기록(trace_XXX.csv) 그림: 정답 궤적과 정책이 본(인식) 궤적을 코스 위에 겹치고, 시간에 따른 오차를 그린다.

무엇인가
  ros_park_eval.py --trace가 판마다 남기는 ros_logs/trace_XXX.csv (정책 단계 0.1초마다 정답 자세, 정책이 관측에 쓴
  추정 자세, 명령)를 읽어, 인식 루프에서 실패한 판이 "실제로는 어디 있었고 정책은 어디 있다고 봤는지"를 보여 준다.

그림 (PNG 1장)
  왼쪽 : 바닥 그림(worlds/course_floor.png) 위에 테이프, 목표 자세, 정답 궤적(파랑), 정책이 본 궤적(주황).
         1초마다 정답 자세의 로봇 외곽(파랑 얇은 선), 마지막 자세는 정답(빨강)과 정책이 본 자세(주황 점선).
  오른쪽: 시간에 따른 값 (정답 = 실선, 정책이 본 값 = 점선)
         1) 좌우: 칸 중심선(u)에서 벗어난 거리 (mm)
         2) 깊이: 목표 v까지 남은 거리 (mm, + = 아직 칸 바깥)
         3) 방향: 목표 방향과의 차이 (°)
         4) 정책이 본 자세의 오차 (위치 mm, 방향 °)
  테이프에 닿은 단계는 세로 빨간 선으로 표시.

사용법 (pinky_train 가상환경, matplotlib 필요):
  python tools/plot_trace.py runs/rl/bc_ppo_seed2_tp10/ros_logs/trace_004.csv
  python tools/plot_trace.py <trace.csv> --out a_rear_ep4.png
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402
from rl_park_env import footprint, target_pose  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FLOOR = REPO / 'worlds/course_floor.png'
MM_PER_PX = 0.5


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f'empty trace: {path}')

    def col(name):
        return np.array([float(r[name]) if r[name] != '' else np.nan for r in rows])
    d = {k: col(k) for k in ('t', 'x', 'y', 'yaw', 'seen_x', 'seen_y', 'seen_yaw', 'v', 'w')}
    d['touching'] = np.array([r['touching'] == 'True' for r in rows])
    d['slot'], d['mode'] = rows[0]['slot'], rows[0]['mode']
    return d


def to_mm(x, y):
    u, v = G.world_to_mm(np.asarray(x), np.asarray(y))
    return np.asarray(u), np.asarray(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('trace', help='ros_logs/trace_XXX.csv from ros_park_eval.py --trace')
    ap.add_argument('--out', default=None, help='output PNG (default: next to the trace, same name)')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = load(args.trace)
    slot = 'ABCD'.index(d['slot'])
    tx, ty, tyaw = target_pose(slot, d['mode'])
    tu, tv = to_mm(tx, ty)
    u, v = to_mm(d['x'], d['y'])
    su, sv = to_mm(d['seen_x'], d['seen_y'])
    yaw_err = np.degrees([G.wrap(a - tyaw) for a in d['yaw']])
    seen_yaw_err = np.degrees([G.wrap(a - tyaw) if a == a else np.nan for a in d['seen_yaw']])
    pos_err = np.hypot(su - u, sv - v)
    head_err = np.degrees([abs(G.wrap(a - b)) if a == a else np.nan for a, b in zip(d['seen_yaw'], d['yaw'])])
    t = d['t']
    hit = np.flatnonzero(d['touching'])
    t_hit = t[hit[0]] if len(hit) else None

    fig = plt.figure(figsize=(15, 8.5))
    ax = fig.add_axes([0.04, 0.07, 0.44, 0.86])

    # 왼쪽: 코스 확대 (궤적과 목표가 모두 들어오게)
    allu = np.concatenate([u, su[~np.isnan(su)], [tu]])
    allv = np.concatenate([v, sv[~np.isnan(sv)], [tv]])
    pad = 120
    u0, u1 = allu.min() - pad, allu.max() + pad
    v0, v1 = max(0, allv.min() - pad), allv.max() + pad
    if FLOOR.is_file():
        img = plt.imread(str(FLOOR))
        h, w = img.shape[:2]
        ax.imshow(img, extent=[0, w * MM_PER_PX, h * MM_PER_PX, 0], zorder=0)
    for uu, vv, ww, hh in G.TAPES:                          # 테이프 (판정에 쓰는 사각형)
        ax.add_patch(plt.Rectangle((uu, vv), ww, hh, fill=False, ec='k', lw=0.6, zorder=1))
    for p in footprint(tx, ty, tyaw):                        # 목표 자세
        ax.fill(p[:, 0], p[:, 1], fc='none', ec='g', lw=1.2, ls='--', zorder=2)
    ax.plot(u, v, '-', c='tab:blue', lw=1.8, label='true path', zorder=3)
    ax.plot(su, sv, '-', c='tab:orange', lw=1.2, label='seen by policy', zorder=3)
    for i in range(0, len(t), 10):                           # 1초마다 정답 외곽
        for p in footprint(d['x'][i], d['y'][i], d['yaw'][i]):
            ax.fill(p[:, 0], p[:, 1], fc='none', ec='tab:blue', lw=0.4, alpha=0.6, zorder=2)
    for p in footprint(d['x'][-1], d['y'][-1], d['yaw'][-1]):  # 마지막 정답 자세
        ax.fill(p[:, 0], p[:, 1], fc='none', ec='r', lw=1.8, zorder=4)
    last_seen = np.flatnonzero(~np.isnan(d['seen_x']))
    if len(last_seen):
        i = last_seen[-1]
        for p in footprint(d['seen_x'][i], d['seen_y'][i], d['seen_yaw'][i]):
            ax.fill(p[:, 0], p[:, 1], fc='none', ec='tab:orange', lw=1.5, ls=':', zorder=4)
    ax.plot(u[0], v[0], 'o', c='tab:blue', ms=6, zorder=5)
    ax.set_xlim(u0, u1)
    ax.set_ylim(v1, v0)
    ax.set_aspect('equal')
    ax.set_xlabel('u (mm)')
    ax.set_ylabel('v (mm)')
    ax.legend(loc='lower right', fontsize=9)
    result = 'contact' if t_hit is not None else 'no contact'
    ax.set_title(f"{Path(args.trace).name}: slot {d['slot']} {d['mode']}, {t[-1]:.1f} s, {result}\n"
                 f"red = final true pose, orange dotted = last pose seen by policy, green = target")

    # 오른쪽: 시간에 따른 값
    panels = [
        ('lateral (u - slot u, mm)', u - tu, su - tu),
        ('depth to target (v - target v, mm, + = outside)', v - tv, sv - tv),
        ('heading vs target (deg)', yaw_err, seen_yaw_err),
    ]
    for k, (title, true_val, seen_val) in enumerate(panels):
        a = fig.add_axes([0.55, 0.76 - k * 0.23, 0.42, 0.17])
        a.plot(t, true_val, '-', c='tab:blue', label='true')
        a.plot(t, seen_val, '--', c='tab:orange', label='seen')
        a.axhline(0, c='gray', lw=0.5)
        if t_hit is not None:
            a.axvline(t_hit, c='r', lw=1)
        a.set_title(title, fontsize=10)
        a.tick_params(labelsize=8, labelbottom=False)
        if k == 0:
            a.legend(fontsize=8, loc='best')
    a = fig.add_axes([0.55, 0.07, 0.42, 0.17])
    a.plot(t, pos_err, '-', c='tab:purple', label='position error (mm)')
    a2 = a.twinx()
    a2.plot(t, head_err, '-', c='tab:green', lw=0.9, label='heading error (deg)')
    if t_hit is not None:
        a.axvline(t_hit, c='r', lw=1)
    a.set_title('error of the pose seen by the policy (vs true)', fontsize=10)
    a.set_xlabel('time since policy start (s)')
    a.set_ylabel('mm', fontsize=8)
    a2.set_ylabel('deg', fontsize=8)
    a.tick_params(labelsize=8)
    a2.tick_params(labelsize=8)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')

    out = Path(args.out) if args.out else Path(args.trace).with_suffix('.png')
    fig.savefig(out, dpi=110)
    print(f'saved {out}')
    # 화면 요약: 마지막 2초 동안의 정답 / 정책이 본 값
    m = t >= t[-1] - 2.0
    print(f"last 2 s: lateral true {np.nanmean((u - tu)[m]):.1f} mm / seen {np.nanmean((su - tu)[m]):.1f} mm, "
          f"heading true {np.nanmean(yaw_err[m]):.1f} deg / seen {np.nanmean(seen_yaw_err[m]):.1f} deg, "
          f"pose error {np.nanmean(pos_err[m]):.1f} mm / {np.nanmean(head_err[m]):.1f} deg")


if __name__ == '__main__':
    main()
