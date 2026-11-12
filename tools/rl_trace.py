#!/usr/bin/env python3
"""강화학습 정책의 주행 궤적을 코스 위에 그려 PNG로 저장 (진단용).

rl_train.py의 evaluate()와 같은 고정 판을 다시 주행한다:
  k번 판 -> 칸 k % 4, 4판마다 전면/후면을 번갈아, 출발 자세는 시드 seed0 + k
  (기본 EVAL_SEED0, --test면 TEST_SEED0). 그중 --slot(과 --mode)에 맞는 판만 골라 --episodes개를 주행.
  같은 시드와 결정적 정책을 쓰므로 결과가 eval_<which>_episodes[_test].csv와 같아야 한다 (화면에 함께 출력).

그림 (PNG 한 장)
  위   : 바닥 좌표(mm) 위의 테이프, 목표 위치, 판마다 궤적 (시작 o, 끝 x, 끝 자세의 로봇 외곽, 결과별 색)
  가운데: 시간에 따른 목표까지 거리
  아래 : 시간에 따른 전진 속도 명령 (앞뒤로 흔들리는지 확인)

사용법:
  python tools/rl_trace.py --name bc_ppo_seed2_cont --which last --slot D --test --episodes 10
  python tools/rl_trace.py --name bc_ppo_seed1 --which last --slot D --test --episodes 10    # 성공한 seed와 비교
출력: runs/rl/<name>/trace_<which>_<slot>[_test][_mode].png
      runs/rl/<name>/trace_<which>_<slot>[_test][_mode].csv (0.1초마다 자세, 명령, 목표까지 거리)
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402
from rl_park_env import ParkEnv, REPO, V_MAX, W_MAX, footprint  # noqa: E402
from rl_train import EVAL_SEED0, TEST_SEED0  # noqa: E402

COLORS = {'success': 'tab:green', 'timeout': 'tab:orange', 'contact': 'tab:red', 'out': 'tab:purple'}


def run_episode(model, vecnorm, env, k, seed0):
    """evaluate()와 같은 방식으로 k번 판을 주행하고 0.1초마다 상태를 기록."""
    slot, mode = k % 4, ('front', 'rear')[(k // 4) % 2]
    obs, _ = env.reset(seed=seed0 + k, options={'slot': slot, 'mode': mode, 'area': 'corridor'})
    tx, ty, _ = env.target

    def record(t, v_cmd, w_cmd):
        x, y, yaw = env.physics.pose()
        u, v = G.world_to_mm(x, y)
        return {'episode': k, 'mode': mode, 't': round(t, 1), 'u': u, 'v': v, 'yaw_deg': math.degrees(yaw),
                'v_cmd': v_cmd, 'w_cmd': w_cmd, 'dist_mm': math.hypot(tx - x, ty - y) * 1000}

    track, done, info = [record(0.0, 0.0, 0.0)], False, None
    while not done:
        action, _ = model.predict(vecnorm.normalize_obs(obs[None]), deterministic=True)
        a = np.clip(action[0], -1.0, 1.0)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        track.append(record(info['time_s'], float(a[0]) * V_MAX, float(a[1]) * W_MAX))
    return mode, info['result'], track


def sign_changes(values, eps=0.005):
    """속도 명령의 부호가 바뀐 횟수 (|값| < eps는 0으로 보고 건너뜀). 앞뒤 흔들림의 지표."""
    s = [math.copysign(1, x) for x in values if abs(x) >= eps]
    return sum(a != b for a, b in zip(s, s[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='runs/rl/<name>')
    ap.add_argument('--which', choices=['best', 'last'], default='last')
    ap.add_argument('--slot', choices=list('ABCD'), default='D')
    ap.add_argument('--mode', choices=['both', 'front', 'rear'], default='both')
    ap.add_argument('--episodes', type=int, default=10, help='number of matching episodes to drive')
    ap.add_argument('--test', action='store_true', help='use the test start poses (TEST_SEED0)')
    ap.add_argument('--noise', choices=['ps', 'none'], default='ps')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')                       # 화면 없이 파일로만 저장
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import torch
    torch.set_num_threads(1)

    out = REPO / 'runs/rl' / args.name
    model = PPO.load(out / f'{args.which}_model', device='cpu')
    vecnorm = VecNormalize.load(str(out / f'{args.which}_vecnormalize.pkl'),
                                DummyVecEnv([lambda: ParkEnv(noise=args.noise)]))
    vecnorm.training = False
    env = ParkEnv(noise=args.noise)
    seed0 = TEST_SEED0 if args.test else EVAL_SEED0
    slot = 'ABCD'.index(args.slot)

    # 조건에 맞는 판 번호 고르기 (evaluate()의 판 배정 규칙과 같음)
    ks, k = [], slot
    while len(ks) < args.episodes:
        if args.mode in ('both', ('front', 'rear')[(k // 4) % 2]):
            ks.append(k)
        k += 4

    runs = []
    print(f'{args.name} / {args.which}_model, slot {args.slot}, mode {args.mode}, '
          f'{"test" if args.test else "eval"} starts, {len(ks)} episodes')
    for k in ks:
        mode, result, track = run_episode(model, vecnorm, env, k, seed0)
        runs.append((k, mode, result, track))
        last = track[-1]
        late = [r['v_cmd'] for r in track if r['t'] >= last['t'] - 20.0]
        print(f'  ep {k:3d} {mode:5s} {result:8s} t {last["t"]:5.1f} s  final dist {last["dist_mm"]:5.0f} mm  '
              f'closest {min(r["dist_mm"] for r in track):5.0f} mm  end (u {last["u"]:.0f}, v {last["v"]:.0f}, '
              f'yaw {last["yaw_deg"]:.0f} deg)  v sign changes in last 20 s: {sign_changes(late)}')

    # ---- 그림
    fig = plt.figure(figsize=(11, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1, 1])
    ax, ax_d, ax_v = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for u, v, w, h in G.TAPES:
        ax.add_patch(Rectangle((u, v), w, h, color='black'))
    for k, mode, result, track in runs:
        c = COLORS.get(result, 'gray')
        uu, vv = [r['u'] for r in track], [r['v'] for r in track]
        ax.plot(uu, vv, color=c, lw=1.0, alpha=0.8, label=f'ep {k} {mode} {result}')
        ax.plot(uu[0], vv[0], 'o', color=c, ms=4)
        ax.plot(uu[-1], vv[-1], 'x', color=c, ms=7)
        env.reset(seed=seed0 + k, options={'slot': k % 4, 'mode': mode, 'area': 'corridor'})
        x, y = G.mm_to_world(track[-1]['u'], track[-1]['v'])
        for poly in footprint(x, y, math.radians(track[-1]['yaw_deg'])):
            ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=c, lw=0.8))
        tu, tv = G.world_to_mm(*env.target[:2])
        ax.plot(tu, tv, '*', color='blue', ms=10)
        t = [r['t'] for r in track]
        ax_d.plot(t, [r['dist_mm'] for r in track], color=c, lw=1.0, alpha=0.8)
        ax_v.plot(t, [r['v_cmd'] for r in track], color=c, lw=1.0, alpha=0.8)
    ax.set_xlim(-20, 1320)
    ax.set_ylim(1220, -20)                      # v는 아래로 커짐 (천장 카메라 그림과 같은 방향)
    ax.set_aspect('equal')
    ax.set_xlabel('u (mm)')
    ax.set_ylabel('v (mm)')
    ax.set_title(f'{args.name} / {args.which}: slot {args.slot}, mode {args.mode} '
                 f'(o start, x end, * target, outline = final pose)')
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax_d.set_ylabel('distance to target (mm)')
    ax_d.set_yscale('log')
    ax_d.grid(alpha=0.3)
    ax_v.set_ylabel('v command (m/s)')
    ax_v.set_xlabel('time (s)')
    ax_v.axhline(0, color='gray', lw=0.5)
    ax_v.grid(alpha=0.3)
    fig.tight_layout()

    suffix = f'{args.which}_{args.slot}' + ('_test' if args.test else '') + \
        ('' if args.mode == 'both' else f'_{args.mode}')
    png, csv_path = out / f'trace_{suffix}.png', out / f'trace_{suffix}.csv'
    fig.savefig(png, dpi=110)
    with open(csv_path, 'w', newline='') as f:
        rows = [r for _, _, _, track in runs for r in track]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'saved {png}\n      {csv_path}')


if __name__ == '__main__':
    main()
