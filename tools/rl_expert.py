#!/usr/bin/env python3
"""주차 규칙 제어기(전문가, 케이스 A)와 그 평가.

무엇인가
  park_closed_loop.py의 FSM과 같은 경유점·같은 방식(제자리 회전 -> 선 따라 주행)의 규칙 제어기를
  강화학습 환경(rl_park_env.ParkEnv)의 0.1초 단계에 맞춘 것. 두 가지 용도로 쓴다.
  1. 모방학습 시연: 정답 자세를 보고 운전하고, 정책이 보는 관측(노이즈 포함)과 행동을 기록 (rl_bc.py)
  2. 케이스 A 평가: 정책과 같은 관측(P-S 노이즈가 섞인 추정 자세)만 보고 운전, 같은 고정 평가 40판

경로 (바닥 mm, park_closed_loop.py와 같음)
  입구 통로 출발: 입구 아래 (346, 700) -> 입구 위 (346, 480) -> 대기점 (칸 u, 420)
                  -> 최종 방향으로 회전 -> 칸 중심선 따라 진입 (전면 = 전진, 후면 = 후진) -> 방향 마무리
  칸 앞 통로 출발: 대기점부터
  칸 앞 진입 구역 출발 (대기점보다 칸 쪽, 칸 축 근처): 바로 칸 진입

사용법 (pinky_train 가상환경, ROS 2 + 이 작업 공간 source):
  python tools/rl_expert.py                         # 케이스 A: 노이즈 자세로 고정 평가 40판 (입구 통로 출발)
  python tools/rl_expert.py --pose truth            # 참고: 정답 자세로 운전
  python tools/rl_expert.py --eval-area slot --mode rear
  python tools/rl_expert.py --test --eval-episodes 200 --brief   # 별도 테스트 200판 (20판마다 진행률)
출력: runs/rl/expert_<pose>/eval_best_episodes[_<area>][_<mode>].csv
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402
import gen_parking_dataset as D  # noqa: E402
from rl_park_env import ParkEnv, PARK_V, STAGING_V, V_MAX, W_MAX, CORRIDOR_V  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ENTRANCE_U, BELOW_GAP_V, ABOVE_GAP_V = 346.0, 700.0, 480.0   # park_closed_loop.py와 같음
V_SLOW = 0.04
TURN_TOL = math.radians(2.0)
SLOT_ZONE_U = 40.0          # 칸 축에서 이 거리 안이고 대기점보다 칸 쪽이면 "바로 진입"으로 시작


def w(u, v):
    x, y = G.mm_to_world(u, v)
    return float(x), float(y)


class Expert:
    """단계 목록(회전 / 선 따라 주행)을 차례로 수행하는 규칙 제어기. act(자세) -> 정규화 행동 [-1, 1]^2."""

    def reset(self, slot, mode, pose):
        su = D.SLOT_U[slot]
        self.final_yaw = 0.0 if mode == 'front' else math.pi
        staging, target = w(su, STAGING_V), w(su, PARK_V[mode])
        sign = +1 if mode == 'front' else -1
        u, v = G.world_to_mm(pose[0], pose[1])
        enter = [('turn', self.final_yaw), ('go', (staging, target, sign)), ('turn', self.final_yaw)]
        if v > CORRIDOR_V:                                    # 입구 통로
            phases = [('turn_to', w(ENTRANCE_U, BELOW_GAP_V)), ('go', (None, w(ENTRANCE_U, BELOW_GAP_V), +1)),
                      ('turn_to', w(ENTRANCE_U, ABOVE_GAP_V)), ('go', (None, w(ENTRANCE_U, ABOVE_GAP_V), +1)),
                      ('turn_to', staging), ('go', (None, staging, +1))] + enter
        elif v <= STAGING_V and abs(u - su) <= SLOT_ZONE_U:   # 칸 앞 진입 구역: 바로 진입
            phases = [('go', (staging, target, sign)), ('turn', self.final_yaw)]
        else:                                                 # 칸 앞 통로
            phases = [('turn_to', staging), ('go', (None, staging, +1))] + enter
        self.phases, self.i, self.line = phases, 0, None

    def _next(self):
        self.i, self.line = self.i + 1, None

    def act(self, pose):
        x, y, yaw = pose
        while self.i < len(self.phases):
            kind, arg = self.phases[self.i]
            if kind in ('turn', 'turn_to'):
                if kind == 'turn_to' and math.hypot(arg[0] - x, arg[1] - y) < 0.01:
                    self._next(); continue
                target = arg if kind == 'turn' else math.atan2(arg[1] - y, arg[0] - x)
                err = G.wrap(target - yaw)
                if abs(err) < TURN_TOL:
                    self._next(); continue
                return np.array([0.0, np.clip(3.0 * err, -W_MAX, W_MAX) / W_MAX])
            a, b, sign = arg
            if self.line is None:
                self.line = (np.array(a if a is not None else (x, y)), np.array(b))
            A, B = self.line
            L = np.linalg.norm(B - A)
            d = (B - A) / max(L, 1e-9)
            p = np.array([x, y]) - A
            along, lateral = p @ d, d[0] * p[1] - d[1] * p[0]
            remaining = L - along
            if L < 0.005 or remaining < 0.003:
                self._next(); continue
            motion = yaw if sign > 0 else G.wrap(yaw + math.pi)
            want = math.atan2(d[1], d[0]) - math.atan(lateral / 0.04)      # 선 위로 돌아가도록 조향
            err = G.wrap(want - motion)
            wv = float(np.clip(3.0 * err, -W_MAX, W_MAX))
            v = sign * float(np.clip(1.2 * remaining, 0.015, V_MAX if remaining > 0.06 else V_SLOW))
            if abs(err) > 0.35:                                            # 방향이 많이 틀어졌으면 먼저 회전
                v = 0.0
            return np.array([v / V_MAX, wv / W_MAX])
        return np.zeros(2)                                                 # 끝: 정지


def pose_from_obs(obs):
    """관측의 추정 자세 (x, y, cos yaw, sin yaw) -> (x, y, yaw). 정책이 보는 것과 같은 값."""
    return float(obs[0]), float(obs[1]), math.atan2(float(obs[3]), float(obs[2]))


class ExpertPolicy:
    """rl_train.evaluate()에 넣을 수 있게 model.predict / vecnorm.normalize_obs 형태로 감쌈."""

    def __init__(self, env, use_truth):
        self.env, self.use_truth, self.expert, self.episode = env, use_truth, Expert(), None

    def normalize_obs(self, obs):
        return obs

    def predict(self, obs, deterministic=True):
        if self.env.steps == 0 or self.episode is not self.env.target:      # 새 판 시작
            self.episode = self.env.target
            pose = self.env.physics.pose() if self.use_truth else pose_from_obs(obs[0])
            self.expert.reset(self.env.slot, self.env.mode, pose)
        pose = self.env.physics.pose() if self.use_truth else pose_from_obs(obs[0])
        return self.expert.act(pose)[None], None


def main():
    from rl_train import evaluate, report, print_summary, EVAL_SEED0, TEST_SEED0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pose', choices=['noisy', 'truth'], default='noisy',
                    help='noisy = same observation as the policy (case A), truth = ground-truth pose')
    ap.add_argument('--noise', choices=['ps', 'none'], default='ps')
    ap.add_argument('--eval-area', choices=['corridor', 'lane', 'slot'], default='corridor')
    ap.add_argument('--mode', choices=['both', 'front', 'rear'], default='both')
    ap.add_argument('--eval-episodes', type=int, default=40)
    ap.add_argument('--progress', type=int, default=20, help='print progress every N episodes (0 = off)')
    ap.add_argument('--brief', action='store_true', help='skip the per-episode table')
    ap.add_argument('--test', action='store_true',
                    help='separate test start poses (seed TEST_SEED0); use with --eval-episodes 200')
    args = ap.parse_args()
    mode = None if args.mode == 'both' else args.mode
    env = ParkEnv(noise=args.noise)
    pol = ExpertPolicy(env, args.pose == 'truth')
    t0 = time.time()
    summary, rows = evaluate(pol, pol, env, args.eval_episodes, args.eval_area, mode,
                             TEST_SEED0 if args.test else EVAL_SEED0, args.progress)
    out = REPO / 'runs/rl' / f'expert_{args.pose}'
    out.mkdir(parents=True, exist_ok=True)
    suffix = (('_test' if args.test else '') + ('' if args.eval_area == 'corridor' else f'_{args.eval_area}')
              + (f'_{mode}' if mode else ''))
    path = out / f'eval_best_episodes{suffix}.csv'
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f'expert (case A), pose={args.pose}, noise={args.noise}, start area {args.eval_area}, mode {args.mode}, '
          f'{args.eval_episodes} fixed episodes ({time.time() - t0:.0f} s)\n')
    report(rows, args.brief)
    print_summary(rows, summary, path, args.test)


if __name__ == '__main__':
    main()
