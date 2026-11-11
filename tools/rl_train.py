#!/usr/bin/env python3
"""MuJoCo 주차 환경(tools/rl_park_env.py)에서 PPO로 주차 정책을 학습한다.

하는 일
  - ParkEnv를 병렬 프로세스 N개(CPU 코어마다 하나)로 돌리며 PPO(Stable-Baselines3)를 학습
  - 관측값과 보상을 정규화(VecNormalize). 정규화 통계는 모델과 함께 저장
  - 학습 판의 출발 구역: 칸 앞 통로(--lane-start-prob), 칸 앞 진입 구역(--slot-start-prob), 나머지는 입구 통로
  - --resume <이름>: 그 실행의 모델과 정규화 통계를 불러와 이어서 학습 (단계 수는 이어서 셈)
  - --eval-every 단계마다: 학습에 쓰지 않은 "고정" 평가 판 목록(칸 A~D x 전면/후면, 고정 출발 자세,
    항상 입구 통로에서 출발)을 현재 정책으로 탐색 무작위성 없이 주행하고,
    성공 / 접촉 / 이탈 / 시간 초과 비율과 성공한 판의 최종 오차를 기록
  - 평가 성공률이 가장 높은 모델과 마지막 모델을 저장

사용법 (pinky_train 가상환경, ROS 2 + 이 작업 공간 source):
  python tools/rl_train.py --name ppo_ps_seed0 --steps 1000000            # 8코어에서 약 5분
  python tools/rl_train.py --name ppo_ps_seed0 --steps 5000000 --seed 0
  python tools/rl_train.py --name ppo_slot_seed0 --resume ppo_route_seed0_3m --steps 3000000   # 이어서 학습
  python tools/rl_train.py --name ppo_ps_seed0 --eval-only                # 저장된 최고 모델 평가:
                                                                          #   판별 결과 + 시간 초과 원인
  tensorboard --logdir runs/rl                                            # 그래프 (브라우저: localhost:6006)
출력: runs/rl/<name>/  best_model.zip, best_vecnormalize.pkl, last_model.zip, last_vecnormalize.pkl,
      eval.csv (평가 1회당 한 줄), args.txt, tb/ (TensorBoard),
      eval_<best|last>_episodes.csv (--eval-only: 판별 최종 상태)
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_park_env import (ParkEnv, CTRL_DT, STOP_V, STOP_W,  # noqa: E402
                         OK_LATERAL_MM, OK_DEPTH_MM, OK_YAW_DEG)

REPO = Path(__file__).resolve().parents[1]
EVAL_SEED0 = 100000                 # 평가 출발 자세는 이 시드부터 만듦 (학습은 다른 시드 사용)


def make_env(rank, seed, noise, lane_start_prob, slot_start_prob):
    def thunk():
        env = ParkEnv(noise=noise, lane_start_prob=lane_start_prob, slot_start_prob=slot_start_prob)
        env.reset(seed=seed * 1000 + rank)
        return env
    return thunk


def _final_state(env):
    """목표 대비 정답 기준 최종 상태: 거리, 부호 있는 깊이 (+ = 덜 들어감), 정지 여부."""
    x, y, yaw = env.physics.pose()
    tx, ty, tyaw = env.target
    v, w = env.physics.twist
    return {'dist_mm': math.hypot(tx - x, ty - y) * 1000, 'depth_signed_mm': (tx - x) * 1000,
            'stopped': bool(abs(v) < STOP_V and abs(w) < STOP_W), 'final_v': float(v), 'final_w': float(w)}


def evaluate(model, vecnorm, env, episodes):
    """고정 평가 판, 항상 입구 통로에서 출발: k번 판 -> 칸 k % 4, 방식은 4판마다 전면/후면을 번갈아,
    출발 자세는 시드 EVAL_SEED0+k로 만듦.
    반환값: (요약, 판마다 최종 상태를 담은 행 목록)."""
    counts = {'success': 0, 'contact': 0, 'out': 0, 'timeout': 0}
    lat, dep, yaw, tim, rows = [], [], [], [], []
    for k in range(episodes):
        obs, start = env.reset(seed=EVAL_SEED0 + k, options={'slot': k % 4, 'mode': ('front', 'rear')[(k // 4) % 2],
                                                                'area': 'corridor'})
        done, min_dist = False, float('inf')
        while not done:
            action, _ = model.predict(vecnorm.normalize_obs(obs[None]), deterministic=True)
            obs, _, term, trunc, info = env.step(action[0])
            done = term or trunc
            x, y, _ = env.physics.pose()
            min_dist = min(min_dist, math.hypot(env.target[0] - x, env.target[1] - y) * 1000)
        counts[info['result']] += 1
        if info['result'] == 'success':
            lat.append(info['lateral_mm']); dep.append(info['depth_mm']); yaw.append(info['yaw_deg'])
            tim.append(info['time_s'])
        rows.append({'episode': k, 'slot': start['slot'], 'mode': start['mode'], 'result': info['result'],
                     'time_s': info['time_s'], 'lateral_mm': info['lateral_mm'], 'depth_mm': info['depth_mm'],
                     'yaw_deg': info['yaw_deg'], **_final_state(env), 'min_dist_mm': min_dist})
    mean = lambda a: float(np.mean(a)) if a else float('nan')
    summary = {**{k: v / episodes for k, v in counts.items()},
               'lateral_mm': mean(lat), 'depth_mm': mean(dep), 'yaw_deg': mean(yaw), 'time_s': mean(tim)}
    return summary, rows


def report(rows):
    """판별 결과 표 + 시간 초과 판이 성공으로 인정되지 않은 이유."""
    print(f"{'ep':>3} {'slot':>4} {'mode':>5} {'result':>8} {'time':>5} {'dist':>6} {'min':>6} "
          f"{'lat':>6} {'depth±':>7} {'head':>5} {'stop':>4}")
    for r in rows:
        print(f"{r['episode']:3d} {r['slot']:>4} {r['mode']:>5} {r['result']:>8} {r['time_s']:5.1f} "
              f"{r['dist_mm']:6.1f} {r['min_dist_mm']:6.1f} {r['lateral_mm']:6.1f} {r['depth_signed_mm']:7.1f} "
              f"{r['yaw_deg']:5.1f} {'yes' if r['stopped'] else 'no':>4}")
    print('(mm / deg / s; dist = final distance to target, min = closest during the episode, '
          'depth± > 0 = not deep enough, stop = stopped at the end)')

    to = [r for r in rows if r['result'] == 'timeout']
    if to:
        near = [r for r in to if r['dist_mm'] <= 50]
        print(f'\ntimeouts: {len(to)}')
        print(f'  ended within 50 mm of the target : {len(near)}')
        for name, bad in (('lateral > 20 mm', lambda r: r['lateral_mm'] > OK_LATERAL_MM),
                          ('depth > 20 mm', lambda r: r['depth_mm'] > OK_DEPTH_MM),
                          ('heading > 5 deg', lambda r: r['yaw_deg'] > OK_YAW_DEG),
                          ('not stopped', lambda r: not r['stopped'])):
            print(f'    {name:16s}: {sum(bad(r) for r in near)}')
        print(f'  ended farther than 50 mm         : {len(to) - len(near)} '
              f"(final distance median {np.median([r['dist_mm'] for r in to if r['dist_mm'] > 50] or [np.nan]):.0f} mm, "
              f"closest-ever median {np.median([r['min_dist_mm'] for r in to if r['dist_mm'] > 50] or [np.nan]):.0f} mm)")
    print('\nsuccess by slot / mode:')
    for mode in ('front', 'rear'):
        cells = []
        for slot in 'ABCD':
            sel = [r for r in rows if r['slot'] == slot and r['mode'] == mode]
            cells.append(f"{slot} {sum(r['result'] == 'success' for r in sel)}/{len(sel)}")
        print(f'  {mode:5s}: ' + '  '.join(cells))


class EvalCallback(BaseCallback):
    def __init__(self, out, every, episodes, noise):
        super().__init__()
        self.out, self.every, self.episodes = out, every, episodes
        self.env = ParkEnv(noise=noise)
        self.next, self.best, self.t0 = every, -1.0, time.time()
        self.csv = open(out / 'eval.csv', 'w', newline='')
        self.writer = None

    def _on_training_start(self):
        # 이어서 학습할 때는 단계 수가 0이 아니므로, 첫 평가를 지금부터 eval_every 뒤로 맞춤
        self.next = self.num_timesteps + self.every

    def _on_step(self):
        if self.num_timesteps < self.next:
            return True
        self.next += self.every
        vecnorm = self.model.get_env()
        r, _ = evaluate(self.model, vecnorm, self.env, self.episodes)
        row = {'steps': self.num_timesteps, 'minutes': round((time.time() - self.t0) / 60, 1), **r}
        if self.writer is None:
            self.writer = csv.DictWriter(self.csv, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.csv.flush()
        for k in ('success', 'contact', 'out', 'timeout', 'lateral_mm', 'depth_mm', 'yaw_deg', 'time_s'):
            self.logger.record(f'eval/{k}', r[k])
        mark = ''
        if r['success'] > self.best:
            self.best = r['success']
            self.model.save(self.out / 'best_model')
            vecnorm.save(str(self.out / 'best_vecnormalize.pkl'))
            mark = '  <- best'
        print(f"[{row['minutes']:6.1f} min] {self.num_timesteps:>9,d} steps  success {r['success']:5.1%}  "
              f"contact {r['contact']:5.1%}  out {r['out']:5.1%}  timeout {r['timeout']:5.1%}  "
              f"| ok: lateral {r['lateral_mm']:5.1f} mm  depth {r['depth_mm']:5.1f} mm  "
              f"heading {r['yaw_deg']:4.1f} deg  time {r['time_s']:5.1f} s{mark}", flush=True)
        return True

    def _on_training_end(self):
        self.csv.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='run name -> runs/rl/<name>')
    ap.add_argument('--steps', type=int, default=1_000_000, help='total environment steps (1 step = 0.1 s)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise', choices=['ps', 'none'], default='ps')
    ap.add_argument('--n-envs', type=int, default=os.cpu_count() or 8, help='parallel environments')
    ap.add_argument('--lane-start-prob', type=float, default=0.35,
                    help='training only: share of episodes that start anywhere in front of the slots')
    ap.add_argument('--slot-start-prob', type=float, default=0.30,
                    help='training only: share of episodes that start in front of the target slot, near the final '
                         'heading (practise entering and stopping); the rest start in the entrance corridor')
    ap.add_argument('--resume', default=None,
                    help='continue training from runs/rl/<RESUME>/<--resume-which>_model + _vecnormalize')
    ap.add_argument('--resume-which', choices=['best', 'last'], default='best')
    ap.add_argument('--eval-every', type=int, default=100_000)
    ap.add_argument('--eval-episodes', type=int, default=40, help='multiple of 8 keeps slots/modes balanced')
    ap.add_argument('--eval-only', action='store_true',
                    help='no training: load runs/rl/<name>/<which>_model + _vecnormalize, run the fixed evaluation, '
                         'print every episode and write eval_<which>_episodes.csv')
    ap.add_argument('--which', choices=['best', 'last'], default='best', help='model for --eval-only')
    args = ap.parse_args()

    out = REPO / 'runs/rl' / args.name
    if args.eval_only:
        from stable_baselines3.common.vec_env import DummyVecEnv
        torch.set_num_threads(1)
        model = PPO.load(out / f'{args.which}_model', device='cpu')
        vecnorm = VecNormalize.load(str(out / f'{args.which}_vecnormalize.pkl'),
                                    DummyVecEnv([lambda: ParkEnv(noise=args.noise)]))
        vecnorm.training = False
        t0 = time.time()
        summary, rows = evaluate(model, vecnorm, ParkEnv(noise=args.noise), args.eval_episodes)
        path = out / f'eval_{args.which}_episodes.csv'
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'{args.name} / {args.which}_model, noise={args.noise}, {args.eval_episodes} fixed episodes '
              f'({time.time() - t0:.0f} s)\n')
        report(rows)
        print(f"\nsuccess {summary['success']:.1%}  contact {summary['contact']:.1%}  out {summary['out']:.1%}  "
              f"timeout {summary['timeout']:.1%}  -> {path}")
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / 'args.txt').write_text('\n'.join(f'{k}={v}' for k, v in vars(args).items()) + '\n')
    torch.set_num_threads(1)        # 작은 신경망: 프로세스당 스레드 1개가 가장 빠름

    env = SubprocVecEnv([make_env(i, args.seed, args.noise, args.lane_start_prob, args.slot_start_prob)
                         for i in range(args.n_envs)])
    env = VecMonitor(env)
    if args.resume:
        # 이전 실행의 정규화 통계와 모델을 불러와 이어서 학습 (하이퍼파라미터는 저장된 값 그대로)
        src = REPO / 'runs/rl' / args.resume
        env = VecNormalize.load(str(src / f'{args.resume_which}_vecnormalize.pkl'), env)
        env.training = True
        model = PPO.load(src / f'{args.resume_which}_model', env=env, device='cpu',
                         tensorboard_log=str(out / 'tb'))
        model.set_random_seed(args.seed)
        start_note = f'resume {args.resume}/{args.resume_which}_model at {model.num_timesteps:,} steps'
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)
        model = PPO('MlpPolicy', env, seed=args.seed, device='cpu', verbose=0,
                    n_steps=1024, batch_size=512, n_epochs=10, gamma=0.995, gae_lambda=0.95,
                    learning_rate=3e-4, clip_range=0.2, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                    tensorboard_log=str(out / 'tb'))
        start_note = 'new model'
    print(f'training {args.name}: {args.steps:,} steps ({args.steps * CTRL_DT / 3600:.0f} h of driving), '
          f'{args.n_envs} envs, noise={args.noise}, starts lane {args.lane_start_prob:.0%} / '
          f'slot {args.slot_start_prob:.0%} / corridor {1 - args.lane_start_prob - args.slot_start_prob:.0%}, '
          f'seed={args.seed}, {start_note} -> {out}', flush=True)
    cb = EvalCallback(out, args.eval_every, args.eval_episodes, args.noise)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb, tb_log_name='ppo', progress_bar=False,
                reset_num_timesteps=not args.resume)
    model.save(out / 'last_model')
    env.save(str(out / 'last_vecnormalize.pkl'))
    dt = time.time() - t0
    print(f'done in {dt / 60:.1f} min ({args.steps / dt:.0f} steps/s); best eval success {cb.best:.1%}; saved to {out}')
    env.close()


if __name__ == '__main__':
    main()
