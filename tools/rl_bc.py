#!/usr/bin/env python3
"""모방학습(행동 복제, 케이스 B0): 규칙 제어기 시연을 따라 하는 정책을 만든다.

하는 일
  1. 시연 수집: rl_expert.Expert가 정답 자세를 보고 운전하고, 정책이 보는 관측(P-S 노이즈 포함)과
     행동을 기록 (병렬 프로세스). 출발 구역 비율은 강화학습 학습과 같다. 성공한 판만 쓴다.
  2. 정규화 통계: 시연 관측으로 VecNormalize의 관측 통계를, 시연의 할인 누적 보상으로 보상 통계를 만든다
  3. 행동 복제: rl_train.py와 같은 PPO 신경망(128-128)의 행동 평균을 시연 행동에 맞춘다(MSE).
     가치 함수도 시연의 할인 누적 보상에 맞춰 둔다 (이어서 강화학습할 때 정책이 처음에 망가지지 않게).
     행동 표준편차는 --log-std로 작게 시작 (탐색 노이즈가 시연 동작을 덮지 않게).
  3b. DAgger (--dagger 회): 현재 정책으로 주행하면서(확률 beta로 교사 행동 실행, beta = 0.5^회차)
     지나간 모든 상태에 교사 행동을 정답으로 붙여 데이터에 더하고 다시 학습한다.
     행동 복제만 하면 시연에 없던 상태(조금 어긋난 자세)에서 오차가 쌓이는 문제를 줄인다.
  4. 단계마다(BC, DAgger 회차) 고정 평가 40판 (입구 통로 출발). 가장 좋은 단계를
     runs/rl/<name>/best_model.zip, best_vecnormalize.pkl로 저장 (rl_train.py --resume, --eval-only와 호환)

사용법 (pinky_train 가상환경, ROS 2 + 이 작업 공간 source):
  python tools/rl_bc.py --name bc_seed0                                     # 시연 수집 + 행동 복제 + 평가
  python tools/rl_train.py --name bc_ppo_seed0 --resume bc_seed0 --steps 3000000 --learning-rate 1e-4
출력: runs/rl/<name>/  demos.npz, best_model.zip, best_vecnormalize.pkl (평가 성공률 최고 단계), best_stage.txt,
      last_*, eval.csv (단계별 평가), args.txt
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_park_env import ParkEnv  # noqa: E402
from rl_expert import Expert  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GAMMA = 0.995                      # rl_train.py와 같음


def _collect(job):
    """작업자 하나: 판들을 돌려 (관측, 교사 행동, 할인 누적 보상, 결과)를 반환.
    policy_dir가 없으면 교사가 운전(시연). 있으면 그 정책이 운전하고(확률 beta로 교사 행동 실행)
    모든 상태에 교사 행동을 붙임(DAgger, 실패한 판도 사용)."""
    seed, episodes, noise, lane_p, slot_p, rear_p, policy_dir, beta = job
    torch.set_num_threads(1)
    env = ParkEnv(noise=noise, lane_start_prob=lane_p, slot_start_prob=slot_p, slot_rear_prob=rear_p)
    expert = Expert()
    model = vn = None
    if policy_dir:
        from stable_baselines3 import PPO
        import pickle
        model = PPO.load(Path(policy_dir) / 'last_model', device='cpu')
        with open(Path(policy_dir) / 'last_vecnormalize.pkl', 'rb') as f:
            vn = pickle.load(f)
    rng = np.random.default_rng(seed)
    obs_all, act_all, ret_all, results = [], [], [], []
    for k in range(episodes):
        obs, info = env.reset(seed=seed * 100000 + k)
        expert.reset(env.slot, env.mode, env.physics.pose())
        O, A, R = [], [], []
        done = False
        while not done:
            a = expert.act(env.physics.pose())                  # 교사는 정답 자세를 봄
            O.append(obs)
            A.append(a)
            run = a
            if model is not None and rng.random() >= beta:      # DAgger: 정책 행동으로 주행
                run = model.predict(vn.normalize_obs(obs[None]), deterministic=True)[0][0]
            obs, r, term, trunc, inf = env.step(run)
            R.append(r)
            done = term or trunc
        results.append(inf['result'])
        if model is None and inf['result'] != 'success':
            continue
        G, g = np.zeros(len(R)), 0.0
        for t in range(len(R) - 1, -1, -1):                     # 할인 누적 보상 (가치 함수 목표)
            g = R[t] + GAMMA * g
            G[t] = g
        obs_all.append(np.array(O, np.float32))
        act_all.append(np.array(A, np.float32))
        ret_all.append(G.astype(np.float32))
    cat = (lambda xs, d: np.concatenate(xs) if xs else np.zeros((0, d), np.float32))
    return cat(obs_all, 17), cat(act_all, 2), (np.concatenate(ret_all) if ret_all else np.zeros(0, np.float32)), results


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rl_train import evaluate

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True)
    ap.add_argument('--episodes', type=int, default=1600, help='demonstration episodes (successful ones are kept)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise', choices=['ps', 'none'], default='ps')
    ap.add_argument('--lane-start-prob', type=float, default=0.35)
    ap.add_argument('--slot-start-prob', type=float, default=0.30)
    ap.add_argument('--slot-rear-prob', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--log-std', type=float, default=-1.5, help='initial log std of the action noise (-1.5 -> 0.22)')
    ap.add_argument('--dagger', type=int, default=4, help='DAgger iterations after behaviour cloning (0 = off)')
    ap.add_argument('--dagger-episodes', type=int, default=400, help='policy-driven episodes per DAgger iteration')
    ap.add_argument('--dagger-epochs', type=int, default=15)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 8)
    ap.add_argument('--eval-episodes', type=int, default=40)
    args = ap.parse_args()
    out = REPO / 'runs/rl' / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'args.txt').write_text('\n'.join(f'{k}={v}' for k, v in vars(args).items()) + '\n')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. 시연 수집 (시드는 학습·평가와 겹치지 않게 1000 + seed 기준)
    t0 = time.time()
    per = [args.episodes // args.workers + (i < args.episodes % args.workers) for i in range(args.workers)]
    jobs = [((1000 + args.seed) * 100 + i, n, args.noise, args.lane_start_prob, args.slot_start_prob,
             args.slot_rear_prob, None, 1.0) for i, n in enumerate(per) if n]
    with mp.get_context('spawn').Pool(len(jobs)) as pool:
        parts = pool.map(_collect, jobs)
    obs = np.concatenate([p[0] for p in parts])
    act = np.concatenate([p[1] for p in parts])
    ret = np.concatenate([p[2] for p in parts])
    results = [r for p in parts for r in p[3]]
    n_ok = results.count('success')
    print(f'demonstrations: {len(results)} episodes, expert success {n_ok}/{len(results)} '
          f'({ {k: results.count(k) for k in set(results)} }), kept {len(obs):,} steps ({time.time() - t0:.0f} s)',
          flush=True)

    # 2. 정규화 통계 + PPO 모델 (rl_train.py와 같은 설정)
    venv = VecNormalize(DummyVecEnv([lambda: ParkEnv(noise=args.noise)]), norm_obs=True, norm_reward=True,
                        clip_obs=10.0, gamma=GAMMA)
    venv.obs_rms.mean, venv.obs_rms.var, venv.obs_rms.count = obs.mean(0), obs.var(0), float(len(obs))
    venv.ret_rms.mean, venv.ret_rms.var, venv.ret_rms.count = float(ret.mean()), float(ret.var()), float(len(ret))
    model = PPO('MlpPolicy', venv, seed=args.seed, device='cpu', verbose=0,
                n_steps=1024, batch_size=512, n_epochs=10, gamma=GAMMA, gae_lambda=0.95,
                learning_rate=3e-4, clip_range=0.2, ent_coef=0.0,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
    pol = model.policy
    with torch.no_grad():
        pol.log_std.fill_(args.log_std)

    # 3. 행동 복제 + 가치 함수 맞춤
    params = [p for n, p in pol.named_parameters() if n != 'log_std']
    opt = torch.optim.Adam(params, lr=args.lr)
    v_scale = np.sqrt(venv.ret_rms.var + venv.epsilon)

    def fit(obs, act, ret, has_ret, epochs):
        X = torch.as_tensor(venv.normalize_obs(obs), dtype=torch.float32)
        Y = torch.as_tensor(act, dtype=torch.float32)
        V = torch.as_tensor(np.clip(ret / v_scale, -venv.clip_reward, venv.clip_reward), dtype=torch.float32)
        M = torch.as_tensor(has_ret, dtype=torch.float32)          # 가치 목표는 교사가 운전한 판에만 있음
        n = len(X)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot_a = tot_v = 0.0
            for i in range(0, n, args.batch):
                idx = perm[i:i + args.batch]
                mean = pol.get_distribution(X[idx]).distribution.mean
                loss_a = ((mean - Y[idx]) ** 2).mean()
                err_v = (pol.predict_values(X[idx]).squeeze(-1) - V[idx]) ** 2
                loss_v = (err_v * M[idx]).sum() / M[idx].sum().clamp(min=1.0)
                opt.zero_grad()
                (loss_a + 0.5 * loss_v).backward()
                opt.step()
                tot_a += loss_a.item() * len(idx)
                tot_v += loss_v.item() * len(idx)
            if ep == 0 or (ep + 1) % 5 == 0:
                print(f'  epoch {ep + 1:3d}/{epochs}: action MSE {tot_a / n:.4f}  value MSE {tot_v / n:.4f}', flush=True)

    best = {'success': -1.0}

    def save_and_eval(label, csv_writer):
        """last_*는 매번 저장(DAgger 주행에 사용), best_*는 평가 성공률이 가장 높을 때만 저장."""
        venv.training = False
        model.save(out / 'last_model')
        venv.save(str(out / 'last_vecnormalize.pkl'))
        r, _ = evaluate(model, venv, ParkEnv(noise=args.noise), args.eval_episodes)
        csv_writer.writerow({'steps': 0, 'minutes': round((time.time() - t_start) / 60, 1), 'stage': label, **r})
        mark = ''
        if r['success'] > best['success']:
            best['success'] = r['success']
            model.save(out / 'best_model')
            venv.save(str(out / 'best_vecnormalize.pkl'))
            (out / 'best_stage.txt').write_text(label + '\n')
            mark = '  <- best'
        print(f"{label}: success {r['success']:.1%} (F {r['success_front']:.0%} R {r['success_rear']:.0%})  "
              f"contact {r['contact']:.1%}  out {r['out']:.1%}  timeout {r['timeout']:.1%}{mark}", flush=True)
        return r

    t_start = time.time()
    data = [obs, act, ret, np.ones(len(obs), np.float32)]
    print(f'behaviour cloning on {len(obs):,} steps', flush=True)
    fit(*data, args.epochs)
    f = open(out / 'eval.csv', 'w', newline='')

    class Writer:
        def __init__(self):
            self.w = None

        def writerow(self, row):
            if self.w is None:
                self.w = csv.DictWriter(f, fieldnames=list(row))
                self.w.writeheader()
            self.w.writerow(row)
            f.flush()
    wr = Writer()
    save_and_eval('BC', wr)

    # 3b. DAgger
    for it in range(1, args.dagger + 1):
        beta = 0.5 ** it
        t0 = time.time()
        per = [args.dagger_episodes // args.workers + (i < args.dagger_episodes % args.workers)
               for i in range(args.workers)]
        jobs = [((2000 + args.seed) * 1000 + it * 100 + i, n, args.noise, args.lane_start_prob,
                 args.slot_start_prob, args.slot_rear_prob, str(out), beta) for i, n in enumerate(per) if n]
        with mp.get_context('spawn').Pool(len(jobs)) as pool:
            parts = pool.map(_collect, jobs)
        o = np.concatenate([p[0] for p in parts])
        a = np.concatenate([p[1] for p in parts])
        res = [r for p in parts for r in p[3]]
        data[0] = np.concatenate([data[0], o])
        data[1] = np.concatenate([data[1], a])
        data[2] = np.concatenate([data[2], np.zeros(len(o), np.float32)])
        data[3] = np.concatenate([data[3], np.zeros(len(o), np.float32)])
        print(f'DAgger {it}/{args.dagger}: beta {beta:.2f}, {len(res)} episodes driven '
              f'({ {k: res.count(k) for k in set(res)} }), +{len(o):,} labelled steps -> {len(data[0]):,} '
              f'({time.time() - t0:.0f} s)', flush=True)
        fit(*data, args.dagger_epochs)
        save_and_eval(f'BC+DAgger{it}', wr)
    f.close()
    np.savez_compressed(out / 'demos.npz', obs=data[0], act=data[1], ret=data[2], has_ret=data[3])
    print(f'done ({(time.time() - t_start) / 60:.1f} min) -> {out}')


if __name__ == '__main__':
    main()
