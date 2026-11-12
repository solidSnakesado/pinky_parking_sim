#!/usr/bin/env python3
"""학습한 주차 정책(Stable-Baselines3 PPO)을 numpy 파일 하나로 변환하고, numpy로 계산하는 정책 클래스를 제공한다.

왜 필요한가
  ROS 2 노드나 실물 쪽 PC에서 PyTorch·Stable-Baselines3 없이 정책을 돌리기 위함. 정책 신경망은
  입력 17 -> 128 -> 128 -> 출력 2 (tanh) 이라 numpy 행렬 곱 세 번이면 계산된다.

저장 내용 (policy_<best|last>.npz)
  W0, b0, W1, b1, W2, b2 : 정책 신경망 가중치 (SB3 mlp_extractor.policy_net + action_net)
  obs_mean, obs_var, epsilon, clip_obs : 관측 정규화 통계 (VecNormalize). 빠뜨리면 정책이 엉뚱하게 동작함
  행동 = clip(W2 · tanh(W1 · tanh(W0 · 정규화 관측 + b0) + b1) + b2, -1, 1)  (탐색 노이즈 없는 평균 행동)

사용법 (pinky_train 가상환경):
  python tools/export_policy.py --name bc_ppo_seed0 --which last
출력: runs/rl/<name>/policy_<which>.npz  (변환 후 SB3 출력과 자동 비교)
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


class NumpyPolicy:
    """policy_*.npz를 불러와 관측 17개 -> 행동 2개 ([-1, 1])를 numpy로 계산."""

    def __init__(self, path):
        d = np.load(path)
        self.W = [d['W0'], d['W1'], d['W2']]
        self.b = [d['b0'], d['b1'], d['b2']]
        self.mean, self.var = d['obs_mean'], d['obs_var']
        self.eps, self.clip = float(d['epsilon']), float(d['clip_obs'])

    def normalize(self, obs):
        return np.clip((np.asarray(obs, np.float64) - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)

    def act(self, obs):
        h = self.normalize(obs)
        h = np.tanh(self.W[0] @ h + self.b[0])
        h = np.tanh(self.W[1] @ h + self.b[1])
        return np.clip(self.W[2] @ h + self.b[2], -1.0, 1.0)


def main():
    import torch
    from stable_baselines3 import PPO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='runs/rl/<name>')
    ap.add_argument('--which', choices=['best', 'last'], default='last')
    args = ap.parse_args()
    run = REPO / 'runs/rl' / args.name
    model = PPO.load(run / f'{args.which}_model', device='cpu')
    with open(run / f'{args.which}_vecnormalize.pkl', 'rb') as f:
        vn = pickle.load(f)

    pol = model.policy
    net = [m for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    acts = [m for m in pol.mlp_extractor.policy_net if not isinstance(m, torch.nn.Linear)]
    assert len(net) == 2 and all(isinstance(a, torch.nn.Tanh) for a in acts), 'unexpected policy network'
    t = lambda x: x.detach().cpu().numpy().astype(np.float64)
    out = run / f'policy_{args.which}.npz'
    np.savez(out, W0=t(net[0].weight), b0=t(net[0].bias), W1=t(net[1].weight), b1=t(net[1].bias),
             W2=t(pol.action_net.weight), b2=t(pol.action_net.bias),
             obs_mean=vn.obs_rms.mean.astype(np.float64), obs_var=vn.obs_rms.var.astype(np.float64),
             epsilon=np.float64(vn.epsilon), clip_obs=np.float64(vn.clip_obs))

    # 확인: 관측 분포 근처의 무작위 관측으로 SB3 결정적 행동과 비교
    npol = NumpyPolicy(out)
    rng = np.random.default_rng(0)
    obs = (vn.obs_rms.mean + rng.standard_normal((5000, vn.obs_rms.mean.size)) * np.sqrt(vn.obs_rms.var) * 1.5)
    obs = obs.astype(np.float32)
    ref, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
    mine = np.array([npol.act(o) for o in obs])
    diff = np.abs(ref - mine).max()
    print(f'saved {out}  ({out.stat().st_size / 1024:.0f} KB, trained {model.num_timesteps:,} steps)')
    print(f'check vs SB3 on 5000 observations: max |action difference| = {diff:.2e} -> {"OK" if diff < 1e-4 else "MISMATCH"}')


if __name__ == '__main__':
    main()
