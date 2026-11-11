#!/usr/bin/env python3
"""주차칸까지 주행을 학습하기 위한 Gymnasium 환경 (강화학습, R1 단계).

무엇인가
  pinky_pro의 MuJoCo 물리 모델(pinky_mujoco.model.Physics: ROS 2 브리지와 같은 로봇, 바퀴 모터,
  가속 제한)을 Gymnasium 환경으로 감싼 것. 실행 중에 ROS 2와 렌더링을 쓰지 않는다. 정책은 숫자만
  보므로 학습이 실제 시간보다 훨씬 빠르다.
  pinky_pro는 수정하지 않는다: 장면은 시뮬이 만든 것(~/.cache/pinky_mujoco/scene.xml)을 쓴다.

한 판(에피소드)
  1. 출발 구역의 무작위 자세에 로봇을 놓고, 칸(A~D)과 방식(전면 / 후면)을 고른다.
     출발 구역: 'corridor' = 입구 통로 (park_closed_loop.py와 같음, 평가에 사용),
                'lane' = 칸 앞 통로 어디든 (학습 전용: 마지막 동작을 연습).
                'slot' = 칸 앞 대기점~칸 입구 사이, 목표 방향에 가까운 자세 (학습 전용: 진입·정지 연습).
     학습 때는 세 구역을 섞는다 (lane_start_prob, slot_start_prob, 나머지는 corridor).
     방식은 반반이지만, 'slot' 출발은 slot_rear_prob 비율로 후면 (후진 진입을 더 자주 연습)
  2. 0.1초(10 Hz)마다 정책이 (직진, 회전) 속도를 내고, 물리를 10 ms씩 10번 진행한다
  3. 성공, 테이프 접촉, 바닥 판 이탈, 시간 초과 중 하나가 되면 판이 끝난다

관측 (정책이 보는 값, 자세 값에는 인식 노이즈가 들어감)
  추정 자세 (x, y, cos yaw, sin yaw), 로봇 기준 목표 위치 (dx, dy), 방향 오차 (cos, sin),
  방식 (+1 전면 / -1 후면), 칸 one-hot (4개), 바퀴 오도메트리로 잰 속도 (v, w), 직전 행동
  노이즈: 'ps'   = P-S 천장 인식 test 오차(results/ps_test.csv)에서 매 단계 하나를 뽑아 더함
          'none' = 정답 그대로
행동
  [-1, 1]^2 를 V_MAX, W_MAX로 곱해 씀 (park_closed_loop.py의 FSM 제어기와 같은 한계)
보상 (정답 자세 사용, 학습에만 씀)
  경로(입구 통로 -> 입구 틈 -> 대기점(칸 u, v 420) -> 칸 목표)를 따라 나아간 만큼
  (route_remaining()의 퍼텐셜 차이, FSM과 같은 경유점), 그리고 최종 방향. 방향 항의 가중치는
  대기점 0.15 m 전부터 커져서 대기점과 칸 진입 구간에서 최대가 된다.
  - 매 단계 작은 시간 비용, - 테이프 접촉 / 이탈 (판 종료), + 성공 (판 종료)
성공 (park_closed_loop.py와 같은 기준): 테이프 접촉 없음, 좌우·깊이 <= 20 mm,
  방향 <= 5°, 정지

사용법 (pinky_train 가상환경, pinky_mujoco를 불러오려면 ROS 2 + 이 작업 공간 source 필요):
  python tools/rl_park_env.py --check                 # 형식 검사 + 무작위 행동 주행 + 속도 측정
  python tools/rl_park_env.py --check --noise none
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402
import gen_parking_dataset as D  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCENE = Path.home() / '.cache/pinky_mujoco/scene.xml'

CTRL_DT = 0.1                      # 정책 한 단계 (10 Hz)
SUB_STEPS = 10                     # 정책 한 단계당 physics.step() 호출 수 (각 10 ms), 호출마다 접촉 확인
V_MAX, W_MAX = 0.10, 1.2           # park_closed_loop.py와 같음 (브리지 한계 0.25 m/s, 1.5 rad/s 안쪽)
TIME_LIMIT_S = 60.0
PARK_V = {'front': 102.5, 'rear': 136.5}   # 몸체가 칸 가운데 오게 하는 바퀴 축 v (park_closed_loop.py)
OK_LATERAL_MM, OK_DEPTH_MM, OK_YAW_DEG = 20.0, 20.0, 5.0
STOP_V, STOP_W = 0.005, 0.05       # 성공 판정의 "정지" 기준 (m/s, rad/s)
LOT_U, LOT_V = (0.0, 1294.0), (0.0, 1200.0)   # 바닥 판이 덮는 범위 (mm)

# 보상 가중치
K_DIST = 10.0                      # 남은 "경로" 거리 1 m당 퍼텐셜 (route_remaining 참고)
K_YAW = 4.0                        # 최종 방향 오차(rad / π)의 퍼텐셜, 대기점부터 최대 가중치
YAW_RAMP_M = 0.15                  # 대기점 전 이 경로 거리 동안 방향 가중치가 0 -> 1로 커짐
TIME_COST = 0.005

# 보상에 쓰는 경로 (바닥 mm, park_closed_loop.py의 FSM과 같은 경유점)
GAP = (346.0, 590.0)               # 입구 틈의 중심 (테이프 끝 u 195와 497, 테이프 줄 v 581..600)
STAGING_V = 420.0                  # 각 칸 앞의 대기점: (칸 u, 420)
SLOT_AXIS_K = 3.0                  # 칸 진입 항에서 칸 축에서 벗어난 1 mm를 거리 3 mm로 계산
CORRIDOR_V = 590.0                 # v > 이 값: 입구 통로 (틈 아래쪽), 그 외: 칸 앞 통로
# 출발 구역 (바닥 mm): corridor = park_closed_loop.py와 같음, lane = 칸 앞 통로 어디든
START_AREAS = {'corridor': ((300, 440), (690, 870)), 'lane': ((60, 840), (260, 540))}
# 'slot' 출발 구역: 칸 축에서 좌우 ±SLOT_START_U mm, 바퀴 축 v는 (목표 v + 40) ~ 대기점 v,
# 방향은 목표 방향 ±SLOT_START_DEG° (전면 = 칸을 향함, 후면 = 칸을 등짐). 테이프에 닿는 자세는 다시 뽑음
SLOT_START_U, SLOT_START_V_MARGIN, SLOT_START_DEG = 30.0, 40.0, 20.0
R_SUCCESS, R_CONTACT, R_OUT = 20.0, -10.0, -10.0


# ---------------------------------------------------------------- 테이프 접촉 (바닥 mm, 정답 자세)
_BOXES = [b for b in G.ROBOT_BOXES if b[4] < 0.1]           # 몸통 + 바퀴 (라이다는 몸통 안쪽)
_CORNERS = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) for x0, x1, y0, y1, _, _ in _BOXES]
_TAPES = np.array(G.TAPES, dtype=float)                     # (u, v, w, h)


def footprint(x, y, yaw):
    """로봇 외곽 다각형 (몸통, 바퀴), 바닥 mm."""
    R = G.rot(yaw)
    out = []
    for c in _CORNERS:
        p = c @ R.T + [x, y]
        out.append(np.column_stack([G.START_U - p[:, 1] * 1000, G.START_V - p[:, 0] * 1000]))
    return out


def _overlaps(poly, rect):
    u, v, w, h = rect
    r = np.array([[u, v], [u + w, v], [u + w, v + h], [u, v + h]])
    for shape in (poly, r):
        for i in range(len(shape)):
            e = shape[(i + 1) % len(shape)] - shape[i]
            n = np.array([-e[1], e[0]])
            a, b = poly @ n, r @ n
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def touching(x, y, yaw):
    """로봇 외곽이 테이프와 겹치면 True (외접 사각형으로 먼저 거른 뒤 분리축 검사)."""
    polys = footprint(x, y, yaw)
    allp = np.vstack(polys)
    lo, hi = allp.min(0), allp.max(0)
    near = ((_TAPES[:, 0] <= hi[0]) & (_TAPES[:, 0] + _TAPES[:, 2] >= lo[0]) &
            (_TAPES[:, 1] <= hi[1]) & (_TAPES[:, 1] + _TAPES[:, 3] >= lo[1]))
    for rect in _TAPES[near]:
        if any(_overlaps(p, rect) for p in polys):
            return True
    return False


def route_remaining(x, y, slot, mode):
    """경로 (입구 통로 -> 입구 틈 -> 대기점 -> 칸 목표)를 따라 남은 거리 (m).
    입구 통로 : |p - 틈| + |틈 - 대기점| + |대기점 - 목표|
    칸 앞 통로 : min( |p - 대기점| + |대기점 - 목표| ,  |p - 목표| + 3 x |칸 축에서 벗어난 거리| )
    로봇이 칸 축 위에 오면 min()이 칸으로 곧장 들어가는 항으로 부드럽게 바뀐다. 두 항은 대기점에서
    같으므로 값이 대기점과 틈 중심에서 끊기지 않는다."""
    u, v = G.world_to_mm(x, y)
    su, tv = D.SLOT_U[slot], PARK_V[mode]
    s_t = abs(STAGING_V - tv)
    to_s = math.hypot(u - su, v - STAGING_V)
    if v > CORRIDOR_V:
        mm = math.hypot(u - GAP[0], v - GAP[1]) + math.hypot(GAP[0] - su, GAP[1] - STAGING_V) + s_t
    else:
        mm = min(to_s + s_t, math.hypot(u - su, v - tv) + SLOT_AXIS_K * abs(u - su))
    return mm / 1000.0


def sample_start(rng, area='corridor', slot=0, mode='front'):
    """출발 구역 안의 무작위 자세.
    corridor = 입구 통로 (park_closed_loop.py와 같음), lane = 칸 앞 통로: 제자리 회전해도 테이프에 닿지 않는 곳.
    slot = 목표 칸 앞 (대기점 ~ 칸 입구, 일부는 칸 안), 목표 방향에 가까운 자세: 그 자세로 테이프에 닿지 않는 곳."""
    if area == 'slot':
        tyaw = 0.0 if mode == 'front' else math.pi
        for _ in range(1000):
            u = D.SLOT_U[slot] + rng.uniform(-SLOT_START_U, SLOT_START_U)
            v = rng.uniform(PARK_V[mode] + SLOT_START_V_MARGIN, STAGING_V)
            yaw = G.wrap(tyaw + math.radians(rng.uniform(-SLOT_START_DEG, SLOT_START_DEG)))
            x, y = G.mm_to_world(u, v)
            if not touching(x, y, yaw):
                return float(x), float(y), float(yaw)
        raise RuntimeError('no free slot start pose')
    (u0, u1), (v0, v1) = START_AREAS[area]
    for _ in range(1000):
        u, v, yaw = rng.uniform(u0, u1), rng.uniform(v0, v1), rng.uniform(-math.pi, math.pi)
        x, y = G.mm_to_world(u, v)
        if all(not touching(x, y, a) for a in np.radians(np.arange(0, 360, 10))):
            return float(x), float(y), float(yaw)
    raise RuntimeError('no free start pose')


def target_pose(slot, mode):
    x, y = G.mm_to_world(D.SLOT_U[slot], PARK_V[mode])
    return float(x), float(y), (0.0 if mode == 'front' else math.pi)


def slot_errors(pose, slot, mode):
    """칸 목표 대비 (좌우 mm, 깊이 mm, 방향 deg), 모두 절댓값."""
    tx, ty, tyaw = target_pose(slot, mode)
    return (abs(pose[1] - ty) * 1000, abs(tx - pose[0]) * 1000, abs(math.degrees(G.wrap(pose[2] - tyaw))))


# ---------------------------------------------------------------- 인식 노이즈
def load_ps_errors(path=REPO / 'results/ps_test.csv'):
    """P-S가 로봇을 찾은 test 이미지마다 (dx, dy, dyaw) = 추정 - 정답.
    앞/뒤 뒤집힘(|dyaw| > 90°)은 되돌림: 필터가 오도메트리로 앞/뒤를 유지하기 때문."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['found'] != 'True':
                continue
            dyaw = G.wrap(float(r['est_yaw']) - float(r['gt_yaw']))
            if abs(dyaw) > math.pi / 2:
                dyaw = G.wrap(dyaw + math.pi)
            rows.append((float(r['est_x']) - float(r['gt_x']), float(r['est_y']) - float(r['gt_y']), dyaw))
    return np.array(rows)


# ---------------------------------------------------------------- 환경
class ParkEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, noise='ps', slot=None, mode=None, scene=SCENE, contact_ends=True, lane_start_prob=0.5,
                 slot_start_prob=0.0, slot_rear_prob=0.5):
        from pinky_mujoco.model import Physics
        self.physics = Physics(scene)
        self.base_z = float(self.physics.model.qpos0[2])
        self.noise = noise
        self.errors = load_ps_errors() if noise == 'ps' else None
        self.fixed_slot, self.fixed_mode, self.contact_ends = slot, mode, contact_ends
        self.lane_start_prob, self.slot_start_prob = lane_start_prob, slot_start_prob
        self.slot_rear_prob = slot_rear_prob
        self.max_steps = int(round(TIME_LIMIT_S / CTRL_DT))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)

    # -- 보조 함수
    def _set_pose(self, x, y, yaw):
        m, d = self.physics.model, self.physics.data
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [x, y, self.base_z]
        d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        self.physics.odom[:] = 0.0
        self.physics.twist[:] = 0.0
        mujoco.mj_forward(m, d)

    def _measured_pose(self, pose):
        if self.errors is None:
            return pose
        e = self.errors[self.np_random.integers(len(self.errors))]
        return np.array([pose[0] + e[0], pose[1] + e[1], G.wrap(pose[2] + e[2])])

    def _potential(self, pose):
        """- 남은 경로 거리, - 대기점에 (거의) 도착한 뒤의 최종 방향 오차."""
        r = route_remaining(pose[0], pose[1], self.slot, self.mode)
        weight = float(np.clip(1.0 - (r - self.staging_to_target) / YAW_RAMP_M, 0.0, 1.0))
        return -K_DIST * r - K_YAW * weight * abs(G.wrap(self.target[2] - pose[2])) / math.pi

    def _obs(self, pose):
        est = self._measured_pose(pose)
        tx, ty, tyaw = self.target
        c, s = math.cos(est[2]), math.sin(est[2])
        dx, dy = tx - est[0], ty - est[1]
        e = G.wrap(tyaw - est[2])
        slot = np.zeros(4)
        slot[self.slot] = 1.0
        v, w = self.physics.twist
        return np.concatenate([[est[0], est[1], c, s, c * dx + s * dy, -s * dx + c * dy, math.cos(e), math.sin(e),
                                1.0 if self.mode == 'front' else -1.0], slot,
                               [v / V_MAX, w / W_MAX], self.prev_action]).astype(np.float32)

    # -- gym 인터페이스
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.slot = options.get('slot', self.fixed_slot)
        self.slot = int(self.np_random.integers(4)) if self.slot is None else int(self.slot)
        area = options.get('area')
        if area is None:
            r = self.np_random.random()
            area = ('slot' if r < self.slot_start_prob else
                    'lane' if r < self.slot_start_prob + self.lane_start_prob else 'corridor')
        # 방식: 지정이 없으면 전면/후면 반반. 단 'slot' 출발은 slot_rear_prob 비율로 후면 (후진 진입 연습)
        rear_prob = self.slot_rear_prob if area == 'slot' else 0.5
        self.mode = options.get('mode', self.fixed_mode) or ('rear' if self.np_random.random() < rear_prob else 'front')
        start = options.get('start') or sample_start(self.np_random, area, self.slot, self.mode)
        self.target = target_pose(self.slot, self.mode)
        self.staging_to_target = abs(STAGING_V - PARK_V[self.mode]) / 1000.0
        self._set_pose(*start)
        self.steps, self.prev_action = 0, np.zeros(2)
        pose = self.physics.pose()
        self.phi = self._potential(pose)
        return self._obs(pose), {'slot': 'ABCD'[self.slot], 'mode': self.mode, 'start': start, 'area': area}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        v, w = a[0] * V_MAX, a[1] * W_MAX
        contact = False
        for _ in range(SUB_STEPS):
            self.physics.step(v, w)
            if touching(*self.physics.pose()):
                contact = True
                if self.contact_ends:
                    break
        self.steps += 1
        self.prev_action = a
        pose = self.physics.pose()
        phi = self._potential(pose)
        reward = phi - self.phi - TIME_COST
        self.phi = phi

        u, vv = G.world_to_mm(pose[0], pose[1])
        out = not (LOT_U[0] <= u <= LOT_U[1] and LOT_V[0] <= vv <= LOT_V[1])
        lat, depth, yaw = slot_errors(pose, self.slot, self.mode)
        mv, mw = self.physics.twist
        stopped = abs(mv) < STOP_V and abs(mw) < STOP_W
        success = (not contact and lat <= OK_LATERAL_MM and depth <= OK_DEPTH_MM and yaw <= OK_YAW_DEG
                   and stopped)

        terminated, result = False, None
        if contact and self.contact_ends:
            reward += R_CONTACT
            terminated, result = True, 'contact'
        elif out:
            reward += R_OUT
            terminated, result = True, 'out'
        elif success:
            reward += R_SUCCESS
            terminated, result = True, 'success'
        truncated = not terminated and self.steps >= self.max_steps
        if truncated:
            result = 'timeout'
        info = {'result': result, 'lateral_mm': lat, 'depth_mm': depth, 'yaw_deg': yaw, 'contact': contact,
                'time_s': self.steps * CTRL_DT}
        return self._obs(pose), float(reward), terminated, truncated, info


# ---------------------------------------------------------------- 확인
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='API check + random-action episodes + speed')
    ap.add_argument('--noise', choices=['ps', 'none'], default='ps')
    ap.add_argument('--episodes', type=int, default=20)
    args = ap.parse_args()
    if not args.check:
        ap.print_help()
        return

    from gymnasium.utils.env_checker import check_env
    env = ParkEnv(noise=args.noise)
    check_env(env, skip_render_check=True)
    print('gymnasium check_env: OK')
    if env.errors is not None:
        e = env.errors
        print(f'noise from P-S test: {len(e)} samples, position error median '
              f'{np.median(np.hypot(e[:, 0], e[:, 1])) * 1000:.2f} mm, heading median {np.degrees(np.median(np.abs(e[:, 2]))):.2f} deg')

    rng = np.random.default_rng(0)
    results, steps, t0 = {}, 0, time.time()
    for k in range(args.episodes):
        obs, info = env.reset(seed=k, options={'area': ('corridor', 'lane', 'slot')[k % 3]})
        ret, done = 0.0, False
        while not done:
            obs, r, term, trunc, inf = env.step(rng.uniform(-1, 1, 2))
            ret += r
            steps += 1
            done = term or trunc
        results[inf['result']] = results.get(inf['result'], 0) + 1
        print(f"episode {k:2d}: {info['area']:8s} slot {info['slot']} {info['mode']:5s} -> {inf['result']:8s} "
              f"t={inf['time_s']:5.1f} s  return={ret:7.2f}")
    dt = time.time() - t0
    print(f'random policy: {results}')
    print(f'speed: {steps} steps in {dt:.1f} s = {steps / dt:.0f} steps/s per process '
          f'({steps * CTRL_DT / dt:.0f}x real time)')


if __name__ == '__main__':
    main()
