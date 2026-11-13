#!/usr/bin/env python3
"""ROS 2 주차 정책 노드: 자세 -> 관측 17개 -> 정책 -> /cmd_vel.

자세 입력 두 가지
  - 기본: /ground_truth (+ 학습 때와 같은 P-S 노이즈). 시뮬 검증용
  - --est-topic /perceived_pose: 인식 노드(pose_perception_node.py)가 낸 추정 자세를 관측에 씀 (노이즈 없음,
    실제 인식 오차가 이미 들어 있음). 판정(접촉, 성공)은 시뮬에서는 계속 /ground_truth로 함

무엇인가
  학습한 정책(export_policy.py로 만든 policy_*.npz)을 ROS 2로 돌린다. 관측은 학습 환경과 같은 함수
  (rl_park_env.make_obs), 판정도 같은 함수(touching, slot_errors)를 쓴다.
  실물 핑키도 같은 /cmd_vel을 받으므로, 자세 입력만 인식 노드로 바꾸면 실물에서도 그대로 쓴다.

한 판의 단계 (ParkEpisode, ROS와 무관한 로직)
  0. wait   : (추정 자세 사용 시) 인식 노드의 첫 추정 자세가 올 때까지 정지 (필터는 정지 상태에서 시작해야 함)
  1. turn   : 출발 방향(start_yaw)까지 제자리 회전 (브리지는 출발 위치만 정할 수 있어서)
  2. policy : 시뮬 시간 0.1초마다 관측 -> 정책 -> 명령 (그 사이에는 같은 명령 유지)
              (추정 자세 사용 시) 추정 자세의 나이(지금 시각 - 추정 자세 시각)가 stale_s보다 크면 정책 대신
              정지 명령 (0, 0)을 내고 그 횟수를 기록한다 (오래된 자세 감시. 인식 노드·통신이 멈췄을 때 보호)
  3. 끝     : 성공(정지 + 좌우·깊이 <= 20 mm, 방향 <= 5°), 테이프 접촉, 이탈, 60초 초과
  자세가 들어올 때마다(50 Hz) 접촉을 확인하고, 성공·이탈은 0.1초 경계에서 확인 (학습 환경과 같은 순서)

토픽
  구독: /ground_truth (geometry_msgs/PoseStamped, 시뮬 시간 도장), /odom (nav_msgs/Odometry, 현재 속도),
        (선택) 추정 자세 토픽 (geometry_msgs/PoseStamped, 영상 시각 도장)
  발행: /cmd_vel (geometry_msgs/Twist), 자세가 들어올 때마다 (브리지 명령 끊김 0.5초 방지)

사용법 (pinky_train 가상환경, ROS 2 + 이 작업 공간 source, 브리지가 실행 중일 때):
  python tools/park_policy_node.py --policy runs/rl/bc_ppo_seed0/policy_last.npz --slot B --mode rear
  python tools/park_policy_node.py --policy ... --slot B --mode rear --est-topic /perceived_pose   # 인식 자세 사용
여러 판 평가는 ros_park_eval.py가 이 파일의 ParkEpisode / ParkPolicyNode를 불러 쓴다.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402
from rl_park_env import (make_obs, target_pose, touching, slot_errors, load_ps_errors,  # noqa: E402
                         CTRL_DT, V_MAX, W_MAX, TIME_LIMIT_S, OK_LATERAL_MM, OK_DEPTH_MM, OK_YAW_DEG,
                         STOP_V, STOP_W, LOT_U, LOT_V)
from export_policy import NumpyPolicy  # noqa: E402

TURN_TOL = math.radians(2.0)       # 출발 방향 맞추기 허용 오차
TURN_SETTLE_W = 0.05               # 회전이 멈췄다고 볼 각속도 (rad/s)
STALE_S = 0.2                      # 추정 자세가 이보다 오래되면 정지 (오래된 자세 감시, 초)


class ParkEpisode:
    """한 판의 주행 로직 (ROS와 무관). on_state(시뮬 시간, 정답 자세, 오도메트리 속도) -> (v, w)."""

    def __init__(self, policy, slot, mode, start_yaw=None, noise='ps', seed=0, use_est=False, stale_s=STALE_S):
        self.policy, self.slot, self.mode = policy, slot, mode
        self.start_yaw = start_yaw
        self.target = target_pose(slot, mode)
        self.use_est = use_est                    # True: 관측에 추정 자세 사용 (노이즈를 더하지 않음)
        self.errors = load_ps_errors() if noise == 'ps' and not use_est else None
        self.rng = np.random.default_rng(seed)
        self.phase = 'wait' if use_est else ('turn' if start_yaw is not None else 'policy')
        self.est_err_mm, self.est_err_deg, self.est_age_ms = [], [], []   # 정책이 쓴 추정 자세의 오차·나이
        self.stale_s = stale_s                    # 오래된 자세 기준 (초, 0 이하면 감시 끔)
        self.stale_steps = 0                      # 오래된 자세 때문에 정지한 정책 단계 수 (0.1초 단위)
        self.stale_hits = 0                       # 오래된 자세를 만난 자세 메시지 수 (50 Hz 단위)
        self.stale_max_ms = 0.0                   # 정책 단계에서 본 추정 자세 나이의 최대값 (ms)
        self.t0 = self.next_act = None
        self.prev_action = np.zeros(2)
        self.command = (0.0, 0.0)
        self.result, self.info = None, {}
        self.min_dist = float('inf')
        self.steps = 0

    def _measured(self, pose):
        if self.errors is None:
            return pose
        e = self.errors[self.rng.integers(len(self.errors))]
        return np.array([pose[0] + e[0], pose[1] + e[1], G.wrap(pose[2] + e[2])])

    def _finish(self, result, t, pose, twist):
        lat, depth, yaw = slot_errors(pose, self.slot, self.mode)
        tx, ty, _ = self.target
        self.result = result
        self.info = {'result': result, 'time_s': round(self.steps * CTRL_DT, 3), 'lateral_mm': lat,
                     'depth_mm': depth, 'yaw_deg': yaw, 'dist_mm': math.hypot(tx - pose[0], ty - pose[1]) * 1000,
                     'depth_signed_mm': (tx - pose[0]) * 1000, 'min_dist_mm': self.min_dist,
                     'stopped': bool(abs(twist[0]) < STOP_V and abs(twist[1]) < STOP_W),
                     'final_v': float(twist[0]), 'final_w': float(twist[1])}
        self.command = (0.0, 0.0)

    def on_state(self, t, pose, twist, est=None, est_t=None):
        """pose = 정답 자세 (판정용), est = 추정 자세 (use_est일 때 관측용), est_t = 추정 자세의 시각."""
        if self.result is not None:
            return 0.0, 0.0
        pose = np.asarray(pose, float)
        if self.phase == 'wait':
            if est is None:
                return 0.0, 0.0
            self.phase = 'turn' if self.start_yaw is not None else 'policy'
            if self.phase == 'policy':
                self.t0 = self.next_act = t
        if self.phase == 'turn':
            err = G.wrap(self.start_yaw - pose[2])
            if abs(err) < TURN_TOL and abs(twist[1]) < TURN_SETTLE_W:
                self.phase, self.t0, self.next_act = 'policy', t, t
            else:
                return 0.0, float(np.clip(3.0 * err, -W_MAX, W_MAX))
        tx, ty, _ = self.target
        self.min_dist = min(self.min_dist, math.hypot(tx - pose[0], ty - pose[1]) * 1000)
        if touching(*pose):                                   # 접촉은 자세가 올 때마다 확인
            self._finish('contact', t, pose, twist)
            return self.command
        stale = (self.use_est and self.stale_s > 0 and est_t is not None and t - est_t > self.stale_s)
        if stale:                                             # 오래된 자세: 0.1초 경계를 기다리지 않고 바로 정지
            self.stale_hits += 1
            self.stale_max_ms = max(self.stale_max_ms, (t - est_t) * 1000)
            self.command = (0.0, 0.0)
        if t + 1e-6 >= self.next_act:                         # 0.1초 경계
            if self.steps > 0:
                u, v = G.world_to_mm(pose[0], pose[1])
                lat, depth, yaw = slot_errors(pose, self.slot, self.mode)
                stopped = abs(twist[0]) < STOP_V and abs(twist[1]) < STOP_W
                if not (LOT_U[0] <= u <= LOT_U[1] and LOT_V[0] <= v <= LOT_V[1]):
                    self._finish('out', t, pose, twist)
                    return self.command
                if lat <= OK_LATERAL_MM and depth <= OK_DEPTH_MM and yaw <= OK_YAW_DEG and stopped:
                    self._finish('success', t, pose, twist)
                    return self.command
                if self.steps * CTRL_DT >= TIME_LIMIT_S - 1e-9:
                    self._finish('timeout', t, pose, twist)
                    return self.command
            if stale:                                         # 정책을 돌리지 않고 정지 명령 유지 (시간은 계속 셈)
                self.prev_action = np.zeros(2)                # 실제로 낸 명령 (0, 0)을 이전 행동으로
                self.next_act += CTRL_DT
                self.steps += 1
                self.stale_steps += 1
                return self.command
            if self.use_est:
                seen = np.asarray(est, float)
                self.est_err_mm.append(math.hypot(seen[0] - pose[0], seen[1] - pose[1]) * 1000)
                self.est_err_deg.append(abs(math.degrees(G.wrap(seen[2] - pose[2]))))
                if est_t is not None:
                    self.est_age_ms.append((t - est_t) * 1000)
            else:
                seen = self._measured(pose)
            obs = make_obs(seen, self.target, self.slot, self.mode, twist, self.prev_action)
            a = self.policy.act(obs)
            self.prev_action = a
            self.command = (float(a[0] * V_MAX), float(a[1] * W_MAX))
            self.next_act += CTRL_DT
            self.steps += 1
        return self.command


def make_node(episode, name='park_policy', est_topic=None):
    """ParkEpisode를 ROS 2 노드로 감싼다 (rclpy.init() 이후 호출). est_topic: 추정 자세 토픽 (없으면 정답만)."""
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry

    class ParkPolicyNode(Node):
        def __init__(self):
            super().__init__(name)
            self.episode = episode
            self.twist = (0.0, 0.0)
            self.got_odom = False
            self.first_pose = None
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.create_subscription(Odometry, '/odom', self.on_odom, 10)
            self.create_subscription(PoseStamped, '/ground_truth', self.on_truth, 10)
            self.est, self.est_t = None, None
            if est_topic:
                self.create_subscription(PoseStamped, est_topic, self.on_est, 10)
            self.compute_ms, self.pub_wall = [], []

        @staticmethod
        def _pose(msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            q = msg.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            return t, (msg.pose.position.x, msg.pose.position.y, yaw)

        def on_est(self, msg):
            self.est_t, self.est = self._pose(msg)

        def on_odom(self, msg):
            self.twist = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
            self.got_odom = True

        def on_truth(self, msg):
            if not self.got_odom:
                return
            c0 = time.perf_counter()
            t, pose = self._pose(msg)
            if self.first_pose is None:
                self.first_pose = pose
            v, w = self.episode.on_state(t, pose, self.twist, self.est, self.est_t)
            cmd = Twist()
            cmd.linear.x, cmd.angular.z = float(v), float(w)
            self.pub.publish(cmd)
            self.compute_ms.append((time.perf_counter() - c0) * 1000)
            self.pub_wall.append(time.monotonic())

        def stop(self):
            for _ in range(3):
                self.pub.publish(Twist())

        def timing(self):
            """명령 발행 간격 (실제 시간, ms)과 노드 처리 시간 (ms) 통계."""
            gaps = np.diff(self.pub_wall) * 1000 if len(self.pub_wall) > 1 else np.zeros(1)
            comp = np.array(self.compute_ms or [0.0])
            out = {'cmd_gap_mean_ms': float(gaps.mean()), 'cmd_gap_max_ms': float(gaps.max()),
                   'compute_mean_ms': float(comp.mean()), 'compute_max_ms': float(comp.max())}
            ep = self.episode
            if ep.est_err_mm:                         # 추정 자세 사용 시: 정책이 본 자세의 오차와 지연
                out.update(est_err_mean_mm=float(np.mean(ep.est_err_mm)), est_err_p95_mm=float(np.percentile(ep.est_err_mm, 95)),
                           est_err_mean_deg=float(np.mean(ep.est_err_deg)))
            if ep.est_age_ms:
                out.update(est_age_mean_ms=float(np.mean(ep.est_age_ms)), est_age_max_ms=float(np.max(ep.est_age_ms)))
            if ep.use_est:                            # 오래된 자세 감시 결과
                out.update(stale_steps=ep.stale_steps, stale_hits=ep.stale_hits, stale_max_ms=ep.stale_max_ms)
            return out

    return ParkPolicyNode()


def main():
    import rclpy
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', required=True, help='policy_*.npz (export_policy.py)')
    ap.add_argument('--slot', choices=list('ABCD'), required=True)
    ap.add_argument('--mode', choices=['front', 'rear'], required=True)
    ap.add_argument('--start-yaw-deg', type=float, default=None, help='turn in place to this heading first')
    ap.add_argument('--noise', choices=['ps', 'none'], default='ps')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--est-topic', default=None, help='use estimated pose from this topic for the policy, e.g. /perceived_pose')
    ap.add_argument('--stale-ms', type=float, default=STALE_S * 1000,
                    help='with --est-topic: stop (0, 0) when the estimated pose is older than this (ms, 0 = off)')
    args = ap.parse_args()
    ep = ParkEpisode(NumpyPolicy(args.policy), 'ABCD'.index(args.slot), args.mode,
                     None if args.start_yaw_deg is None else math.radians(args.start_yaw_deg), args.noise, args.seed,
                     use_est=args.est_topic is not None, stale_s=args.stale_ms / 1000)
    rclpy.init()
    node = make_node(ep, est_topic=args.est_topic)
    try:
        while rclpy.ok() and ep.result is None:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        print(ep.info or 'interrupted', node.timing())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
