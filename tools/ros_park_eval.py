#!/usr/bin/env python3
"""ROS 2 + MuJoCo 브리지에서 주차 정책을 여러 판 평가한다 (학습 환경 테스트와 같은 출발 자세).

판마다
  1. 출발 자세 계산: rl_train.evaluate와 같은 규칙 (k번 판 -> 칸 k % 4, 방식은 4판마다 전면/후면,
     출발 자세는 시드 TEST_SEED0+k(--test) 또는 EVAL_SEED0+k의 입구 통로 자세)
  2. 출발 위치를 적은 MuJoCo 설정 파일을 임시로 만들고 (pinky_pro 기본 설정 복사 + spawn.position,
     카메라 끔) 브리지를 가상환경 Python으로 실행. pinky_pro는 수정하지 않음 (브리지의 --physics-config 옵션)
  3. 정책 노드(park_policy_node.ParkEpisode): 출발 방향까지 제자리 회전 -> 정책 주행 -> 판정
  4. 브리지 종료
끝나면 rl_train.py --eval-only와 같은 형식의 요약(칸·방식별 성공, Wilson 95% 신뢰구간) + 명령 주기·처리 시간

--perception pa|ps: 정답 자세 대신 인식 루프로 주행 (실물과 같은 구성의 시뮬):
           브리지 -> ceiling_cam_sim_node (천장 영상) -> pose_perception_node (인식 + 필터 -> /perceived_pose)
           -> 정책 노드는 추정 자세를 관측에 씀 (노이즈 추가 없음). 판정은 계속 정답 자세로 함.
           세 노드를 판마다 새로 띄우고, 로그는 ros_logs/{camera,perception}_XXX.log.
           요약에 인식 오차(정책이 본 자세 vs 정답), 인식 지연(시뮬 시간), 인식 처리 시간, 검출률을 더함

--dry-run: ROS 없이 같은 로직을 MuJoCo 물리에 직접 연결해 실행 (브리지와 같은 10 ms 물리, 50 Hz 자세,
           한 메시지 늦게 적용되는 명령). ROS 문제와 정책 문제를 나눠 보기 위함

사용법 (pinky_train 가상환경, ROS 2 + 이 작업 공간 source):
  python tools/export_policy.py --name bc_ppo_seed0 --which last           # 먼저 한 번
  python tools/ros_park_eval.py --name bc_ppo_seed0 --which last --test --episodes 2     # 동작 확인
  python tools/ros_park_eval.py --name bc_ppo_seed0 --which last --test --episodes 40
  python tools/ros_park_eval.py --name bc_ppo_seed0 --which last --test --episodes 40 --dry-run
  python tools/ros_park_eval.py --name bc_ppo_seed2_tp10 --which last --test --episodes 2 --perception pa
  python tools/ros_park_eval.py --name bc_ppo_seed2_tp10 --which last --test --episodes 40 --perception ps --weights best.pt
출력: runs/rl/<name>/ros_eval_<which>[_test][_dry|_pa|_ps].csv, 로그 runs/rl/<name>/ros_logs/
"""
import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gymnasium.utils import seeding  # noqa: E402
from rl_park_env import sample_start, SCENE  # noqa: E402
from rl_train import EVAL_SEED0, TEST_SEED0, report, print_summary  # noqa: E402
from export_policy import NumpyPolicy  # noqa: E402
from park_policy_node import ParkEpisode, make_node  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPAWN_Z = 0.030


def start_pose(k, seed0):
    """rl_train.evaluate와 같은 k번째 판 (칸, 방식, 출발 자세)."""
    slot, mode = k % 4, ('front', 'rear')[(k // 4) % 2]
    rng, _ = seeding.np_random(seed0 + k)
    return slot, mode, sample_start(rng, 'corridor', slot, mode)


def run_dry(policy, slot, mode, start, noise, seed):
    """ROS 없이: 브리지와 같은 주기로 물리를 돌리며 ParkEpisode를 실행."""
    import mujoco
    from pinky_mujoco.model import Physics
    ph = Physics(SCENE)
    m, d = ph.model, ph.data
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [start[0], start[1], float(m.qpos0[2])]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]                      # 브리지처럼 방향 0으로 시작
    mujoco.mj_forward(m, d)
    ep = ParkEpisode(policy, slot, mode, start[2], noise, seed)
    cmd, tick = (0.0, 0.0), 0
    while ep.result is None and d.time < 200.0:
        ph.step(*cmd)                                        # 10 ms
        tick += 1
        if tick % 2 == 0:                                    # 50 Hz 자세 -> 다음 물리 단계부터 명령 적용
            cmd = ep.on_state(d.time, ph.pose(), tuple(ph.twist))
    return ep.info or {'result': 'error'}, {}


def run_ros(policy, slot, mode, start, noise, seed, k, args, tmp, logdir):
    """브리지를 띄우고 정책 노드로 한 판 주행."""
    import rclpy
    from ament_index_python.packages import get_package_share_directory
    cfg = yaml.safe_load((Path(get_package_share_directory('pinky_mujoco')) / 'config/mujoco.yaml').read_text())
    cfg['spawn']['position'] = [float(start[0]), float(start[1]), SPAWN_Z]
    if cfg.get('camera'):
        cfg['camera']['enabled'] = False
    cfg_path = Path(tmp) / f'physics_{k}.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))
    log = open(logdir / f'bridge_{k:03d}.log', 'w')
    # 브리지 main()을 가상환경 Python으로 직접 실행 (Jetson은 MuJoCo가 가상환경에만 있어 ros2 run 불가)
    proc = subprocess.Popen([sys.executable, '-c', 'from pinky_mujoco.bridge import main; main()',
                             '--no-camera', '--world', str(args.world),
                             '--physics-config', str(cfg_path), '--output-dir', str(Path(tmp) / 'model')],
                            stdout=log, stderr=subprocess.STDOUT)
    helpers, stats_path = [], Path(tmp) / f'perception_{k}.json'
    if args.perception != 'none':                         # 천장 카메라 시뮬 + 인식 노드
        tools = Path(__file__).resolve().parent
        cam_cmd = [sys.executable, str(tools / 'ceiling_cam_sim_node.py'), '--seed', str(seed)]
        perc_cmd = [sys.executable, str(tools / 'pose_perception_node.py'), '--method', args.perception,
                    '--stats-file', str(stats_path)] + (['--weights', args.weights] if args.weights else [])
        for name, cmd in (('camera', cam_cmd), ('perception', perc_cmd)):
            lg = open(logdir / f'{name}_{k:03d}.log', 'w')
            helpers.append((subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT), lg))
    use_est = args.perception != 'none'
    ep = ParkEpisode(policy, slot, mode, start[2], noise, seed, use_est=use_est)
    node = make_node(ep, name=f'park_policy_{k}', est_topic='/perceived_pose' if use_est else None)
    t_start = time.monotonic()
    info, timing = None, {}
    try:
        while ep.result is None:
            rclpy.spin_once(node, timeout_sec=0.05)
            if proc.poll() is not None:
                info = {'result': 'bridge_exit'}
                break
            if any(h.poll() is not None for h, _ in helpers):
                info = {'result': 'helper_exit'}
                break
            if time.monotonic() - t_start > args.wall_timeout:
                info = {'result': 'wall_timeout'}
                break
        node.stop()
        timing = node.timing()
        if node.first_pose is not None:
            timing['spawn_error_mm'] = math.hypot(node.first_pose[0] - start[0], node.first_pose[1] - start[1]) * 1000
    finally:
        node.destroy_node()
        for p_, lg in helpers + [(proc, log)]:            # 인식·카메라 노드 먼저, 브리지 마지막
            if p_.poll() is None:
                p_.send_signal(signal.SIGINT)
                try:
                    p_.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p_.kill()
                    p_.wait()
            lg.close()
    if stats_path.is_file():
        timing.update({k2: v for k2, v in json.loads(stats_path.read_text()).items() if k2 != 'method'})
    timing['wall_s'] = time.monotonic() - t_start
    return info or ep.info, timing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='runs/rl/<name> (policy_<which>.npz from export_policy.py)')
    ap.add_argument('--which', choices=['best', 'last'], default='last')
    ap.add_argument('--episodes', type=int, default=40)
    ap.add_argument('--first', type=int, default=0, help='start from episode index k (resume)')
    ap.add_argument('--test', action='store_true', help='test start poses (seed TEST_SEED0) instead of EVAL_SEED0')
    ap.add_argument('--noise', choices=['ps', 'none'], default='ps')
    ap.add_argument('--dry-run', action='store_true', help='no ROS: same logic on MuJoCo physics in-process')
    ap.add_argument('--world', default=None, help='world file for the bridge (default: worlds/parking_course_local.sdf '
                                                  'if it exists, else worlds/parking_course.sdf)')
    ap.add_argument('--wall-timeout', type=float, default=180.0, help='max real seconds per episode (bridge start incl.)')
    ap.add_argument('--brief', action='store_true', help='skip the per-episode table')
    ap.add_argument('--perception', choices=['none', 'pa', 'ps'], default='none',
                    help='drive on perceived pose (ceiling camera sim + recognition node) instead of ground truth')
    ap.add_argument('--weights', default=None, help='YOLO-seg weights for --perception ps')
    args = ap.parse_args()
    if args.perception == 'ps' and not args.weights:
        raise SystemExit('--weights is required for --perception ps')
    if args.perception != 'none' and args.dry_run:
        raise SystemExit('--perception needs ROS (not --dry-run)')
    if args.world is None:
        local = REPO / 'worlds/parking_course_local.sdf'
        args.world = local if local.is_file() else REPO / 'worlds/parking_course.sdf'

    run = REPO / 'runs/rl' / args.name
    policy = NumpyPolicy(run / f'policy_{args.which}.npz')
    seed0 = TEST_SEED0 if args.test else EVAL_SEED0
    suffix = ('_test' if args.test else '') + ('_dry' if args.dry_run else '') + \
        ('' if args.perception == 'none' else f'_{args.perception}')
    out = run / f'ros_eval_{args.which}{suffix}.csv'
    logdir = run / 'ros_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    print(f'{args.name} / policy_{args.which}.npz, {"dry run (no ROS)" if args.dry_run else "ROS 2 + MuJoCo bridge"}, '
          f'{"noise=" + args.noise if args.perception == "none" else "perception=" + args.perception}, start poses seed {seed0}+k, episodes {args.first}..{args.episodes - 1}, '
          f'world {Path(args.world).name}', flush=True)

    if not args.dry_run:
        import rclpy
        rclpy.init()
    rows, t0 = [], time.time()
    tmp = tempfile.mkdtemp(prefix='ros_park_eval_')
    try:
        for k in range(args.first, args.episodes):
            slot, mode, start = start_pose(k, seed0)
            if args.dry_run:
                info, timing = run_dry(policy, slot, mode, start, args.noise, seed0 + k)
            else:
                info, timing = run_ros(policy, slot, mode, start, args.noise, seed0 + k, k, args, tmp, logdir)
            row = {'episode': k, 'slot': 'ABCD'[slot], 'mode': mode, **{
                'time_s': np.nan, 'lateral_mm': np.nan, 'depth_mm': np.nan, 'yaw_deg': np.nan, 'dist_mm': np.nan,
                'depth_signed_mm': np.nan, 'min_dist_mm': np.nan, 'stopped': False}, **info, **timing}
            rows.append(row)
            n_ok = sum(r['result'] == 'success' for r in rows)
            el = time.time() - t0
            eta = el / len(rows) * (args.episodes - args.first - len(rows))
            extra = (f"  cmd gap {timing['cmd_gap_mean_ms']:.0f}/{timing['cmd_gap_max_ms']:.0f} ms, "
                     f"compute {timing['compute_mean_ms']:.1f} ms" if 'cmd_gap_mean_ms' in timing else '')
            if 'est_err_mean_mm' in timing:
                extra += (f", est err {timing['est_err_mean_mm']:.1f} mm / {timing['est_err_mean_deg']:.1f} deg, "
                          f"age {timing.get('est_age_mean_ms', float('nan')):.0f} ms, "
                          f"perception {timing.get('perc_compute_mean_ms', float('nan')):.0f} ms")
            print(f"  [{k + 1:4d}/{args.episodes}] {'ABCD'[slot]} {mode:5s} -> {row['result']:12s} "
                  f"t={row['time_s'] if row['time_s'] == row['time_s'] else 0:5.1f} s | success {n_ok}/{len(rows)} "
                  f"| {el / 60:.1f} min, ~{eta / 60:.1f} min left{extra}", flush=True)
            with open(out, 'w', newline='') as f:                # 판마다 저장 (중간에 멈춰도 남음)
                keys = sorted({k2 for r in rows for k2 in r}, key=lambda c: list(rows[0]).index(c)
                              if c in rows[0] else 999)
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    finally:
        if not args.dry_run:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()

    counts = {r: sum(x['result'] == r for x in rows) / len(rows) for r in ('success', 'contact', 'out', 'timeout')}
    other = [x['result'] for x in rows if x['result'] not in counts]
    if other:
        print(f'\nother results (setup problems): { {k: other.count(k) for k in set(other)} }')
    report([r for r in rows if r['result'] in counts], args.brief)
    print_summary(rows, counts, out, args.test)
    if not args.dry_run:
        g = [r for r in rows if 'cmd_gap_mean_ms' in r]
        if g:
            print(f"ROS timing: command gap mean {np.mean([r['cmd_gap_mean_ms'] for r in g]):.1f} ms "
                  f"(max {max(r['cmd_gap_max_ms'] for r in g):.0f} ms), node compute mean "
                  f"{np.mean([r['compute_mean_ms'] for r in g]):.2f} ms (max {max(r['compute_max_ms'] for r in g):.1f} ms), "
                  f"wall per episode {np.mean([r['wall_s'] for r in g]):.0f} s, "
                  f"spawn error max {max(r.get('spawn_error_mm', 0) for r in g):.1f} mm")
        e = [r for r in rows if 'est_err_mean_mm' in r]
        if e:
            print(f"perception ({args.perception}): pose error seen by the policy mean "
                  f"{np.mean([r['est_err_mean_mm'] for r in e]):.1f} mm (p95 per episode up to "
                  f"{max(r['est_err_p95_mm'] for r in e):.1f} mm), heading {np.mean([r['est_err_mean_deg'] for r in e]):.2f} deg, "
                  f"age mean {np.nanmean([r.get('est_age_mean_ms', np.nan) for r in e]):.0f} ms "
                  f"(max {np.nanmax([r.get('est_age_max_ms', np.nan) for r in e]):.0f} ms), "
                  f"compute mean {np.nanmean([r.get('perc_compute_mean_ms', np.nan) for r in e]):.1f} ms, "
                  f"detect rate {np.nanmean([r.get('detect_rate', np.nan) for r in e]):.1%}")


if __name__ == '__main__':
    main()
