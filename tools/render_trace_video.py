#!/usr/bin/env python3
"""주행 기록(trace) -> 주차 영상(mp4): ros_park_eval.py --trace로 남긴 한 판을 천장 카메라 시점으로 다시 그린다.

무엇인가
  ros_park_eval.py --trace는 정책 단계마다 정답 자세(x, y, yaw), 정책이 본 자세(seen_*), 명령(v, w), 접촉 여부를
  ros_logs/trace_XXX.csv에 남긴다. 이 스크립트는 그 기록을 읽어 MuJoCo 장면(천장 카메라 노드와 같은 장면·보정
  카메라·렌즈 왜곡)에 로봇을 정답 자세대로 놓고 렌더링한 뒤, 다음을 겹쳐 그린다.
    - 노란 상자: 목표 칸
    - 초록: 정답 자세의 로봇 외곽 + 지나온 경로 / 빨강: 정책이 본(인식) 자세의 로봇 외곽 + 경로
    - 접촉 중이면 화면 테두리를 빨갛게
    - 오른쪽 패널: 판 번호, 칸·방식, 시간, 명령, 목표 대비 오차(정답 / 인식), 결과
  다시 주행하지 않으므로 평가한 바로 그 주행이 나온다. 단, 기록은 정책 단계부터라서 시작 대기와
  처음 제자리 회전은 영상에 없다. 바퀴 회전도 기록에 없어서 바퀴는 돌지 않는다.

사용법 (pinky_train 가상환경):
  python tools/render_trace_video.py --run runs/rl/bc_ppo_seed2_tp10 --episodes 3,5,29,37,0,2,4,7,11,23
  python tools/render_trace_video.py --run runs/rl/bc_ppo_seed2_tp10 --episodes 0 --fps 20
  (--episodes는 trace_XXX.csv의 번호 = 0부터 센 판 번호. 화면에는 1부터 센 번호로 표시)
결과 판정은 --results의 CSV(episode, result 열)에서 읽는다. 기본: <run>/ros_eval_last_test_ps_repro_*.csv
화면 없는 환경에서 렌더링 오류가 나면 MUJOCO_GL=egl (또는 osmesa)를 지정.
출력: <run>/videos/ep_XXX.mp4 (XXX = 1부터 센 판 번호)
"""
import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_parking_dataset as D  # noqa: E402
import parking_geom as G  # noqa: E402
from rl_park_env import footprint, slot_errors, OK_LATERAL_MM, OK_DEPTH_MM, OK_YAW_DEG  # noqa: E402

PANEL_W, HOLD_S = 300, 2.0         # 정보 패널 폭(px), 마지막 장면을 결과와 함께 더 보여 주는 시간(초)
GREEN, RED, YELLOW, BLACK, GREY = (0, 200, 0), (0, 0, 255), (0, 200, 255), (20, 20, 20), (110, 110, 110)


def load_trace(path):
    """trace CSV -> (slot 번호, 방식, 행 목록). 행 = dict(t, true=(x,y,yaw), seen=(x,y,yaw)|None, v, w, touching)."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            seen = (float(r['seen_x']), float(r['seen_y']), float(r['seen_yaw'])) if r['seen_x'] != '' else None
            rows.append({'t': float(r['t']), 'true': (float(r['x']), float(r['y']), float(r['yaw'])),
                         'seen': seen, 'v': float(r['v']), 'w': float(r['w']),
                         'touching': r['touching'] == 'True', 'slot': r['slot'], 'mode': r['mode']})
    if not rows:
        raise SystemExit(f'empty trace: {path}')
    return 'ABCD'.index(rows[0]['slot']), rows[0]['mode'], rows


def load_results(paths):
    """결과 CSV들 -> {판 번호(0부터): 결과 문자열}."""
    out = {}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get('episode', '') != '' and r.get('result'):
                    out[int(r['episode'])] = r['result']
    return out


def world_pts(poly_mm, z=0.002):
    """바닥 mm 다각형 -> 월드 점 (N,3)."""
    xy = np.array([G.mm_to_world(u, v) for u, v in poly_mm])
    return np.column_stack([xy, np.full(len(xy), z)])


class Drawer:
    def __init__(self, cam, slot, mode):
        self.cam, self.slot, self.mode = cam, slot, mode
        su = D.SLOT_U[slot]
        self.box_px = self._px(world_pts([(su - 95, 19), (su + 95, 19), (su + 95, 220), (su - 95, 220)]))

    def _px(self, pts):
        return np.round(self.cam.project(pts)).astype(np.int32)

    def robot(self, img, pose, color, thick):
        """로봇 외곽(몸통 + 바퀴)과 방향 화살표."""
        for poly in footprint(*pose):
            cv2.polylines(img, [self._px(world_pts(poly, 0.03))], True, color, thick, cv2.LINE_AA)
        x, y, yaw = pose
        q = self._px(np.array([[x, y, 0.03], [x + 0.08 * math.cos(yaw), y + 0.08 * math.sin(yaw), 0.03]]))
        cv2.arrowedLine(img, tuple(q[0]), tuple(q[1]), color, 2, cv2.LINE_AA, tipLength=0.3)

    def trail(self, img, poses, color):
        if len(poses) >= 2:
            cv2.polylines(img, [self._px(np.array([[p[0], p[1], 0.002] for p in poses]))], False, color, 1, cv2.LINE_AA)

    def frame(self, base, row, true_trail, seen_trail, ep_no, result, show_result):
        img = base.copy()
        cv2.polylines(img, [self.box_px], True, YELLOW, 2, cv2.LINE_AA)
        self.trail(img, true_trail, GREEN)
        self.trail(img, seen_trail, RED)
        if row['seen'] is not None:
            self.robot(img, row['seen'], RED, 1)
        self.robot(img, row['true'], GREEN, 2)
        if row['touching']:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), RED, 6)
        panel = np.full((img.shape[0], PANEL_W, 3), 255, np.uint8)
        self.panel(panel, row, ep_no, result, show_result)
        return np.hstack([img, panel])

    def panel(self, p, row, ep_no, result, show_result):
        y = [24]

        def text(s, color=BLACK, scale=0.55, gap=24, x=12, bold=1):
            cv2.putText(p, s, (x, y[0]), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)
            y[0] += gap

        text(f'episode {ep_no}', scale=0.7, gap=28, bold=2)
        text(f'target: slot {"ABCD"[self.slot]} {self.mode}')
        text(f't = {row["t"]:5.1f} s (policy phase)')
        text(f'cmd v {row["v"]:+.3f} m/s  w {row["w"]:+.2f}', gap=34)
        text('error to target', GREY, 0.5, 22)
        text('truth', GREY, 0.45, 0, x=120)
        text('seen', GREY, 0.45, 20, x=205)
        g = slot_errors(row['true'], self.slot, self.mode)
        e = slot_errors(row['seen'], self.slot, self.mode) if row['seen'] is not None else None
        limits = (OK_LATERAL_MM, OK_DEPTH_MM, OK_YAW_DEG)
        for i, (name, unit) in enumerate((('lateral', 'mm'), ('depth', 'mm'), ('heading', 'deg'))):
            text(name, gap=0)
            text(f'{g[i]:6.1f} {unit}', RED if g[i] > limits[i] else BLACK, gap=0, x=110)
            text('   -' if e is None else f'{e[i]:6.1f} {unit}', GREY, x=195)
        text(f'tape contact: {"YES" if row["touching"] else "no"}', RED if row['touching'] else BLACK, gap=30)
        text('limits: 20 mm, 20 mm, 5 deg', GREY, 0.45, 20)
        text('green = truth (scored)', GREY, 0.45, 20)
        text('red = pose the policy saw', GREY, 0.45, 20)
        text('yellow = target slot', GREY, 0.45, 30)
        if show_result and result:
            text(f'RESULT: {result.upper()}', (20, 140, 40) if result == 'success' else RED, 0.8, 30, bold=2)


def resample(rows, fps):
    """기록 시각 t를 기준으로 영상 프레임마다 그 시각까지의 마지막 행 번호."""
    ts = np.array([r['t'] for r in rows])
    frames = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / fps)
    return np.searchsorted(ts, frames, side='right') - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='run folder, e.g. runs/rl/bc_ppo_seed2_tp10 (reads ros_logs/trace_XXX.csv)')
    ap.add_argument('--episodes', required=True, help='comma list of 0-based episode indices (trace file numbers)')
    ap.add_argument('--results', nargs='*', default=None,
                    help='result CSVs with episode,result columns (default: <run>/ros_eval_last_test_ps_repro_*.csv)')
    ap.add_argument('--model', default='~/.cache/pinky_mujoco/scene.xml', help='MuJoCo scene with the parking course')
    ap.add_argument('--fps', type=float, default=20.0, help='video frames per second (real-time playback)')
    ap.add_argument('--out', default=None, help='output folder (default: <run>/videos)')
    args = ap.parse_args()

    from ceiling_cam_sim_node import CeilingRenderer      # rclpy는 그 파일의 main()에서만 불러옴
    run = Path(args.run)
    out = Path(args.out) if args.out else run / 'videos'
    out.mkdir(parents=True, exist_ok=True)
    res_paths = args.results if args.results is not None else sorted(glob.glob(str(run / 'ros_eval_last_test_ps_repro_*.csv')))
    results = load_results(res_paths)
    renderer = CeilingRenderer(args.model, noise_sd=0.0, jpeg=0)   # 보기용: 노이즈·압축 없음
    cam = G.Camera(D.CAM_POS, D.CAM_QUAT)
    episodes = [int(s) for s in args.episodes.split(',') if s.strip()]
    print(f'{len(episodes)} episodes, results from {len(res_paths)} file(s), output {out}')
    for k in episodes:
        path = run / 'ros_logs' / f'trace_{k:03d}.csv'
        if not path.exists():
            print(f'  episode {k + 1}: missing {path}, skipped')
            continue
        slot, mode, rows = load_trace(path)
        drawer = Drawer(cam, slot, mode)
        result = results.get(k, '')
        idx = resample(rows, args.fps)
        dst = out / f'ep_{k + 1:03d}.mp4'
        writer = None
        true_trail, seen_trail, last_i = [], [], -1
        for n, i in enumerate(idx):
            for j in range(last_i + 1, i + 1):                # 이번 프레임까지의 기록을 경로에 추가
                true_trail.append(rows[j]['true'])
                if rows[j]['seen'] is not None:
                    seen_trail.append(rows[j]['seen'])
            last_i = max(last_i, i)
            base = renderer.render(*rows[i]['true'])
            img = drawer.frame(base, rows[i], true_trail, seen_trail, k + 1, result, False)
            if writer is None:
                writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                                         (img.shape[1], img.shape[0]))
            writer.write(img)
        final = drawer.frame(base, rows[idx[-1]], true_trail, seen_trail, k + 1, result, True)
        for _ in range(int(HOLD_S * args.fps)):
            writer.write(final)
        writer.release()
        print(f'  episode {k + 1}: slot {"ABCD"[slot]} {mode}, {rows[-1]["t"]:.1f} s, {result or "result ?"} -> {dst}')


if __name__ == '__main__':
    main()
