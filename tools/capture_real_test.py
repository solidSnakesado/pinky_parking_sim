#!/usr/bin/env python3
"""실물 테스트 사진 촬영 + 자로 잰 정답 자세 기록 (P-A / P-B / P-S / P-K 공통 평가용).

정답 자세 측정 (자로 실측)
- 기준점 (실측 2026-11-13, 로봇 좌표: 원점 = 바퀴 축 중심, x 앞, y 왼쪽)
  RL / RR: 뒤 금속 기둥 바깥 모서리 (x = -72 mm, y = ±38.5 mm). 뒤쪽·바깥 옆쪽으로 잰다.
  FL / FR: 맨 아래 라인 센서 기판의 앞 모서리 (x = +28.5 mm, y = ±30 mm). 앞쪽으로만 잰다
           (옆으로는 바퀴 앞면과 1~2 mm밖에 떨어져 있지 않아 쓰지 않음).
  실측값: 기둥 폭 77, 바퀴 중심 -> 앞 기둥 20, 앞 기둥 -> 뒤 기둥 92, 바퀴 중심 -> 기판 앞 28~29, 기판 폭 60 (mm).
  앞 기둥 모서리는 바로 아래 바닥이 바퀴(반지름 약 27 mm) 안이라 쓰지 않는다.
  기둥은 바닥에서 떠 있으므로 삼각자를 세워 기둥 모서리에 대고, 바닥에서 테이프 가장자리까지 잰다.
- 자가 지나는 길이 바퀴·기판·몸통·다른 테이프에 막히는 조합은 고르지 않는다.
- 자세마다 측정 3개: (모서리 -> 테이프 가장자리 선) 수직 거리 mm. 기준선 1개에 모서리 2개, 그와 수직인
  선에 모서리 1개. 어떤 모서리·선을 잴지는 목표 자세에서 자동으로 고르고 화면에 표시한다.
- 3개 거리로 x, y, yaw를 최소제곱으로 푼다(시작값 = 목표 자세). 정답은 목표가 아니라 실측값이다.

촬영 흐름 (키는 영상 창에서)
  n / p   다음 / 이전 자세 (노란 외곽 = 목표 자세, 그 위에 로봇을 대략 놓는다)
  m       측정값 입력 (터미널에서 3개 입력) -> 실측 자세 계산, 초록 외곽으로 표시
  1 / 2   조명 번호 선택 (예: 1 = 낮 자연광, 2 = 실내등만)
  c       현재 조명으로 사진 저장 (프레임 5장 중앙값). 같은 자세에서 조명만 바꿔 다시 c
  b       현재 조명의 빈 주차장 배경 저장 (로봇을 치우고) -> P-B용
  q       종료: poses.csv, background/test/<이름>.png 정리
진행 상태는 <out>/session.json에 저장되어 다시 실행하면 이어서 한다.

출력 (기존 데이터셋과 같은 형식, eval_pose.py --data <out> --split test로 바로 평가):
  <out>/images/test/<이름>.jpg, <out>/poses.csv, <out>/background/test/<이름>.png, <out>/measurements.csv
  이름 = <번호>_<종류>_L<조명>, poses.csv의 light_scale 열 = 조명 번호, 카메라 열 = --calib 보정값.

사용법 (pinky_train 가상환경):
  python tools/capture_real_test.py --device 0 --calib config/ceiling_camera_calib.json [--count 20]
  python tools/capture_real_test.py --plan-only            # 자세 목록과 측정 안내만 출력
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402

TAPE_NAMES = ['P1', 'P2', 'P3', 'S1', 'S2', 'S3', 'S4', 'S5', 'P4', 'P5',
              'L1', 'L2', 'L3', 'L4', 'L5', 'T1', 'T2', 'T3', 'T4', 'T5']   # course_tape_positions.md 순서
POST_FRONT, POST_REAR, POST_HALF_W = 0.020, -0.072, 0.0385          # 실측 기둥 바깥면 (m)
BOARD_FRONT, BOARD_HALF_W = 0.0285, 0.030                            # 실측 라인 센서 기판 앞 가장자리 (m)
CORNERS = {'FL': (BOARD_FRONT, BOARD_HALF_W), 'FR': (BOARD_FRONT, -BOARD_HALF_W),
           'RL': (POST_REAR, POST_HALF_W), 'RR': (POST_REAR, -POST_HALF_W)}
# 모서리마다 잴 수 있는 방향 (로봇 좌표 단위 벡터). 이 방향에서 45° 이내의 자만 허용
CORNER_DIRS = {'FL': [(1, 0)], 'FR': [(1, 0)], 'RL': [(-1, 0), (0, 1)], 'RR': [(-1, 0), (0, -1)]}
# 자를 막는 실물 부분 (x0, x1, y0, y1) m: 기둥 사각형, 좌우 바퀴(반지름 약 28 mm), 라인 센서 기판
REAL_BLOCKERS = [(POST_REAR, POST_FRONT, -POST_HALF_W, POST_HALF_W),
                 (-0.028, 0.028, 0.026, 0.055), (-0.028, 0.028, -0.055, -0.026),
                 (POST_FRONT, BOARD_FRONT, -BOARD_HALF_W, BOARD_HALF_W)]
SLOT_U = (136.5, 345.5, 554.5, 763.5)
PARK_V = {'front': 102.5, 'rear': 136.5}           # park_closed_loop.py와 같은 칸 중앙 바퀴 축 v
MAX_DIST_MM = 300.0                                  # 자 길이 (이보다 먼 선은 쓰지 않음)
CLEAR_MM = 8.0                                       # 목표 자세와 테이프 사이 최소 여유


# ---------------------------------------------------------------- 좌표 (바닥 mm u, v / 월드 m x, y)
def pose_to_uv(x, y):
    return G.world_to_mm(x, y)


def corner_uv(x, y, yaw, name):
    cx, cy = CORNERS[name]
    wx, wy = G.rot(yaw) @ [cx, cy] + [x, y]
    return np.array(G.world_to_mm(wx, wy))


def tape_lines():
    """테이프 가장자리 선 목록: (축 'u' 또는 'v', 좌표 mm, 범위 (a, b), 바깥 방향 부호, 설명, 테이프 번호)."""
    lines = []
    for t, (name, (u, v, w, h)) in enumerate(zip(TAPE_NAMES, G.TAPES)):
        lines += [('u', u, (v, v + h), -1, f'{name} left edge (u={u:g})', t),
                  ('u', u + w, (v, v + h), +1, f'{name} right edge (u={u + w:g})', t),
                  ('v', v, (u, u + w), -1, f'{name} top edge (v={v:g})', t),
                  ('v', v + h, (u, u + w), +1, f'{name} bottom edge (v={v + h:g})', t)]
    return lines


LINES = tape_lines()


def inside_rect(pts, rect, eps=0.5):
    u, v, w, h = rect
    return (pts[:, 0] > u + eps) & (pts[:, 0] < u + w - eps) & (pts[:, 1] > v + eps) & (pts[:, 1] < v + h - eps)


def inside_poly(pts, poly):
    """볼록 다각형 안 (꼭짓점 순서와 관계없이, 경계는 제외)."""
    area = sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
               for i in range(len(poly)))
    sign = 1.0 if area > 0 else -1.0                 # 꼭짓점 순서 방향
    ok = np.ones(len(pts), bool)
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        cr = e[0] * (pts[:, 1] - poly[i][1]) - e[1] * (pts[:, 0] - poly[i][0])
        ok &= cr * sign > 1e-9
    return ok


def corner_line_dist(c_uv, line, body=None):
    """모서리에서 선까지 수직 거리 mm. 다음을 모두 만족할 때만, 아니면 None:
    선이 모서리를 바라봄, 수선의 발이 테이프 위, 자가 지나는 길에 다른 테이프나 로봇 몸이 없음."""
    axis, c, (a, b), side, _, tape = line
    k, other = (0, 1) if axis == 'u' else (1, 0)
    if not (a + 1 <= c_uv[other] <= b - 1):
        return None
    d = (c_uv[k] - c) * side
    if d <= 0:
        return None
    t = np.linspace(3.0, d - 0.5, max(2, int(d / 2)))[:, None]      # 자가 지나는 점들 (모서리 -> 선)
    step = np.zeros(2)
    step[k] = -side
    pts = c_uv + t * step
    if any(inside_rect(pts, r).any() for j, r in enumerate(G.TAPES) if j != tape):
        return None
    if body is not None and any(inside_poly(pts, poly).any() for poly in body):
        return None
    return d


def dir_ok(pose, name, line):
    """자 방향(모서리 -> 선)이 그 모서리에서 허용된 방향에서 45° 이내인지."""
    axis, _, _, side, _, _ = line
    d_uv = np.array([-side, 0.0]) if axis == 'u' else np.array([0.0, -side])
    d_world = np.array([-d_uv[1], -d_uv[0]])               # u, v (mm) 증가 방향 = 월드 -y, -x
    d_robot = G.rot(-pose[2]) @ d_world
    return any(np.dot(d_robot, a) > math.cos(math.radians(45)) for a in CORNER_DIRS[name])


def choose_measurements(pose):
    """측정 3개: 기준선 1개(모서리 2개) + 수직선 1개(모서리 1개). 가장 가까운 조합. 없으면 None."""
    cuv = {n: corner_uv(*pose, n) for n in CORNERS}
    body = blockers_uv(*pose)
    best = None
    for i, l1 in enumerate(LINES):
        ds = sorted((d, n) for n in CORNERS if dir_ok(pose, n, l1)
                    and (d := corner_line_dist(cuv[n], l1, body)) is not None and d <= MAX_DIST_MM)
        if len(ds) < 2:
            continue
        pair = ds[:2]
        for l2 in LINES:
            if l2[0] == l1[0]:
                continue
            ds2 = sorted((d, n) for n in CORNERS if dir_ok(pose, n, l2)
                         and (d := corner_line_dist(cuv[n], l2, body)) is not None and d <= MAX_DIST_MM)
            if not ds2:
                continue
            cost = max(pair[1][0], ds2[0][0])
            if best is None or cost < best[0]:
                best = (cost, [(pair[0][1], i), (pair[1][1], i), (ds2[0][1], LINES.index(l2))])
    if best is None:
        return None
    meas = best[1]
    if solve_pose(pose, meas, [corner_line_dist(cuv[n], LINES[j]) for n, j in meas], check=True) is None:
        return None
    return meas


def solve_pose(init, meas, dists, check=False):
    """측정 거리로 (x, y, yaw) 풀기. check=True면 조건수만 확인(나쁘면 None)."""
    def residual(p):
        r = []
        for (n, j), d in zip(meas, dists):
            axis, c, _, side, _, _ = LINES[j]
            cu = corner_uv(p[0], p[1], p[2], n)
            r.append(((cu[0] if axis == 'u' else cu[1]) - c) * side - d)
        return np.array(r)
    sol = least_squares(residual, np.asarray(init, float), x_scale=[0.01, 0.01, 0.05])
    sv = np.linalg.svd(sol.jac * [1e-3, 1e-3, math.radians(1)], compute_uv=False)   # 열 단위: mm, mm, °
    if sv.min() < 0.3:          # 1 mm / 1° 당 0.3 mm 미만이면 측정 조합이 자세를 잘 정하지 못함
        return None
    return sol.x if not check else True


# ---------------------------------------------------------------- 목표 자세 목록
def footprint_uv(x, y, yaw, grow=0.0):
    polys = []
    for x0, x1, y0, y1, z0, z1 in G.ROBOT_BOXES:
        if z0 >= 0.1:
            continue
        c = np.array([[x0 - grow, y0 - grow], [x1 + grow, y0 - grow], [x1 + grow, y1 + grow], [x0 - grow, y1 + grow]])
        c = c @ G.rot(yaw).T + [x, y]
        polys.append(np.column_stack(G.world_to_mm(c[:, 0], c[:, 1])))
    return polys


def blockers_uv(x, y, yaw):
    """자를 막는 실물 부분들의 바닥 다각형 (mm). 자 두께·놓는 오차를 생각해 2 mm 넓힘."""
    out = []
    g = 0.002
    for x0, x1, y0, y1 in REAL_BLOCKERS:
        c = np.array([[x0 - g, y0 - g], [x1 + g, y0 - g], [x1 + g, y1 + g], [x0 - g, y1 + g]]) @ G.rot(yaw).T + [x, y]
        out.append(np.column_stack(G.world_to_mm(c[:, 0], c[:, 1])))
    return out


def overlaps(poly, rect):
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


def pose_ok(pose, cam):
    if any(overlaps(p, t) for p in footprint_uv(*pose, grow=CLEAR_MM / 1000) for t in G.TAPES):
        return False
    pts = []
    for x0, x1, y0, y1, z0, z1 in G.ROBOT_BOXES:
        for a in (x0, x1):
            for b in (y0, y1):
                for z in (z0, z1):
                    pts.append([*(G.rot(pose[2]) @ [a, b] + pose[:2]), z])
    pix = cam.project(np.array(pts))
    return bool(np.all((pix[:, 0] > 15) & (pix[:, 0] < G.WIDTH - 15) & (pix[:, 1] > 15) & (pix[:, 1] < G.HEIGHT - 15)))


def make_plan(cam, seed=600, count=40):
    """40자세: 칸 안 16, 칸 앞 접근 12, 입구 통로 6, 가로 통로 6 (count=20이면 절반: 8 / 6 / 3 / 3).
    각각 테이프 여유·화면 안·측정 가능 확인."""
    half = count == 20
    rng = np.random.default_rng(seed)
    plan = []

    def add(kind, sampler, n):
        got = 0
        for _ in range(5000):
            if got == n:
                return
            u, v, yaw = sampler()
            x, y = G.mm_to_world(u, v)
            pose = (float(x), float(y), float(G.wrap(yaw)))
            if pose_ok(np.array(pose), cam) and (meas := choose_measurements(pose)) is not None:
                plan.append(dict(kind=kind, pose=pose, meas=meas))
                got += 1
        raise RuntimeError(f'could not place {kind}')

    for slot in range(4):
        for mode in ('front', 'rear'):
            for _ in range(1 if half else 2):
                base = 0.0 if mode == 'front' else math.pi
                add('parked', lambda: (SLOT_U[slot] + rng.uniform(-12, 12), PARK_V[mode] + rng.uniform(-10, 10),
                                       base + math.radians(rng.uniform(-4, 4))), 1)
    add('approach', lambda: (SLOT_U[rng.integers(4)] + rng.uniform(-40, 40), rng.uniform(200, 420),
                             rng.uniform(-math.pi, math.pi)), 6 if half else 12)
    add('entrance', lambda: (rng.uniform(260, 430), rng.uniform(640, 920), rng.uniform(-math.pi, math.pi)),
        3 if half else 6)
    add('free', lambda: (rng.uniform(600, 900), rng.uniform(740, 920), rng.uniform(-math.pi, math.pi)),
        3 if half else 6)
    return plan


def describe(meas, pose):
    cuv = {n: corner_uv(*pose, n) for n in CORNERS}
    return [f'{n} -> {LINES[j][4]}  (expect {corner_line_dist(cuv[n], LINES[j]):.0f} mm)' for n, j in meas]


# ---------------------------------------------------------------- 그리기 / 촬영
def draw_outline(img, cam, pose, color):
    m = G.silhouette(cam, *pose).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, color, 1, cv2.LINE_AA)
    tip = G.rot(pose[2]) @ [0.09, 0.0] + pose[:2]                 # 앞쪽 표시
    a, b = cam.project([[pose[0], pose[1], 0.1], [tip[0], tip[1], 0.1]]).astype(int)
    cv2.arrowedLine(img, tuple(map(int, a)), tuple(map(int, b)), color, 1, cv2.LINE_AA, tipLength=0.3)


def put(img, text, y, color=(255, 255, 255)):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def grab_median(cap, n=5):
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, (G.WIDTH, G.HEIGHT)))
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def ask_distances(lines_text):
    print('\nEnter measured distances in mm (empty line = cancel):')
    out = []
    for t in lines_text:
        s = input(f'  {t}: ').strip()
        if not s:
            return None
        out.append(float(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='0', help='camera index (check with v4l2-ctl --list-devices)')
    ap.add_argument('--calib', default='config/ceiling_camera_calib.json')
    ap.add_argument('--out', default='~/dev_ws/pinky_parking_sim/datasets/real_test')
    ap.add_argument('--seed', type=int, default=600, help='seed for the target pose list')
    ap.add_argument('--count', type=int, default=40, choices=[20, 40], help='number of poses')
    ap.add_argument('--plan-only', action='store_true', help='print the pose list and measurement guide, then exit')
    args = ap.parse_args()

    calib = json.load(open(args.calib))
    cam = G.Camera(calib['pos'], calib['quat_wxyz'])
    out = Path(os.path.expanduser(args.out))
    plan = make_plan(cam, args.seed, args.count)
    if args.plan_only:
        for k, p in enumerate(plan):
            u, v = pose_to_uv(*p['pose'][:2])
            print(f"{k:02d} {p['kind']:8s} u={u:6.1f} v={v:6.1f} yaw={math.degrees(p['pose'][2]):7.1f}")
            for t in describe(p['meas'], p['pose']):
                print('     ' + t)
        return

    (out / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (out / 'background' / 'test').mkdir(parents=True, exist_ok=True)
    sess_path = out / 'session.json'
    sess = (json.load(open(sess_path)) if sess_path.exists()
            else dict(seed=args.seed, count=args.count, poses={}, shots=[], bg={}))
    if sess['seed'] != args.seed or sess.get('count', 40) != args.count:
        raise SystemExit(f'session in {out} used seed {sess["seed"]} count {sess.get("count", 40)}')

    def save_session():
        json.dump(sess, open(sess_path, 'w'), indent=1)

    dev = int(args.device) if str(args.device).isdigit() else args.device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, G.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, G.HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f'cannot open camera {args.device}')

    k, light = 0, 1
    while k < len(plan) and str(k) in sess['poses'] and any(s['k'] == k for s in sess['shots']):
        k += 1                                       # 이어서 하기: 끝난 자세는 건너뜀
    k = min(k, len(plan) - 1)
    while True:
        ok, frame = cap.read()
        if not ok:
            print('frame grab failed')
            break
        vis = cv2.resize(frame, (G.WIDTH, G.HEIGHT))
        p = plan[k]
        draw_outline(vis, cam, p['pose'], (0, 255, 255))
        meas_pose = sess['poses'].get(str(k))
        if meas_pose:
            draw_outline(vis, cam, meas_pose['pose'], (0, 255, 0))
        shots = [s['light'] for s in sess['shots'] if s['k'] == k]
        put(vis, f"pose {k + 1}/{len(plan)} {p['kind']}  light L{light}  measured={'yes' if meas_pose else 'no'}  "
                 f"shots={shots}  bg L{light}={'yes' if str(light) in sess['bg'] else 'no'}", 16)
        for i, t in enumerate(describe(p['meas'], p['pose'])):
            put(vis, f'{i + 1}) {t}', 36 + 18 * i, (0, 255, 255))
        put(vis, 'n/p pose  m measure  1/2 light  c capture  b background  q quit', G.HEIGHT - 8, (200, 200, 200))
        cv2.imshow('capture real test', vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('n'):
            k = min(k + 1, len(plan) - 1)
        elif key == ord('p'):
            k = max(k - 1, 0)
        elif key in (ord('1'), ord('2')):
            light = key - ord('0')
        elif key == ord('m'):
            d = ask_distances(describe(p['meas'], p['pose']))
            if d is None:
                continue
            sol = solve_pose(p['pose'], p['meas'], d)
            if sol is None:
                print('  could not solve (bad geometry) - check the numbers')
                continue
            du, dv = np.subtract(pose_to_uv(*sol[:2]), pose_to_uv(*p['pose'][:2]))
            print(f'  measured pose: offset from target du={du:+.1f} dv={dv:+.1f} mm, '
                  f'dyaw={math.degrees(G.wrap(sol[2] - p["pose"][2])):+.1f} deg')
            if math.hypot(du, dv) > 60 or abs(math.degrees(G.wrap(sol[2] - p['pose'][2]))) > 20:
                print('  warning: far from the target - check that the green outline matches the robot')
            sess['poses'][str(k)] = dict(pose=[float(v) for v in sol], dist=d, meas=p['meas'], kind=p['kind'])
            save_session()
        elif key == ord('c'):
            if not meas_pose:
                print('  measure first (m)')
                continue
            name = f"{k:03d}_{p['kind']}_L{light}"
            cv2.imwrite(str(out / 'images' / 'test' / f'{name}.jpg'), grab_median(cap), [cv2.IMWRITE_JPEG_QUALITY, 95])
            sess['shots'] = [s for s in sess['shots'] if s['name'] != name] + [dict(k=k, light=light, name=name)]
            save_session()
            print(f'  saved {name}.jpg')
        elif key == ord('b'):
            path = out / f'background_L{light}.png'
            cv2.imwrite(str(path), grab_median(cap, 10))
            sess['bg'][str(light)] = path.name
            save_session()
            print(f'  saved background for light L{light} (robot must be out of view)')
    cap.release()
    cv2.destroyAllWindows()

    # 정리: poses.csv, measurements.csv, 배경 복사
    c = calib
    with open(out / 'poses.csv', 'w', newline='') as fh, open(out / 'measurements.csv', 'w', newline='') as fm:
        wr = csv.writer(fh)
        wr.writerow(['split', 'name', 'kind', 'x_m', 'y_m', 'yaw_rad', 'u_mm', 'v_mm', 'cam_x', 'cam_y', 'cam_z',
                     'cam_qw', 'cam_qx', 'cam_qy', 'cam_qz', 'light_scale'])
        wm = csv.writer(fm)
        wm.writerow(['name', 'pose_index', 'light', 'corner_1', 'line_1', 'dist_1_mm', 'corner_2', 'line_2', 'dist_2_mm',
                     'corner_3', 'line_3', 'dist_3_mm'])
        missing_bg = set()
        for s in sorted(sess['shots'], key=lambda s: s['name']):
            mp = sess['poses'][str(s['k'])]
            x, y, yaw = mp['pose']
            u, v = pose_to_uv(x, y)
            wr.writerow(['test', s['name'], mp['kind'], f'{x:.5f}', f'{y:.5f}', f'{yaw:.5f}', f'{u:.1f}', f'{v:.1f}',
                         *c['pos'], *c['quat_wxyz'], s['light']])
            wm.writerow([s['name'], s['k'], s['light']] + sum(([n, LINES[j][4], dd] for (n, j), dd in
                                                               zip(mp['meas'], mp['dist'])), []))
            bg = sess['bg'].get(str(s['light']))
            if bg:
                shutil.copyfile(out / bg, out / 'background' / 'test' / f"{s['name']}.png")
            else:
                missing_bg.add(s['light'])
    print(f"wrote {len(sess['shots'])} images to {out}/poses.csv")
    if missing_bg:
        print(f'warning: no background for light(s) {sorted(missing_bg)} - P-B cannot run on those yet (press b)')


if __name__ == '__main__':
    main()
