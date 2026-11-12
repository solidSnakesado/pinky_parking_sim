#!/usr/bin/env python3
"""시뮬 폐루프 주차 (운동학): 로봇이 천장 카메라 인식만으로 주차한다.

시행 1회
  1. 입구 통로의 무작위 위치·방향에 로봇을 놓는다(제자리 회전해도 테이프에 닿지 않는 곳만).
     칸(A~D)과 방식(전면 / 후면)을 고른다(또는 옵션으로 지정).
  2. 0.1초마다: 천장 이미지 렌더링(보정 카메라, 렌즈 왜곡, 데이터셋과 같은 조명 / 노이즈)
     -> 인식(pa / pb / ps) -> 필터(바퀴 오도메트리가 예측하고 카메라가 보정, 출발 후에는 앞/뒤를
     오도메트리로 유지) -> 제어기는 추정 자세만 보고 움직임
  3. 테이프를 피하는 경유점을 따라 이동:
       입구 통로 -> 입구 중심(u 346 mm) -> 입구 틈을 지나 위로
       -> 칸 앞 대기점 -> 칸 축 방향으로 회전 -> 칸 중심선을 따라 진입
       (전진 = 전면 주차, 후진 = 후면 주차) -> 정지
  4. 판정은 정답 자세로 한다: 테이프 접촉(로봇 외곽 vs 테이프, 10 ms마다),
     최종 좌우 / 깊이 / 방향 오차, 앞/뒤, 시간.
     판정 두 가지: 정밀 성공(좌우·깊이 ≤ 20 mm, 방향 ≤ 5°, 접촉 없음) /
     주차 가능(접촉 없음 + 로봇 외곽이 칸 안에 완전히 들어감, 칸 경계까지의 여유 거리도 기록).
로봇은 운동학적으로 움직인다(자세를 직접 설정, 바퀴 속도 노이즈 + 오도메트리 오차는
gen_drive_sequences.py와 같음). 접촉 물리가 없으므로 접촉은 "느끼는" 것이 아니라 판정만 한다.

사용법:
  python tools/park_closed_loop.py --method pa --trials 1 --viewer           # 1회 보기
  python tools/park_closed_loop.py --method pa --trials 20 --video 3         # 20회, 앞 3회는 mp4 저장
  python tools/park_closed_loop.py --method ps --weights runs/segment/runs/ps/yolo11n_seed0/weights/best.pt
  python tools/park_closed_loop.py --slot B --mode rear --trials 1 --video 1
출력: results/park_<method>/trials.csv, trial_XXX.mp4 (추정 / 정답을 표시한 천장 카메라 영상)
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_parking_dataset as D  # noqa: E402
import gen_drive_sequences as S  # noqa: E402
import parking_geom as G  # noqa: E402
from eval_sequence import detector  # noqa: E402

CTRL_DT = S.DT                     # 10 ms 제어 / 움직임 간격
FRAME_DT = 0.1                     # 카메라 10 Hz
V_MAX, V_SLOW, W_MAX = 0.10, 0.04, 1.2
ENTRANCE_U = 346.0                 # 입구 틈의 중심 (테이프 끝이 u 195와 497)
BELOW_GAP_V, ABOVE_GAP_V = 700.0, 480.0
STAGING_V = 420.0
PARK_V = {'front': 102.5, 'rear': 136.5}   # 몸체(길이 118 mm)가 칸(v 19..220) 가운데 오게 하는 바퀴 축 v
SETTLE_S, HOLD_S, TIME_LIMIT_S = 0.5, 1.0, 60.0
# 성공 기준 (첫 시행 전에 고정)
OK_LATERAL_MM, OK_DEPTH_MM, OK_YAW_DEG = 20.0, 20.0, 5.0


# ---------------------------------------------------------------- 형상 / 접촉 (바닥 mm)
def footprint(x, y, yaw):
    """로봇 외곽 상자(몸통 + 양쪽 바퀴)를 바닥 mm 다각형으로."""
    R = G.rot(yaw)
    polys = []
    for x0, x1, y0, y1, z0, z1 in G.ROBOT_BOXES:
        if z0 >= 0.1:            # 라이다는 몸통 외곽 안쪽에 있음
            continue
        c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ R.T + [x, y]
        polys.append(np.column_stack([D.START_U - c[:, 1] * 1000, D.START_V - c[:, 0] * 1000]))
    return polys


def overlaps(poly, rect):
    """분리축 검사: 볼록 다각형 (N,2) vs 축 정렬 사각형 (u, v, w, h)."""
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
    """정답 자세의 로봇 외곽이 닿은 테이프 번호 목록."""
    hit = []
    for k, rect in enumerate(D.TAPES):
        if any(overlaps(p, rect) for p in footprint(x, y, yaw)):
            hit.append(k)
    return hit


def slot_rect(slot):
    """칸 안쪽 범위 (u0, u1, v0, v1) mm: 좌우 = 양쪽 구분선의 안쪽 가장자리, 깊이 = 위 벽 테이프 안쪽 ~ 구분선 끝."""
    dividers = sorted((t for t in D.TAPES if t[1] == 0 and t[3] == 220), key=lambda t: t[0])   # S1~S5
    wall = next(t for t in D.TAPES if t[0] == 0 and t[1] == 0 and t[2] == 900)               # P1 (위 벽)
    left, right = dividers[slot], dividers[slot + 1]
    return left[0] + left[2], right[0], wall[1] + wall[3], left[1] + left[3]


def park_margin(x, y, yaw, slot):
    """로봇 외곽(몸통 + 바퀴) 꼭짓점에서 칸 경계까지의 최소 거리 mm. 음수 = 칸 밖으로 나감.
    외곽 상자들은 볼록이고 칸은 축 정렬 사각형이므로 꼭짓점만 보면 정확하다."""
    u0, u1, v0, v1 = slot_rect(slot)
    pts = np.vstack(footprint(x, y, yaw))
    return float(np.min(np.minimum.reduce([pts[:, 0] - u0, u1 - pts[:, 0], pts[:, 1] - v0, v1 - pts[:, 1]])))


def sample_start(rng):
    """입구 통로의 무작위 자세 (제자리 회전해도 테이프에 닿지 않는 곳)."""
    for _ in range(1000):
        u, v, yaw = rng.uniform(300, 440), rng.uniform(690, 870), rng.uniform(-math.pi, math.pi)
        x, y = D.mm_to_world(u, v)[:2]
        if all(not touching(x, y, a) for a in np.radians(np.arange(0, 360, 10))):
            return x, y, yaw
    raise RuntimeError('no free start pose')


# ---------------------------------------------------------------- 필터 (실시간)
class Filter:
    """오도메트리가 예측하고 카메라가 보정한다. 앞/뒤는 출발 시(정지 상태의 첫 측정들의 다수결)
    정하고 이후에는 오도메트리로 유지한다. 반대 방향을 가리키는 카메라 자세는 180° 돌려서 쓴다.
    예측과 크게 다른 방향은 무시한다(위치는 사용). 예측에서 먼 위치는 건너뛰며, 이런 프레임이
    30번 연속이면(3초: 사람이 로봇을 옮김) 처음부터 다시 시작한다."""
    GAIN_POS, GAIN_YAW, GATE_M, GATE_DEG, INIT_N, RESET_AFTER = 0.4, 0.3, 0.04, 30.0, 3, 30

    def __init__(self):
        self.state, self.buf, self.prev, self.skips = None, [], None, 0

    def predict(self, odom):
        if self.prev is not None and self.state is not None:
            d = S_body_delta(self.prev, odom)
            c, s = math.cos(self.state[2]), math.sin(self.state[2])
            self.state = [self.state[0] + c * d[0] - s * d[1], self.state[1] + s * d[0] + c * d[1],
                          G.wrap(self.state[2] + d[2])]
        self.prev = odom

    def update(self, meas):
        if meas is None:
            return
        if self.state is None:
            self.buf.append(list(meas))
            if len(self.buf) >= self.INIT_N:
                ref = self.buf[0][2]
                same = [m for m in self.buf if abs(G.wrap(m[2] - ref)) < math.pi / 2]
                other = [m for m in self.buf if abs(G.wrap(m[2] - ref)) >= math.pi / 2]
                group = same if len(same) >= len(other) else other
                yaw = math.atan2(np.mean([math.sin(m[2]) for m in group]), np.mean([math.cos(m[2]) for m in group]))
                self.state = [float(np.median([m[0] for m in self.buf])),
                              float(np.median([m[1] for m in self.buf])), yaw]
                self.buf = []
            return
        m = list(meas)
        if abs(G.wrap(m[2] - self.state[2])) > math.pi / 2:      # 앞/뒤는 카메라가 아니라 오도메트리 기준
            m[2] = G.wrap(m[2] + math.pi)
        far = math.hypot(m[0] - self.state[0], m[1] - self.state[1]) > self.GATE_M
        if far:                                                   # 위치가 맞지 않음: 이상값 또는 사람이 옮김
            self.skips += 1
            if self.skips >= self.RESET_AFTER:                    # 3초 동안 불일치: 다시 시작
                self.state, self.skips = None, 0
            return
        self.skips = 0
        if abs(math.degrees(G.wrap(m[2] - self.state[2]))) > self.GATE_DEG:
            m[2] = self.state[2]                                  # 방향 이상값: 위치만 사용
        self.state[0] += self.GAIN_POS * (m[0] - self.state[0])
        self.state[1] += self.GAIN_POS * (m[1] - self.state[1])
        self.state[2] = G.wrap(self.state[2] + self.GAIN_YAW * G.wrap(m[2] - self.state[2]))


def S_body_delta(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = math.cos(a[2]), math.sin(a[2])
    return c * dx + s * dy, -s * dx + c * dy, G.wrap(b[2] - a[2])


# ---------------------------------------------------------------- 제어기 (추정 자세로 동작)
def plan(slot, mode):
    """경유점 단계 목록 (월드 미터)."""
    w = lambda u, v: tuple(D.mm_to_world(u, v)[:2])
    su = D.SLOT_U[slot]
    final_yaw = 0.0 if mode == 'front' else math.pi
    return [('settle', SETTLE_S),
            ('turn_to', w(ENTRANCE_U, BELOW_GAP_V)), ('go', (None, w(ENTRANCE_U, BELOW_GAP_V), +1)),
            ('turn_to', w(ENTRANCE_U, ABOVE_GAP_V)), ('go', (None, w(ENTRANCE_U, ABOVE_GAP_V), +1)),
            ('turn_to', w(su, STAGING_V)), ('go', (None, w(su, STAGING_V), +1)),
            ('turn', final_yaw),
            ('go', (w(su, STAGING_V), w(su, PARK_V[mode]), +1 if mode == 'front' else -1)),
            ('turn', final_yaw),                       # 마지막 몇 도는 제자리 회전 (칸 안에서 안전)
            ('hold', HOLD_S)]


class Controller:
    def __init__(self, phases):
        self.phases, self.i, self.t, self.line = phases, 0, 0.0, None

    @property
    def name(self):
        return self.phases[self.i][0] if self.i < len(self.phases) else 'done'

    def command(self, est):
        """추정 자세로 (v, w)를 계산하고, 단계를 넘김."""
        while self.i < len(self.phases):
            kind, arg = self.phases[self.i]
            if kind in ('settle', 'hold'):
                if self.t >= arg and (kind == 'hold' or est is not None):
                    self._next(); continue
                return 0.0, 0.0
            if est is None:                                   # 위치를 잃음: 멈추고 필터를 기다림
                return 0.0, 0.0
            x, y, yaw = est
            if kind in ('turn', 'turn_to'):
                target = arg if kind == 'turn' else math.atan2(arg[1] - y, arg[0] - x)
                err = G.wrap(target - yaw)
                if abs(err) < math.radians(2.0) or (kind == 'turn_to' and math.hypot(arg[0] - x, arg[1] - y) < 0.01):
                    self._next(); continue
                return 0.0, float(np.clip(3.0 * err, -W_MAX, W_MAX))
            if kind == 'go':
                a, b, sign = arg
                if self.line is None:
                    self.line = (np.array(a if a is not None else (x, y)), np.array(b))
                A, B = self.line
                L = np.linalg.norm(B - A)
                if L < 0.005:
                    self._next(); continue
                d = (B - A) / L
                p = np.array([x, y]) - A
                along, lateral = p @ d, d[0] * p[1] - d[1] * p[0]
                remaining = L - along
                if remaining < 0.003:
                    self._next(); continue
                motion = yaw if sign > 0 else G.wrap(yaw + math.pi)
                want = math.atan2(d[1], d[0]) - math.atan(lateral / 0.04)    # 선 위로 돌아가도록 조향
                w = float(np.clip(3.0 * G.wrap(want - motion), -W_MAX, W_MAX))
                v = sign * float(np.clip(1.2 * remaining, 0.015, V_MAX if remaining > 0.06 else V_SLOW))
                return v, w
        return 0.0, 0.0

    def tick(self):
        self.t += CTRL_DT

    def _next(self):
        self.i, self.t, self.line = self.i + 1, 0.0, None


# ---------------------------------------------------------------- 렌더링 / 영상
PANEL_W = 300
METHOD_NAME = {'pa': 'P-A 규칙', 'pb': 'P-B 배경 차분', 'ps': 'P-S 세그멘테이션'}
PHASE_NAME = {'settle': '초기 확인', 'turn_to': '회전', 'turn': '정렬 회전', 'go': '이동', 'hold': '주차 완료', 'done': '종료'}


def find_korean_font():
    """한글이 들어 있는 글꼴 파일 (fontconfig 먼저, 그다음 흔한 경로). None이면 영어 표시."""
    import subprocess
    try:
        out = subprocess.run(['fc-list', ':lang=ko', 'file'], capture_output=True, text=True, timeout=5).stdout
        files = [l.split(':')[0].strip() for l in out.splitlines() if l.strip()]
        files.sort(key=lambda f: (('Sans' not in f and 'Gothic' not in f), 'Bold' in f, f))
        if files:
            return files[0]
    except Exception:
        pass
    for f in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'):
        if os.path.exists(f):
            return f
    return None


def target_errors(pose, slot, mode):
    """칸 목표 대비 자세의 (좌우 mm, 깊이 mm, 방향 deg). 깊이 > 0 = 덜 들어감."""
    tx, ty = D.mm_to_world(D.SLOT_U[slot], PARK_V[mode])[:2]
    final_yaw = 0.0 if mode == 'front' else math.pi
    return (abs(pose[1] - ty) * 1000, (tx - pose[0]) * 1000,
            abs(math.degrees(G.wrap(pose[2] - final_yaw))))


class Overlay:
    """칸 / 화살표를 그린 천장 이미지 + 실시간 오차와 판정을 보여 주는 한글 정보 패널."""

    def __init__(self, method):
        from PIL import ImageFont
        path = find_korean_font()
        self.ko = path is not None
        self.font = ImageFont.truetype(path, 17) if path else ImageFont.load_default()
        self.small = ImageFont.truetype(path, 14) if path else ImageFont.load_default()
        self.big = ImageFont.truetype(path, 26) if path else ImageFont.load_default()
        self.method = METHOD_NAME[method] if self.ko else method.upper()

    def draw(self, img, cam, gt, est, slot, mode, phase, contact_now, contact_ms, t, verdict):
        from PIL import Image, ImageDraw
        vis = img.copy()
        su = D.SLOT_U[slot]
        box = [D.mm_to_world(u, v) for u, v in ((su - 95, 19), (su + 95, 19), (su + 95, 220), (su - 95, 220))]
        pix = cam.project(np.array([[p[0], p[1], 0.002] for p in box]))
        cv2.polylines(vis, [np.round(pix).astype(np.int32)], True, (0, 200, 255), 2)

        def arrow(p, color):
            q = cam.project(np.array([[p[0], p[1], 0.03],
                                      [p[0] + 0.08 * math.cos(p[2]), p[1] + 0.08 * math.sin(p[2]), 0.03]]))
            cv2.arrowedLine(vis, tuple(np.round(q[0]).astype(int)), tuple(np.round(q[1]).astype(int)), color, 2,
                            tipLength=0.3)
        arrow(gt, (0, 200, 0))
        if est is not None:
            arrow(est, (0, 0, 255))
        if contact_now:
            cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 0, 255), 6)

        canvas = Image.new('RGB', (vis.shape[1] + PANEL_W, vis.shape[0]), 'white')
        canvas.paste(Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)), (0, 0))
        d = ImageDraw.Draw(canvas)
        x0, y = vis.shape[1] + 14, 12
        black, grey, red, green = (20, 20, 20), (110, 110, 110), (210, 30, 30), (20, 140, 40)
        L = (lambda ko, en: ko if self.ko else en)
        mode_txt = L('전면' if mode == 'front' else '후면', mode)
        d.text((x0, y), self.method, font=self.font, fill=black); y += 26
        d.text((x0, y), L(f'목표: {"ABCD"[slot]}칸 {mode_txt} 주차', f'target: slot {"ABCD"[slot]} {mode_txt}'),
               font=self.font, fill=black); y += 26
        d.text((x0, y), L(f'단계: {PHASE_NAME.get(phase, phase)}   {t:4.1f} s', f'phase: {phase}  {t:4.1f} s'),
               font=self.font, fill=black); y += 36

        d.text((x0 + 92, y), L('실제', 'truth'), font=self.small, fill=grey)
        d.text((x0 + 182, y), L('추정', 'estimate'), font=self.small, fill=grey); y += 22
        g = target_errors(gt, slot, mode)
        e = target_errors(est, slot, mode) if est is not None else None
        limits = (OK_LATERAL_MM, OK_DEPTH_MM, OK_YAW_DEG)
        rows = [(L('좌우 오차', 'lateral'), '{:.1f} mm'), (L('깊이 오차', 'depth'), '{:.1f} mm'),
                (L('방향 오차', 'heading'), '{:.1f}°')]
        for i, (name, fmt) in enumerate(rows):
            d.text((x0, y), name, font=self.font, fill=black)
            gv = abs(g[i])
            d.text((x0 + 92, y), fmt.format(gv), font=self.font, fill=red if gv > limits[i] else black)
            d.text((x0 + 182, y), '-' if e is None else fmt.format(abs(e[i])), font=self.font, fill=grey)
            y += 26
        d.text((x0, y), L('테이프 접촉', 'tape contact'), font=self.font, fill=black)
        d.text((x0 + 92, y), L('있음', 'yes') + f' ({contact_ms} ms)' if contact_ms else L('없음', 'none'),
               font=self.font, fill=red if contact_ms else black); y += 30
        d.text((x0, y), L('기준: 좌우·깊이 ≤ 20 mm, 방향 ≤ 5°', 'limits: 20 mm, 20 mm, 5°'), font=self.small,
               fill=grey); y += 20
        d.text((x0, y), L('실제 = 정답 자세 기준(판정에 사용)', 'truth = ground truth (scored)'), font=self.small,
               fill=grey); y += 18
        d.text((x0, y), L('추정 = 제어기가 보는 값', 'estimate = what the controller sees'), font=self.small,
               fill=grey); y += 30
        d.text((x0, y), L('초록 화살표 = 정답, 빨강 = 추정', 'green = truth, red = estimate'), font=self.small,
               fill=grey); y += 18
        d.text((x0, y), L('노란 상자 = 목표 칸', 'yellow box = target slot'), font=self.small, fill=grey); y += 30
        if verdict is not None:
            ok, reasons, parkable, margin = verdict
            d.text((x0, y), L('결과: 성공', 'SUCCESS') if ok else L('결과: 실패', 'FAIL'), font=self.big,
                   fill=green if ok else red); y += 34
            for r in reasons:
                d.text((x0, y), r, font=self.small, fill=red); y += 18
            d.text((x0, y), L(f'주차 가능: {"예" if parkable else "아니오"} (여유 {margin:.1f} mm)',
                              f'parkable: {"yes" if parkable else "no"} (margin {margin:.1f} mm)'),
                   font=self.font, fill=green if parkable else red); y += 22
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def verdict_of(gt, slot, mode, contact_ms, finished, ko=True):
    lat, dep, yaw = target_errors(gt, slot, mode)
    reasons = []
    if contact_ms:
        reasons.append('테이프 접촉' if ko else 'tape contact')
    if lat > OK_LATERAL_MM:
        reasons.append(f'좌우 오차 {lat:.1f} mm > {OK_LATERAL_MM:.0f}' if ko else f'lateral {lat:.1f} mm')
    if abs(dep) > OK_DEPTH_MM:
        reasons.append(f'깊이 오차 {abs(dep):.1f} mm > {OK_DEPTH_MM:.0f}' if ko else f'depth {abs(dep):.1f} mm')
    if yaw > OK_YAW_DEG:
        reasons.append(f'방향 오차 {yaw:.1f}° > {OK_YAW_DEG:.0f}°' if ko else f'heading {yaw:.1f} deg')
    if not finished:
        reasons.append('시간 초과' if ko else 'time out')
    margin = park_margin(*gt, slot)
    parkable = finished and not contact_ms and margin >= 0
    return (not reasons, reasons, parkable, margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/.cache/pinky_mujoco/scene.xml')
    ap.add_argument('--method', default='pa', choices=['pa', 'pb', 'ps'])
    ap.add_argument('--weights', default=None)
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--seed', type=int, default=500)
    ap.add_argument('--slot', default=None, choices=list('ABCD'))
    ap.add_argument('--mode', default=None, choices=['front', 'rear'])
    ap.add_argument('--video', type=int, default=1, help='save mp4 for the first N trials')
    ap.add_argument('--viewer', action='store_true', help='show the MuJoCo viewer (real-time speed)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.method == 'ps' and not args.weights:
        raise SystemExit('--weights is required for ps')

    out = Path(args.out or f'results/park_{args.method}')
    out.mkdir(parents=True, exist_ok=True)
    detect = detector(args.method, args.weights)
    overlay = Overlay(args.method)
    W, H = 640, 480
    model, f, rw, rh = D.build(os.path.expanduser(args.model), W, H)
    data = mujoco.MjData(model)
    cam_id = model.camera('ceiling_camera').id
    map_x, map_y = D.distortion_map(W, H, f, rw, rh)
    wheel_q = [model.joint(f'{s}_wheel_joint').qposadr[0] for s in ('l', 'r')]
    base_z = float(model.qpos0[2])
    light0 = model.light_diffuse.copy()
    head0 = (model.vis.headlight.diffuse.copy(), model.vis.headlight.ambient.copy())
    renderer = mujoco.Renderer(model, rh, rw)
    cam = G.Camera(D.CAM_POS, D.CAM_QUAT)                 # 보정 카메라 = 추정에 쓰는 카메라
    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    def set_pose(robot, light, head):
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = [robot.x, robot.y, base_z] if robot else [-5.0, -5.0, base_z]
        data.qpos[3:7] = D.yaw_quat(robot.yaw if robot else 0.0)
        if robot:
            data.qpos[wheel_q] = [robot.wl, robot.wr]
        model.light_diffuse[:] = light0 * light
        model.vis.headlight.diffuse[:] = head[0]
        model.vis.headlight.ambient[:] = head[1]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)

    def ceiling():
        renderer.update_scene(data, camera='ceiling_camera')
        return D.remap_bilinear(renderer.render(), map_x, map_y)

    rows = []
    for k in range(args.trials):
        rng = np.random.default_rng([args.seed, k])
        start = sample_start(rng)
        slot = 'ABCD'.index(args.slot) if args.slot else int(rng.integers(4))
        mode = args.mode or str(rng.choice(['front', 'rear']))
        robot = S.Robot(start, rng)
        light = rng.uniform(0.7, 1.2)
        head = (head0[0] * rng.uniform(0.7, 1.2), head0[1] * rng.uniform(0.7, 1.3))
        contrast, offset, cast = rng.uniform(0.85, 1.15), rng.uniform(-20, 20), rng.uniform(0.97, 1.03, 3)
        noise_sd = rng.uniform(1, 5)
        model.cam_pos[cam_id], model.cam_quat[cam_id] = D.CAM_POS, D.CAM_QUAT
        set_pose(None, 1.0, head0)
        bg = cv2.cvtColor(np.clip(ceiling(), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        filt, ctrl = Filter(), Controller(plan(slot, mode))
        writer = None
        if k < args.video:
            writer = cv2.VideoWriter(str(out / f'trial_{k:03d}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (W + PANEL_W, H))
        t, next_frame, contact_steps, touched, n_meas, n_frames = 0.0, 0.0, 0, set(), 0, 0
        t_wall = time.perf_counter()
        while ctrl.name != 'done' and t < TIME_LIMIT_S:
            filt.predict((robot.ox, robot.oy, robot.oyaw))
            hit = touching(robot.x, robot.y, robot.yaw)
            if hit:
                contact_steps += 1
                touched.update(hit)
            if t >= next_frame - 1e-9:                                   # 카메라 프레임
                set_pose(robot, light, head)
                rgb = ceiling()
                rgb = (rgb - 128) * contrast + 128 + offset
                rgb = rgb * cast + rng.normal(0, noise_sd, rgb.shape)
                img = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                img = cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], cv2.IMREAD_COLOR)
                wheels, body = detect(img, cam, bg)
                pose = G.pose_from_masks(cam, wheels, body)
                n_frames += 1
                n_meas += pose is not None
                filt.update(None if pose is None else pose[:3])
                if os.environ.get('PARK_DEBUG') and filt.state is not None:
                    print(f'{t:5.1f} {ctrl.name:8s} gt=({robot.x:.3f},{robot.y:.3f},{math.degrees(robot.yaw):6.1f}) '
                          f'est=({filt.state[0]:.3f},{filt.state[1]:.3f},{math.degrees(filt.state[2]):6.1f}) '
                          f'meas={None if pose is None else (round(pose[0],3), round(pose[1],3), round(math.degrees(pose[2]),1))}')
                if writer is not None:
                    vd = (verdict_of((robot.x, robot.y, robot.yaw), slot, mode, contact_steps * 10, True, overlay.ko)
                          if ctrl.name == 'hold' else None)
                    writer.write(overlay.draw(img, cam, (robot.x, robot.y, robot.yaw), filt.state, slot, mode,
                                              ctrl.name, bool(hit), contact_steps * 10, t, vd))
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    time.sleep(max(0.0, FRAME_DT - (time.perf_counter() - t_wall)))
                    t_wall = time.perf_counter()
                next_frame += FRAME_DT
            v, w = ctrl.command(filt.state)
            robot.step(v, w)
            ctrl.tick()
            t += CTRL_DT
        if writer is not None:
            writer.release()

        # 정답 자세로 판정
        su = D.SLOT_U[slot]
        tx, ty = D.mm_to_world(su, PARK_V[mode])[:2]
        final_yaw = 0.0 if mode == 'front' else math.pi
        lateral = abs(robot.y - ty) * 1000              # 칸 축은 월드 x 방향
        depth = abs(robot.x - tx) * 1000
        yaw_err = abs(math.degrees(G.wrap(robot.yaw - final_yaw)))
        done = ctrl.name == 'done'
        success = (done and contact_steps == 0 and lateral <= OK_LATERAL_MM and depth <= OK_DEPTH_MM
                   and yaw_err <= OK_YAW_DEG)
        margin = park_margin(robot.x, robot.y, robot.yaw, slot)    # 주차 가능 기준: 외곽이 칸 안
        parkable = done and contact_steps == 0 and margin >= 0
        row = dict(trial=k, slot='ABCD'[slot], mode=mode, start_x=round(start[0], 4), start_y=round(start[1], 4),
                   start_yaw_deg=round(math.degrees(start[2]), 1), finished=done, success=success,
                   contact_ms=contact_steps * 10, tapes_touched=' '.join(map(str, sorted(touched))),
                   lateral_mm=round(lateral, 1), depth_mm=round(depth, 1), yaw_err_deg=round(yaw_err, 1),
                   front_back_ok=yaw_err < 90, time_s=round(t, 1), detect_rate=round(n_meas / max(1, n_frames), 3),
                   parkable=parkable, margin_mm=round(margin, 1), final_x=round(robot.x, 5), final_y=round(robot.y, 5),
                   final_yaw_deg=round(math.degrees(robot.yaw), 2))
        rows.append(row)
        print(f"trial {k}: slot {row['slot']} {mode:5s} {'OK  ' if success else 'FAIL'} contact={row['contact_ms']} ms "
              f"lateral={row['lateral_mm']} mm depth={row['depth_mm']} mm yaw={row['yaw_err_deg']} deg "
              f"time={row['time_s']} s parkable={'yes' if parkable else 'no'} margin={row['margin_mm']} mm", flush=True)

    with open(out / 'trials.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    n = len(rows)
    ok = sum(r['success'] for r in rows)
    print(f'method={args.method} success {ok}/{n}  parkable {sum(r["parkable"] for r in rows)}/{n}  contact-free {sum(r["contact_ms"] == 0 for r in rows)}/{n}  '
          f'front/back ok {sum(r["front_back_ok"] for r in rows)}/{n}  -> {out}')
    if viewer is not None:
        viewer.close()


if __name__ == '__main__':
    main()
