#!/usr/bin/env python3
"""Closed-loop parking in the sim (kinematic): the robot parks using ONLY the ceiling-camera recognition.

Each trial
  1. the robot is put at a random position and heading in the entrance corridor (only where it can turn
     in place without touching tape); a slot (A-D) and a mode (front / rear) are chosen (or given)
  2. every 0.1 s: ceiling image rendered (calibrated camera, lens distortion, lighting / noise as the
     dataset) -> recognition (pa / pb / ps) -> filter (wheel odometry predicts, camera corrects;
     front/back kept by odometry after the start) -> controller acts on the ESTIMATED pose only
  3. route through waypoints that keep tape clear:
       entrance corridor -> entrance centre (u 346 mm) -> up through the entrance gap
       -> staging point in front of the slot -> turn to the slot axis -> slot centre line in
       (forward = front-in, backward = rear-in) -> stop
  4. scoring uses the ground-truth pose: tape contact (robot footprint vs tape, every 10 ms),
     final lateral / depth / heading error, front/back, time.
The robot moves kinematically (pose set directly, wheel speed noise + odometry error as in
gen_drive_sequences.py); no contact physics, so contact is detected, not felt.

usage:
  python tools/park_closed_loop.py --method pa --trials 1 --viewer           # watch one trial
  python tools/park_closed_loop.py --method pa --trials 20 --video 3         # 20 trials, mp4 of the first 3
  python tools/park_closed_loop.py --method ps --weights runs/segment/runs/ps/yolo11n_seed0/weights/best.pt
  python tools/park_closed_loop.py --slot B --mode rear --trials 1 --video 1
output: results/park_<method>/trials.csv, trial_XXX.mp4 (ceiling camera with estimate / ground truth)
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

CTRL_DT = S.DT                     # 10 ms control / motion step
FRAME_DT = 0.1                     # camera 10 Hz
V_MAX, V_SLOW, W_MAX = 0.10, 0.04, 1.2
ENTRANCE_U = 346.0                 # centre of the entrance gap (tape ends at u 195 and 497)
BELOW_GAP_V, ABOVE_GAP_V = 700.0, 480.0
STAGING_V = 420.0
PARK_V = {'front': 102.5, 'rear': 136.5}   # axle v that centres the body (118 mm long) in the slot (v 19..220)
SETTLE_S, HOLD_S, TIME_LIMIT_S = 0.5, 1.0, 60.0
# success limits (fixed before the first trial)
OK_LATERAL_MM, OK_DEPTH_MM, OK_YAW_DEG = 20.0, 20.0, 5.0


# ---------------------------------------------------------------- geometry / contact (floor mm)
def footprint(x, y, yaw):
    """Robot footprint boxes (chassis + both wheels) as floor-mm polygons."""
    R = G.rot(yaw)
    polys = []
    for x0, x1, y0, y1, z0, z1 in G.ROBOT_BOXES:
        if z0 >= 0.1:            # lidar sits inside the chassis outline
            continue
        c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ R.T + [x, y]
        polys.append(np.column_stack([D.START_U - c[:, 1] * 1000, D.START_V - c[:, 0] * 1000]))
    return polys


def overlaps(poly, rect):
    """Separating-axis test: convex polygon (N,2) vs axis-aligned rectangle (u, v, w, h)."""
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
    """Names of the tapes the ground-truth footprint touches."""
    hit = []
    for k, rect in enumerate(D.TAPES):
        if any(overlaps(p, rect) for p in footprint(x, y, yaw)):
            hit.append(k)
    return hit


def sample_start(rng):
    """Random pose in the entrance corridor where an in-place turn touches no tape."""
    for _ in range(1000):
        u, v, yaw = rng.uniform(300, 440), rng.uniform(690, 870), rng.uniform(-math.pi, math.pi)
        x, y = D.mm_to_world(u, v)[:2]
        if all(not touching(x, y, a) for a in np.radians(np.arange(0, 360, 10))):
            return x, y, yaw
    raise RuntimeError('no free start pose')


# ---------------------------------------------------------------- filter (online)
class Filter:
    """Odometry predicts; camera corrects. Front/back is fixed at the start (majority of the first
    measurements while standing still) and then kept by odometry; a camera pose pointing the other way
    is turned around; a heading far from the prediction is ignored (position still used); a position
    far from the prediction is skipped, and 30 such frames in a row (3 s: robot moved by hand) -> start again."""
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
        if abs(G.wrap(m[2] - self.state[2])) > math.pi / 2:      # front/back from odometry, not the camera
            m[2] = G.wrap(m[2] + math.pi)
        far = math.hypot(m[0] - self.state[0], m[1] - self.state[1]) > self.GATE_M
        if far:                                                   # position disagrees: outlier or moved by hand
            self.skips += 1
            if self.skips >= self.RESET_AFTER:                    # 3 s of disagreement: start again
                self.state, self.skips = None, 0
            return
        self.skips = 0
        if abs(math.degrees(G.wrap(m[2] - self.state[2]))) > self.GATE_DEG:
            m[2] = self.state[2]                                  # heading outlier: use the position only
        self.state[0] += self.GAIN_POS * (m[0] - self.state[0])
        self.state[1] += self.GAIN_POS * (m[1] - self.state[1])
        self.state[2] = G.wrap(self.state[2] + self.GAIN_YAW * G.wrap(m[2] - self.state[2]))


def S_body_delta(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = math.cos(a[2]), math.sin(a[2])
    return c * dx + s * dy, -s * dx + c * dy, G.wrap(b[2] - a[2])


# ---------------------------------------------------------------- controller (acts on the estimate)
def plan(slot, mode):
    """Waypoint phases in world metres."""
    w = lambda u, v: tuple(D.mm_to_world(u, v)[:2])
    su = D.SLOT_U[slot]
    final_yaw = 0.0 if mode == 'front' else math.pi
    return [('settle', SETTLE_S),
            ('turn_to', w(ENTRANCE_U, BELOW_GAP_V)), ('go', (None, w(ENTRANCE_U, BELOW_GAP_V), +1)),
            ('turn_to', w(ENTRANCE_U, ABOVE_GAP_V)), ('go', (None, w(ENTRANCE_U, ABOVE_GAP_V), +1)),
            ('turn_to', w(su, STAGING_V)), ('go', (None, w(su, STAGING_V), +1)),
            ('turn', final_yaw),
            ('go', (w(su, STAGING_V), w(su, PARK_V[mode]), +1 if mode == 'front' else -1)),
            ('turn', final_yaw),                       # last few degrees in place (safe inside the slot)
            ('hold', HOLD_S)]


class Controller:
    def __init__(self, phases):
        self.phases, self.i, self.t, self.line = phases, 0, 0.0, None

    @property
    def name(self):
        return self.phases[self.i][0] if self.i < len(self.phases) else 'done'

    def command(self, est):
        """(v, w) from the estimated pose; advances phases."""
        while self.i < len(self.phases):
            kind, arg = self.phases[self.i]
            if kind in ('settle', 'hold'):
                if self.t >= arg and (kind == 'hold' or est is not None):
                    self._next(); continue
                return 0.0, 0.0
            if est is None:                                   # lost: stop and wait for the filter
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
                want = math.atan2(d[1], d[0]) - math.atan(lateral / 0.04)    # steer back onto the line
                w = float(np.clip(3.0 * G.wrap(want - motion), -W_MAX, W_MAX))
                v = sign * float(np.clip(1.2 * remaining, 0.015, V_MAX if remaining > 0.06 else V_SLOW))
                return v, w
        return 0.0, 0.0

    def tick(self):
        self.t += CTRL_DT

    def _next(self):
        self.i, self.t, self.line = self.i + 1, 0.0, None


# ---------------------------------------------------------------- rendering / video
def draw_overlay(img, cam, gt, est, slot, mode, phase, contact, t):
    vis = img.copy()
    su = D.SLOT_U[slot]
    box = [D.mm_to_world(u, v) for u, v in ((su - 95, 19), (su + 95, 19), (su + 95, 220), (su - 95, 220))]
    pix = cam.project(np.array([[p[0], p[1], 0.002] for p in box]))
    cv2.polylines(vis, [np.round(pix).astype(np.int32)], True, (0, 200, 255), 2)

    def arrow(p, color):
        q = cam.project(np.array([[p[0], p[1], 0.03], [p[0] + 0.08 * math.cos(p[2]), p[1] + 0.08 * math.sin(p[2]), 0.03]]))
        cv2.arrowedLine(vis, tuple(np.round(q[0]).astype(int)), tuple(np.round(q[1]).astype(int)), color, 2, tipLength=0.3)
    arrow(gt, (0, 200, 0))
    if est is not None:
        arrow(est, (0, 0, 255))
    cv2.putText(vis, f'slot {"ABCD"[slot]} {mode}  {phase}  t={t:4.1f}s', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, 'green: truth  red: estimate', (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if contact:
        cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 0, 255), 6)
        cv2.putText(vis, 'TAPE CONTACT', (420, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


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
    cam = G.Camera(D.CAM_POS, D.CAM_QUAT)                 # calibrated camera = what the estimate uses
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
            writer = cv2.VideoWriter(str(out / f'trial_{k:03d}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H))
        t, next_frame, contact_steps, touched, n_meas, n_frames = 0.0, 0.0, 0, set(), 0, 0
        t_wall = time.perf_counter()
        while ctrl.name != 'done' and t < TIME_LIMIT_S:
            filt.predict((robot.ox, robot.oy, robot.oyaw))
            hit = touching(robot.x, robot.y, robot.yaw)
            if hit:
                contact_steps += 1
                touched.update(hit)
            if t >= next_frame - 1e-9:                                   # camera frame
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
                    writer.write(draw_overlay(img, cam, (robot.x, robot.y, robot.yaw), filt.state, slot, mode,
                                              ctrl.name, bool(hit), t))
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

        # score with the ground truth
        su = D.SLOT_U[slot]
        tx, ty = D.mm_to_world(su, PARK_V[mode])[:2]
        final_yaw = 0.0 if mode == 'front' else math.pi
        lateral = abs(robot.y - ty) * 1000              # slot axis runs along world x
        depth = abs(robot.x - tx) * 1000
        yaw_err = abs(math.degrees(G.wrap(robot.yaw - final_yaw)))
        done = ctrl.name == 'done'
        success = (done and contact_steps == 0 and lateral <= OK_LATERAL_MM and depth <= OK_DEPTH_MM
                   and yaw_err <= OK_YAW_DEG)
        row = dict(trial=k, slot='ABCD'[slot], mode=mode, start_x=round(start[0], 4), start_y=round(start[1], 4),
                   start_yaw_deg=round(math.degrees(start[2]), 1), finished=done, success=success,
                   contact_ms=contact_steps * 10, tapes_touched=' '.join(map(str, sorted(touched))),
                   lateral_mm=round(lateral, 1), depth_mm=round(depth, 1), yaw_err_deg=round(yaw_err, 1),
                   front_back_ok=yaw_err < 90, time_s=round(t, 1), detect_rate=round(n_meas / max(1, n_frames), 3))
        rows.append(row)
        print(f"trial {k}: slot {row['slot']} {mode:5s} {'OK  ' if success else 'FAIL'} contact={row['contact_ms']} ms "
              f"lateral={row['lateral_mm']} mm depth={row['depth_mm']} mm yaw={row['yaw_err_deg']} deg "
              f"time={row['time_s']} s", flush=True)

    with open(out / 'trials.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    n = len(rows)
    ok = sum(r['success'] for r in rows)
    print(f'method={args.method} success {ok}/{n}  contact-free {sum(r["contact_ms"] == 0 for r in rows)}/{n}  '
          f'front/back ok {sum(r["front_back_ok"] for r in rows)}/{n}  -> {out}')
    if viewer is not None:
        viewer.close()


if __name__ == '__main__':
    main()
