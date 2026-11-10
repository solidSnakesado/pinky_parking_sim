#!/usr/bin/env python3
"""Continuous parking drives in the sim, for the +F experiment (temporal filter / odometry fusion).

Each run: the robot starts at a random pose in the entrance area, gets a random slot (A-D) and a
random mode (front / rear), and drives a simple parking motion:
  turn to the staging point in front of the slot -> drive there -> turn to the slot axis
  -> drive into the slot (forward = front-in, backward = rear-in) -> hold.
Frames are saved at 10 Hz (same as the ceiling camera node) with
  - the ceiling camera image (same calibrated camera, lens distortion, colours as the dataset)
  - ground-truth pose
  - wheel odometry (integrated from wheel encoders with a per-run wheel radius / track error and
    wheel speed noise, so odometry drifts like a real robot)
The robot is moved kinematically (pose set directly, no contact physics): enough for recognition,
and the motion stays exactly known.

usage (pinky_venv, needs mujoco):
  python tools/gen_drive_sequences.py --runs 20             # background, fast
  python tools/gen_drive_sequences.py --runs 1 --viewer     # watch one run in the MuJoCo viewer
output: <out>/run_XXX/images/NNNNN.jpg, background.png, frames.csv ; <out>/runs.csv
Each run uses its own random generator (seed, run index), so --runs 40 with the same seed
contains the --runs 20 runs as its first 20.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_parking_dataset as D  # noqa: E402

WHEEL_R = 0.028          # wheel radius (m)
TRACK = 0.0812           # distance between the wheels (m)
DT = 0.01                # motion integration step (s)
FRAME_DT = 0.1           # camera frame period (10 Hz)
V_MAX, W_MAX = 0.10, 1.2     # m/s, rad/s
STAGING_V, PARKED_V = 420.0, 120.0     # floor v (mm) of the staging point and of the parked axle centre
HOLD_S, MAX_S = 1.0, 40.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def plan(rng):
    """Start pose (x, y, yaw), slot index, mode and the list of motion phases."""
    u, v = rng.uniform(250, 445), rng.uniform(560, 760)
    yaw = wrap(rng.choice([0.0, math.pi]) + rng.normal(0, math.radians(25)))
    x, y = D.mm_to_world(u, v)[:2]
    slot = int(rng.integers(4))
    mode = str(rng.choice(['front', 'rear']))
    sx, sy = D.mm_to_world(D.SLOT_U[slot], STAGING_V)[:2]
    px, py = D.mm_to_world(D.SLOT_U[slot] + rng.normal(0, 3), PARKED_V)[:2]
    final_yaw = 0.0 if mode == 'front' else math.pi          # yaw 0 = facing the slots
    phases = [('turn_to', (sx, sy)), ('drive', (sx, sy, +1)), ('turn', final_yaw),
              ('drive', (px, py, +1 if mode == 'front' else -1)), ('hold', HOLD_S)]
    return (x, y, yaw), slot, mode, phases


class Robot:
    """Differential drive moved by wheel speeds; true pose and encoder odometry kept separately."""

    def __init__(self, pose, rng):
        self.x, self.y, self.yaw = pose
        self.ox, self.oy, self.oyaw = pose               # odometry starts at the true pose
        self.r_err = 1 + rng.normal(0, 0.02)             # wheel radius error used by odometry
        self.l_err = 1 + rng.normal(0, 0.03)             # track error used by odometry
        self.noise = 0.03                                # wheel speed noise (slip, motor)
        self.rng = rng
        self.wl = self.wr = 0.0                          # wheel angles (for the rendered wheels)

    def step(self, v, w):
        wl = (v - w * TRACK / 2) / WHEEL_R * (1 + self.rng.normal(0, self.noise))
        wr = (v + w * TRACK / 2) / WHEEL_R * (1 + self.rng.normal(0, self.noise))
        self.wl += wl * DT
        self.wr += wr * DT
        # true motion
        vt, wt = WHEEL_R * (wl + wr) / 2, WHEEL_R * (wr - wl) / TRACK
        self.x += vt * math.cos(self.yaw + wt * DT / 2) * DT
        self.y += vt * math.sin(self.yaw + wt * DT / 2) * DT
        self.yaw = wrap(self.yaw + wt * DT)
        # odometry from the same encoder readings, with the wrong radius / track
        r, l = WHEEL_R * self.r_err, TRACK * self.l_err
        vo, wo = r * (wl + wr) / 2, r * (wr - wl) / l
        self.ox += vo * math.cos(self.oyaw + wo * DT / 2) * DT
        self.oy += vo * math.sin(self.oyaw + wo * DT / 2) * DT
        self.oyaw = wrap(self.oyaw + wo * DT)


def command(robot, phase, t_phase):
    """(v, w, done) for the current phase, using the true pose (the motion itself, not the experiment)."""
    kind, arg = phase
    if kind in ('turn', 'turn_to'):
        target = arg if kind == 'turn' else math.atan2(arg[1] - robot.y, arg[0] - robot.x)
        err = wrap(target - robot.yaw)
        return 0.0, float(np.clip(3.0 * err, -W_MAX, W_MAX)), abs(err) < math.radians(1.5)
    if kind == 'drive':
        gx, gy, sign = arg
        dx, dy = gx - robot.x, gy - robot.y
        dist = math.hypot(dx, dy)
        heading = math.atan2(dy, dx) if sign > 0 else wrap(math.atan2(dy, dx) + math.pi)
        err = wrap(heading - robot.yaw)
        v = sign * min(V_MAX, 1.5 * dist + 0.02)
        return v, float(np.clip(4.0 * err, -W_MAX, W_MAX)) if dist > 0.02 else 0.0, dist < 0.004
    return 0.0, 0.0, t_phase >= arg                                   # hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/.cache/pinky_mujoco/scene.xml')
    ap.add_argument('--out', default='~/dev_ws/pinky_parking_sim/datasets/drive_seq')
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=300)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--cam-jitter-m', type=float, default=0.03)
    ap.add_argument('--cam-jitter-deg', type=float, default=3.0)
    ap.add_argument('--viewer', action='store_true', help='show the MuJoCo viewer (real-time speed)')
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    model, f, rw, rh = D.build(os.path.expanduser(args.model), args.width, args.height)
    data = mujoco.MjData(model)
    cam_id = model.camera('ceiling_camera').id
    map_x, map_y = D.distortion_map(args.width, args.height, f, rw, rh)
    wheel_q = [model.joint(f'{s}_wheel_joint').qposadr[0] for s in ('l', 'r')]
    base_z = float(model.qpos0[2])
    light0 = model.light_diffuse.copy()
    head0 = (model.vis.headlight.diffuse.copy(), model.vis.headlight.ambient.copy())
    renderer = mujoco.Renderer(model, rh, rw)
    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    runs_csv = open(out / 'runs.csv', 'w')
    runs_csv.write('run,slot,mode,start_x,start_y,start_yaw,frames,cam_x,cam_y,cam_z,cam_qw,cam_qx,cam_qy,cam_qz,'
                   'light_scale,odom_r_err,odom_l_err\n')

    def render(robot, light_scale, head):
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = [robot.x, robot.y, base_z] if robot else [-5.0, -5.0, base_z]
        data.qpos[3:7] = D.yaw_quat(robot.yaw if robot else 0.0)
        if robot:
            data.qpos[wheel_q] = [robot.wl, robot.wr]
        model.light_diffuse[:] = light0 * light_scale
        model.vis.headlight.diffuse[:] = head[0]
        model.vis.headlight.ambient[:] = head[1]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        renderer.update_scene(data, camera='ceiling_camera')
        return D.remap_bilinear(renderer.render(), map_x, map_y)

    for k in range(args.runs):
        rng = np.random.default_rng([args.seed, k])
        start, slot, mode, phases = plan(rng)
        robot = Robot(start, rng)
        model.cam_pos[cam_id] = D.CAM_POS + rng.uniform(-args.cam_jitter_m, args.cam_jitter_m, 3)
        model.cam_quat[cam_id] = D.quat_mul(D.CAM_QUAT, D.small_rotation(rng, args.cam_jitter_deg))
        scale = rng.uniform(0.7, 1.2)
        head = (head0[0] * rng.uniform(0.7, 1.2), head0[1] * rng.uniform(0.7, 1.3))
        contrast, offset, cast = rng.uniform(0.85, 1.15), rng.uniform(-20, 20), rng.uniform(0.97, 1.03, 3)
        noise_sd = rng.uniform(1, 5)

        run_dir = out / f'run_{k:03d}'
        (run_dir / 'images').mkdir(parents=True, exist_ok=True)
        # empty-lot background for P-B: same camera, nominal lighting, no noise
        bg = render(None, 1.0, head0)
        Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).save(run_dir / 'background.png')

        frames = open(run_dir / 'frames.csv', 'w')
        frames.write('frame,t,phase,gt_x,gt_y,gt_yaw,odom_x,odom_y,odom_yaw\n')
        t, t_phase, p, n = 0.0, 0.0, 0, 0
        next_frame = 0.0
        while p < len(phases) and t < MAX_S:
            if t >= next_frame - 1e-9:
                rgb = render(robot, scale, head)
                rgb = (rgb - 128) * contrast + 128 + offset
                rgb = rgb * cast + rng.normal(0, noise_sd, rgb.shape)
                Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(run_dir / 'images' / f'{n:05d}.jpg',
                                                                             quality=95)
                frames.write(f'{n},{t:.2f},{phases[p][0]},{robot.x:.5f},{robot.y:.5f},{robot.yaw:.5f},'
                             f'{robot.ox:.5f},{robot.oy:.5f},{robot.oyaw:.5f}\n')
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    time.sleep(FRAME_DT)
                n += 1
                next_frame += FRAME_DT
            v, w, done = command(robot, phases[p], t_phase)
            if done:
                p, t_phase = p + 1, 0.0
                continue
            robot.step(v, w)
            t += DT
            t_phase += DT
        frames.close()
        cp, cq = model.cam_pos[cam_id], model.cam_quat[cam_id]
        runs_csv.write(f'{k},{"ABCD"[slot]},{mode},{start[0]:.4f},{start[1]:.4f},{start[2]:.4f},{n},'
                       f'{cp[0]:.5f},{cp[1]:.5f},{cp[2]:.5f},{cq[0]:.6f},{cq[1]:.6f},{cq[2]:.6f},{cq[3]:.6f},'
                       f'{scale:.3f},{robot.r_err:.4f},{robot.l_err:.4f}\n')
        runs_csv.flush()
        print(f'run {k + 1}/{args.runs}: slot {"ABCD"[slot]} {mode}, {n} frames ({t:.1f} s)', flush=True)
    runs_csv.close()
    if viewer is not None:
        viewer.close()
    print(f'done: {out}')


if __name__ == '__main__':
    main()
