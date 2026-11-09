#!/usr/bin/env python3
"""Generate a YOLO-seg dataset of the ceiling camera view of the parking board (MuJoCo, no ROS).

Classes: 0 wheel, 1 line (black tape), 2 body (robot body, wheels excluded).
Robot poses are sampled in the parking board (random, approaching, parked front-in / rear-in),
with small camera, light and image randomisation. pinky_pro is not modified.

usage: python gen_parking_dataset.py --out ~/dev_ws/pinky_parking_sim/datasets/parking_seg --count 2000
"""
import argparse
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

CLASSES = ['wheel', 'line', 'body']

# Ceiling camera (same as parking_ceiling_camera node).
CAM_POS = np.array([0.8, 1.05, 0.98])
CAM_QUAT = np.array([0.7071068, 0.0, 0.0, 0.7071068])
HFOV_DEG = 60.0

# Black tape rectangles in floor-image mm (u from left, v from top, w, h); see worlds/course_floor.png.
TAPES = [
    (22.5, 0, 19, 220), (231.5, 0, 19, 220), (440.5, 0, 19, 220), (649.5, 0, 19, 220), (858.5, 0, 19, 220),
    (0, 0, 900, 19), (0, 0, 19, 600), (881, 0, 19, 600), (0, 581, 317.1, 19), (578.5, 581, 321.5, 19),
    (298.1, 581, 19, 409.5), (578.5, 581, 19, 107.5), (578.5, 669.5, 630.5, 19), (298.1, 971.5, 850.9, 19),
    (1341, 111, 317, 19), (1341, 111, 19, 569), (1209, 670, 151, 19), (1639, 111, 19, 889), (1149, 971.5, 200, 19),
]
FLOOR_Z = 0.002
SLOT_Y = [(1500 - u) / 1000 for u in (136.5, 345.5, 554.5, 763.5)]   # A..D slot centres (world y)

REAL_COLORS = {'wheel': (0.05, 0.42, 0.72), 'body': (0.06, 0.06, 0.07), 'lidar': (0.04, 0.04, 0.05),
               'rear': (0.62, 0.62, 0.66), 'front': (0.45, 0.45, 0.45)}
COLOR_RULES = [('l_wheel_visual_0_4', 'wheel'), ('r_wheel_visual_0_4', 'wheel'),
               ('l_wheel_visual', 'body'), ('r_wheel_visual', 'body'),
               ('robot_lamp_visual', 'rear'), ('screen_mount_visual', 'front'),
               ('rplidar_mount_visual', 'lidar'), ('base_link_visual', 'body'),
               ('front_camera_mount_visual', 'body'), ('caster_', 'body')]


def mm_to_world(u, v):
    return np.array([(1100 - v) / 1000, (1500 - u) / 1000, FLOOR_Z])


def yaw_quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def small_rotation(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = math.radians(rng.uniform(-max_deg, max_deg))
    return np.array([math.cos(angle / 2), *(axis * math.sin(angle / 2))])


def build(model_path, width, height):
    spec = mujoco.MjSpec.from_file(model_path)
    for key, rgb in REAL_COLORS.items():
        spec.add_material(name=f'real_{key}', rgba=[*rgb, 1.0], specular=0.05, shininess=0.1, reflectance=0.0)
    for geom in spec.geoms:
        if 'visual' in geom.name:
            for prefix, key in COLOR_RULES:
                if geom.name.startswith(prefix):
                    geom.material, geom.rgba = f'real_{key}', [*REAL_COLORS[key], 1.0]
                    break
    fovy = math.degrees(2 * math.atan(math.tan(math.radians(HFOV_DEG) / 2) * height / width))
    spec.worldbody.add_camera(name='ceiling_camera', pos=CAM_POS.tolist(), quat=CAM_QUAT.tolist(), fovy=fovy)
    model = spec.compile()
    return model, fovy


def geom_classes(model):
    """geom id -> (class id, instance key) for robot visual geoms."""
    table = {}
    for i in range(model.ngeom):
        if model.geom_group[i] != 2 or model.geom_bodyid[i] == 0:
            continue
        name = model.geom(i).name
        if name.startswith(('l_wheel_visual', 'r_wheel_visual')):
            table[i] = (0, name[0])          # whole wheel, 'l' / 'r' instance
        else:
            table[i] = (2, 'body')
    return table


def sample_pose(rng):
    kind = rng.choice(['free', 'approach', 'parked'], p=[0.45, 0.25, 0.30])
    if kind == 'free':                       # anywhere in the board, any heading
        x, y, yaw = rng.uniform(0.56, 1.04), rng.uniform(0.66, 1.44), rng.uniform(-math.pi, math.pi)
    elif kind == 'approach':                 # between entrance and slots, roughly facing a slot
        y = rng.choice(SLOT_Y) + rng.normal(0, 0.08)
        x = rng.uniform(0.6, 0.9)
        yaw = rng.choice([0.0, math.pi]) + rng.normal(0, math.radians(35))
    else:                                    # parked front-in (yaw 0) or rear-in (yaw pi), small errors
        y = rng.choice(SLOT_Y) + rng.normal(0, 0.015)
        x = rng.uniform(0.95, 1.01)
        yaw = rng.choice([0.0, math.pi]) + rng.normal(0, math.radians(6))
    return x, y, yaw, kind


def project(points, data, cam_id, f, width, height):
    pos, rot = data.cam_xpos[cam_id], data.cam_xmat[cam_id].reshape(3, 3)
    local = (points - pos) @ rot              # world -> camera frame (camera looks along -Z, +Y up)
    u = width / 2 + f * local[:, 0] / -local[:, 2]
    v = height / 2 - f * local[:, 1] / -local[:, 2]
    return np.column_stack([u, v])


def clip_polygon(poly, width, height):
    """Sutherland-Hodgman clip of a polygon to the image rectangle."""
    def clip(points, inside, cross):
        out = []
        for i in range(len(points)):
            a, b = points[i - 1], points[i]
            if inside(b):
                if not inside(a):
                    out.append(cross(a, b))
                out.append(b)
            elif inside(a):
                out.append(cross(a, b))
        return out

    def at_x(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

    def at_y(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)

    pts = [tuple(p) for p in poly]
    for inside, cross in ((lambda p: p[0] >= 0, at_x(0)), (lambda p: p[0] <= width, at_x(width)),
                          (lambda p: p[1] >= 0, at_y(0)), (lambda p: p[1] <= height, at_y(height))):
        pts = clip(pts, inside, cross)
        if not pts:
            return []
    return pts


def mask_polygon(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return []
    pts = np.column_stack([xs, ys]).astype(float)
    pts = np.vstack([pts, pts + [1, 0], pts + [0, 1], pts + [1, 1]])   # pixel corners
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    return pts[hull.vertices].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='~/.cache/pinky_mujoco/scene.xml')
    parser.add_argument('--out', default='~/dev_ws/pinky_parking_sim/datasets/parking_seg')
    parser.add_argument('--count', type=int, default=2000)
    parser.add_argument('--val', type=float, default=0.1)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', type=int, default=10, help='overlay images to write')
    args = parser.parse_args()

    out = Path(os.path.expanduser(args.out))
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    (out / 'debug').mkdir(exist_ok=True)
    (out / 'data.yaml').write_text(
        f'path: {out}\ntrain: images/train\nval: images/val\nnames:\n'
        + ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASSES)))

    model, fovy = build(os.path.expanduser(args.model), args.width, args.height)
    data = mujoco.MjData(model)
    cam_id = model.camera('ceiling_camera').id
    f = args.height / 2 / math.tan(math.radians(fovy) / 2)
    table = geom_classes(model)
    wheel_q = [model.joint(f'{s}_wheel_joint').qposadr[0] for s in ('l', 'r')]
    base_z = float(model.qpos0[2])
    light0 = model.light_diffuse.copy()
    head0 = (model.vis.headlight.diffuse.copy(), model.vis.headlight.ambient.copy())
    rgb_renderer = mujoco.Renderer(model, args.height, args.width)
    seg_renderer = mujoco.Renderer(model, args.height, args.width)
    seg_renderer.enable_segmentation_rendering()
    tapes = [np.array([mm_to_world(u, v), mm_to_world(u + w, v), mm_to_world(u + w, v + h), mm_to_world(u, v + h)])
             for u, v, w, h in TAPES]
    rng = np.random.default_rng(args.seed)
    n_val = int(round(args.count * args.val))

    for k in range(args.count):
        split = 'val' if k < n_val else 'train'
        x, y, yaw, kind = sample_pose(rng)
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = [x, y, base_z]
        data.qpos[3:7] = yaw_quat(yaw)
        data.qpos[wheel_q] = rng.uniform(-math.pi, math.pi, 2)
        # camera jitter: +-1 cm, +-1 deg (mount not perfectly placed)
        model.cam_pos[cam_id] = CAM_POS + rng.uniform(-0.01, 0.01, 3)
        model.cam_quat[cam_id] = quat_mul(CAM_QUAT, small_rotation(rng, 1.0))
        scale = rng.uniform(0.7, 1.2)
        model.light_diffuse[:] = light0 * scale
        model.vis.headlight.diffuse[:] = head0[0] * rng.uniform(0.7, 1.2)
        model.vis.headlight.ambient[:] = head0[1] * rng.uniform(0.7, 1.3)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)

        rgb_renderer.update_scene(data, camera='ceiling_camera')
        rgb = rgb_renderer.render().astype(np.float32)
        seg_renderer.update_scene(data, camera='ceiling_camera')
        seg = seg_renderer.render()
        # image randomisation: brightness/contrast, colour cast, sensor noise
        rgb = (rgb - 128) * rng.uniform(0.85, 1.15) + 128 + rng.uniform(-20, 20)
        rgb *= rng.uniform(0.97, 1.03, 3)
        rgb += rng.normal(0, rng.uniform(1, 5), rgb.shape)
        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

        labels = []
        geom_ids = seg[:, :, 0]
        geom_types = seg[:, :, 1]
        masks = {}
        for gid, (cls, inst) in table.items():
            hit = (geom_ids == gid) & (geom_types == int(mujoco.mjtObj.mjOBJ_GEOM))
            if hit.any():
                key = (cls, inst)
                masks[key] = masks.get(key, False) | hit
        for (cls, _), mask in sorted(masks.items(), key=lambda item: item[0][0]):
            poly = mask_polygon(mask)
            if poly:
                labels.append((cls, poly))
        for tape in tapes:
            poly = clip_polygon(project(tape, data, cam_id, f, args.width, args.height), args.width, args.height)
            if len(poly) >= 3:
                area = 0.5 * abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1])))
                if area > 30:
                    labels.append((1, poly))

        name = f'{k:05d}_{kind}'
        image.save(out / 'images' / split / f'{name}.jpg', quality=95)
        with open(out / 'labels' / split / f'{name}.txt', 'w') as fh:
            for cls, poly in labels:
                coords = ' '.join(f'{min(max(px / args.width, 0), 1):.5f} {min(max(py / args.height, 0), 1):.5f}'
                                  for px, py in poly)
                fh.write(f'{cls} {coords}\n')
        if k < args.debug:
            overlay = image.convert('RGBA')
            layer = Image.new('RGBA', overlay.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            colors = {0: (0, 120, 255, 110), 1: (255, 0, 0, 90), 2: (0, 200, 0, 90)}
            for cls, poly in labels:
                draw.polygon([tuple(p) for p in poly], fill=colors[cls], outline=colors[cls][:3] + (255,))
            Image.alpha_composite(overlay, layer).convert('RGB').save(out / 'debug' / f'{name}.jpg')
        if (k + 1) % 100 == 0 or k + 1 == args.count:
            print(f'{k + 1}/{args.count}', flush=True)
    print(f'done: {out}')


if __name__ == '__main__':
    main()
