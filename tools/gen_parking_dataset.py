#!/usr/bin/env python3
"""주차판을 내려다보는 천장 카메라 시점의 YOLO-seg 데이터셋을 만든다 (MuJoCo, ROS 불필요).

클래스: 0 바퀴, 1 라인(검은 테이프), 2 몸통(로봇 몸체, 바퀴 제외).
로봇 자세는 주차판 안에서 뽑는다(무작위, 접근 중, 전면 주차 / 후면 주차 완료).
카메라·조명·이미지를 무작위로 바꾼다. 카메라는 보정한 실물 웹캠(NOVOMAX NC-150, 640x480)이며
방사 렌즈 왜곡까지 넣는다. pinky_pro는 수정하지 않는다.
이미지마다 로봇 정답 자세와 실제(흔들림이 들어간) 카메라 자세를 poses.csv로 data.yaml 옆에 저장한다.

사용법:
  train+val : python gen_parking_dataset.py --count 2000                      (val = --val 비율, seed 0)
  test만    : python gen_parking_dataset.py --count 300 --split test --seed 100
따로 만드는 분할마다 다른 --seed를 써서, 자세·조명 샘플이 겹치지 않게 한다.
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

# 천장 카메라: 보정한 실물 웹캠 (parking_ceiling_camera 노드와 같은 값).
CAM_POS = np.array([-0.2894, 0.5706, 1.815])
CAM_QUAT = np.array([0.69746, 0.15253, -0.16265, -0.68105])
HFOV_DEG = 37.21
K1 = -0.2184

# 검은 테이프 사각형, 바닥 mm 좌표 (u 왼쪽부터, v 위쪽부터, 폭 w, 높이 h). 바닥 1350 x 1200,
# worlds/course_floor.png와 문서 course_tape_positions.md 참고.
TAPES = [
    (0, 0, 900, 19), (0, 0, 19, 600), (881, 0, 19, 600),
    (22.5, 0, 19, 220), (231.5, 0, 19, 220), (440.5, 0, 19, 220), (649.5, 0, 19, 220), (858.5, 0, 19, 220),
    (0, 581, 195, 19), (497, 581, 403, 19),
    (176, 581, 19, 419), (497, 581, 19, 99), (497, 661, 479, 19), (176, 981, 800, 19), (540.5, 680, 19, 301),
    (957, 100, 19, 580), (957, 100, 337, 19), (1275, 100, 19, 1100), (957, 981, 19, 219), (957, 1181, 337, 19),
]
FLOOR_Z = 0.002
START_U, START_V = 1125.0, 1090.0                                     # 시뮬 원점(로봇 출발점)의 바닥 mm 좌표
SLOT_U = (136.5, 345.5, 554.5, 763.5)                                 # A~D 칸 중심 (바닥 u)

REAL_COLORS = {'wheel': (0.05, 0.42, 0.72), 'body': (0.06, 0.06, 0.07), 'lidar': (0.04, 0.04, 0.05),
               'rear': (0.62, 0.62, 0.66), 'front': (0.45, 0.45, 0.45)}
COLOR_RULES = [('l_wheel_visual_0_4', 'wheel'), ('r_wheel_visual_0_4', 'wheel'),
               ('l_wheel_visual', 'body'), ('r_wheel_visual', 'body'),
               ('robot_lamp_visual', 'rear'), ('screen_mount_visual', 'front'),
               ('rplidar_mount_visual', 'lidar'), ('base_link_visual', 'body'),
               ('front_camera_mount_visual', 'body'), ('caster_', 'body')]


def mm_to_world(u, v):
    return np.array([(START_V - v) / 1000, (START_U - u) / 1000, FLOOR_Z])


def world_to_mm(x, y):
    return START_U - y * 1000, START_V - x * 1000


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


def render_size(width, height, f):
    """왜곡 전 렌더링을 출력보다 조금 크게 잡아, 배럴 왜곡을 적용할 때 원본 픽셀이 모자라지 않게 함."""
    rd = math.hypot(width / 2, height / 2) / f
    ru = rd
    for _ in range(20):
        ru = rd / (1 + K1 * ru * ru)
    margin = max(1.0, ru / rd) * 1.08          # 카메라 흔들림용 여유는 필요 없음. 8% 안전 여유
    return int(math.ceil(width * margin / 2)) * 2, int(math.ceil(height * margin / 2)) * 2


def distortion_map(width, height, f, rw, rh):
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    xd, yd = (xs - width / 2) / f, (ys - height / 2) / f
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        scale = 1 + K1 * (xu * xu + yu * yu)
        xu, yu = xd / scale, yd / scale
    return xu * f + rw / 2, yu * f + rh / 2


def remap_bilinear(img, mx, my):
    h, w = img.shape[:2]
    x = np.clip(mx, 0, w - 1.001); y = np.clip(my, 0, h - 1.001)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    a = img.astype(np.float32)
    top = a[y0, x0] * (1 - fx) + a[y0, x0 + 1] * fx
    bot = a[y0 + 1, x0] * (1 - fx) + a[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def remap_nearest(img, mx, my):
    h, w = img.shape[:2]
    x = np.clip(np.rint(mx), 0, w - 1).astype(np.int32); y = np.clip(np.rint(my), 0, h - 1).astype(np.int32)
    return img[y, x]


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
    f = width / 2 / math.tan(math.radians(HFOV_DEG) / 2)
    rw, rh = render_size(width, height, f)
    fovy = math.degrees(2 * math.atan(rh / 2 / f))
    spec.worldbody.add_camera(name='ceiling_camera', pos=CAM_POS.tolist(), quat=CAM_QUAT.tolist(), fovy=fovy)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, rw)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, rh)
    model = spec.compile()
    return model, f, rw, rh


def geom_classes(model):
    """로봇 시각 geom의 geom id -> (클래스 id, 인스턴스 키)."""
    table = {}
    for i in range(model.ngeom):
        if model.geom_group[i] != 2 or model.geom_bodyid[i] == 0:
            continue
        name = model.geom(i).name
        if name.startswith(('l_wheel_visual', 'r_wheel_visual')):
            table[i] = (0, name[0])          # 바퀴 전체, 'l' / 'r' 인스턴스
        else:
            table[i] = (2, 'body')
    return table


def sample_pose(rng):
    """로봇 중심(바닥 mm u, v)과 방향. yaw 0 = 칸 쪽(-v)을 봄, 즉 전면 주차 방향."""
    kind = rng.choice(['free', 'entrance', 'approach', 'parked'], p=[0.35, 0.15, 0.20, 0.30])
    if kind == 'free':                       # 주차판 어디든, 방향 무작위
        u, v, yaw = rng.uniform(60, 840), rng.uniform(60, 560), rng.uniform(-math.pi, math.pi)
    elif kind == 'entrance':                 # 입구 / 아래쪽 통로, 대략 위 또는 아래 방향
        u, v = rng.uniform(250, 445), rng.uniform(560, 760)
        yaw = rng.choice([0.0, math.pi]) + rng.normal(0, math.radians(25))
    elif kind == 'approach':                 # 입구와 칸 사이, 대략 칸을 향함
        u = rng.choice(SLOT_U) + rng.normal(0, 80)
        v = rng.uniform(250, 500)
        yaw = rng.choice([0.0, math.pi]) + rng.normal(0, math.radians(35))
    else:                                    # 전면 주차(yaw 0) 또는 후면 주차(yaw π) 완료, 작은 오차 포함
        u = rng.choice(SLOT_U) + rng.normal(0, 15)
        v = rng.uniform(90, 150)
        yaw = rng.choice([0.0, math.pi]) + rng.normal(0, math.radians(6))
    x, y = mm_to_world(u, v)[:2]
    return x, y, (yaw + math.pi) % (2 * math.pi) - math.pi, kind


def project(points, data, cam_id, f, width, height):
    """월드 점 -> 왜곡된 출력 픽셀 (실물 웹캠과 같은 모델)."""
    pos, rot = data.cam_xpos[cam_id], data.cam_xmat[cam_id].reshape(3, 3)
    local = (points - pos) @ rot              # 월드 -> 카메라 좌표 (카메라는 -Z 방향을 봄, +Y가 위)
    xu = local[:, 0] / -local[:, 2]
    yu = -local[:, 1] / -local[:, 2]
    scale = 1 + K1 * (xu * xu + yu * yu)
    return np.column_stack([width / 2 + f * xu * scale, height / 2 + f * yu * scale])


def tape_outline(u, v, w, h, step=10.0):
    """사각형 둘레를 `step` mm마다 샘플링 (왜곡으로 휘어진 이미지 모양을 따라가기 위함)."""
    pts = []
    for (u0, v0), (u1, v1) in (((u, v), (u + w, v)), ((u + w, v), (u + w, v + h)),
                               ((u + w, v + h), (u, v + h)), ((u, v + h), (u, v))):
        n = max(1, int(math.ceil(math.hypot(u1 - u0, v1 - v0) / step)))
        for t in np.arange(n) / n:
            pts.append(mm_to_world(u0 + (u1 - u0) * t, v0 + (v1 - v0) * t))
    return np.array(pts)


def clip_polygon(poly, width, height):
    """다각형을 이미지 사각형으로 자름 (Sutherland-Hodgman 방식)."""
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
    pts = np.vstack([pts, pts + [1, 0], pts + [0, 1], pts + [1, 1]])   # 픽셀 모서리
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
    parser.add_argument('--val', type=float, default=0.1, help='val fraction when --split auto')
    parser.add_argument('--split', default='auto', choices=['auto', 'train', 'val', 'test'],
                        help='auto: train/val by --val; otherwise write every image to this split')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', type=int, default=10, help='overlay images to write')
    parser.add_argument('--cam-jitter-m', type=float, default=0.03, help='camera position jitter (+- m)')
    parser.add_argument('--cam-jitter-deg', type=float, default=3.0, help='camera rotation jitter (+- deg)')
    args = parser.parse_args()

    out = Path(os.path.expanduser(args.out))
    for split in ('train', 'val', 'test'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    (out / 'debug').mkdir(exist_ok=True)
    (out / 'data.yaml').write_text(
        f'path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'
        + ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASSES)))

    model, f, rw, rh = build(os.path.expanduser(args.model), args.width, args.height)
    data = mujoco.MjData(model)
    cam_id = model.camera('ceiling_camera').id
    map_x, map_y = distortion_map(args.width, args.height, f, rw, rh)
    table = geom_classes(model)
    wheel_q = [model.joint(f'{s}_wheel_joint').qposadr[0] for s in ('l', 'r')]
    base_z = float(model.qpos0[2])
    light0 = model.light_diffuse.copy()
    head0 = (model.vis.headlight.diffuse.copy(), model.vis.headlight.ambient.copy())
    rgb_renderer = mujoco.Renderer(model, rh, rw)
    seg_renderer = mujoco.Renderer(model, rh, rw)
    seg_renderer.enable_segmentation_rendering()
    tapes = [tape_outline(u, v, w, h) for u, v, w, h in TAPES]
    pose_csv = out / 'poses.csv'
    new_csv = not pose_csv.exists()
    csv = open(pose_csv, 'a')
    if new_csv:
        csv.write('split,name,kind,x_m,y_m,yaw_rad,u_mm,v_mm,cam_x,cam_y,cam_z,cam_qw,cam_qx,cam_qy,cam_qz,light_scale\n')
    rng = np.random.default_rng(args.seed)
    n_val = int(round(args.count * args.val))

    for k in range(args.count):
        if args.split == 'auto':
            split = 'val' if k < n_val else 'train'
        else:
            split = args.split
        x, y, yaw, kind = sample_pose(rng)
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = [x, y, base_z]
        data.qpos[3:7] = yaw_quat(yaw)
        data.qpos[wheel_q] = rng.uniform(-math.pi, math.pi, 2)
        # 카메라 흔들림 (거치대가 조금 움직이거나 다시 설치됨): 기본 ±3 cm, ±3°
        cam_d = rng.uniform(-args.cam_jitter_m, args.cam_jitter_m, 3)
        model.cam_pos[cam_id] = CAM_POS + cam_d
        model.cam_quat[cam_id] = quat_mul(CAM_QUAT, small_rotation(rng, args.cam_jitter_deg))
        scale = rng.uniform(0.7, 1.2)
        model.light_diffuse[:] = light0 * scale
        model.vis.headlight.diffuse[:] = head0[0] * rng.uniform(0.7, 1.2)
        model.vis.headlight.ambient[:] = head0[1] * rng.uniform(0.7, 1.3)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)

        rgb_renderer.update_scene(data, camera='ceiling_camera')
        rgb = remap_bilinear(rgb_renderer.render(), map_x, map_y)          # 렌즈 왜곡
        seg_renderer.update_scene(data, camera='ceiling_camera')
        seg = remap_nearest(seg_renderer.render(), map_x, map_y)
        # 이미지 무작위화: 밝기/대비, 색 틀어짐, 센서 노이즈
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
        u_mm, v_mm = world_to_mm(x, y)
        cp, cq = model.cam_pos[cam_id], model.cam_quat[cam_id]     # 실제(흔들림이 들어간) 카메라 자세, 월드 좌표
        csv.write(f'{split},{name},{kind},{x:.4f},{y:.4f},{yaw:.4f},{u_mm:.1f},{v_mm:.1f},'
                  f'{cp[0]:.5f},{cp[1]:.5f},{cp[2]:.5f},{cq[0]:.6f},{cq[1]:.6f},{cq[2]:.6f},{cq[3]:.6f},{scale:.3f}\n')
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
            Image.alpha_composite(overlay, layer).convert('RGB').save(out / 'debug' / f'{split}_{name}.jpg')
        if (k + 1) % 100 == 0 or k + 1 == args.count:
            print(f'{k + 1}/{args.count}', flush=True)
    csv.close()
    print(f'done: {out}')


if __name__ == '__main__':
    main()
