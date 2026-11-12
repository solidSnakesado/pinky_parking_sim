#!/usr/bin/env python3
"""시뮬 천장 카메라 노드: /ground_truth -> 천장 카메라 영상 렌더링 -> /ceiling/image_raw.

무엇인가
  실물에서는 천장 USB 카메라 드라이버가 영상을 발행한다. 시뮬에는 그 카메라가 없으므로(브리지의 카메라는
  핑키 몸체 카메라), 이 노드가 대신 만든다. 데이터셋·폐루프와 같은 방식으로 렌더링한다:
  gen_parking_dataset.build()의 장면(실물 색, 보정 카메라 'ceiling_camera'), 배럴 왜곡, 센서 노이즈, JPEG.
  로봇 자세는 브리지가 내는 정답 자세를 그대로 놓는다(물리는 브리지가 돌리고, 여기서는 그리기만 함).

동작
  /ground_truth를 받다가 시뮬 시간이 다음 프레임 시각(--rate, 기본 10 Hz)을 지나면 한 장 렌더링해 발행.
  영상 시각 도장 = 그 정답 자세의 시뮬 시각 (인식 지연을 시뮬 시간으로 잴 수 있음).
  종료(Ctrl+C / SIGINT) 시 렌더링 시간 통계를 출력.

토픽
  구독: /ground_truth (geometry_msgs/PoseStamped)
  발행: /ceiling/image_raw (sensor_msgs/Image, bgr8, 640x480)

사용법 (pinky_train 가상환경, ROS 2 + 이 작업 공간 source, 브리지가 실행 중일 때):
  python tools/ceiling_cam_sim_node.py
  python tools/ceiling_cam_sim_node.py --noise-sd 3 --rate 10
화면 없는 환경에서 렌더링 오류가 나면 MUJOCO_GL=egl (또는 osmesa)를 지정.
ros_park_eval.py --perception pa|ps가 판마다 이 노드를 띄운다.
"""
import argparse
import math
import os
import sys
import time

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_parking_dataset as D  # noqa: E402

W, H = 640, 480


class CeilingRenderer:
    """정답 자세 -> 천장 카메라 영상 (BGR uint8). 데이터셋과 같은 장면·왜곡·노이즈."""

    def __init__(self, model_path, noise_sd=3.0, jpeg=95, seed=0):
        self.model, f, rw, rh = D.build(os.path.expanduser(model_path), W, H)
        self.data = mujoco.MjData(self.model)
        mx, my = D.distortion_map(W, H, f, rw, rh)
        # cv2.remap (쌍선형)으로 왜곡 적용: D.remap_bilinear와 같은 계산을 훨씬 빠르게 (실시간 10 Hz용)
        self.map_x, self.map_y = mx.astype(np.float32), my.astype(np.float32)
        self.renderer = mujoco.Renderer(self.model, rh, rw)
        self.base_z = float(self.model.qpos0[2])
        self.noise_sd, self.jpeg = noise_sd, jpeg
        self.rng = np.random.default_rng(seed)

    def render(self, x, y, yaw):
        m, d = self.model, self.data
        d.qpos[:] = m.qpos0
        d.qpos[0:3] = [x, y, self.base_z]
        d.qpos[3:7] = D.yaw_quat(yaw)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        self.renderer.update_scene(d, camera='ceiling_camera')
        rgb = cv2.remap(self.renderer.render(), self.map_x, self.map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        if self.noise_sd > 0:
            rgb += self.noise_sd * self.rng.standard_normal(rgb.shape, dtype=np.float32)
        img = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if self.jpeg:                                       # 웹캠처럼 압축 흔적을 넣음 (데이터셋과 같음)
            img = cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg])[1], cv2.IMREAD_COLOR)
        return img


def to_image_msg(img, stamp, frame_id='ceiling_camera'):
    """BGR uint8 영상 -> sensor_msgs/Image (cv_bridge 없이)."""
    from sensor_msgs.msg import Image
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding, msg.is_bigendian, msg.step = 'bgr8', 0, img.shape[1] * 3
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='~/.cache/pinky_mujoco/scene.xml', help='MuJoCo scene with the parking course')
    ap.add_argument('--rate', type=float, default=10.0, help='frames per second of simulated time')
    ap.add_argument('--noise-sd', type=float, default=3.0, help='sensor noise sigma (dataset range 1..5)')
    ap.add_argument('--jpeg', type=int, default=95, help='JPEG quality round trip (0 = off)')
    ap.add_argument('--topic', default='/ceiling/image_raw')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    cam = CeilingRenderer(args.model, args.noise_sd, args.jpeg, args.seed)

    class CeilingCamSimNode(Node):
        def __init__(self):
            super().__init__('ceiling_cam_sim')
            self.pub = self.create_publisher(Image, args.topic, 2)
            # 대기열 1: 렌더링이 밀려도 쌓인 옛 자세가 아니라 가장 최근 자세로 그림 (지연이 쌓이지 않게)
            self.create_subscription(PoseStamped, '/ground_truth', self.on_truth, 1)
            self.next_t, self.render_ms = None, []

        def on_truth(self, msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if self.next_t is None:
                self.next_t = t
            if t + 1e-6 < self.next_t:
                return
            self.next_t += 1.0 / args.rate
            if t > self.next_t:                                    # 크게 밀렸으면 다시 맞춤
                self.next_t = t + 1.0 / args.rate
            q = msg.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            c0 = time.perf_counter()
            img = cam.render(msg.pose.position.x, msg.pose.position.y, yaw)
            self.pub.publish(to_image_msg(img, msg.header.stamp))
            self.render_ms.append((time.perf_counter() - c0) * 1000)

    rclpy.init()
    node = CeilingCamSimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        r = np.array(node.render_ms or [0.0])
        print(f'ceiling_cam_sim: {len(node.render_ms)} frames, render mean {r.mean():.1f} ms (max {r.max():.1f} ms)',
              flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
