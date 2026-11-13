#!/usr/bin/env python3
"""자세 인식 노드: 천장 카메라 영상 -> 인식(P-A 규칙 / P-S 세그멘테이션) -> 필터 -> /perceived_pose.

무엇인가
  폐루프 실험(park_closed_loop.py)의 인식 + 필터를 ROS 2 노드로 옮긴 것. 시뮬과 실물에서 같은 노드를 쓴다.
    시뮬: 영상 = ceiling_cam_sim_node.py, 카메라 = 시뮬 보정 카메라 (기본)
    실물: 영상 = USB 카메라 드라이버, 카메라 = --calib config/ceiling_camera_calib.json (calib_from_tape.py)
  인식 : pa_rule.detect (가중치 필요 없음) 또는 ps_seg.detect (--weights) -> parking_geom.pose_from_masks
  필터 : park_closed_loop.Filter. 바퀴 오도메트리(/odom)가 예측하고 카메라가 보정. 앞/뒤는 출발 시
         정지 상태의 첫 측정 3개의 다수결로 정하고 이후에는 오도메트리로 유지
  지연 보정: 영상은 찍힌 뒤 늦게 도착한다(렌더링·전송·인식). 영상 시각부터 지금까지의 오도메트리 이동량을
         측정 자세에 더해 "지금" 자세로 옮긴 뒤 필터에 넣는다 (--no-latency-comp로 끌 수 있음).
         영상 시각과 /odom 시각은 같은 시계여야 함 (시뮬: 둘 다 시뮬 시간, 실물: PC와 핑키 시계 동기화 필요)
  필터가 시작된 뒤(첫 측정 3개 이후) /odom이 올 때마다(시뮬 50 Hz) 오도메트리로 예측한 추정 자세를 발행한다.
  영상(10 Hz) 사이에도 자세가 갱신되므로 정책이 보는 자세가 오래되지 않는다. 시각 도장 = 그 /odom 시각.
  스레드: 인식(영상 -> 자세)은 별도 작업 스레드에서 가장 최근 영상 1장만 처리한다. ROS 콜백(영상 받기,
         /odom 예측·발행)은 인식을 기다리지 않으므로, 인식이 영상 간격보다 오래 걸려도 자세 발행이 밀리지 않는다
         (밀리면 옛 /odom 시각이 붙은 오래된 자세가 나가고, 정책 노드의 오래된 자세 감시에 걸림).
         처리 중에 새 영상이 오면 기다리던 영상은 버리고 새 영상으로 바꾼다 (버린 수 = dropped).
         필터·오도메트리 기록은 잠금(lock)으로 보호하고, 무거운 인식은 잠금 밖에서 한다.
  종료(SIGINT) 시 처리 시간·검출률 통계를 출력하고, --stats-file이 있으면 JSON으로 저장.

토픽
  구독: /ceiling/image_raw (sensor_msgs/Image, bgr8 또는 rgb8), /odom (nav_msgs/Odometry)
  발행: /perceived_pose (geometry_msgs/PoseStamped, frame 'world')

사용법 (pinky_train 가상환경, ROS 2 + 이 작업 공간 source):
  python tools/pose_perception_node.py --method pa
  python tools/pose_perception_node.py --method ps --weights runs/segment/runs/ps/yolo11n_seed0/weights/best.pt
  python tools/pose_perception_node.py --method ps --weights ... --calib config/ceiling_camera_calib.json   # 실물
ros_park_eval.py --perception pa|ps가 판마다 이 노드를 띄운다.
"""
import argparse
import bisect
import json
import math
import os
import sys
import threading
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402
import gen_parking_dataset as D  # noqa: E402
from park_closed_loop import Filter, S_body_delta  # noqa: E402
from eval_sequence import detector  # noqa: E402


def load_camera(calib):
    """보정 카메라. calib 없음 = 시뮬 카메라 (데이터셋과 같음), 있으면 calib_from_tape.py의 JSON."""
    if not calib:
        return G.Camera(D.CAM_POS, D.CAM_QUAT)
    c = json.loads(open(os.path.expanduser(calib)).read())
    return G.Camera(c['pos'], c['quat_wxyz'])


def image_to_bgr(msg):
    """sensor_msgs/Image -> BGR uint8 (cv_bridge 없이). bgr8 / rgb8만 지원."""
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
    img = a.reshape(msg.height, msg.width, 3)
    if msg.encoding == 'rgb8':
        img = img[:, :, ::-1]
    elif msg.encoding != 'bgr8':
        raise ValueError(f'unsupported image encoding {msg.encoding}')
    return np.ascontiguousarray(img)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def shift_by_odom(meas, odom_then, odom_now):
    """측정 자세(영상 시각)를 그 뒤의 오도메트리 이동량만큼 옮겨 지금 자세로 만든다."""
    d = S_body_delta(odom_then, odom_now)                   # 영상 시각의 로봇 좌표로 본 이동 (dx, dy, dyaw)
    c, s = math.cos(meas[2]), math.sin(meas[2])
    return (meas[0] + c * d[0] - s * d[1], meas[1] + s * d[0] + c * d[1], G.wrap(meas[2] + d[2]))


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', choices=['pa', 'ps'], default='pa')
    ap.add_argument('--weights', default=None, help='YOLO-seg weights for ps')
    ap.add_argument('--calib', default=None, help='camera calibration JSON (real camera); default = simulated camera')
    ap.add_argument('--image-topic', default='/ceiling/image_raw')
    ap.add_argument('--out-topic', default='/perceived_pose')
    ap.add_argument('--stats-file', default=None, help='write timing / detection stats as JSON on exit')
    ap.add_argument('--no-latency-comp', action='store_true', help='use measurements as they are (no odometry shift)')
    args = ap.parse_args()
    if args.method == 'ps' and not args.weights:
        raise SystemExit('--weights is required for ps')

    detect = detector(args.method, args.weights)
    cam = load_camera(args.calib)

    class PosePerceptionNode(Node):
        def __init__(self):
            super().__init__('pose_perception')
            self.filt = Filter()
            self.pub = self.create_publisher(PoseStamped, args.out_topic, 10)
            self.create_subscription(Image, args.image_topic, self.on_image, 1)    # 가장 최근 영상만
            self.create_subscription(Odometry, '/odom', self.on_odom, 20)
            self.got_odom = False
            self.odom_t, self.odom = deque(maxlen=300), deque(maxlen=300)    # 최근 오도메트리 (약 6초)
            self.compute_ms, self.n_frames, self.n_meas, self.n_pub = [], 0, 0, 0
            self.comp_lag_ms, self.n_comp_miss = [], 0
            self.n_dropped = 0                         # 처리하기 전에 새 영상으로 바뀌어 버린 영상 수
            self.lock = threading.Lock()               # 필터·오도메트리 기록 보호 (콜백 <-> 작업 스레드)
            self.pending = None                        # 작업 스레드가 처리할 가장 최근 영상
            self.cv = threading.Condition()            # 새 영상 알림
            self.running = True
            self.worker = threading.Thread(target=self.work, daemon=True)
            self.worker.start()

        def on_odom(self, msg):
            p = msg.pose.pose
            o = (p.position.x, p.position.y, yaw_of(p.orientation))
            with self.lock:
                self.filt.predict(o)
                self.odom_t.append(stamp_s(msg.header.stamp))
                self.odom.append(o)
                self.got_odom = True
                self.publish(msg.header.stamp)

        def publish(self, stamp):
            """필터의 현재 추정 자세 발행 (필터 시작 전에는 발행하지 않음)."""
            if self.filt.state is None:
                return
            x, y, yaw = self.filt.state
            out = PoseStamped()
            out.header.stamp = stamp
            out.header.frame_id = 'world'
            out.pose.position.x, out.pose.position.y = float(x), float(y)
            out.pose.orientation.z, out.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.pub.publish(out)
            self.n_pub += 1

        def compensate(self, meas, t_img):
            """(지금 자세로 옮긴 측정, 기준 시각). 영상 시각의 오도메트리가 없으면 옮기지 않음."""
            t_now = self.odom_t[-1]
            if args.no_latency_comp:
                return meas, t_img
            i = bisect.bisect_right(list(self.odom_t), t_img) - 1
            if i < 0 or t_img > t_now + 0.05:
                self.n_comp_miss += 1
                return meas, t_img
            m = list(meas)
            if self.filt.state is not None and abs(G.wrap(m[2] - self.filt.state[2])) > math.pi / 2:
                m[2] = G.wrap(m[2] + math.pi)                # 앞/뒤를 필터 기준으로 맞춘 뒤 옮김
            self.comp_lag_ms.append((t_now - t_img) * 1000)
            return shift_by_odom(m, self.odom[i], self.odom[-1]), t_now

        def on_image(self, msg):
            """영상은 받기만 하고 작업 스레드에 넘긴다 (기다리던 영상이 있으면 새 영상으로 바꿈)."""
            if not self.got_odom:                       # 필터 예측에 오도메트리가 필요
                return
            with self.cv:
                if self.pending is not None:
                    self.n_dropped += 1
                self.pending = msg
                self.cv.notify()

        def work(self):
            """작업 스레드: 가장 최근 영상 -> 인식 (잠금 밖) -> 지연 보정 + 필터 보정 (잠금 안)."""
            while True:
                with self.cv:
                    while self.pending is None and self.running:
                        self.cv.wait(timeout=0.5)
                    if not self.running:
                        return
                    msg, self.pending = self.pending, None
                c0 = time.perf_counter()
                img = image_to_bgr(msg)
                wheels, body = detect(img, cam, None)
                pose = G.pose_from_masks(cam, wheels, body)
                t_ref = stamp_s(msg.header.stamp)
                with self.lock:
                    self.n_frames += 1
                    self.n_meas += pose is not None
                    if pose is not None:
                        pose, _ = self.compensate(pose[:3], t_ref)
                    self.filt.update(pose)
                    self.compute_ms.append((time.perf_counter() - c0) * 1000)

        def stop_worker(self):
            with self.cv:
                self.running = False
                self.cv.notify()
            self.worker.join(timeout=2.0)

        def stats(self):
            c = np.array(self.compute_ms or [0.0])
            return {'method': args.method, 'frames': self.n_frames, 'published': self.n_pub,
                    'detect_rate': self.n_meas / max(1, self.n_frames),
                    'perc_compute_mean_ms': float(c.mean()), 'perc_compute_max_ms': float(c.max()),
                    'comp_lag_mean_ms': float(np.mean(self.comp_lag_ms)) if self.comp_lag_ms else float('nan'),
                    'comp_miss': self.n_comp_miss, 'dropped': self.n_dropped}

    rclpy.init()
    node = PosePerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_worker()
        st = node.stats()
        print(f"pose_perception ({st['method']}): {st['frames']} frames, detected {st['detect_rate']:.1%}, "
              f"published {st['published']}, compute mean {st['perc_compute_mean_ms']:.1f} ms "
              f"(max {st['perc_compute_max_ms']:.1f} ms), latency compensated {st['comp_lag_mean_ms']:.0f} ms "
              f"(missed {st['comp_miss']}), dropped {st['dropped']}", flush=True)
        if args.stats_file:
            with open(args.stats_file, 'w') as f:
                json.dump(st, f)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
