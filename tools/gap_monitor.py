#!/usr/bin/env python3
"""토픽 간격 감시 노드: /ground_truth, /odom 메시지가 늦게 오는 순간을 실제 시각과 함께 기록한다.

무엇인가
  인식 루프 평가에서 정책 노드가 약 340 ms 동안 /ground_truth를 받지 못하는 끊김이 판 시작 약 3초 뒤에 생긴다.
  이 노드는 평가와 별개의 프로세스에서 같은 토픽을 받아, 같은 순간에 여기서도 끊김이 보이는지 확인한다.
    - 여기서도 같은 시각에 보임 -> 보내는 쪽(브리지 발행 또는 ROS 통신)이 막힌 것
    - 여기서는 안 보임        -> 평가 프로세스(정책 노드) 쪽이 멈춘 것
  /odom도 같이 끊기는지 보면 브리지의 발행 전체가 막힌 것인지 /ground_truth만인지 구분된다.

기록 (간격이 --gap-ms보다 클 때마다 한 줄)
  wall  = 실제 시각 (time.time(), 정책 노드 끊김 기록의 시각과 비교용)
  gap   = 실제 시간 간격 (ms), sim = 같은 두 메시지의 시뮬 시간 간격 (ms)
  burst = 끊김 뒤 5 ms보다 짧은 간격으로 몰려 온 메시지 수 (밀렸다가 한꺼번에 온 것)
  at    = 그 브리지가 처음 보낸 메시지부터의 시뮬 시간 (판 시작 후 시간)
  브리지가 다시 시작되면(시뮬 시간이 뒤로 감 또는 간격 1.5초 이상) 끊김이 아니라 restart로 표시한다.

사용법 (ROS 2 source, 평가와 다른 터미널에서 먼저 실행, Ctrl+C로 종료):
  python3 tools/gap_monitor.py | tee ~/gap_monitor.txt
"""
import argparse
import time
from datetime import datetime

RESTART_S = 1.5        # 이보다 긴 간격 또는 시뮬 시간이 뒤로 가면 브리지 재시작으로 봄
BURST_S = 0.005        # 끊김 뒤 이보다 짧은 간격 = 몰려 온 메시지


def clock(t):
    """실제 시각 -> 'HH:MM:SS.mmm'."""
    return datetime.fromtimestamp(t).strftime('%H:%M:%S.%f')[:-3]


class TopicWatch:
    """토픽 하나의 간격 감시 (ROS와 무관한 로직)."""

    def __init__(self, name, gap_s):
        self.name, self.gap_s = name, gap_s
        self.prev_wall = self.prev_t = self.first_t = None
        self.pending = None                    # 몰려 온 메시지 수를 세는 중인 끊김
        self.n_msgs, self.n_gaps, self.n_restarts = 0, 0, 0

    def on_msg(self, wall, t):
        self.n_msgs += 1
        if self.prev_wall is None or t < self.prev_t or wall - self.prev_wall > RESTART_S:
            if self.prev_wall is not None:
                self.n_restarts += 1
                print(f'{clock(wall)}  {self.name:13s} restart', flush=True)
            self._close()
            self.first_t = t
        else:
            gap = wall - self.prev_wall
            if gap > self.gap_s:
                self._close()
                self.n_gaps += 1
                self.pending = {'wall': wall, 'gap': gap * 1000, 'sim': (t - self.prev_t) * 1000,
                                'at': t - self.first_t, 'burst': 0}
            elif self.pending is not None:
                if gap < BURST_S:
                    self.pending['burst'] += 1
                else:
                    self._close()
        self.prev_wall, self.prev_t = wall, t

    def _close(self):
        """세던 끊김을 출력한다."""
        e, self.pending = self.pending, None
        if e is not None:
            print(f"{clock(e['wall'])}  {self.name:13s} gap {e['gap']:4.0f} ms  sim {e['sim']:4.0f} ms  "
                  f"burst {e['burst']:2d}  at {e['at']:5.1f} s", flush=True)


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gap-ms', type=float, default=100.0, help='log intervals longer than this (ms)')
    args = ap.parse_args()

    def stamp_s(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    class GapMonitor(Node):
        def __init__(self):
            super().__init__('gap_monitor')
            self.watch = {'/ground_truth': TopicWatch('/ground_truth', args.gap_ms / 1000),
                          '/odom': TopicWatch('/odom', args.gap_ms / 1000)}
            # 대기열을 넉넉히 (몰려 온 메시지를 버리지 않고 모두 세기 위해)
            self.create_subscription(PoseStamped, '/ground_truth',
                                     lambda m: self.watch['/ground_truth'].on_msg(time.time(), stamp_s(m)), 50)
            self.create_subscription(Odometry, '/odom',
                                     lambda m: self.watch['/odom'].on_msg(time.time(), stamp_s(m)), 50)

    rclpy.init()
    node = GapMonitor()
    print(f'{clock(time.time())}  gap_monitor started (gap > {args.gap_ms:.0f} ms)', flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for w in node.watch.values():
            w._close()
            print(f'{w.name}: {w.n_msgs} msgs, {w.n_gaps} gaps, {w.n_restarts} restarts', flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
