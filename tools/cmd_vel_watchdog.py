#!/usr/bin/env python3
"""명령 감시 노드: 정책의 명령을 받아 로봇에 넘기고, 명령이 끊기면 로봇을 세운다 (받는 쪽 안전장치).

왜 필요한가
  실물 핑키의 모터 노드(pinky_bringup)는 /cmd_vel을 받을 때 모터 속도를 정하고, 다음 명령이 올 때까지 그 속도를
  계속 유지한다 (명령 끊김 시 정지 기능 없음). 그래서 PC 쪽이 멈추면(Fast DDS 발행 지연, Wi-Fi 끊김, 프로그램 정지)
  로봇이 마지막 속도로 계속 움직인다. 보내는 쪽(PC)의 안전장치는 PC가 멈추면 같이 멈추므로, 받는 쪽에서 막아야 한다.
  pinky_pro는 수정하지 않고, 이 노드를 핑키에서 따로 실행한다 (핑키 안에서만 /cmd_vel을 주고받음).

동작
  - 입력 토픽(기본 /cmd_vel_policy)으로 명령을 받으면 바로 출력 토픽(기본 /cmd_vel)으로 넘긴다.
  - 타이머(기본 50 Hz, 실제 시간)가 마지막 명령을 받은 뒤 --timeout-ms가 지났으면 정지 명령 (0, 0)을 낸다.
    명령이 다시 오면 그대로 넘긴다. 첫 명령을 받기 전에는 아무것도 내지 않는다.
  - 끝낼 때(Ctrl+C, kill) 정지 명령을 3번 내고, 정지 횟수·최대 공백을 출력한다 (--stats-file이면 JSON으로도 저장).

시뮬에서는 ros_park_eval.py --watchdog이 판마다 이 노드를 띄우고, 정책 노드는 /cmd_vel_policy로 명령을 낸다.

사용법 (핑키 또는 PC, ROS 2 source):
  python3 tools/cmd_vel_watchdog.py                          # /cmd_vel_policy -> /cmd_vel, 200 ms
  python3 tools/cmd_vel_watchdog.py --timeout-ms 200 --in-topic /cmd_vel_policy --out-topic /cmd_vel
  (정책 노드: python tools/park_policy_node.py ... --cmd-topic /cmd_vel_policy)
"""
import argparse
import json
import time


class Watch:
    """명령 공백 판정 (ROS와 무관한 로직). on_cmd(실제 시각), tick(실제 시각) -> 정지 명령을 낼지."""

    def __init__(self, timeout_s):
        self.timeout_s = timeout_s
        self.last = None                       # 마지막 명령을 받은 실제 시각
        self.stopped = False                   # 지금 정지 명령을 내는 중인지
        self.n_cmds, self.events, self.stop_pubs, self.max_gap = 0, 0, 0, 0.0

    def on_cmd(self, now):
        if self.last is not None:
            self.max_gap = max(self.max_gap, now - self.last)
        self.last = now
        self.stopped = False
        self.n_cmds += 1

    def tick(self, now):
        if self.last is None or now - self.last <= self.timeout_s:
            return False
        if not self.stopped:
            self.events += 1
            self.stopped = True
        self.stop_pubs += 1
        return True

    def stats(self):
        return {'cmds': self.n_cmds, 'timeout_events': self.events, 'stop_pubs': self.stop_pubs,
                'max_gap_ms': self.max_gap * 1000}


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from geometry_msgs.msg import Twist

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-topic', default='/cmd_vel_policy', help='commands from the policy (PC)')
    ap.add_argument('--out-topic', default='/cmd_vel', help='commands to the robot motor node')
    ap.add_argument('--timeout-ms', type=float, default=200.0, help='publish stop when no command for this long (real ms)')
    ap.add_argument('--rate', type=float, default=50.0, help='check rate (Hz)')
    ap.add_argument('--stats-file', default=None, help='write stats as JSON on exit')
    args = ap.parse_args()

    class CmdVelWatchdog(Node):
        def __init__(self):
            super().__init__('cmd_vel_watchdog')
            self.watch = Watch(args.timeout_ms / 1000)
            self.pub = self.create_publisher(Twist, args.out_topic, 10)
            self.create_subscription(Twist, args.in_topic, self.on_cmd, 10)
            self.create_timer(1.0 / args.rate, self.on_tick)

        def on_cmd(self, msg):
            self.watch.on_cmd(time.monotonic())
            self.pub.publish(msg)                  # 받은 명령은 그대로 넘김

        def on_tick(self):
            if self.watch.tick(time.monotonic()):
                self.pub.publish(Twist())          # 명령 끊김: 정지

    # rclpy 기본 Ctrl+C 처리는 통신(context)을 먼저 닫아 마지막 정지 명령을 못 보내므로 끄고 직접 처리.
    # 백그라운드(&)로 띄우면 SIGINT가 무시된 채 시작되므로, SIGINT와 SIGTERM(kill, 서비스 종료)에 처리기를 직접 단다
    import signal
    from rclpy.signals import SignalHandlerOptions

    def _stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CmdVelWatchdog()
    print(f'cmd_vel_watchdog: {args.in_topic} -> {args.out_topic}, stop after {args.timeout_ms:.0f} ms without commands',
          flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            for _ in range(3):                     # 끝낼 때 정지 명령
                node.pub.publish(Twist())
        except Exception as e:                     # 통신이 이미 닫혔어도 통계는 남김
            print(f'cmd_vel_watchdog: final stop command failed ({e})', flush=True)
        st = node.watch.stats()
        print(f"cmd_vel_watchdog: {st['cmds']} commands, {st['timeout_events']} timeout events "
              f"({st['stop_pubs']} stop commands), max gap {st['max_gap_ms']:.0f} ms", flush=True)
        if args.stats_file:
            with open(args.stats_file, 'w') as f:
                json.dump(st, f)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
