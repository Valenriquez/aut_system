#!/usr/bin/env python3
"""
policy_runner_variant_A.py  –  AlphaBot2 (ROS 2 Humble)

STRATEGY: Timed-primary, sensor-confirmed.

Instead of relying on sensor detection to STOP the robot, the primary
signal is a calibrated timer (one cell = CELL_SIZE / speed seconds).
Sensors are used for:
  (a) lateral steering correction throughout the move
  (b) early stop — but ONLY after the robot has travelled ≥ GATE_FRACTION
      of the expected cell distance (prevents false triggers on the
      starting tape line)

This is more robust when sensor readings are noisy or inconsistent,
because the robot always has a safe fallback duration.

Failure mode this fixes vs grid_maze.py:
  If DEPARTING phase never triggers (sensors stay on black because the
  white cell is too narrow), the two-phase code drives forever.
  Variant A always stops within CELL_SIZE / speed + BUFFER seconds.

Topics
  subscribe : /alphabot2/line_sensors  (std_msgs/msg/Int32MultiArray)
  publish   : /alphabot2/cmd_vel       (geometry_msgs/msg/Twist)
"""

import time
import math

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data


# ===================== IR SENSOR TUNING =====================
THRESHOLD     = 700
KP            = 0.30
WEIGHTS       = [-2, -1, 0, 1, 2]
ON_LINE_COUNT = 3      # ≥3 sensors on black → on line

# ===================== MOTION TUNING ========================
CELL_SIZE     = 0.18        # m  ← measure your physical cell
QUARTER_TURN  = math.pi / 2

FAST_LINEAR   = 0.12        # m/s
SLOW_LINEAR   = 0.07        # m/s
ANGULAR_SPEED = 0.70        # rad/s
SETTLE_TIME   = 0.5         # s

# ── KEY PARAMETER ───────────────────────────────────────────
# Sensor gating: ignore line detection for the first GATE_FRACTION
# of the expected travel time.  0.55 = ignore first 55%.
# Raise toward 0.7 if the starting line keeps causing false stops.
# Lower toward 0.4 if the robot consistently overshoots the target line.
GATE_FRACTION = 0.55

# Hard timeout buffer added on top of the expected cell time (seconds).
# Robot always stops within  (CELL_SIZE / speed) + TIMEOUT_BUFFER.
TIMEOUT_BUFFER = 1.2
# ────────────────────────────────────────────────────────────

GRID_SIZE = 7
START     = (0, 0)
GOAL      = (6, 6)
OBSTACLES = {
    (1, 0), (1, 2), (1, 3), (1, 4), (1, 6),
    (3, 1), (3, 2), (3, 3), (3, 5),
    (4, 3), (4, 5),
    (5, 1), (5, 5),
    (6, 1), (6, 3), (6, 5),
}

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_TO_HEADING = {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}
ACTION_DELTA      = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
HEADING_DELTA     = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
HEADING_NAME      = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}

policy = np.array([
    [    RIGHT,  DOWN, RIGHT, RIGHT, RIGHT,  DOWN,  LEFT],
    [       -1,  DOWN,    -1,    -1,    -1,  DOWN,    -1],
    [    RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT,  DOWN],
    [       UP,    -1,    -1,    -1,    UP,    -1,  DOWN],
    [       UP,  LEFT,  DOWN,    -1,    UP,    -1,  DOWN],
    [       UP,    -1, RIGHT, RIGHT,    UP,    -1,  DOWN],
    [       UP,    -1,    UP,    -1,    UP,    -1,    -1],
], dtype=int)


def compute_policy_path(pol, start, goal, max_len=200):
    path, seen, pos = [start], {start}, start
    while pos != goal and len(path) < max_len:
        action = int(pol[pos])
        if action not in ACTION_DELTA:
            break
        dr, dc = ACTION_DELTA[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if nxt in seen:
            break
        path.append(nxt); seen.add(nxt); pos = nxt
    return path


class PolicyRunner(Node):

    def __init__(self):
        super().__init__('policy_runner_A')
        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)
        self.sub = self.create_subscription(
            Int32MultiArray,
            '/alphabot2/line_sensors',
            self._sensor_cb, 10,
            qos_profile_sensor_data,
        )
        self._sensor_data   = [999, 999, 999, 999, 999]
        self._on_line       = False
        self.get_logger().info(
            'Variant A ready  —  timed-primary / halfway-gated sensor'
        )

    def _sensor_cb(self, msg: Int32MultiArray):
        if len(msg.data) != 5:
            return
        self._sensor_data = list(msg.data)
        count = sum(1 for v in self._sensor_data if v < THRESHOLD)
        self._on_line = count >= ON_LINE_COUNT
        self.get_logger().info(
            f'raw={self._sensor_data} count={count}',
            throttle_duration_sec=0.5,
        )

    def _line_error(self):
        binary = [1 if v < THRESHOLD else 0 for v in self._sensor_data]
        count  = sum(binary)
        if count == 0:
            return None
        return sum(WEIGHTS[i] * binary[i] for i in range(5)) / count

    def _stop(self):
        self.pub.publish(Twist())

    def _rotate(self, angular_z: float, duration: float):
        msg = Twist()
        msg.angular.z = angular_z
        end = time.time() + duration
        while time.time() < end:
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
        self._stop()
        time.sleep(SETTLE_TIME)

    def _drive_one_cell(self, linear_speed: float, expected_cell=None):
        """
        Drive one cell using a timed gate:
          - Steer with IR sensors throughout
          - Ignore line detection for first GATE_FRACTION of expected time
          - Accept line detection after the gate opens
          - Hard stop at expected_time + TIMEOUT_BUFFER regardless
        """
        expected_time = CELL_SIZE / linear_speed
        gate_time     = expected_time * GATE_FRACTION
        hard_stop     = expected_time + TIMEOUT_BUFFER

        start      = time.time()
        gate_open  = False
        sensor_stopped = False

        self.get_logger().info(
            f'  [A] driving to {expected_cell}  '
            f'expected {expected_time:.2f}s  gate opens at {gate_time:.2f}s'
        )

        while True:
            elapsed = time.time() - start

            # hard timeout
            if elapsed >= hard_stop:
                self.get_logger().warn(
                    f'  [A] hard timeout at {elapsed:.2f}s → dead-reckoning'
                )
                break

            # open the gate after GATE_FRACTION of travel
            if not gate_open and elapsed >= gate_time:
                gate_open = True
                self.get_logger().info('  [A] gate open — watching for line')

            # sensor stop (only after gate)
            if gate_open and self._on_line:
                self.get_logger().info(
                    f'  [A] line detected at {elapsed:.2f}s → {expected_cell}'
                )
                sensor_stopped = True
                break

            # steering + forward
            error = self._line_error()
            cmd   = Twist()
            cmd.linear.x  = linear_speed
            cmd.angular.z = (-KP * error) if error is not None else 0.0
            self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.04)

        self._stop()
        if not sensor_stopped:
            self.get_logger().warn('  [A] no sensor confirmation — position estimated')
        time.sleep(SETTLE_TIME)

    def face(self, current: int, desired: int) -> int:
        diff = (desired - current) % 4
        if diff == 0:
            return desired
        dur = QUARTER_TURN / ANGULAR_SPEED
        if diff == 1:
            self._rotate(-ANGULAR_SPEED, dur)
        elif diff == 2:
            self._rotate(-ANGULAR_SPEED, 2 * dur)
        else:
            self._rotate(+ANGULAR_SPEED, dur)
        return desired

    def _in_bounds_and_free(self, pos) -> bool:
        r, c = pos
        return (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE
                and pos not in OBSTACLES)

    def _turn_coming(self, cur, nxt) -> bool:
        if nxt == GOAL or nxt in OBSTACLES:
            return True
        na = int(policy[nxt])
        if na == -1:
            return True
        return na != int(policy[cur])

    def run(self):
        time.sleep(1.5)
        pos, heading = START, 0
        self.get_logger().info(f'Start {pos}  heading {HEADING_NAME[heading]}')

        for step in range(1, 100):
            if pos == GOAL:
                self.get_logger().info(f'*** Goal in {step-1} steps ***')
                return

            action = int(policy[pos])
            if action == -1:
                self.get_logger().warn(f'No action at {pos}')
                return

            heading = self.face(heading, ACTION_TO_HEADING[action])
            dr, dc  = HEADING_DELTA[heading]
            target  = (pos[0] + dr, pos[1] + dc)

            if not self._in_bounds_and_free(target):
                self.get_logger().warn(f'Obstacle at {target}')
                return

            linear = SLOW_LINEAR if self._turn_coming(pos, target) else FAST_LINEAR
            tag    = 'slow' if linear == SLOW_LINEAR else 'fast'
            self.get_logger().info(
                f'step {step:2d}: {pos} → {target}  {HEADING_NAME[heading]}  ({tag})'
            )

            self._drive_one_cell(linear, expected_cell=target)
            pos = target

        self.get_logger().warn('Step limit reached.')


def main():
    path = compute_policy_path(policy, START, GOAL)
    print(f'[variant A] path ({len(path)} cells): {path}')
    rclpy.init()
    node = PolicyRunner()
    try:
        node.run()
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()