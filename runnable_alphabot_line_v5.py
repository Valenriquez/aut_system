#!/usr/bin/env python3
"""
policy_runner_v5.py  –  AlphaBot2 (ROS 2 Humble)

Corrected from Variant A. Three fixes:

  1. STEERING SIGN FLIPPED
     WEIGHTS is now [2, 1, 0, -1, -2] (was [-2,-1,0,1,2]).
     The robot was steering AWAY from the line ("escaping"); the
     correction direction was inverted for this sensor mounting.
     Now `angular.z = -KP*error` pushes back ONTO the line.

  2. HIGHER SPEEDS (above motor deadband)
     0.07 m/s was likely below the AlphaBot2's minimum PWM, so the
     wheels buzzed without turning. Floored well above that.

  3. TIMING-PRIMARY + RISING-EDGE RE-ANCHOR (line sensors only)
     The timer decides expected arrival. On a GAPPED move (cross
     white, then a new line) the rising edge into that line is
     ground truth -> stop there (kills along-track drift). On a
     CONTINUOUS-tape move (never goes white) no edge fires and the
     timer stops it. Sensors also steer the whole time.

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
THRESHOLD      = 700
KP             = 0.30
WEIGHTS        = [2, 1, 0, -1, -2]   # FLIPPED: fixes "escaping the line"
ON_LINE_COUNT  = 3      # >=3 sensors on black -> on a line
OFF_LINE_COUNT = 1      # <=1 sensor on black  -> in white (off the line)

# ===================== MOTION TUNING ========================
CELL_SIZE     = 0.18        # m  <- MEASURE intersection-to-intersection pitch
QUARTER_TURN  = math.pi / 2

# Raised above the deadband. If it still won't move, raise further.
FAST_LINEAR   = 0.18        # m/s  straightaways
SLOW_LINEAR   = 0.12        # m/s  cell before a turn / goal
ANGULAR_SPEED = 0.80        # rad/s
SETTLE_TIME   = 0.4         # s    pause after each primitive

# Measured REAL ground speeds (m/s) for each commanded speed.
# Until you calibrate, these fall back to the commanded value, so the
# timer will be approximate. To calibrate: command a speed for 5 s,
# tape-measure the distance, real = distance / 5, put it here.
REAL_SPEED = {
    # FAST_LINEAR: 0.14,   # <- fill in from a calibration drive
    # SLOW_LINEAR: 0.095,  # <- fill in from a calibration drive
}

ARRIVE_GATE_FRAC = 0.5      # ignore line crossings for first half of the move
ARRIVE_GRACE     = 0.4      # extra seconds allowed past expected to catch
                            # a slightly-late edge on a gapped move

# ===================== GRID / WORLD =========================
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
        super().__init__('policy_runner_v5')
        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)
        self.sub = self.create_subscription(
            Int32MultiArray,
            '/alphabot2/line_sensors',
            self._sensor_cb,
            qos_profile_sensor_data,   # match typical BEST_EFFORT sensor publisher
        )
        self._sensor_data = [999, 999, 999, 999, 999]
        self._count       = 0
        self._on_line     = False
        self.get_logger().info('PolicyRunner v5 ready')

    # ── sensing ────────────────────────────────────
    def _sensor_cb(self, msg: Int32MultiArray):
        if len(msg.data) != 5:
            return
        self._sensor_data = list(msg.data)
        self._count   = sum(1 for v in self._sensor_data if v < THRESHOLD)
        self._on_line = self._count >= ON_LINE_COUNT
        self.get_logger().info(
            f'raw={self._sensor_data} count={self._count}',
            throttle_duration_sec=0.5,
        )

    def _line_error(self):
        binary = [1 if v < THRESHOLD else 0 for v in self._sensor_data]
        count  = sum(binary)
        if count == 0:
            return None
        return sum(WEIGHTS[i] * binary[i] for i in range(5)) / count

    # ── primitives ─────────────────────────────────
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
        Timing-primary, line-re-anchored cell traversal.

        - GAPPED move: rising edge (white -> line) is ground truth -> stop.
        - CONTINUOUS-tape move: no edge -> timer stops it.
        - Sensors steer the whole time (corrected sign).
        """
        real_speed = REAL_SPEED.get(linear_speed, linear_speed)
        expected   = CELL_SIZE / real_speed
        gate_time  = expected * ARRIVE_GATE_FRAC
        hard_time  = expected + ARRIVE_GRACE

        start            = time.time()
        prev_on          = self._on_line
        seen_white_after = False

        self.get_logger().info(
            f'  [drive] -> {expected_cell}  real={real_speed:.3f}  '
            f'expected={expected:.2f}s'
        )

        while True:
            elapsed = time.time() - start
            on  = self._on_line
            off = self._count <= OFF_LINE_COUNT

            if elapsed >= gate_time:
                if off:
                    seen_white_after = True
                if seen_white_after and on and not prev_on:
                    self.get_logger().info(
                        f'  [drive] line edge at {elapsed:.2f}s '
                        f'-> {expected_cell} (re-anchored)'
                    )
                    break

            if elapsed >= expected and not seen_white_after:
                self.get_logger().info(
                    f'  [drive] timer at {elapsed:.2f}s '
                    f'-> {expected_cell} (continuous tape, dead-reckoned)'
                )
                break

            if elapsed >= hard_time:
                self.get_logger().warn(
                    f'  [drive] hard cap at {elapsed:.2f}s '
                    f'-> {expected_cell} (dead-reckoned)'
                )
                break

            prev_on = on
            error = self._line_error()
            cmd = Twist()
            cmd.linear.x  = linear_speed
            cmd.angular.z = (-KP * error) if error is not None else 0.0
            self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.04)

        self._stop()
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

    # ── policy helpers ─────────────────────────────
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

    # ── main loop ──────────────────────────────────
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
                f'step {step:2d}: {pos} -> {target}  '
                f'{HEADING_NAME[heading]}  ({tag})'
            )

            self._drive_one_cell(linear, expected_cell=target)
            pos = target

        self.get_logger().warn('Step limit reached.')


def main():
    path = compute_policy_path(policy, START, GOAL)
    print(f'[v5] path ({len(path)} cells): {path}')
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