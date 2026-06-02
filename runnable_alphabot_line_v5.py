#!/usr/bin/env python3
"""
policy_runner_variant_B.py  –  AlphaBot2 (ROS 2 Humble)

STRATEGY: Debounced detection + post-turn heading correction.

Two additions over grid_maze.py:

  1. DEBOUNCED CHECKPOINT DETECTION
     A single noisy reading cannot trigger a checkpoint.
     The robot must see ≥ ON_LINE_COUNT sensors on black for
     DEBOUNCE_HITS consecutive sensor callbacks before the
     checkpoint is accepted.  This eliminates false stops on
     tape edges, reflections, and sensor glitches.

  2. POST-TURN HEADING CORRECTION  (re-align to line after every turn)
     After each rotation the robot's heading has a small error from
     motor speed mismatch.  This accumulates across multiple turns
     and causes the robot to approach cell lines at an angle,
     triggering only 1-2 sensors instead of all 5.
     Fix: after every turn, creep forward slowly until the sensor
     array detects the nearest line, then rotate in small increments
     until the error signal is minimised (robot centred and square
     to the line).  Only then does the main forward move begin.

Failure mode this fixes:
  Heading drift causes missed checkpoints → robot overshoots cells.
  Noisy sensors cause early stops mid-cell.

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

# ===================== IR SENSOR TUNING =====================
THRESHOLD     = 700
KP            = 0.30
WEIGHTS       = [-2, -1, 0, 1, 2]
ON_LINE_COUNT = 3      # ≥3 sensors on black → on line
OFF_LINE_COUNT = 1     # ≤1 sensor on black  → fully in white cell

# ── DEBOUNCE ─────────────────────────────────────────────────
# How many consecutive on-line callbacks required to accept a checkpoint.
# Raise if you get false stops; lower if the robot overshoots.
DEBOUNCE_HITS = 4

# ── POST-TURN ALIGNMENT ──────────────────────────────────────
# After a turn, creep forward at this speed looking for the current line.
ALIGN_CREEP_SPEED   = 0.05   # m/s  (very slow)
ALIGN_ROTATE_SPEED  = 0.20   # rad/s (gentle nudge)
ALIGN_MAX_TIME      = 2.0    # s  give up if line not found
# Error magnitude below which heading is considered "square"
ALIGN_ERROR_THRESH  = 0.4    # out of [-2, +2]

# ===================== MOTION TUNING ========================
CELL_SIZE     = 0.18
QUARTER_TURN  = math.pi / 2

FAST_LINEAR   = 0.12
SLOW_LINEAR   = 0.07
ANGULAR_SPEED = 0.70
SETTLE_TIME   = 0.5

MAX_CELL_TIME = CELL_SIZE / SLOW_LINEAR + 3.0

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
        super().__init__('policy_runner_B')
        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)
        self.sub = self.create_subscription(
            Int32MultiArray,
            '/alphabot2/line_sensors',
            self._sensor_cb, 10,
        )
        self._sensor_data    = [999, 999, 999, 999, 999]
        self._on_line_count  = 0     # consecutive on-line callbacks
        self._off_line_count = 0     # consecutive off-line callbacks

        # debounced flags (updated in callback)
        self._debounced_on_line  = False
        self._debounced_off_line = True

        self.get_logger().info(
            'Variant B ready  —  debounced detection + post-turn alignment'
        )

    # ──────────────────────────────────────────────
    # SENSOR CALLBACK
    # ──────────────────────────────────────────────

    def _sensor_cb(self, msg: Int32MultiArray):
        if len(msg.data) != 5:
            return
        self._sensor_data = list(msg.data)
        count = sum(1 for v in self._sensor_data if v < THRESHOLD)

        # ── debounce ON_LINE ────────────────────────
        if count >= ON_LINE_COUNT:
            self._on_line_count  += 1
            self._off_line_count  = 0
        else:
            self._off_line_count += 1
            self._on_line_count   = 0

        self._debounced_on_line  = self._on_line_count  >= DEBOUNCE_HITS
        self._debounced_off_line = self._off_line_count >= DEBOUNCE_HITS

    def _line_error(self):
        binary = [1 if v < THRESHOLD else 0 for v in self._sensor_data]
        count  = sum(binary)
        if count == 0:
            return None
        return sum(WEIGHTS[i] * binary[i] for i in range(5)) / count

    def _reset_debounce(self):
        self._on_line_count      = 0
        self._off_line_count     = 0
        self._debounced_on_line  = False
        self._debounced_off_line = False

    # ──────────────────────────────────────────────
    # MOTION PRIMITIVES
    # ──────────────────────────────────────────────

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

    def _align_to_line(self):
        """
        Post-turn heading correction.
        Step 1: creep forward until sensors detect the current cell line.
        Step 2: rotate slowly until the lateral error is near zero
                (robot is square to the tape line).
        This corrects the small heading error left after each timed turn.
        """
        self.get_logger().info('  [B] aligning to line after turn...')
        end = time.time() + ALIGN_MAX_TIME

        # ── Step 1: find the line ──────────────────
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.04)
            if self._debounced_on_line:
                self._stop()
                self.get_logger().info('  [B] line found for alignment')
                break
            cmd = Twist()
            cmd.linear.x = ALIGN_CREEP_SPEED
            self.pub.publish(cmd)
        else:
            self._stop()
            self.get_logger().warn('  [B] alignment creep timed out')
            return

        # ── Step 2: rotate until centred ──────────
        align_end = time.time() + 1.5   # max 1.5 s of nudging
        while time.time() < align_end:
            rclpy.spin_once(self, timeout_sec=0.04)
            error = self._line_error()
            if error is None:
                break
            if abs(error) < ALIGN_ERROR_THRESH:
                self.get_logger().info(
                    f'  [B] aligned  error={error:.2f}'
                )
                break
            cmd = Twist()
            cmd.linear.x  = 0.0
            cmd.angular.z = -ALIGN_ROTATE_SPEED * (1 if error > 0 else -1)
            self.pub.publish(cmd)

        self._stop()
        time.sleep(SETTLE_TIME)

    def _drive_to_next_line(self, linear_speed: float, expected_cell=None):
        """
        Two-phase forward drive with DEBOUNCED checkpoint detection.
        Phase 1 DEPARTING: drive until sensors are debounce-confirmed OFF line.
        Phase 2 SEEKING:   drive until sensors are debounce-confirmed ON line.
        """
        self._reset_debounce()
        phase = 'DEPARTING'
        end   = time.time() + MAX_CELL_TIME

        self.get_logger().info(
            f'  [B] driving to {expected_cell}  phase=DEPARTING'
        )

        while time.time() < end:
            error = self._line_error()
            cmd   = Twist()
            cmd.linear.x  = linear_speed
            cmd.angular.z = (-KP * error) if error is not None else 0.0
            self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.04)

            if phase == 'DEPARTING':
                if self._debounced_off_line:
                    phase = 'SEEKING'
                    self._reset_debounce()
                    self.get_logger().info('  [B] phase=SEEKING')

            elif phase == 'SEEKING':
                if self._debounced_on_line:
                    self.get_logger().info(
                        f'  [B] debounced checkpoint → {expected_cell}'
                    )
                    break
        else:
            self.get_logger().warn('  [B] timed out — dead-reckoning')

        self._stop()
        time.sleep(SETTLE_TIME)

    def face(self, current: int, desired: int, do_align: bool = False) -> int:
        """Rotate to face desired heading; optionally align to line after."""
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
        if do_align:
            self._align_to_line()
        return desired

    # ──────────────────────────────────────────────
    # POLICY HELPERS
    # ──────────────────────────────────────────────

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

    # ──────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────

    def run(self):
        time.sleep(1.5)
        pos, heading = START, 0
        self.get_logger().info(
            f'Start {pos}  heading {HEADING_NAME[heading]}'
        )

        for step in range(1, 100):
            if pos == GOAL:
                self.get_logger().info(f'*** Goal in {step-1} steps ***')
                return

            action = int(policy[pos])
            if action == -1:
                self.get_logger().warn(f'No action at {pos}')
                return

            desired_heading = ACTION_TO_HEADING[action]
            turn_needed     = (desired_heading - heading) % 4 != 0

            # post-turn alignment only when we actually rotate
            heading = self.face(heading, desired_heading, do_align=turn_needed)

            dr, dc = HEADING_DELTA[heading]
            target = (pos[0] + dr, pos[1] + dc)

            if not self._in_bounds_and_free(target):
                self.get_logger().warn(f'Obstacle at {target}')
                return

            linear = SLOW_LINEAR if self._turn_coming(pos, target) else FAST_LINEAR
            tag    = 'slow' if linear == SLOW_LINEAR else 'fast'
            self.get_logger().info(
                f'step {step:2d}: {pos} → {target}  '
                f'{HEADING_NAME[heading]}  ({tag})'
            )

            self._drive_to_next_line(linear, expected_cell=target)
            pos = target

        self.get_logger().warn('Step limit reached.')


def main():
    path = compute_policy_path(policy, START, GOAL)
    print(f'[variant B] path ({len(path)} cells): {path}')
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