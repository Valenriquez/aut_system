#!/usr/bin/env python3
"""
policy_runner_grid_maze.py  –  AlphaBot2 (ROS 2 Humble)

Built specifically for a maze where black tape marks EVERY cell boundary
(i.e. the grid lines are always visible at the start AND end of each move).

The key fix over previous versions:
  Two-phase checkpoint detection:
    Phase 1: wait for sensors to go WHITE  (robot left the starting line)
    Phase 2: wait for sensors to go BLACK  (robot reached the next cell line)
  Without this, the robot fires a checkpoint immediately on every move
  because it starts each step already sitting on a black tape line.

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
THRESHOLD  = 700     # below → on black line
KP         = 0.30    # proportional lateral steering gain
WEIGHTS    = [-2, -1, 0, 1, 2]

# how many sensors must be on black to count as "on line"
ON_LINE_COUNT  = 3   # ≥3 sensors on black → on line
OFF_LINE_COUNT = 1   # ≤1 sensor on black  → fully off line (white cell)

# ===================== MOTION TUNING ========================
CELL_SIZE     = 0.18        # m  ← MEASURE YOUR PHYSICAL CELLS and set this
QUARTER_TURN  = math.pi / 2 # rad

FAST_LINEAR   = 0.12        # m/s
SLOW_LINEAR   = 0.07        # m/s  (cell before turn / goal)
ANGULAR_SPEED = 0.70        # rad/s
SETTLE_TIME   = 0.5         # s – pause after each primitive

# safety: if checkpoint never fires, stop after this many seconds
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

#         col:   0      1      2      3      4      5      6
policy = np.array([
    [    RIGHT,  DOWN, RIGHT, RIGHT, RIGHT,  DOWN,  LEFT],  # row 0
    [       -1,  DOWN,    -1,    -1,    -1,  DOWN,    -1],  # row 1
    [    RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT,  DOWN],  # row 2
    [       UP,    -1,    -1,    -1,    UP,    -1,  DOWN],  # row 3
    [       UP,  LEFT,  DOWN,    -1,    UP,    -1,  DOWN],  # row 4
    [       UP,    -1, RIGHT, RIGHT,    UP,    -1,  DOWN],  # row 5
    [       UP,    -1,    UP,    -1,    UP,    -1,    -1],  # row 6
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


# ────────────────────────────────────────────────────────────
class PolicyRunner(Node):

    def __init__(self):
        super().__init__('policy_runner')

        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)
        self.sub = self.create_subscription(
            Int32MultiArray,
            '/alphabot2/line_sensors',
            self._sensor_cb,
            10,
        )

        # shared sensor state
        self._sensor_data    = [999, 999, 999, 999, 999]
        self._sensors_on_line  = False   # True  = ≥ ON_LINE_COUNT sensors on black
        self._sensors_off_line = True    # True  = ≤ OFF_LINE_COUNT sensors on black

        self.get_logger().info('PolicyRunner ready — grid-maze mode')

    # ──────────────────────────────────────────────
    # SENSOR CALLBACK
    # ──────────────────────────────────────────────

    def _sensor_cb(self, msg: Int32MultiArray):
        if len(msg.data) != 5:
            return
        self._sensor_data = list(msg.data)
        count = sum(1 for v in self._sensor_data if v < THRESHOLD)
        self._sensors_on_line  = count >= ON_LINE_COUNT
        self._sensors_off_line = count <= OFF_LINE_COUNT

    def _line_error(self):
        binary = [1 if v < THRESHOLD else 0 for v in self._sensor_data]
        count  = sum(binary)
        if count == 0:
            return None
        return sum(WEIGHTS[i] * binary[i] for i in range(5)) / count

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

    def _drive_to_next_line(self, linear_speed: float, expected_cell=None):
        """
        Drive forward using two-phase checkpoint detection:

          Phase 1  DEPARTING  – robot is on the starting tape line.
                               Keep driving until sensors go white.
          Phase 2  SEEKING    – robot is in the white cell.
                               Keep driving until sensors hit the next line.

        Falls back to timed motion if the line is never found within
        MAX_CELL_TIME seconds.
        """
        # ── Phase 1: wait until robot moves OFF the starting line ──
        phase = 'DEPARTING'
        self.get_logger().info('  [sensor] phase=DEPARTING (leaving start line)')

        end = time.time() + MAX_CELL_TIME

        while time.time() < end:
            error = self._line_error()
            cmd = Twist()
            cmd.linear.x  = linear_speed
            cmd.angular.z = (-KP * error) if error is not None else 0.0
            self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.04)

            if phase == 'DEPARTING':
                if self._sensors_off_line:
                    # sensors now in white cell → switch to seeking
                    phase = 'SEEKING'
                    self.get_logger().info('  [sensor] phase=SEEKING (in white cell)')

            elif phase == 'SEEKING':
                if self._sensors_on_line:
                    # hit the next cell boundary line
                    self.get_logger().info(
                        f'  [sensor] checkpoint → {expected_cell}'
                    )
                    break

        else:
            self.get_logger().warn(
                '  [sensor] timed out — using dead-reckoning position'
            )

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

    # ──────────────────────────────────────────────
    # POLICY HELPERS
    # ──────────────────────────────────────────────

    def _in_bounds_and_free(self, pos) -> bool:
        r, c = pos
        return (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE
                and pos not in OBSTACLES)

    def _turn_coming(self, current_pos, next_pos) -> bool:
        if next_pos == GOAL or next_pos in OBSTACLES:
            return True
        nxt_a = int(policy[next_pos])
        if nxt_a == -1:
            return True
        return nxt_a != int(policy[current_pos])

    # ──────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────

    def run(self):
        time.sleep(1.5)   # let connections settle

        pos     = START
        heading = 0       # North
        self.get_logger().info(
            f'Start {pos}  heading {HEADING_NAME[heading]}'
        )

        for step in range(1, 100):

            if pos == GOAL:
                self.get_logger().info(f'*** Goal reached in {step-1} steps ***')
                return

            action = int(policy[pos])
            if action == -1:
                self.get_logger().warn(f'No action at {pos}')
                return

            # ── rotate to face policy direction ──────────
            heading = self.face(heading, ACTION_TO_HEADING[action])

            # ── compute target cell ───────────────────────
            dr, dc = HEADING_DELTA[heading]
            target = (pos[0] + dr, pos[1] + dc)

            if not self._in_bounds_and_free(target):
                self.get_logger().warn(f'Obstacle at {target}, aborting.')
                return

            # ── choose speed ──────────────────────────────
            linear = SLOW_LINEAR if self._turn_coming(pos, target) else FAST_LINEAR
            tag    = 'slow' if linear == SLOW_LINEAR else 'fast'

            # ── drive to next cell line ───────────────────
            self.get_logger().info(
                f'step {step:2d}: {pos} → {target}  '
                f'{HEADING_NAME[heading]}  ({tag})'
            )
            self._drive_to_next_line(linear, expected_cell=target)
            pos = target

        self.get_logger().warn('Step limit reached.')


# ────────────────────────────────────────────────────────────
def main():
    path = compute_policy_path(policy, START, GOAL)
    print(f'[policy_runner] path ({len(path)} cells): {path}')
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