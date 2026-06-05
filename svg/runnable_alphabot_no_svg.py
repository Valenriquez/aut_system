#!/usr/bin/env python3
"""
policy_runner for the AlphaBot2 (ROS 2 Humble) -- MOTION-ONLY TEST.

This is the same as runnable_alphabot.py but with the SVG / ArUco / camera
parts removed. Use it as a FIRST test: get the robot driving the policy
correctly before adding marker-based localization.

The robot follows a hardcoded value-iteration policy across a 7x7 grid by
pure DEAD RECKONING -- it just turns and drives forward for fixed times.
There is no camera and no position feedback, so nothing corrects drift:
this test only checks that

  * /alphabot2/cmd_vel is reaching the robot,
  * the forward / turn directions are right, and
  * the speed/time pairs are roughly calibrated.

Once the motion looks good, switch to runnable_alphabot.py, which adds the
ArUco SVG markers for localization.

Topic (AlphaBot2):
  motion : /alphabot2/cmd_vel   (geometry_msgs/msg/Twist)
"""
import time
import math

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ===================== MOTION TUNING =====================
# Speed is variable per step: FAST on open straight stretches, SLOW on the cell
# right before a turn (or approaching the goal). Each step always covers exactly
# one cell -- the duration is computed from the chosen speed.
CELL_SIZE     = 0.225          # m -- one grid cell (must match maze.sdf)
QUARTER_TURN  = math.pi / 2    # rad -- one 90 deg turn

FAST_LINEAR   = 0.15           # m/s -- open straight stretches (gentler ramp)
SLOW_LINEAR   = 0.08           # m/s -- last cell before a turn / near the goal
ANGULAR_SPEED = 0.7854         # rad/s -- 90 deg in 2 sec (calm)
SETTLE_TIME   = 0.8            # s   -- pause after each motion to let robot fully stop

# ===================== GRID / WORLD =====================
GRID_SIZE = 7
START = (0, 0)
GOAL  = (6, 6)
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
ACTION_NAME       = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT', -1: '--'}
HEADING_DELTA     = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
HEADING_NAME      = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}

# ===================== POLICY =====================
policy = np.array([
    [    RIGHT, DOWN, RIGHT, RIGHT, RIGHT, DOWN,  LEFT],   # row 0
    [       -1, DOWN,    -1,    -1,    -1, DOWN,    -1],   # row 1
    [    RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN],  # row 2
    [       UP,   -1,    -1,    -1,    UP,    -1, DOWN],   # row 3
    [       UP, LEFT,  DOWN,    -1,    UP,    -1, DOWN],   # row 4
    [       UP,   -1, RIGHT, RIGHT,    UP,    -1, DOWN],   # row 5
    [       UP,   -1,    UP,    -1,    UP,    -1,   -1],   # row 6
], dtype=int)


def compute_policy_path(pol, start, goal, max_len=200):
    path = [start]
    seen = {start}
    pos = start
    while pos != goal and len(path) < max_len:
        action = int(pol[pos])
        if action not in ACTION_DELTA:
            break
        dr, dc = ACTION_DELTA[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if nxt in seen:
            break
        path.append(nxt)
        seen.add(nxt)
        pos = nxt
    return path


class PolicyRunner(Node):
    def __init__(self):
        super().__init__('policy_runner')
        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)
        self.get_logger().info(
            "MOTION-ONLY test: publishing Twist on /alphabot2/cmd_vel "
            "(no camera / no ArUco localization)"
        )

    def publish_twist(self, linear_x, angular_z, duration):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        end_time = time.time() + duration
        while time.time() < end_time:
            self.pub.publish(msg)
            time.sleep(0.1)
        self.pub.publish(Twist())
        time.sleep(SETTLE_TIME)

    def face(self, current, desired):
        diff = (desired - current) % 4
        if diff == 0:
            return desired
        duration = QUARTER_TURN / ANGULAR_SPEED
        if diff == 1:
            self.publish_twist(0.0, -ANGULAR_SPEED, duration)
        elif diff == 2:
            self.publish_twist(0.0, -ANGULAR_SPEED, 2 * duration)
        else:
            self.publish_twist(0.0, ANGULAR_SPEED, duration)
        return desired

    def in_bounds_and_free(self, pos):
        r, c = pos
        return (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE
                and pos not in OBSTACLES)

    def turn_coming(self, current_pos, next_pos):
        """True if the move AFTER this one will require a 90 deg turn -- so
        the robot should slow down for this step to set up cleanly."""
        if next_pos == GOAL or next_pos in OBSTACLES:
            return True
        next_action = int(policy[next_pos])
        if next_action == -1:
            return True
        return next_action != int(policy[current_pos])

    def run(self):
        time.sleep(1.0)
        pos = START
        heading = 0
        self.get_logger().info(f"start at {pos}, facing {HEADING_NAME[heading]}")

        for step in range(1, 60):
            if pos == GOAL:
                self.get_logger().info(
                    f"*** reached goal in {step - 1} moves ***")
                return

            action = int(policy[pos])
            if action == -1:
                self.get_logger().warn(f"no action at {pos}, stopping")
                return

            heading = self.face(heading, ACTION_TO_HEADING[action])
            dr, dc = HEADING_DELTA[heading]
            target = (pos[0] + dr, pos[1] + dc)

            if self.in_bounds_and_free(target):
                linear   = SLOW_LINEAR if self.turn_coming(pos, target) else FAST_LINEAR
                duration = CELL_SIZE / linear
                self.publish_twist(linear, 0.0, duration)
                tag = "slow" if linear == SLOW_LINEAR else "fast"
                pos = target
                self.get_logger().info(
                    f"step {step:2d}: -> {pos}, facing {HEADING_NAME[heading]} ({tag})"
                )
            else:
                self.get_logger().info(
                    f"step {step:2d}: bounce (stayed at {pos})")


def main():
    path = compute_policy_path(policy, START, GOAL)
    print("[policy_runner] MOTION-ONLY TEST (no camera, no ArUco markers)")
    print(f"[policy_runner] policy path ({len(path)} cells): {path}")
    print("[policy_runner] the robot will dead-reckon this path open-loop")

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
