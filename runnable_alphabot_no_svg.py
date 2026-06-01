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
  * FORWARD_TIME and TURN_90_TIME are roughly calibrated.

Once the motion looks good, switch to runnable_alphabot.py, which adds the
ArUco SVG markers for localization.

Topic (AlphaBot2):
  motion : /alphabot2/cmd_vel   (geometry_msgs/msg/Twist)
"""
import time

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ===================== MOTION TUNING =====================
# No odometry on the AlphaBot2 -> these timings are the only thing that
# decides how far/much the robot moves. Calibrate them on the real robot.
LINEAR_SPEED  = 0.15
ANGULAR_SPEED = 1.5
FORWARD_TIME  = 1.5     # seconds to drive forward one grid cell
TURN_90_TIME  = 1.0     # seconds to rotate 90 degrees

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
# Hardcoded from value-iteration output. Indexed as policy[row, col].
# Action codes: UP=0, DOWN=1, LEFT=2, RIGHT=3, obstacle/goal = -1.
#
#         col:  0   1   2   3   4   5   6
policy = np.array([
    [    RIGHT, DOWN, RIGHT, RIGHT, RIGHT, DOWN,  LEFT],   # row 0
    [       -1, DOWN,    -1,    -1,    -1, DOWN,    -1],   # row 1
    [    RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN],  # row 2
    [       UP,   -1,    -1,    -1,    UP,    -1, DOWN],   # row 3
    [       UP, LEFT,  DOWN,    -1,    UP,    -1, DOWN],   # row 4
    [       UP,   -1, RIGHT, RIGHT,    UP,    -1, DOWN],   # row 5
    [       UP,   -1,    UP,    -1,    UP,    -1,   -1],   # row 6 (goal at 6,6)
], dtype=int)
# ===================================================


# ===================== POLICY PATH =====================

def compute_policy_path(pol, start, goal, max_len=200):
    """Follow `pol` from `start` to `goal`, returning the ordered cells."""
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
    # ---------- motion ----------
    def publish_twist(self, linear_x, angular_z, duration):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        end_time = time.time() + duration
        while time.time() < end_time:
            self.pub.publish(msg)
            time.sleep(0.1)
        self.pub.publish(Twist())   # stop
        time.sleep(0.3)

    def face(self, current, desired):
        diff = (desired - current) % 4
        if diff == 0:
            return desired
        if diff == 1:
            self.publish_twist(0.0, -ANGULAR_SPEED, TURN_90_TIME)
        elif diff == 2:
            self.publish_twist(0.0, -ANGULAR_SPEED, 2 * TURN_90_TIME)
        else:
            self.publish_twist(0.0, ANGULAR_SPEED, TURN_90_TIME)
        return desired

    def in_bounds_and_free(self, pos):
        r, c = pos
        return (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and pos not in OBSTACLES)

    # ---------- main loop ----------
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
                self.publish_twist(LINEAR_SPEED, 0.0, FORWARD_TIME)
                pos = target
                self.get_logger().info(
                    f"step {step:2d}: -> {pos}, facing {HEADING_NAME[heading]}"
                )
            else:
                self.get_logger().info(
                    f"step {step:2d}: bounce (stayed at {pos})")


def main():
    path = compute_policy_path(policy, START, GOAL)
    print("[policy_runner] MOTION-ONLY TEST (no camera, no ArUco markers)")
    print(f"[policy_runner] policy path ({len(path)} cells): {path}")
    print("[policy_runner] the robot will dead-reckon this path open-loop")

    rclpy.init()  # Starts ROS 2 (initializes the ROS client library)
    node = PolicyRunner() # run de instance Creates an instance of your class PolicyRunner.
    try:
        PolicyRunner.run(node) # “Go execute the run() function defined inside PolicyRunner.”
    finally:
        node.pub.publish(Twist())   # make sure the robot is stopped
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
