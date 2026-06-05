import time
import math

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

CELL_SIZE     = 0.225
QUARTER_TURN  = math.pi / 2

FAST_LINEAR   = 0.15
SLOW_LINEAR   = 0.08
ANGULAR_SPEED = 0.7854
SETTLE_TIME   = 0.8

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

policy = np.array([
    [    RIGHT, DOWN, RIGHT, RIGHT, RIGHT, DOWN,  LEFT],
    [       -1, DOWN,    -1,    -1,    -1, DOWN,    -1],
    [    RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN],
    [       UP,   -1,    -1,    -1,    UP,    -1, DOWN],
    [       UP, LEFT,  DOWN,    -1,    UP,    -1, DOWN],
    [       UP,   -1, RIGHT, RIGHT,    UP,    -1, DOWN],
    [       UP,   -1,    UP,    -1,    UP,    -1,   -1],
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
