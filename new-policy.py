#!/usr/bin/env python3

import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ===================== TUNE THESE IF MOTION IS OFF =====================
LINEAR_SPEED  = 0.15
ANGULAR_SPEED = 1.5
FORWARD_TIME  = 1.5
TURN_90_TIME  = 1.0
# =======================================================================

GRID_SIZE = 7

GOAL = (6, 6)

OBSTACLES = {
    (1, 0), (1, 2), (1, 3), (1, 4), (1, 6),
    (3, 1), (3, 2), (3, 3), (3, 5),
    (4, 3), (4, 5),
    (5, 1), (5, 5),
    (6, 1), (6, 3), (6, 5),
}

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Headings clockwise
ACTION_TO_HEADING = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
}

HEADING_DELTA = {
    0: (-1, 0),   # N
    1: (0, 1),    # E
    2: (1, 0),    # S
    3: (0, -1),   # W
}

HEADING_NAME = {
    0: 'N',
    1: 'E',
    2: 'S',
    3: 'W'
}

# Load policy
policy = np.load('policy.npy')


class PolicyRunner(Node):

    def __init__(self):
        super().__init__('policy_runner')

        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)

    def publish_twist(self, linear_x, angular_z, duration):

        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z

        end_time = time.time() + duration

        while time.time() < end_time:

            self.pub.publish(msg)
            time.sleep(0.1)

        # stop robot
        self.pub.publish(Twist())
        time.sleep(0.3)

    def face(self, current, desired):

        diff = (desired - current) % 4

        if diff == 0:
            return desired

        if diff == 1:
            # right
            self.publish_twist(0.0, -ANGULAR_SPEED, TURN_90_TIME)

        elif diff == 2:
            # 180
            self.publish_twist(0.0, -ANGULAR_SPEED, 2 * TURN_90_TIME)

        else:
            # left
            self.publish_twist(0.0, ANGULAR_SPEED, TURN_90_TIME)

        return desired

    def in_bounds_and_free(self, pos):

        r, c = pos

        return (
            0 <= r < GRID_SIZE and
            0 <= c < GRID_SIZE and
            pos not in OBSTACLES
        )

    def run(self):

        time.sleep(1.0)

        pos = (0, 0)
        heading = 0

        self.get_logger().info(
            f"start at {pos}, facing {HEADING_NAME[heading]}"
        )

        for step in range(1, 60):

            if pos == GOAL:

                self.get_logger().info(
                    f"*** reached goal in {step - 1} moves ***"
                )
                return

            action = int(policy[pos])

            if action == -1:

                self.get_logger().warn(
                    f"no action at {pos}, stopping"
                )
                return

            heading = self.face(
                heading,
                ACTION_TO_HEADING[action]
            )

            dr, dc = HEADING_DELTA[heading]

            target = (
                pos[0] + dr,
                pos[1] + dc
            )

            if self.in_bounds_and_free(target):

                self.publish_twist(
                    LINEAR_SPEED,
                    0.0,
                    FORWARD_TIME
                )

                pos = target

                self.get_logger().info(
                    f"step {step:2d}: -> {pos}, "
                    f"facing {HEADING_NAME[heading]}"
                )

            else:

                self.get_logger().info(
                    f"step {step:2d}: bounce "
                    f"(stayed at {pos})"
                )


def main():

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
