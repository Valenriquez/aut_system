#!/usr/bin/env python3

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

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_TO_HEADING = {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}
HEADING_DELTA = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1)
}
HEADING_NAME = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}


class PolicyRunner(Node):

    def __init__(self):
        super().__init__('policy_runner')

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate_hz = 10

        self.policy = np.load('policy.npy')

        self.timer = self.create_timer(0.1, self.loop)  # 10 Hz

        self.pos = (0, 0)
        self.heading = 0
        self.step = 0
        self.done = False

        self.get_logger().info(f"start at {self.pos}, facing {HEADING_NAME[self.heading]}")

    # ---------------- movement helper ----------------

    def publish(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.pub.publish(msg)

    def stop(self):
        self.pub.publish(Twist())

    # ---------------- logic helpers ----------------

    def face(self, desired):
        diff = (desired - self.heading) % 4

        if diff == 0:
            return desired

        msg = Twist()

        if diff == 1:
            msg.angular.z = -ANGULAR_SPEED
        elif diff == 2:
            msg.angular.z = -ANGULAR_SPEED
        else:
            msg.angular.z = ANGULAR_SPEED

        duration = TURN_90_TIME * (2 if diff == 2 else 1)

        end_time = self.get_clock().now().seconds_nanoseconds()[0] + duration

        while self.get_clock().now().seconds_nanoseconds()[0] < end_time:
            self.pub.publish(msg)

        self.stop()
        return desired

    def in_bounds_and_free(self, pos):
        r, c = pos
        return (0 <= r < GRID_SIZE and
                0 <= c < GRID_SIZE and
                pos not in OBSTACLES)

    # ---------------- main loop ----------------

    def loop(self):
        if self.done:
            return

        if self.pos == GOAL:
            self.get_logger().info(f"*** reached goal in {self.step} moves ***")
            self.stop()
            self.done = True
            return

        self.step += 1

        action = int(self.policy[self.pos])

        if action == -1:
            self.get_logger().warn(f"no action at {self.pos}, stopping")
            self.done = True
            return

        desired_heading = ACTION_TO_HEADING[action]
        self.heading = self.face(desired_heading)

        dr, dc = HEADING_DELTA[self.heading]
        target = (self.pos[0] + dr, self.pos[1] + dc)

        if self.in_bounds_and_free(target):
            self.publish(LINEAR_SPEED, 0.0)
            self.pos = target
            self.get_logger().info(
                f"step {self.step}: -> {self.pos}, facing {HEADING_NAME[self.heading]}"
            )
        else:
            self.get_logger().info(
                f"step {self.step}: bounce (stayed at {self.pos})"
            )


def main():
    rclpy.init()
    node = PolicyRunner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()