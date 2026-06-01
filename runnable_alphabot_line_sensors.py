#!/usr/bin/env python3
"""
line_follower.py  –  AlphaBot2 (ROS 2 Humble)

Follows a black line using 5 IR sensors with proportional steering.
When the line is lost, enters a recovery search instead of stopping:
  1. Rotates toward the direction the line was last seen
  2. If still not found after RECOVERY_TIMEOUT steps, reverses direction
  3. Resumes normal following as soon as sensors detect the line again

Topics
  subscribe : /alphabot2/line_sensors  (std_msgs/msg/Int32MultiArray)
  publish   : /alphabot2/cmd_vel       (geometry_msgs/msg/Twist)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist

# ===================== TUNING ================================
THRESHOLD        = 700    # sensor value below this → on black line
BASE_SPEED       = 0.12   # m/s  forward speed while following
KP               = 0.35   # proportional steering gain
WEIGHTS          = [-2, -1, 0, 1, 2]   # sensor positions left → right

RECOVERY_ANGULAR = 0.45   # rad/s  rotation speed during search
RECOVERY_TIMEOUT = 25     # sensor callbacks before reversing search direction
# =============================================================


class LineFollower(Node):

    def __init__(self):
        super().__init__('line_follower')

        self.sub = self.create_subscription(
            Int32MultiArray,
            '/alphabot2/line_sensors',
            self.sensor_callback,
            10,
        )
        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)

        # --- recovery state ---
        self.last_error       = 0.0   # last known weighted error
        self.search_direction = 1     # +1 = left,  -1 = right
        self.recovery_count   = 0     # steps spent searching
        self.recovering       = False

        self.get_logger().info('Line follower started  (with recovery)')

    # ──────────────────────────────────────────────
    # SENSOR CALLBACK
    # ──────────────────────────────────────────────

    def sensor_callback(self, msg: Int32MultiArray):
        data = list(msg.data)
        if len(data) != 5:
            return

        binary = [1 if v < THRESHOLD else 0 for v in data]
        count  = sum(binary)

        if count == 0:
            # line completely lost → search
            self._recover()
            return

        # ── line found ──────────────────────────────────
        if self.recovering:
            self.get_logger().info('Line re-acquired  → resuming follow')
            self.recovering     = False
            self.recovery_count = 0

        # weighted lateral error
        error = sum(WEIGHTS[i] * binary[i] for i in range(5)) / count

        # remember which side the line was on (used by recovery)
        self.last_error       = error
        self.search_direction = 1 if error >= 0 else -1

        self._follow(error)

    # ──────────────────────────────────────────────
    # NORMAL FOLLOWING
    # ──────────────────────────────────────────────

    def _follow(self, error: float):
        """Publish forward + proportional steering command."""
        cmd = Twist()
        cmd.linear.x  = BASE_SPEED
        cmd.angular.z = -KP * error
        self.pub.publish(cmd)

    # ──────────────────────────────────────────────
    # RECOVERY (search for lost line)
    # ──────────────────────────────────────────────

    def _recover(self):
        """
        Rotate in place toward the last known line position.
        After RECOVERY_TIMEOUT steps without finding the line,
        reverse the search direction and try the other side.
        """
        self.recovering      = True
        self.recovery_count += 1

        # flip search direction after timeout
        if self.recovery_count > RECOVERY_TIMEOUT:
            self.search_direction *= -1
            self.recovery_count    = 0
            self.get_logger().warn(
                'Recovery timeout  → reversing search direction'
            )

        side = 'left' if self.search_direction > 0 else 'right'
        self.get_logger().warn(
            f'Line lost  → searching {side}  '
            f'(step {self.recovery_count}/{RECOVERY_TIMEOUT})'
        )

        cmd = Twist()
        cmd.linear.x  = 0.0   # stay in place while searching
        cmd.angular.z = RECOVERY_ANGULAR * self.search_direction
        self.pub.publish(cmd)


# ────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())   # safety stop
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()