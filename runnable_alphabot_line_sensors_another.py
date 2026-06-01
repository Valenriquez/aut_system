#!/usr/bin/env python3
"""
line_follower_v2.py  –  AlphaBot2 (ROS 2 Humble)

Two key differences from line_follower.py (v1):

  1. VARIABLE SPEED
     Speed decreases as lateral error increases.
     When centered → full BASE_SPEED.
     When drifting → speed drops toward MIN_SPEED.
     This reduces overshoot on curves and gives the steering
     more time to correct before the robot leaves the line.

  2. BACK-UP RECOVERY
     When the line is lost, the robot first reverses a short
     distance (in case it overshot the line), then rotates
     toward the last known side.  Rotation-only recovery (v1)
     fails when the robot has driven past the line; backing up
     first brings the sensors back over it.

Topics
  subscribe : /alphabot2/line_sensors  (std_msgs/msg/Int32MultiArray)
  publish   : /alphabot2/cmd_vel       (geometry_msgs/msg/Twist)
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist

# ===================== TUNING ================================
THRESHOLD   = 700     # sensor value below this → on black line
BASE_SPEED  = 0.15    # m/s  max forward speed (centered on line)
MIN_SPEED   = 0.05    # m/s  min forward speed (maximum drift)
KP          = 0.40    # proportional steering gain
WEIGHTS     = [-2, -1, 0, 1, 2]   # sensor positions left → right

# recovery
BACKUP_SPEED    = -0.08   # m/s  reverse speed during back-up phase
BACKUP_DURATION = 0.4     # s    how long to back up before rotating
RECOVERY_ANGULAR = 0.45   # rad/s rotation speed while searching
RECOVERY_TIMEOUT = 25     # callbacks before reversing search direction
# =============================================================

# recovery phases
PHASE_FOLLOW  = 'FOLLOW'
PHASE_BACKUP  = 'BACKUP'
PHASE_ROTATE  = 'ROTATE'


class LineFollowerV2(Node):

    def __init__(self):
        super().__init__('line_follower_v2')

        self.sub = self.create_subscription(
            Int32MultiArray,
            '/alphabot2/line_sensors',
            self.sensor_callback,
            10,
        )
        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)

        # --- state ---
        self.phase            = PHASE_FOLLOW
        self.last_error       = 0.0
        self.search_direction = 1      # +1 = left,  -1 = right
        self.recovery_count   = 0
        self.backup_end_time  = 0.0    # when back-up phase ends

        self.get_logger().info(
            'LineFollower v2 started  '
            '(variable speed + back-up recovery)'
        )

    # ──────────────────────────────────────────────
    # SENSOR CALLBACK
    # ──────────────────────────────────────────────

    def sensor_callback(self, msg: Int32MultiArray):
        data = list(msg.data)
        if len(data) != 5:
            return

        binary = [1 if v < THRESHOLD else 0 for v in data]
        count  = sum(binary)

        # ── currently in back-up phase ─────────────────
        # keep reversing regardless of sensor reading;
        # the sensors may briefly see the line during backup
        if self.phase == PHASE_BACKUP:
            if time.time() < self.backup_end_time:
                self._do_backup()
                return
            else:
                # back-up done → switch to rotate
                self.get_logger().info('Back-up done  → rotating to search')
                self.phase = PHASE_ROTATE

        # ── line found ─────────────────────────────────
        if count > 0:
            if self.phase != PHASE_FOLLOW:
                self.get_logger().info(
                    f'Line re-acquired  (was in {self.phase})'
                    f'  → resuming follow'
                )
            self.phase          = PHASE_FOLLOW
            self.recovery_count = 0

            error = sum(WEIGHTS[i] * binary[i] for i in range(5)) / count

            # save side for recovery direction hint
            self.last_error       = error
            self.search_direction = 1 if error >= 0 else -1

            self._follow(error)
            return

        # ── line completely lost ────────────────────────
        if self.phase == PHASE_FOLLOW:
            # just lost the line → start back-up immediately
            self.get_logger().warn(
                'Line lost  → backing up '
                f'({BACKUP_DURATION} s) before searching'
            )
            self.phase           = PHASE_BACKUP
            self.backup_end_time = time.time() + BACKUP_DURATION
            self._do_backup()
            return

        # ── rotating to search ──────────────────────────
        self._do_rotate()

    # ──────────────────────────────────────────────
    # MOTION HELPERS
    # ──────────────────────────────────────────────

    def _follow(self, error: float):
        """
        Proportional steering + variable speed.
        Speed is scaled down linearly with |error|:
          error = 0  → BASE_SPEED
          error = ±2 → MIN_SPEED
        """
        # normalise error to [0, 1]
        error_norm = min(abs(error) / 2.0, 1.0)
        speed = BASE_SPEED - error_norm * (BASE_SPEED - MIN_SPEED)

        cmd = Twist()
        cmd.linear.x  = speed
        cmd.angular.z = -KP * error
        self.pub.publish(cmd)

    def _do_backup(self):
        """Drive in reverse – used in BACKUP phase."""
        cmd = Twist()
        cmd.linear.x  = BACKUP_SPEED
        cmd.angular.z = 0.0
        self.pub.publish(cmd)

    def _do_rotate(self):
        """
        Rotate in place toward last known line side.
        After RECOVERY_TIMEOUT callbacks, reverse direction.
        """
        self.recovery_count += 1

        if self.recovery_count > RECOVERY_TIMEOUT:
            self.search_direction *= -1
            self.recovery_count    = 0
            self.get_logger().warn(
                'Recovery timeout  → reversing search direction'
            )

        side = 'left' if self.search_direction > 0 else 'right'
        self.get_logger().warn(
            f'Searching {side}  '
            f'(step {self.recovery_count}/{RECOVERY_TIMEOUT})'
        )

        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = RECOVERY_ANGULAR * self.search_direction
        self.pub.publish(cmd)


# ────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = LineFollowerV2()
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