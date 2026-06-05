#!/usr/bin/env python3
"""IR line-sensor drift correction for the AlphaBot2."""

from geometry_msgs.msg import Twist

TAU          = 700
KP           = 0.6
OMEGA_EDGE   = 0.4
LINEAR_SPEED = 0.15
WEIGHTS      = [-2, -1, 0, 1, 2]


def line_error(readings):
    on = [r < TAU for r in readings]
    n_on = sum(on)
    if n_on == 0:
        return 0.0
    return sum(w for w, o in zip(WEIGHTS, on) if o) / n_on


def ir_callback(self, readings):
    on = [r < TAU for r in readings]
    twist = Twist()
    twist.linear.x = LINEAR_SPEED

    if all(on):
        self.state = self.expected_cell
        twist.angular.z = 0.0

    elif (not on[0]) and all(on[1:]):
        twist.angular.z = +OMEGA_EDGE

    elif all(on[:4]) and (not on[4]):
        twist.angular.z = -OMEGA_EDGE

    else:
        e = line_error(readings)
        twist.angular.z = -KP * e

    self.cmd_pub.publish(twist)
