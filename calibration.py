#!/usr/bin/env python3
"""
IR line-sensor drift correction for the AlphaBot2.

Drop-in callback for the PolicyRunner node. The robot's open-loop timing
(FORWARD_TIME / TURN_90_TIME) can't hold a straight line on its own, so the
five downward-facing IR sensors are used to nudge heading and to bound the
accumulated dead-reckoning error.

Sensor layout (left -> right): s0 s1 s2 s3 s4
A reading BELOW the threshold means that sensor is OVER the black line.
"""

from geometry_msgs.msg import Twist

# ===================== TUNE THESE =====================
TAU         = 700      # threshold: reading < TAU => sensor is on the line
KP          = 0.6      # proportional gain for angular correction
OMEGA_EDGE  = 0.4      # fixed angular nudge when only an outer sensor falls off
LINEAR_SPEED = 0.15    # m/s forward velocity while correcting
WEIGHTS     = [-2, -1, 0, 1, 2]   # sensor position weights for the error term
# ======================================================


def line_error(readings):
    """
    Weighted-centroid error of the line under the sensor bar.
    e in [-2, +2]: negative = line is to the LEFT, positive = to the RIGHT.
    Returns 0.0 if no sensor sees the line (avoids divide-by-zero).
    """
    on = [r < TAU for r in readings]
    n_on = sum(on)
    if n_on == 0:
        return 0.0
    return sum(w for w, o in zip(WEIGHTS, on) if o) / n_on


def ir_callback(self, readings):
    """
    self : the PolicyRunner node (needs self.cmd_pub, self.state, self.expected_cell)
    readings : list of 5 raw IR values [s0, s1, s2, s3, s4]
    """
    on = [r < TAU for r in readings]
    twist = Twist()
    twist.linear.x = LINEAR_SPEED

    if all(on):
        # All five on the line => crossed a cell boundary.
        # Snap the dead-reckoning estimate to the known checkpoint.
        self.state = self.expected_cell
        twist.angular.z = 0.0

    elif (not on[0]) and all(on[1:]):
        # Outermost-left sensor fell off => robot drifted RIGHT => turn left.
        twist.angular.z = +OMEGA_EDGE

    elif all(on[:4]) and (not on[4]):
        # Outermost-right sensor fell off => robot drifted LEFT => turn right.
        twist.angular.z = -OMEGA_EDGE

    else:
        # Intermediate drift: proportional correction.
        e = line_error(readings)
        twist.angular.z = -KP * e

    self.cmd_pub.publish(twist)