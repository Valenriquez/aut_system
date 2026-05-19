#!/usr/bin/env python3
"""
policy_runner.py — Drive the AlphaBot2 through a precomputed grid policy.

Expects 'policy.npy' (produced by the training script on the laptop) to be
in the same folder as this file.

Requires the lab's motor driver to be running in another SSH session:
    cd ~/catkin_ws && source devel/setup.bash
    roslaunch web_control web_control.launch
"""

import numpy as np
import rospy
from geometry_msgs.msg import Twist

# ===================== TUNE THESE IF MOTION IS OFF =====================
LINEAR_SPEED  = 0.15   # m/s
ANGULAR_SPEED = 1.5    # rad/s
FORWARD_TIME  = 1.5    # seconds to drive one cell
TURN_90_TIME  = 1.0    # seconds to rotate 90 degrees
# =======================================================================

# Must match the training script
GRID_SIZE = 7
GOAL = (6, 6)
OBSTACLES = {
    (1, 0), (1, 2), (1, 3), (1, 4), (1, 6),
    (3, 1), (3, 2), (3, 3), (3, 5),
    (4, 3), (4, 5),
    (5, 1), (5, 5),
    (6, 1), (6, 3), (6, 5),
}

# Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
# Headings clockwise: 0=N, 1=E, 2=S, 3=W
ACTION_TO_HEADING = {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}
HEADING_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
HEADING_NAME  = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}

# Load the policy produced by the training script
policy = np.load('policy.npy')

rospy.init_node('policy_runner', anonymous=True)
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
rate = rospy.Rate(10)


def publish_twist(linear_x, angular_z, duration):
    msg = Twist()
    msg.linear.x = linear_x
    msg.angular.z = angular_z
    end = rospy.Time.now() + rospy.Duration(duration)
    while not rospy.is_shutdown() and rospy.Time.now() < end:
        pub.publish(msg)
        rate.sleep()
    pub.publish(Twist())
    rospy.sleep(0.3)


def face(current, desired):
    diff = (desired - current) % 4
    if diff == 0:
        return desired
    if diff == 1:
        publish_twist(0, -ANGULAR_SPEED, TURN_90_TIME)        # right
    elif diff == 2:
        publish_twist(0, -ANGULAR_SPEED, 2 * TURN_90_TIME)    # 180
    else:
        publish_twist(0,  ANGULAR_SPEED, TURN_90_TIME)        # left
    return desired


def in_bounds_and_free(pos):
    r, c = pos
    return (0 <= r < GRID_SIZE
            and 0 <= c < GRID_SIZE
            and pos not in OBSTACLES)


def main():
    rospy.sleep(1.0)
    pos = (0, 0)
    heading = 0   # NORTH
    rospy.loginfo(f"start at {pos}, facing {HEADING_NAME[heading]}")

    for step in range(1, 60):
        if pos == GOAL:
            rospy.loginfo(f"*** reached goal in {step - 1} moves ***")
            return
        action = int(policy[pos])
        if action == -1:
            rospy.logwarn(f"no action at {pos}, stopping")
            return
        heading = face(heading, ACTION_TO_HEADING[action])
        dr, dc = HEADING_DELTA[heading]
        target = (pos[0] + dr, pos[1] + dc)
        if in_bounds_and_free(target):
            publish_twist(LINEAR_SPEED, 0, FORWARD_TIME)
            pos = target
            rospy.loginfo(f"step {step:2d}: -> {pos}, facing {HEADING_NAME[heading]}")
        else:
            rospy.loginfo(f"step {step:2d}: bounce (stayed at {pos})")

    rospy.logwarn(f"max steps reached, final pos {pos}")


if __name__ == '__main__':
    try:
        main()
    finally:
        pub.publish(Twist())