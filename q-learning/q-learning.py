#!/usr/bin/env python3
"""Q-learning trainer and policy runner for the AlphaBot2.

Modes:
    python3 q-learning.py run     # drive the robot (default)
    python3 q-learning.py train   # re-derive the policy
"""

import os
import sys
import math
import time
import argparse
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Int32MultiArray
    from geometry_msgs.msg import Twist
    _ROS_OK = True
except Exception:
    _ROS_OK = False
    Node = object


GRID_SIZE = 7
START     = (0, 0)
GOAL      = (6, 6)
OBSTACLES = {
    (1, 0), (1, 2), (1, 3), (1, 4), (1, 6),
    (3, 1), (3, 2), (3, 3), (3, 5),
    (4, 3), (4, 5),
    (5, 1), (5, 5),
    (6, 1), (6, 3), (6, 5),
}

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS           = [UP, DOWN, LEFT, RIGHT]
ACTION_DELTA      = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
ACTION_TO_HEADING = {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}
HEADING_DELTA     = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
HEADING_NAME      = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
ARROW             = {UP: '^', DOWN: 'v', LEFT: '<', RIGHT: '>', -1: ' '}

_POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'learned_policy.txt')


def compute_policy_path(pol, start=START, goal=GOAL, max_len=200):
    path, seen, pos = [start], {start}, start
    while pos != goal and len(path) < max_len:
        action = int(pol[pos])
        if action not in ACTION_DELTA:
            break
        dr, dc = ACTION_DELTA[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if (not (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE)
                or nxt in OBSTACLES or nxt in seen):
            break
        path.append(nxt); seen.add(nxt); pos = nxt
    return path


GOAL_REWARD  = 100.0
STEP_PENALTY = -20.0
WALL_PENALTY = -10.0
ALPHA         = 0.10
GAMMA         = 0.95
N_EPISODES    = 3000
MAX_STEPS     = 200
EPS_START     = 1.00
EPS_END       = 0.05
EPS_DECAY_EPS = 2000
SEED          = 0


def env_step(state, action):
    dr, dc = ACTION_DELTA[action]
    nr, nc = state[0] + dr, state[1] + dc
    nxt = (nr, nc)
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or nxt in OBSTACLES:
        return state, WALL_PENALTY, False
    if nxt == GOAL:
        return nxt, GOAL_REWARD, True
    return nxt, STEP_PENALTY, False


def train_q():
    rng = np.random.default_rng(SEED)
    Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    returns = []
    for ep in range(N_EPISODES):
        frac    = min(1.0, ep / EPS_DECAY_EPS)
        epsilon = EPS_START + frac * (EPS_END - EPS_START)
        state, total = START, 0.0
        for _ in range(MAX_STEPS):
            if rng.random() < epsilon:
                action = int(rng.integers(len(ACTIONS)))
            else:
                action = int(np.argmax(Q[state[0], state[1]]))
            nxt, reward, done = env_step(state, action)
            total += reward
            best_next = 0.0 if done else float(np.max(Q[nxt[0], nxt[1]]))
            td_error  = reward + GAMMA * best_next - Q[state[0], state[1], action]
            Q[state[0], state[1], action] += ALPHA * td_error
            state = nxt
            if done:
                break
        returns.append(total)
    return Q, returns


def extract_policy(Q):
    pol = -np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) in OBSTACLES or (r, c) == GOAL:
                continue
            pol[r, c] = int(np.argmax(Q[r, c]))
    return pol


def pretty_grid(pol, path):
    path_set = set(path)
    lines = []
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            if   (r, c) == START:      row.append('S')
            elif (r, c) == GOAL:       row.append('G')
            elif (r, c) in OBSTACLES:  row.append('#')
            elif (r, c) in path_set:   row.append(ARROW[int(pol[r, c])])
            else:                      row.append('.')
        lines.append(' '.join(row))
    return '\n'.join(lines)


def run_training():
    print('Training Q-learning ...')
    Q, returns = train_q()
    pol  = extract_policy(Q)
    path = compute_policy_path(pol)
    print(f'\nLast-100-episode mean return: {np.mean(returns[-100:]):.1f}')
    print(f'Greedy path: {len(path)} cells, reaches goal: {path[-1] == GOAL}')
    print(f'{path}\n')
    print(pretty_grid(pol, path))
    np.savetxt(_POLICY_PATH, pol, fmt='%d')
    print(f'\nSaved policy -> {_POLICY_PATH}')
    print('(RUN mode will load this automatically.)')
    return pol


THRESHOLD = 700
KP        = 0.10
WEIGHTS   = [-2, -1, 0, 1, 2]

YAW_SIGN      = +1
STEER_SIGN    = +1
STRAIGHT_TRIM = 0.0

CELL_SIZE     = 0.225
QUARTER_TURN  = math.pi / 2
FAST_LINEAR   = 0.22
SLOW_LINEAR   = 0.15
ANGULAR_SPEED = 0.80
SETTLE_TIME   = 0.9

TURN_TIME_SCALE = 1.3
FWD_TIME_SCALE  = 1.0


_FALLBACK_POLICY = np.array([
    [ 3,  1,  3,  3,  3,  1,  2],
    [-1,  1, -1, -1, -1,  1, -1],
    [ 3,  3,  3,  3,  3,  3,  1],
    [ 0, -1, -1, -1,  0, -1,  1],
    [ 0,  2,  1, -1,  0, -1,  1],
    [ 0, -1,  3,  3,  0, -1,  1],
    [ 0, -1,  0, -1,  0, -1, -1],
], dtype=int)

if os.path.exists(_POLICY_PATH):
    policy = np.loadtxt(_POLICY_PATH, dtype=int)
    _POLICY_SRC = 'learned_policy.txt'
else:
    policy = _FALLBACK_POLICY
    _POLICY_SRC = 'embedded fallback'


class PolicyRunner(Node):

    def __init__(self):
        super().__init__('policy_runner_qlearning')
        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)
        self.create_subscription(
            Int32MultiArray, '/alphabot2/line_sensors',
            self._sensor_cb, qos_profile_sensor_data)

        self._sensor_data = [999, 999, 999, 999, 999]
        self._count       = 0
        self.get_logger().info(
            f'PolicyRunner (Q-learning) ready - policy: {_POLICY_SRC}')

    def _sensor_cb(self, msg):
        if len(msg.data) != 5:
            return
        self._sensor_data = list(msg.data)
        self._count = sum(1 for v in self._sensor_data if v < THRESHOLD)

    def _line_error(self):
        binary = [1 if v < THRESHOLD else 0 for v in self._sensor_data]
        count  = sum(binary)
        if count == 0:
            return None
        return sum(WEIGHTS[i] * binary[i] for i in range(5)) / count

    def _stop(self):
        self.pub.publish(Twist())

    def _turn_timed(self, angular_z, duration):
        cmd = Twist(); cmd.angular.z = YAW_SIGN * angular_z
        end = time.time() + duration
        while time.time() < end:
            self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
        self._stop(); time.sleep(SETTLE_TIME)

    def _drive_cell(self, speed):
        duration = (CELL_SIZE / speed) * FWD_TIME_SCALE
        end = time.time() + duration
        while time.time() < end:
            error = self._line_error()
            cmd = Twist()
            cmd.linear.x  = speed
            cmd.angular.z = (STEER_SIGN * (-KP * error) + STRAIGHT_TRIM
                             if error is not None else STRAIGHT_TRIM)
            self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.04)
        self._stop(); time.sleep(SETTLE_TIME)

    def face(self, current, desired):
        diff = (desired - current) % 4
        if diff == 0:
            return desired
        dur = TURN_TIME_SCALE * QUARTER_TURN / ANGULAR_SPEED
        if diff == 1:
            self._turn_timed(-ANGULAR_SPEED, dur)
        elif diff == 2:
            self._turn_timed(-ANGULAR_SPEED, 2 * dur)
        else:
            self._turn_timed(+ANGULAR_SPEED, dur)
        return desired

    def _in_bounds_and_free(self, pos):
        r, c = pos
        return (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE
                and pos not in OBSTACLES)

    def _turn_coming(self, cur, nxt):
        if nxt == GOAL or nxt in OBSTACLES:
            return True
        na = int(policy[nxt])
        if na == -1:
            return True
        return na != int(policy[cur])

    def run(self):
        time.sleep(1.5)
        pos, heading = START, 0
        self.get_logger().info(f'Start {pos}  heading {HEADING_NAME[heading]}')

        for step in range(1, 100):
            if pos == GOAL:
                self.get_logger().info(f'*** Goal reached in {step-1} steps ***')
                return
            action = int(policy[pos])
            if action == -1:
                self.get_logger().warn(f'No action at {pos}')
                return

            heading = self.face(heading, ACTION_TO_HEADING[action])
            dr, dc  = HEADING_DELTA[heading]
            target  = (pos[0] + dr, pos[1] + dc)
            if not self._in_bounds_and_free(target):
                self.get_logger().warn(f'Obstacle at {target}, aborting.')
                return

            speed = SLOW_LINEAR if self._turn_coming(pos, target) else FAST_LINEAR
            self.get_logger().info(
                f'step {step:2d}: {pos} -> {target}  {HEADING_NAME[heading]}')
            self._drive_cell(speed)
            pos = target

        self.get_logger().warn('Step limit reached.')


def run_robot():
    if not _ROS_OK:
        sys.exit('ERROR: rclpy not found. Run inside ROS 2 first:\n'
                 '  source /opt/ros/humble/setup.bash')
    path = compute_policy_path(policy, START, GOAL)
    print(f'[policy_runner] policy source: {_POLICY_SRC}')
    print(f'[policy_runner] path ({len(path)} cells): {path}')
    rclpy.init()
    node = PolicyRunner()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', nargs='?', default='run', choices=['train', 'run'])
    args = ap.parse_args()
    run_training() if args.mode == 'train' else run_robot()


if __name__ == '__main__':
    main()
