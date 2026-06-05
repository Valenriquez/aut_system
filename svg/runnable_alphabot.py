import os
import re
import glob
import time
import threading
import xml.etree.ElementTree as ET

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image

try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
except ImportError:
    HAS_CV_BRIDGE = False

LINEAR_SPEED  = 0.15
ANGULAR_SPEED = 1.5
FORWARD_TIME  = 1.5
TURN_90_TIME  = 1.0

CAMERA_TOPIC          = '/camera/compressed'
CAMERA_COMPRESSED     = True
CORRECT_FROM_CAMERA   = True
FRAME_WAIT_AFTER_MOVE = 0.2

SVG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svgs')

# ArUco id : (row, col) on the 7x7 grid
MARKER_TO_CELL = {
    0: (0, 0),
    1: (0, 1),
    2: (2, 1),
    3: (2, 3),
    4: (2, 6),
    5: (4, 6),
    6: (6, 6),
}

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


def load_svg_markers(svg_dir):
    markers = []
    dict_name = 'DICT_4X4_1000'
    if not os.path.isdir(svg_dir):
        return markers, dict_name

    pat = re.compile(r'(\d+)x(\d+)_(\d+)[-_](\d+)\.svg$', re.IGNORECASE)
    for path in sorted(glob.glob(os.path.join(svg_dir, '*.svg'))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        bw, bh, size, marker_id = m.groups()
        dict_name = f'DICT_{bw}X{bh}_{size}'
        markers.append((int(marker_id), path))

    markers.sort(key=lambda t: t[0])
    return markers, dict_name


def decode_svg_grid(svg_path):
    with open(svg_path, 'r') as f:
        text = f.read()
    text = re.sub(r'\sxmlns="[^"]+"', '', text, count=1)
    root = ET.fromstring(text)

    white = []
    for el in root.iter('rect'):
        if (el.get('fill') or '').lower() != 'white':
            continue
        white.append((
            float(el.get('x', 0)), float(el.get('y', 0)),
            float(el.get('width', 0)), float(el.get('height', 0)),
        ))

    grid = [[0] * 6 for _ in range(6)]
    for r in range(6):
        for c in range(6):
            cx, cy = c + 0.5, r + 0.5
            for (x, y, w, h) in white:
                if x <= cx <= x + w and y <= cy <= y + h:
                    grid[r][c] = 1
                    break
    return grid


def grid_to_ascii(grid):
    return '\n'.join(
        '    ' + ''.join('  ' if cell else '##' for cell in row)
        for row in grid
    )


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


def _make_aruco_detector(dict_name):
    dict_id = getattr(cv2.aruco, dict_name, None)
    if dict_id is None:
        dict_id = cv2.aruco.DICT_4X4_1000

    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        adict = cv2.aruco.getPredefinedDictionary(dict_id)
    else:
        adict = cv2.aruco.Dictionary_get(dict_id)

    if hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
        return ('new', cv2.aruco.ArucoDetector(adict, params))
    params = cv2.aruco.DetectorParameters_create()
    return ('old', (adict, params))


def _detect(detector, gray):
    kind, obj = detector
    if kind == 'new':
        return obj.detectMarkers(gray)
    adict, params = obj
    return cv2.aruco.detectMarkers(gray, adict, parameters=params)


class PolicyRunner(Node):
    def __init__(self, marker_to_cell, marker_files, aruco_dict_name):
        super().__init__('policy_runner')
        cb_group = ReentrantCallbackGroup()

        self.marker_to_cell = marker_to_cell
        self.marker_files = marker_files

        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)

        msg_type = CompressedImage if CAMERA_COMPRESSED else Image
        self.sub = self.create_subscription(
            msg_type, CAMERA_TOPIC, self.image_cb,
            qos_profile_sensor_data, callback_group=cb_group,
        )

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.bridge = CvBridge() if HAS_CV_BRIDGE else None
        self.aruco = _make_aruco_detector(aruco_dict_name)

        self.get_logger().info(
            f"camera subscriber on {CAMERA_TOPIC} "
            f"({'compressed' if CAMERA_COMPRESSED else 'raw'}, "
            f"cv_bridge={'yes' if HAS_CV_BRIDGE else 'no, manual decode'})"
        )
        self.get_logger().info(f"ArUco dictionary: {aruco_dict_name}")
        self._log_marker_map()

    def _log_marker_map(self):
        if not self.marker_to_cell:
            self.get_logger().warn(
                f"no SVG markers loaded from {SVG_DIR} -- "
                f"camera localization disabled"
            )
            return
        self.get_logger().info("SVG marker -> policy-path cell linkage:")
        for mid in sorted(self.marker_to_cell):
            cell = self.marker_to_cell[mid]
            act = int(policy[cell])
            tag = 'START' if cell == START else ('GOAL' if cell == GOAL else '')
            fname = os.path.basename(self.marker_files.get(mid, f'id {mid}'))
            self.get_logger().info(
                f"  {fname:<18} id={mid} -> cell {cell} "
                f"policy={ACTION_NAME[act]:<5} {tag}"
            )

    def image_cb(self, msg):
        try:
            if CAMERA_COMPRESSED:
                if self.bridge is not None:
                    frame = self.bridge.compressed_imgmsg_to_cv2(
                        msg, desired_encoding='bgr8')
                else:
                    arr = np.frombuffer(msg.data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                if self.bridge is not None:
                    frame = self.bridge.imgmsg_to_cv2(
                        msg, desired_encoding='bgr8')
                else:
                    arr = np.frombuffer(msg.data, dtype=np.uint8)
                    frame = arr.reshape((msg.height, msg.width, -1))
                    if msg.encoding == 'rgb8':
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().warn(f"image_cb failed: {e}")

    def observe_position(self):
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = _detect(self.aruco, gray)
        if ids is None or len(ids) == 0:
            return None

        areas = [cv2.contourArea(c.reshape(-1, 2).astype(np.float32))
                 for c in corners]
        idx = int(np.argmax(areas))
        marker_id = int(ids[idx][0])

        cell = self.marker_to_cell.get(marker_id)
        if cell is None:
            self.get_logger().info(
                f"saw marker id={marker_id} -- not on the policy path")
            return None
        self.get_logger().info(
            f"saw SVG marker id={marker_id} -> policy-path cell {cell}")
        return cell

    def publish_twist(self, linear_x, angular_z, duration):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        end_time = time.time() + duration
        while time.time() < end_time:
            self.pub.publish(msg)
            time.sleep(0.1)
        self.pub.publish(Twist())
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
        return (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE
                and pos not in OBSTACLES)

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

            observed = self.observe_position()
            if observed is not None and observed != pos:
                self.get_logger().warn(
                    f"  SVG marker re-syncs position: {pos} -> {observed}")
                if CORRECT_FROM_CAMERA:
                    pos = observed
            elif observed == pos:
                self.get_logger().info(f"  SVG marker confirms {pos}")

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

            time.sleep(FRAME_WAIT_AFTER_MOVE)


def main():
    markers, aruco_dict_name = load_svg_markers(SVG_DIR)
    marker_files = {mid: path for mid, path in markers}

    path = compute_policy_path(policy, START, GOAL)
    print(f"[policy_runner] loaded {len(markers)} SVG markers from {SVG_DIR}")
    print(f"[policy_runner] ArUco dictionary: {aruco_dict_name}")
    print(f"[policy_runner] policy path ({len(path)} cells): {path}")
    print(f"[policy_runner] fixed SVG marker positions:")
    for mid, cell in sorted(MARKER_TO_CELL.items()):
        fname = os.path.basename(marker_files.get(mid, f'<no SVG for id {mid}>'))
        flags = []
        if mid not in marker_files:
            flags.append('!! no matching SVG file in svgs/')
        if cell not in path:
            flags.append('!! cell is NOT on the policy path')
        note = ('  ' + ' '.join(flags)) if flags else ''
        print(f"[policy_runner]   {fname} (id {mid}) -> fixed cell {cell}{note}")
        if mid in marker_files:
            try:
                print(grid_to_ascii(decode_svg_grid(marker_files[mid])))
            except Exception as e:
                print(f"    (could not decode SVG: {e})")

    rclpy.init()
    node = PolicyRunner(MARKER_TO_CELL, marker_files, aruco_dict_name)

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run()
    finally:
        node.pub.publish(Twist())
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
