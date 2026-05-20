#!/usr/bin/env python3
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2

try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
except ImportError:
    HAS_CV_BRIDGE = False

# ===================== MOTION TUNING =====================
LINEAR_SPEED  = 0.15
ANGULAR_SPEED = 1.5
FORWARD_TIME  = 1.5
TURN_90_TIME  = 1.0

# ===================== CAMERA / LOCALIZATION =====================
CAMERA_TOPIC          = '/alphabot2/image_raw'
ARUCO_DICT            = cv2.aruco.DICT_4X4_50   # change if your markers use another dict
CORRECT_FROM_CAMERA   = False  # True = overwrite pos with camera reading; False = just log
FRAME_WAIT_AFTER_MOVE = 0.2    # extra settle time before reading a frame

# Map ArUco IDs -> grid cells (row, col). FILL THIS IN for your lab setup.
MARKER_TO_CELL = {
    # 0: (0, 0),
    # 1: (0, 1),
    # 10: (6, 6),  # goal marker, etc.
}
# =================================================================

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
HEADING_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
HEADING_NAME  = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}

policy = np.load('policy.npy')


def _make_aruco_detector(dict_name):
    """Handles both old (<=4.6) and new (>=4.7) cv2.aruco APIs."""
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        adict = cv2.aruco.getPredefinedDictionary(dict_name)
    else:
        adict = cv2.aruco.Dictionary_get(dict_name)

    if hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
        return ('new', cv2.aruco.ArucoDetector(adict, params))
    else:
        params = cv2.aruco.DetectorParameters_create()
        return ('old', (adict, params))


def _detect(detector, gray):
    kind, obj = detector
    if kind == 'new':
        return obj.detectMarkers(gray)
    adict, params = obj
    return cv2.aruco.detectMarkers(gray, adict, parameters=params)


class PolicyRunner(Node):
    def __init__(self):
        super().__init__('policy_runner')
        cb_group = ReentrantCallbackGroup()

        self.pub = self.create_publisher(Twist, '/alphabot2/cmd_vel', 10)
        self.sub = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_cb, 10,
            callback_group=cb_group,
        )

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.bridge = CvBridge() if HAS_CV_BRIDGE else None
        self.aruco = _make_aruco_detector(ARUCO_DICT)

        self.get_logger().info(
            f"camera subscriber on {CAMERA_TOPIC} "
            f"(cv_bridge={'yes' if HAS_CV_BRIDGE else 'no, using manual decode'})"
        )

    # ---------- camera ----------

    def image_cb(self, msg: Image):
        try:
            if self.bridge is not None:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
        """Return (row, col) of the largest known marker in view, or None."""
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = _detect(self.aruco, gray)
        if ids is None or len(ids) == 0:
            return None

        # pick the closest-looking marker = largest area in the image
        areas = [cv2.contourArea(c.reshape(-1, 2).astype(np.float32)) for c in corners]
        idx = int(np.argmax(areas))
        marker_id = int(ids[idx][0])

        cell = MARKER_TO_CELL.get(marker_id)
        if cell is None:
            self.get_logger().info(f"saw unmapped marker id={marker_id}")
            return None
        return cell

    # ---------- motion ----------

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
        return (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and pos not in OBSTACLES)

    # ---------- main loop ----------

    def run(self):
        time.sleep(1.0)
        pos = (0, 0)
        heading = 0
        self.get_logger().info(f"start at {pos}, facing {HEADING_NAME[heading]}")

        for step in range(1, 60):
            if pos == GOAL:
                self.get_logger().info(f"*** reached goal in {step - 1} moves ***")
                return

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
                self.get_logger().info(f"step {step:2d}: bounce (stayed at {pos})")

            # --- camera localization check, after the move has settled ---
            time.sleep(FRAME_WAIT_AFTER_MOVE)
            observed = self.observe_position()
            if observed is None:
                self.get_logger().info(f"  camera: no known marker visible")
            elif observed != pos:
                self.get_logger().warn(
                    f"  camera says {observed}, policy thinks {pos}"
                )
                if CORRECT_FROM_CAMERA:
                    pos = observed
                    self.get_logger().warn(f"  -> corrected pos to {pos}")
            else:
                self.get_logger().info(f"  camera confirms {pos}")


def main():
    rclpy.init()
    node = PolicyRunner()

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