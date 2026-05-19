import rospy
import json
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

# ── 1. LOAD THE POLICY ────────────────────────────────────────────────────────
with open("policy.json") as f:
    data = json.load(f)

POLICY    = data["policy"]
GOAL      = tuple(data["goal"])
GRID_SIZE = data["grid_size"]

# ── 2. MAP ARUCO MARKER IDs TO GRID CELLS ────────────────────────────────────
# You place a physical ArUco marker on the floor of each cell.
# This dictionary tells the robot: "if I see marker 5, I'm at cell (0,3)"
# Fill this in based on your actual maze layout!
MARKER_TO_CELL = {
    0:  (3, 0),   # start cell
    1:  (3, 1),
    2:  (3, 2),
    3:  (2, 0),
    4:  (2, 2),
    5:  (1, 0),
    6:  (1, 2),
    7:  (0, 0),
    8:  (0, 1),
    9:  (0, 2),
    10: (0, 3),   # goal cell
}

# ── 3. MAP POLICY ACTIONS TO WHEEL COMMANDS ───────────────────────────────────
# Twist message: linear.x = forward speed, angular.z = turning speed
# Tune these values for your AlphaBot's actual speed!
LINEAR_SPEED  = 0.15   # m/s forward
ANGULAR_SPEED = 1.2    # rad/s turning
MOVE_DURATION = 1.0    # seconds per move (tune this per cell size)
TURN_DURATION = 1.3    # seconds per 90° turn

# The robot always faces 'up' initially.
# We track its current facing so we know how to turn.
# Directions: 0=up, 1=right, 2=down, 3=left
FACING = {'up': 0, 'right': 1, 'down': 2, 'left': 3}

# ── 4. ARUCO DETECTOR SETUP ───────────────────────────────────────────────────
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ── 5. STATE VARIABLES ────────────────────────────────────────────────────────
current_facing = 0          # starts facing 'up'
is_moving      = False      # prevents processing new images while moving
goal_reached   = False

# ── 6. MOVEMENT FUNCTIONS ─────────────────────────────────────────────────────
def publish_velocity(linear, angular, duration):
    """Send a movement command for a fixed duration, then stop."""
    cmd = Twist()
    cmd.linear.x  = linear
    cmd.angular.z = angular
    
    end_time = rospy.Time.now() + rospy.Duration(duration)
    rate = rospy.Rate(10)
    while rospy.Time.now() < end_time:
        pub.publish(cmd)
        rate.sleep()
    
    # Stop
    pub.publish(Twist())
    rospy.sleep(0.3)   # short pause before next action

def turn_to_face(target_action):
    """
    Turn the robot so it faces the direction the policy wants.
    Uses the shortest rotation (left or right).
    """
    global current_facing
    
    target_facing = FACING[target_action]
    
    # How many 90° turns needed?
    diff = (target_facing - current_facing) % 4
    
    if diff == 0:
        return   # already facing the right way
    elif diff == 1:
        # Turn right once
        rospy.loginfo("Turning right")
        publish_velocity(0, -ANGULAR_SPEED, TURN_DURATION)
        current_facing = target_facing
    elif diff == 3:
        # Turn left once (faster than turning right 3 times)
        rospy.loginfo("Turning left")
        publish_velocity(0, ANGULAR_SPEED, TURN_DURATION)
        current_facing = target_facing
    elif diff == 2:
        # Turn 180° — do it as two right turns
        rospy.loginfo("Turning 180°")
        publish_velocity(0, -ANGULAR_SPEED, TURN_DURATION)
        publish_velocity(0, -ANGULAR_SPEED, TURN_DURATION)
        current_facing = target_facing

def execute_action(action):
    """Turn to face the right direction, then move forward one cell."""
    global is_moving
    
    is_moving = True
    
    turn_to_face(action)
    
    rospy.loginfo(f"Moving {action}")
    publish_velocity(LINEAR_SPEED, 0, MOVE_DURATION)
    
    is_moving = False

# ── 7. IMAGE CALLBACK ─────────────────────────────────────────────────────────
def image_callback(msg):
    """
    Called every time the camera publishes a new frame.
    This is the brain of the robot:
    image → detect marker → find cell → look up policy → move
    """
    global goal_reached, is_moving

    # Don't process new frames while already moving
    if is_moving or goal_reached:
        return

    # ── Decode compressed image ───────────────────────────────────────────
    np_arr = np.frombuffer(msg.data, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Detect ArUco markers ──────────────────────────────────────────────
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        rospy.logwarn_throttle(3, "No ArUco marker detected — robot lost!")
        return

    # Use the first detected marker
    marker_id = int(ids[0][0])
    rospy.loginfo(f"Detected marker ID: {marker_id}")

    # ── Map marker to grid cell ───────────────────────────────────────────
    if marker_id not in MARKER_TO_CELL:
        rospy.logwarn(f"Marker {marker_id} not in map — ignoring")
        return

    row, col = MARKER_TO_CELL[marker_id]
    rospy.loginfo(f"Robot is at cell ({row}, {col})")

    # ── Check if goal reached ─────────────────────────────────────────────
    if (row, col) == GOAL:
        rospy.loginfo("GOAL REACHED!")
        goal_reached = True
        pub.publish(Twist())   # make sure robot stops
        return

    # ── Look up policy ────────────────────────────────────────────────────
    cell_key = f"{row},{col}"
    action   = POLICY.get(cell_key)

    if action is None or action == 'goal':
        rospy.logwarn(f"No valid action for cell {cell_key}")
        return

    rospy.loginfo(f"Policy says: {action}")

    # ── Execute the action ────────────────────────────────────────────────
    execute_action(action)

# ── 8. MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node('maze_navigator')
    rospy.loginfo("Maze navigator started!")

    # Publisher: sends movement commands to the robot
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    # Subscriber: receives camera images
    rospy.Subscriber('/camera/compressed', CompressedImage, image_callback)

    rospy.loginfo("Waiting for camera images...")
    rospy.spin()   # keeps node alive until Ctrl+C