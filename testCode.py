import signal
import time
import sys
import os
import threading
import json

import cv2
import numpy as np

# Add path to access ros_robot_controller_sdk and takePhoto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/Motor Control/board_demo')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/cameraCode')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/Inverse Kinematics')

import ros_robot_controller_sdk as rrc
import takePhoto
import inverseKinematics

# Global control variables
running = True
servo_ids = [1, 2, 3]  # IDs of the three bus servos in IK order
MOVE_DURATION_S = 0.3
SERVO_ZERO_OFFSETS_DEG = {1: 90.0, 2: 90.0, 3: 90.0}
SERVO_DIRECTIONS = {1: -1.0, 2: -1.0, 3: -1.0}
SERVO_ANGLE_SCALES = {1: 1.0, 2: 1.0, 3: 1.0}
SERVO_STARTUP_ENABLE_DELAY_S = 0.35
SERVO_TORQUE_STAGE_DELAY_S = 0.35
SQUARE_DWELL_S = 0.5
UPDOWN_Z_LOW = -600.0
UPDOWN_Z_HIGH = -400.0
UPDOWN_DWELL_S = 0.6
UPDOWN_DEFAULT_CYCLES = 5
NPATH_X_MM = 200.0
NPATH_Y_MM = 200.0
NPATH_Z_LOW = -550.0
NPATH_Z_HIGH = -450.0
TORQUE_ON_HOLD_DURATION_S = 0.35
TORQUE_ON_SETTLE_S = 0.08
TORQUE_ON_AUTO_HOLD_ZERO = True
POSITION_TOLERANCE_RAW = 5
FEEDBACK_MAX_ITERS = 10
FEEDBACK_SETTLE_S = 0.15
FEEDBACK_GAIN = 1.0
CAMERA_XYZ_MAX_ITERS = 10
CAMERA_XYZ_GAIN = 0.2
CAMERA_XYZ_TOLERANCE_MM = 12.0
CAMERA_SETTLE_S = 0.4
CAMERA_INTRINSICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameraCode", "camera_intrinsics.json")
CAMERA_3D_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameraCode", "aruco_end_effector_calibration_3d.json")


# Board initialization
board = None
try:
    print("Initializing board...")
    board = rrc.Board()
    board.enable_reception()
    print("Board initialized successfully")
except Exception as e:
    print(f"WARNING: Could not initialize board: {e}")
    print("Continuing in camera-only mode (servos disabled)")

motion_lock = threading.Lock()
sequence_running = False
camera = None
target_format = None
camera_matrix = None
dist_coeffs = None
camera_to_robot_rotation = None
camera_to_robot_translation = None
aruco_tag_id = 0
aruco_tag_size_mm = 20.0
aruco_dictionary = None
aruco_detector = None
aruco_parameters = None


def ik_angles_to_servo_positions(angles):
    return {servo_id: angle for servo_id, angle in zip(servo_ids, angles)}

def read_servos(servo_ids):
    for servo_id in servo_ids:
        try:
            pos = board.bus_servo_read_position(servo_id)
            if pos:
                print(f"Servo {servo_id} current position: {pos[0]}")
        except Exception as e:
            print(f"Could not read position for servo {servo_id}: {e}")


def initialize_camera_once():
    global camera, target_format
    if camera is None or target_format is None:
        print("Initializing camera...")
        camera, target_format = takePhoto.initialzeCamera()
        print("Camera initialized successfully")


def load_camera_feedback_calibration():
    global camera_matrix, dist_coeffs, camera_to_robot_rotation, camera_to_robot_translation
    global aruco_tag_id, aruco_tag_size_mm, aruco_dictionary, aruco_detector, aruco_parameters

    if camera_matrix is not None:
        return

    intrinsics = json.loads(open(CAMERA_INTRINSICS_PATH, "r").read())
    calib3d = json.loads(open(CAMERA_3D_CALIB_PATH, "r").read())

    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.asarray(intrinsics["dist_coeffs"], dtype=np.float32)
    camera_to_robot_rotation = np.asarray(calib3d["camera_to_robot_rotation"], dtype=np.float32)
    camera_to_robot_translation = np.asarray(calib3d["camera_to_robot_translation_mm"], dtype=np.float32).reshape(3)
    aruco_tag_id = int(calib3d.get("aruco_tag_id", 0))
    aruco_tag_size_mm = float(calib3d.get("aruco_tag_size_mm", 20.0))
    dict_name = calib3d.get("aruco_dict", "DICT_ARUCO_ORIGINAL")

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available")
    dict_id = getattr(cv2.aruco, dict_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    elif hasattr(cv2.aruco, "Dictionary_get"):
        aruco_dictionary = cv2.aruco.Dictionary_get(dict_id)
    else:
        raise RuntimeError("No supported ArUco dictionary API found")

    if hasattr(cv2.aruco, "DetectorParameters"):
        aruco_parameters = cv2.aruco.DetectorParameters()
    elif hasattr(cv2.aruco, "DetectorParameters_create"):
        aruco_parameters = cv2.aruco.DetectorParameters_create()
    else:
        aruco_parameters = None

    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)


def estimate_robot_xyz_from_camera():
    initialize_camera_once()
    load_camera_feedback_calibration()

    frame = takePhoto.takePhoto(camera, target_format, save_photo=False)
    if frame is None:
        return None

    if aruco_detector is not None:
        corners, ids, _ = aruco_detector.detectMarkers(frame)
    elif hasattr(cv2.aruco, "detectMarkers"):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dictionary, parameters=aruco_parameters)
    else:
        raise RuntimeError("No supported ArUco detectMarkers API found in this OpenCV build")

    if ids is None:
        return None
    ids_flat = ids.flatten().tolist()
    if aruco_tag_id not in ids_flat:
        return None

    idx = ids_flat.index(aruco_tag_id)
    marker_corners = corners[idx][0].astype(np.float32)
    half = aruco_tag_size_mm / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    ok, _rvec, tvec = cv2.solvePnP(
        object_points,
        marker_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None

    camera_point = np.asarray(tvec, dtype=np.float32).reshape(3)
    robot_point = camera_to_robot_rotation @ camera_point + camera_to_robot_translation
    return float(robot_point[0]), float(robot_point[1]), float(robot_point[2])


def logical_angle_to_raw(servo_id, logical_angle):
    direction = SERVO_DIRECTIONS.get(servo_id, 1.0)
    scale = SERVO_ANGLE_SCALES.get(servo_id, 1.0)
    physical_angle = SERVO_ZERO_OFFSETS_DEG.get(servo_id, 0.0) + (direction * scale * float(logical_angle))
    physical_angle = max(0.0, min(240.0, physical_angle))
    return int((physical_angle / 240.0) * 1000)


def read_servo_raw_positions(target_ids):
    positions = {}
    for servo_id in target_ids:
        try:
            pos = board.bus_servo_read_position(servo_id)
            if pos:
                positions[servo_id] = int(pos[0])
        except Exception as e:
            print(f"Could not read position for servo {servo_id}: {e}")
    return positions


def drive_to_targets_with_feedback(target_raw, update_activity=True):
    remaining = dict(target_raw)
    commanded_raw = dict(target_raw)
    for iteration in range(1, FEEDBACK_MAX_ITERS + 1):
        if not remaining:
            return True
        command = [[servo_id, commanded_raw[servo_id]] for servo_id in remaining]
        board.bus_servo_set_position(MOVE_DURATION_S * 3, command)
        time.sleep(MOVE_DURATION_S * 3 + FEEDBACK_SETTLE_S)

        measured = read_servo_raw_positions(remaining.keys())
        next_remaining = {}
        for servo_id, desired_raw in remaining.items():
            actual_raw = measured.get(servo_id)
            if actual_raw is None:
                next_remaining[servo_id] = desired_raw
                continue
            error = desired_raw - actual_raw
            if abs(error) > POSITION_TOLERANCE_RAW:
                next_remaining[servo_id] = desired_raw
                corrected = commanded_raw[servo_id] + int(round(FEEDBACK_GAIN * error))
                commanded_raw[servo_id] = max(0, min(1000, corrected))
            if update_activity:
                print(
                    f"Servo {servo_id} feedback iter={iteration} "
                    f"target={desired_raw} actual={actual_raw} error={error} "
                    f"next_cmd={commanded_raw[servo_id]}"
                )
        remaining = next_remaining

    if update_activity and remaining:
        print(f"Feedback loop ended with remaining error on servos: {sorted(remaining)}")
    return not remaining

def move_servos(servo_positions, update_activity=True):
    if board is None:
        print("Board not initialized, skipping servo movement")
        return
    
    try:
        with motion_lock:
            # Convert angles to servo positions and move servos
            target_raw = {}
            for servo_id in servo_ids:
                if servo_id in servo_positions:
                    target_raw[servo_id] = logical_angle_to_raw(servo_id, servo_positions[servo_id])

            if target_raw:
                if update_activity:
                    print(f"Moving servos simultaneously: {sorted(target_raw.items())}")
                drive_to_targets_with_feedback(target_raw, update_activity=update_activity)
                if update_activity:
                    read_servos(servo_ids)
    
    except Exception as e:
        print(f"Error moving servos: {e}")


def set_torque(state, servo_id=None):
    if board is None:
        print("Board not initialized, cannot change torque")
        return
    torque_value = 0 if state == "on" else 1
    target_ids = [servo_id] if servo_id is not None else servo_ids
    target_positions = {}
    if state == "on":
        for sid in target_ids:
            try:
                pos = board.bus_servo_read_position(sid)
                if pos:
                    target_positions[sid] = int(max(0, min(1000, pos[0])))
            except Exception:
                pass
    for sid in target_ids:
        board.bus_servo_enable_torque(sid, torque_value)
        print(f"Servo {sid} torque {state.upper()}")
        if state == "on":
            hold_pos = target_positions.get(sid)
            if hold_pos is not None:
                board.bus_servo_set_position(TORQUE_ON_HOLD_DURATION_S, [[sid, hold_pos]])
            if servo_id is None:
                time.sleep(SERVO_TORQUE_STAGE_DELAY_S)
            else:
                time.sleep(TORQUE_ON_SETTLE_S)
    if state == "on" and servo_id is None and TORQUE_ON_AUTO_HOLD_ZERO:
        print("Applying post-torque zero hold (a,0,0,0)")
        move_servos({servo_id: 0.0 for servo_id in servo_ids})


def build_square_points(cx, cy, z, side_len):
    half = side_len / 2.0
    return [
        (cx - half, cy - half, z),
        (cx + half, cy - half, z),
        (cx + half, cy + half, z),
        (cx - half, cy + half, z),
    ]


def run_square_path(cx, cy, z, side_len, dwell_s=SQUARE_DWELL_S):
    global sequence_running
    sequence_running = True
    points = build_square_points(cx, cy, z, side_len)
    if points:
        points.append(points[0])
    print(f"Square path center=({cx:.1f},{cy:.1f},{z:.1f}) side={side_len:.1f} mm")
    try:
        for idx, (px, py, pz) in enumerate(points, start=1):
            angles = inverseKinematics.getAngles(px, py, pz)
            print(f"  P{idx}: ({px:.1f},{py:.1f},{pz:.1f}) -> IK {angles}")
            if angles is None:
                print(f"  P{idx} skipped: no valid IK solution")
                continue
            servo_positions = ik_angles_to_servo_positions(angles)
            move_servos(servo_positions, update_activity=False)
            time.sleep(max(0.0, dwell_s))
    finally:
        sequence_running = False


def run_updown_test(cycles=UPDOWN_DEFAULT_CYCLES, dwell_s=UPDOWN_DWELL_S):
    global sequence_running
    sequence_running = True
    print(
        f"Up/down test at x=0, y=0 between z={UPDOWN_Z_LOW:.1f} and z={UPDOWN_Z_HIGH:.1f}, "
        f"cycles={cycles}, dwell={dwell_s:.2f}s"
    )
    waypoints = [(0.0, 0.0, UPDOWN_Z_LOW), (0.0, 0.0, UPDOWN_Z_HIGH)]
    try:
        for i in range(cycles):
            for idx, (x, y, z) in enumerate(waypoints, start=1):
                angles = inverseKinematics.getAngles(x, y, z)
                print(f"  Cycle {i+1} P{idx}: ({x:.1f},{y:.1f},{z:.1f}) -> IK {angles}")
                if angles is None:
                    print("  Skipped: no valid IK solution")
                    continue
                move_servos(ik_angles_to_servo_positions(angles), update_activity=False)
                time.sleep(max(0.0, dwell_s))
    finally:
        sequence_running = False


def run_n_path(x_mm=NPATH_X_MM, y_mm=NPATH_Y_MM, dwell_s=SQUARE_DWELL_S):
    global sequence_running
    sequence_running = True
    anchors = [
        (x_mm, 0.0),
        (-x_mm, 0.0),
        (0.0, y_mm),
        (0.0, -y_mm),
    ]
    points = []
    z_pairs = [
        (NPATH_Z_LOW, NPATH_Z_HIGH),
        (NPATH_Z_HIGH, NPATH_Z_LOW),
        (NPATH_Z_LOW, NPATH_Z_HIGH),
        (NPATH_Z_HIGH, NPATH_Z_LOW),
    ]
    for (ax, ay), (z1, z2) in zip(anchors, z_pairs):
        points.append((ax, ay, z1))
        points.append((ax, ay, z2))
    print(
        "N-path test anchors: "
        f"({x_mm:.0f},0)->({-x_mm:.0f},0)->(0,{y_mm:.0f})->(0,{-y_mm:.0f}) "
        f"with z order [{NPATH_Z_LOW:.0f},{NPATH_Z_HIGH:.0f},{NPATH_Z_HIGH:.0f},{NPATH_Z_LOW:.0f},"
        f"{NPATH_Z_LOW:.0f},{NPATH_Z_HIGH:.0f},{NPATH_Z_HIGH:.0f},{NPATH_Z_LOW:.0f}]"
    )
    try:
        for idx, (x, y, z) in enumerate(points, start=1):
            angles = inverseKinematics.getAngles(x, y, z)
            print(f"  P{idx}: ({x:.1f},{y:.1f},{z:.1f}) -> IK {angles}")
            if angles is None:
                print(f"  P{idx} skipped: no valid IK solution")
                continue
            move_servos(ik_angles_to_servo_positions(angles), update_activity=False)
            time.sleep(max(0.0, dwell_s))
    finally:
        sequence_running = False


def run_camera_xyz_feedback(target_xyz):
    desired = np.asarray(target_xyz, dtype=np.float32)
    commanded = np.asarray(target_xyz, dtype=np.float32)
    print(f"Camera-closed-loop target xyz=({desired[0]:.1f}, {desired[1]:.1f}, {desired[2]:.1f})")

    for iteration in range(1, CAMERA_XYZ_MAX_ITERS + 1):
        angles = inverseKinematics.getAngles(float(commanded[0]), float(commanded[1]), float(commanded[2]))
        print(
            f"  Iter {iteration} command xyz=({commanded[0]:.1f}, {commanded[1]:.1f}, {commanded[2]:.1f}) "
            #f"-> IK {angles}"
        )
        if angles is None:
            print("  No valid IK solution for corrected command; aborting")
            return

        move_servos(ik_angles_to_servo_positions(angles), update_activity=False)
        time.sleep(CAMERA_SETTLE_S)

        measured = estimate_robot_xyz_from_camera()
        if measured is None:
            print("  Camera could not estimate ArUco robot XYZ; aborting")
            return

        measured_vec = np.asarray(measured, dtype=np.float32)
        error = desired - measured_vec
        error_norm = float(np.linalg.norm(error))
        print(
            f"  Measured xyz=({measured_vec[0]:.1f}, {measured_vec[1]:.1f}, {measured_vec[2]:.1f}) "
            f"error=({error[0]:.1f}, {error[1]:.1f}, {error[2]:.1f}) |e|={error_norm:.1f}"
        )
        if error_norm <= CAMERA_XYZ_TOLERANCE_MM:
            print("  Camera feedback converged")
            return

        commanded = commanded + (CAMERA_XYZ_GAIN * error)

    print("  Camera feedback loop reached max iterations without converging")


def parse_control_input(text):
    """
    Supported formats:
      - "x, y, z" (default XYZ mode for backward compatibility)
      - "x, x, y, z" (explicit XYZ mode)
      - "c, x, y, z" (camera-closed-loop XYZ correction using ArUco)
      - "a, a1, a2, a3" (direct motor angles for servos 1/2/3)
      - "t, on|off[, servo_id]" (manual torque control)
      - "sq, cx, cy, z, side_len[, dwell_s]" (run 4-point square via IK)
      - "ud[, cycles[, dwell_s]]" (x=0,y=0 alternating z up/down test)
      - "n[, x, y[, dwell_s]]" (pick/place N-path sequence)
    """
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty input")

    mode_token = parts[0].lower()
    if mode_token in ("x", "xyz"):
        if len(parts) != 4:
            raise ValueError("xyz mode requires: x, x_val, y_val, z_val")
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        return ("xyz", (x, y, z))

    if mode_token in ("c", "cam", "camera"):
        if len(parts) != 4:
            raise ValueError("camera mode requires: c, x_val, y_val, z_val")
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        return ("camera_xyz", (x, y, z))

    if mode_token in ("a", "ang", "angle", "angles"):
        if len(parts) != 4:
            raise ValueError("angle mode requires: a, a1, a2, a3")
        a1, a2, a3 = float(parts[1]), float(parts[2]), float(parts[3])
        return ("angles", (a1, a2, a3))

    if mode_token in ("t", "torque"):
        if len(parts) not in (2, 3):
            raise ValueError("torque mode requires: t, on|off[, servo_id]")
        state = parts[1].lower()
        if state not in ("on", "off"):
            raise ValueError("torque state must be 'on' or 'off'")
        servo_id = int(parts[2]) if len(parts) == 3 else None
        return ("torque", (state, servo_id))

    if mode_token in ("sq", "square"):
        if len(parts) not in (5, 6):
            raise ValueError("square mode requires: sq, cx, cy, z, side_len[, dwell_s]")
        cx, cy, z = float(parts[1]), float(parts[2]), float(parts[3])
        side_len = float(parts[4])
        dwell_s = float(parts[5]) if len(parts) == 6 else SQUARE_DWELL_S
        return ("square", (cx, cy, z, side_len, dwell_s))

    if mode_token in ("ud", "updown"):
        if len(parts) > 3:
            raise ValueError("up/down mode requires: ud[, cycles[, dwell_s]]")
        cycles = int(parts[1]) if len(parts) >= 2 else UPDOWN_DEFAULT_CYCLES
        dwell_s = float(parts[2]) if len(parts) == 3 else UPDOWN_DWELL_S
        return ("updown", (cycles, dwell_s))

    if mode_token in ("n", "npath"):
        if len(parts) not in (1, 2, 3, 4):
            raise ValueError("n-path mode requires: n[, x, y[, dwell_s]]")
        if len(parts) == 1:
            return ("npath", (NPATH_X_MM, NPATH_Y_MM, SQUARE_DWELL_S))
        if len(parts) == 2:
            dwell_s = float(parts[1])
            return ("npath", (NPATH_X_MM, NPATH_Y_MM, dwell_s))
        x_mm = float(parts[1])
        y_mm = float(parts[2])
        dwell_s = float(parts[3]) if len(parts) == 4 else SQUARE_DWELL_S
        return ("npath", (x_mm, y_mm, dwell_s))

    if len(parts) == 3:
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        return ("xyz", (x, y, z))

    raise ValueError("invalid format; use 'x,x,y,z', 'a,a1,a2,a3', or plain 'x,y,z'")

if __name__ == '__main__':
    print(f"Servo IDs: {servo_ids}")
    
    try:
        
        # Servo initialization block intentionally disabled for manual testing.
        # Uncomment to re-enable startup torque-on sequence.
        # if board is not None:
        #     print(f"Initializing servos sequentially (delay={SERVO_STARTUP_ENABLE_DELAY_S:.2f}s)...")
        #     for servo_id in servo_ids:
        #         try:
        #             board.bus_servo_enable_torque(servo_id, 0)
        #             print(f"Servo {servo_id} torque ON")
        #             time.sleep(SERVO_STARTUP_ENABLE_DELAY_S)
        #         except Exception as e:
        #             print(f"Warning: Could not enable servo {servo_id}: {e}")
        # else:
        #     print("Board not initialized, skipping servo initialization")
        
        time.sleep(0.5)
        
        print("Starting main loop. Press Ctrl+C to stop.")
        print("Input formats:")
        print("  XYZ IK:     x,100,0,120   or   100,0,120")
        print("  Camera XYZ: c,100,0,-550  (ArUco feedback-corrected XYZ)")
        print("  Angles:     a,10,0,-5     (maps directly to servos 1,2,3)")
        print("  Torque:     t,on          or   t,off,1")
        print("  Square IK:  sq,0,0,-550,40,0.6")
        print("  Up/Down IK: ud,5,0.6      (x=0,y=0,z=-600<->-400)")
        print("  N Path IK:  n,200,200,0.6 (anchors: (x,0),(-x,0),(0,y),(0,-y))")
        # Main loop - capture, analyze, and move servos
        while running:
            try:
                
                print("enter command: ", end="")
                mode, values = parse_control_input(input())

                if mode == "xyz":
                    x, y, z = values
                    angles = inverseKinematics.getAngles(x, y, z)
                    print(f"IK angles: {angles}")
                    if angles is None:
                        print(f"no valid solution for coordinates x={x}, y={y}, z={z}")
                    else:
                        servo_positions = ik_angles_to_servo_positions(angles)
                        move_servos(servo_positions)
                elif mode == "camera_xyz":
                    run_camera_xyz_feedback(values)
                else:
                    if mode == "angles":
                        a1, a2, a3 = values
                        servo_positions = {1: a1, 2: a2, 3: a3}
                        print(f"Direct angles -> servo_positions: {servo_positions}")
                        move_servos(servo_positions)
                    elif mode == "torque":
                        state, servo_id = values
                        set_torque(state, servo_id)
                    elif mode == "square":
                        cx, cy, z, side_len, dwell_s = values
                        run_square_path(cx, cy, z, side_len, dwell_s)
                    elif mode == "updown":
                        cycles, dwell_s = values
                        run_updown_test(cycles, dwell_s)
                    elif mode == "npath":
                        x_mm, y_mm, dwell_s = values
                        run_n_path(x_mm, y_mm, dwell_s)
                
                
                # Tempporary sleep to slow main loop
                time.sleep(1)

            
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(0.5)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        running = False
    
    finally:
        print("\nShutting down...")
        
        # Stop all servos
        if board is not None:
            try:
                for servo_id in servo_ids:
                    board.bus_servo_enable_torque(servo_id, 1)  # Torque OFF
                    print(f"Servo {servo_id} torque OFF")
            except Exception as e:
                print(f"Error disabling servos: {e}")

        
        print("Shutdown complete")
