import signal
import time
import sys
import os
import threading

# Add path to access ros_robot_controller_sdk and takePhoto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/Motor Control/board_demo')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/cameraCode')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/Inverse Kinematics')

import ros_robot_controller_sdk as rrc
import takePhoto
import inverseKinematics

# Global control variables
running = True
servo_ids = [3, 4, 5]  # IDs of the three bus servos
MOVE_DURATION_S = 0.3
SERVO_ZERO_OFFSETS_DEG = {3: 87.0, 4: 90.0, 5: 90.0}
SERVO_DIRECTIONS = {3: -1.0, 4: -1.0, 5: -1.0}
SERVO_ANGLE_SCALES = {3: 1.0, 4: 1.0, 5: 1.0}
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
NPATH_Z_HIGH = -350.0
TORQUE_ON_HOLD_DURATION_S = 0.35
TORQUE_ON_SETTLE_S = 0.08
TORQUE_ON_AUTO_HOLD_ZERO = True

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

def read_servos(servo_ids):
    for servo_id in servo_ids:
        try:
            pos = board.bus_servo_read_position(servo_id)
            if pos:
                print(f"Servo {servo_id} current position: {pos[0]}")
        except Exception as e:
            print(f"Could not read position for servo {servo_id}: {e}")

def move_servos(servo_positions, update_activity=True):
    if board is None:
        print("Board not initialized, skipping servo movement")
        return
    
    try:
        with motion_lock:
            # Convert angles to servo positions and move servos
            servo_commands = []
            for servo_id in servo_ids:
                if servo_id in servo_positions:
                    logical_angle = float(servo_positions[servo_id])
                    direction = SERVO_DIRECTIONS.get(servo_id, 1.0)
                    scale = SERVO_ANGLE_SCALES.get(servo_id, 1.0)
                    physical_angle = SERVO_ZERO_OFFSETS_DEG.get(servo_id, 0.0) + (direction * scale * logical_angle)
                    physical_angle = max(0.0, min(240.0, physical_angle))
                    raw_pos = int((physical_angle / 240.0) * 1000)
                    servo_commands.append([servo_id, raw_pos])
            
            if servo_commands:
                if update_activity:
                    print(f"Moving servos simultaneously: {servo_commands}")
                board.bus_servo_set_position(MOVE_DURATION_S * 3, servo_commands)
                time.sleep(MOVE_DURATION_S * 3 + 0.05)
                # Read back positions for confirmation
                if update_activity:
                    read_servos(servo_ids)
                if update_activity:
                    pass
    
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
        move_servos({3: 0.0, 4: 0.0, 5: 0.0})


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
            servo_positions = {3: angles[0], 4: angles[1], 5: angles[2]}
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
                move_servos({3: angles[0], 4: angles[1], 5: angles[2]}, update_activity=False)
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
            move_servos({3: angles[0], 4: angles[1], 5: angles[2]}, update_activity=False)
            time.sleep(max(0.0, dwell_s))
    finally:
        sequence_running = False


def parse_control_input(text):
    """
    Supported formats:
      - "x, y, z" (default XYZ mode for backward compatibility)
      - "x, x, y, z" (explicit XYZ mode)
      - "a, a3, a4, a5" (direct motor angles for servos 3/4/5)
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

    if mode_token in ("a", "ang", "angle", "angles"):
        if len(parts) != 4:
            raise ValueError("angle mode requires: a, a3, a4, a5")
        a3, a4, a5 = float(parts[1]), float(parts[2]), float(parts[3])
        return ("angles", (a3, a4, a5))

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

    raise ValueError("invalid format; use 'x,x,y,z', 'a,a3,a4,a5', or plain 'x,y,z'")

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
        print("  Angles:     a,10,0,-5     (maps directly to servos 3,4,5)")
        print("  Torque:     t,on          or   t,off,3")
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
                        servo_positions = {3: angles[0], 4: angles[1], 5: angles[2]}
                        move_servos(servo_positions)
                else:
                    if mode == "angles":
                        a3, a4, a5 = values
                        servo_positions = {3: a3, 4: a4, 5: a5}
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
