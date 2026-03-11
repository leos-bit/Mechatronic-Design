import signal
import time
import sys
import os

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
MOVE_DURATION_S = 0.8

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


def move_servos(servo_positions):
    if board is None:
        print("Board not initialized, skipping servo movement")
        return
    
    try:
        # Convert angles to servo positions and move servos
        servo_commands = []
        for servo_id in servo_ids:
            if servo_id in servo_positions:
                angle = servo_positions[servo_id]
                # Clamp angle to valid range (0-240)
                angle = max(0, min(240, angle))
                # Convert angle to position (0-1000)
                raw_pos = int((angle / 240.0) * 1000)
                servo_commands.append([servo_id, raw_pos])
        
        if servo_commands:
            print(f"Moving servos: {servo_commands}")
            board.bus_servo_set_position(MOVE_DURATION_S * 3, servo_commands)
            time.sleep(MOVE_DURATION_S * 3 + 0.05)
            
            # Read back positions for confirmation
            for servo_id in servo_ids:
                try:
                    pos = board.bus_servo_read_position(servo_id)
                    if pos:
                        print(f"Servo {servo_id} current position: {pos[0]}")
                except Exception as e:
                    print(f"Could not read position for servo {servo_id}: {e}")
    
    except Exception as e:
        print(f"Error moving servos: {e}")

if __name__ == '__main__':
    print(f"Servo IDs: {servo_ids}")
    
    try:
        
        # Initialize servos - set to torque ON (0)
        if board is not None:
            print("Initializing servos...")
            for servo_id in servo_ids:
                try:
                    board.bus_servo_enable_torque(servo_id, 0)
                    print(f"Servo {servo_id} torque ON")
                except Exception as e:
                    print(f"Warning: Could not enable servo {servo_id}: {e}")
        else:
            print("Board not initialized, skipping servo initialization")
        
        time.sleep(0.5)
        
        print("Starting main loop. Press Ctrl+C to stop.")
        
        # Main loop - capture, analyze, and move servos
        while running:
            try:
                
                print("enter coords as x, y, z: ", end="")
                coords = input().split(",")
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])

                angles = inverseKinematics.getAngles(x, y, z)
                if angles is None:
                    print(f"no valid solution for coordinates x={x}, y={y}, z={z}")
                else:
                    # Not sure which angle corresponds to which motor
                    servo_positions = {3: angles[0], 4: angles[1], 5: angles[2]}
                    move_servos(servo_positions)
                
                
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