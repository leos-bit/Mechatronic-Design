import signal
import time
import sys
import os

# Add path to access ros_robot_controller_sdk and takePhoto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/Motor Control/board_demo')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/cameraCode')

import ros_robot_controller_sdk as rrc
import takePhoto

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

# Camera initialization
camera = None
target_format = None


def stop_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nStopping...")
    running = False


signal.signal(signal.SIGINT, stop_handler)


def initialize_camera():
    try:
        print("Initializing camera...")
        camera, target_format = takePhoto.initialzeCamera()
        print("Camera initialized successfully")
        return camera, target_format
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return None, None


def capture_photo(camera, target_format):
    try:
        if camera is None or target_format is None:
            return None
        
        bgr = takePhoto.takePhoto(camera, target_format)
        
        if bgr is None:
            print(f"Photo not taken")
            return 
        else:
            return bgr
        
        
    except Exception as e:
        print(f"Error capturing photo: {e}")
        return None


def analyze_photo(image):
    try:
        if image is None:
            print("[ANALYSIS] No image to analyze")
            return None, {3: 120, 4: 120, 5: 120}
        
        import cv2
        
        # Make a copy to draw on
        image_with_centroid = image.copy()
        
        # Simple centroid-based analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] > 0:
                # Calculate centroid
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw centroid on image
                cv2.circle(image_with_centroid, (cx, cy), 5, (0, 255, 0), -1)  # Green filled circle
                cv2.circle(image_with_centroid, (cx, cy), 15, (0, 255, 0), 2)  # Green circle outline
                
                # Map centroid to servo angles
                # Servo 3: X position
                # Servo 4: Y position
                # Servo 5: Size/confidence
                
                h, w = image.shape[:2]
                angle_3 = (cx / w) * 240  # 0-240 degree range
                angle_4 = (cy / h) * 240
                angle_5 = 120  # Default center position
                
                analysis_results = {
                    3: angle_3,
                    4: angle_4,
                    5: angle_5
                }
                
                print(f"Detected object at ({cx}, {cy}) -> angles: {analysis_results}")
                return image_with_centroid, analysis_results
        
        # Default positions if nothing detected
        print("No objects detected, returning neutral positions")
        return image_with_centroid, {3: 120, 4: 120, 5: 120}
        
    except Exception as e:
        print(f"Error analyzing photo: {e}")
        return image, {3: 120, 4: 120, 5: 120}


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
        # Initialize camera
        camera, target_format = initialize_camera()
        
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
        frame_count = 0
        while running:
            try:
                # Capture photo
                frame_count += 1
                print(f"\n=== Frame {frame_count} ===")
                image = capture_photo(camera, target_format)
                
                if image is not None:
                    # Analyze photo (returns image with centroid drawn and servo positions)
                    image_with_centroid, servo_positions = analyze_photo(image)
                    
                    # Save image with centroid to photos directory
                    try:
                        import cv2
                        photo_path = os.path.dirname(os.path.abspath(__file__)) + '/Mechatronic-Design/cameraCode/photos/default.jpg'
                        cv2.imwrite(photo_path, image_with_centroid)
                        print(f"Saved photo to {photo_path}")
                    except Exception as e:
                        print(f"Error saving photo: {e}")
                    
                    # Move servos
                    
                    move_servos(servo_positions)
                    # Tempporary sleep to slow main loop
                    time.sleep(1)
                else:
                    print("No image captured, skipping analysis")
                    time.sleep(0.1)
            
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
        
        # Close camera
        if camera is not None:
            try:
                takePhoto.closeCamera(camera)
                print("Camera closed")
            except Exception as e:
                print(f"Error closing camera: {e}")
        
        print("Shutdown complete")