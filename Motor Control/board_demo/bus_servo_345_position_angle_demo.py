import signal
import time

import ros_robot_controller_sdk as rrc

"""Live position monitor for bus servo 5 while moving it by hand."""

board = rrc.Board()
board.enable_reception()
running = True
SERVO_ID = 5
READ_INTERVAL_S = 0.05
TARGET_POSITION = 1193
MOVE_DURATION_S = 0.8


def stop_handler(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, stop_handler)


def read_pos_or_raise(servo_id):
    pos = board.bus_servo_read_position(servo_id)
    if pos is None:
        raise RuntimeError(f"failed to read position for servo {servo_id}")
    return int(pos[0])


if __name__ == "__main__":
    try:
        board.bus_servo_stop([SERVO_ID])
        time.sleep(0.05)
        # Hardware-specific semantics verified on this setup:
        # 0 -> torque ON (stiff), 1 -> torque OFF (limp)
        board.bus_servo_enable_torque(SERVO_ID, 0)
        board.bus_servo_set_position(MOVE_DURATION_S, [[SERVO_ID, TARGET_POSITION]])
        time.sleep(MOVE_DURATION_S + 0.1)
        print(f"servo {SERVO_ID} commanded target={TARGET_POSITION}. Press Ctrl+C to stop.")
        while running:
            current = read_pos_or_raise(SERVO_ID)
            print(f"servo {SERVO_ID} position={current}")
            time.sleep(READ_INTERVAL_S)
    finally:
        board.bus_servo_stop([SERVO_ID])
        time.sleep(0.05)
        board.bus_servo_enable_torque(SERVO_ID, 1)
