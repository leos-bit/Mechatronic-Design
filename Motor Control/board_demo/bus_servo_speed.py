import sys
import time
import signal
import threading
import ros_robot_controller_sdk as rrc

print('''
**********************************************************
********功能:幻尔科技树莓派扩展板，控制总线舵机转动**********
**********************************************************
----------------------------------------------------------
Official website:https://www.hiwonder.com
Online mall:https://hiwonder.tmall.com
----------------------------------------------------------
Tips:
 * 按下Ctrl+C可关闭此次程序运行，若失败请多次尝试！
----------------------------------------------------------
''')
board = rrc.Board()
board.enable_reception()
start = True

# 关闭前处理
def Stop(signum, frame):
    global start
    start = False

signal.signal(signal.SIGINT, Stop)

def get_bus_servo_id(board):
    servo_id = board.bus_servo_read_id()
    if servo_id == None:
        return
    servo_id = servo_id[0]
    print("id:", servo_id)
    return servo_id


if __name__ == '__main__':
    servo_ids = set()
    while True:
        # if len(servo_ids) < 3:
        #     servo_id = get_bus_servo_id(board)
        #     if servo_id is not None:
        #         servo_ids.add(get_bus_servo_id(board))
                
        for val in servo_ids:
            board.bus_servo_set_position(1, [[val, 0]])
            time.sleep(0.1)
            board.bus_servo_set_position(2, [[val, 1000]])
            time.sleep(1)
            board.bus_servo_stop([val])
            time.sleep(1)

        # board.bus_servo_set_offset(5, 127)
        time.sleep(2)
        
        if not start:
            time.sleep(1)
            print('已关闭')
            break