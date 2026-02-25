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
start = True

# 关闭前处理
def Stop(signum, frame):
    global start
    start = False
    print('关闭中...')

signal.signal(signal.SIGINT, Stop)

if __name__ == '__main__':
    while True:
       board.bus_servo_set_position(1, [[4, 0]])
       time.sleep(2)
       board.bus_servo_set_position(1, [[4, 180]])
       time.sleep(2)
       board.bus_servo_stop([4])
       time.sleep(1)
       if not start:
        board.bus_servo_set_position(1, [[4, 0]])
        time.sleep(2)
        board.bus_servo_stop([4])
        time.sleep(1)
        print('已关闭')
        break