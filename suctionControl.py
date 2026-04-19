import serial
import time

def initializeSerial():
    ser = serial.Serial(
        port='/dev/ttyACM1',
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
    )
    time.sleep(2)
    return ser

def turnOnSuction(ser):
    ser.write(b"on\n")
    ser.flush()

def turnOffSuction(ser):
    ser.write(b"off\n")
    ser.flush()
