import sys
import cv2
import time

# echo 'options uvcvideo quirks=128' | sudo tee /etc/modprobe.d/uvcvideo.conf
# sudo reboot

# 93cm

def initialzeCamera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    time.sleep(2)  # warm-up time
    return cap

def takePhoto(cam, save_photo=False, destination="/home/collector/Documents/deltaArmControl/Mechatronic-Design/cameraCode/photos/", name="default.jpg"):
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return None
    
    ret, frame = cam.read()

    if ret:
        if save_photo:
            cv2.imwrite(destination + name, frame)
        return frame
    else:
        print("Error: Could not read frame.")
        return None

def closeCamera(cam):
    cam.release()

def main():
    camera = initialzeCamera()
    takePhoto(camera, save_photo=True)
    closeCamera(camera)

if __name__ == "__main__":
    main()