import sys
import cv2
import time

# echo 'options uvcvideo quirks=128' | sudo tee /etc/modprobe.d/uvcvideo.conf
# sudo reboot

# 93cm
CAMERA_WARMUP_S = 2.0
CAMERA_FLUSH_FRAMES = 5

def initialzeCamera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(CAMERA_WARMUP_S)  # warm-up time
    return cap

def takePhoto(cam, save_photo=False, destination="/home/collector/Documents/deltaArmControl/Mechatronic-Design/cameraCode/photos/", name="default.jpg"):
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return None

    # Flush a few queued frames so each capture is as fresh as possible.
    for _ in range(CAMERA_FLUSH_FRAMES):
        ok = cam.grab()
        if not ok:
            break

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
