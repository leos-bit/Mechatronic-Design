import pyvizionsdk
from pyvizionsdk import VX_UVC_IMAGE_PROPERTIES, VX_IMAGE_FORMAT, VX_CAPTURE_RESULT
import sys

def initialzeCamera():
    result, camera_list = pyvizionsdk.VxDiscoverCameraDevices()
    if result == 0:
        print("No device detected.")
        sys.exit()

    # Find the AR1335 USB camera specifically
    target_camera = None
    for i, cam in enumerate(camera_list):
        if "AR1335" in str(cam) or "video0" in str(cam):
            target_camera = i
            break

    if target_camera is None:
        print("AR1335 camera not found in device list")
        sys.exit()

    camera = pyvizionsdk.VxInitialCameraDevice(target_camera)
    # open camera
    result = pyvizionsdk.VxOpen(camera)
    # get the camera device name 
    result, name = pyvizionsdk.VxGetDeviceName(camera)
    # get interface type
    result, tyname = pyvizionsdk.VxGetDeviceInterfaceType(camera)
    result, speed = pyvizionsdk.VxGetUSBDeviceSpeed(camera)
    result, format_list = pyvizionsdk.VxGetFormatList(camera)
    mjpg_format = None
    min_resolution = float('inf')
    for format in format_list:
        # get mjpg format and minimum resolution
        if format.format == VX_IMAGE_FORMAT.VX_IMAGE_FORMAT_MJPG:
            resolution = format.width * format.height
            if resolution < min_resolution:
                min_resolution = resolution
                mjpg_format = format
    uyvy_format = None
    for fmt in format_list:
        if fmt.format == pyvizionsdk.VX_IMAGE_FORMAT.VX_IMAGE_FORMAT_UYVY:
            if uyvy_format is None:
                uyvy_format = fmt

    target_format = uyvy_format or mjpg_format
    result = pyvizionsdk.VxSetFormat(camera, target_format)
    result = pyvizionsdk.VxStartStreaming(camera)
    return camera, target_format

def takePhoto(camera, target_format, destination = "/home/collector/Documents/deltaArmControl/Mechatronic-Design/cameraCode/photos/", name = "default.jpg"):
    # Then get image
    result, image_data = pyvizionsdk.VxGetImage(camera, 1000, target_format)

    if result == pyvizionsdk.VX_CAPTURE_RESULT.VX_SUCCESS and image_data.size > 0:
        import numpy as np
        import cv2

        w = target_format.width
        h = target_format.height

        # Convert raw buffer to numpy array
        uyvy = np.frombuffer(image_data, dtype=np.uint8).reshape((h, w, 2))

        # Convert UYVY → BGR
        bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)

        cv2.imwrite(destination + name, bgr)
        # print("Photo saved (converted)!")

def closeCamera(camera):
    pyvizionsdk.VxStopStreaming(camera)
    pyvizionsdk.VxClose(camera)

def main():
    camera, target_format = initialzeCamera()
    takePhoto(camera, target_format)
    closeCamera(camera)

if __name__ == "__main__":
    main()