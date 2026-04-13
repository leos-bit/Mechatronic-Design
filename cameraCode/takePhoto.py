import pyvizionsdk
from pyvizionsdk import VX_UVC_IMAGE_PROPERTIES, VX_IMAGE_FORMAT, VX_CAPTURE_RESULT
import time
import re
import subprocess
import sys

ARM_CAMERA_PATH = "usb-xhci-hcd.1-2"
MOUNT_CAMERA_PATH = "usb-xhci-hcd.1-1"
STREAM_START_RETRIES = 3
FRAME_CAPTURE_RETRIES = 3
RETRY_DELAY_SECONDS = 0.3


def discoverCameraDevices():
    result, camera_list = pyvizionsdk.VxDiscoverCameraDevices()
    if result == 0:
        print("No device detected.")
        sys.exit()
    return camera_list


def resolveVideoNodeFromCameraHint(camera_hint):
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print("v4l2-ctl not found; cannot resolve camera hint to a video node")
        sys.exit()
    except subprocess.CalledProcessError as exc:
        print(f"v4l2-ctl --list-devices failed: {exc}")
        sys.exit()

    current_header = None
    for raw_line in result.stdout.splitlines():
        line = raw_line.rstrip()
        if not line:
            current_header = None
            continue
        if not raw_line.startswith("\t"):
            current_header = line
            continue
        if current_header and camera_hint in current_header:
            device = line.strip()
            if device.startswith("/dev/video"):
                return device

    print(f"Could not map camera hint '{camera_hint}' to a /dev/video node")
    sys.exit()


def extractVideoNode(device_description):
    match = re.search(r"(\(/dev/video\d+\))", device_description)
    if not match:
        return None
    return match.group(0)[1:-1]


def selectCameraIndex(camera_list, camera_hint=None):
    if camera_hint:
        resolved_video_node = resolveVideoNodeFromCameraHint(camera_hint)
        for i, cam in enumerate(camera_list):
            camera_video_node = extractVideoNode(str(cam))
            if camera_video_node == resolved_video_node:
                return i
        print(
            f"Camera with hint '{camera_hint}' resolved to '{resolved_video_node}', "
            "but no matching SDK device was found"
        )
        for i, cam in enumerate(camera_list):
            print(f"[{i}] {cam}")
        sys.exit()

    # Find the AR1335 USB camera specifically
    for i, cam in enumerate(camera_list):
        if "AR1335" in str(cam) or "video0" in str(cam):
            return i

    print("AR1335 camera not found in device list")
    for i, cam in enumerate(camera_list):
        print(f"[{i}] {cam}")
    sys.exit()


def initialzeCamera(camera_hint=None):
    camera_list = discoverCameraDevices()
    target_camera = selectCameraIndex(camera_list, camera_hint=camera_hint)
    camera = pyvizionsdk.VxInitialCameraDevice(target_camera)
    result = pyvizionsdk.VxOpen(camera)
    if result != 0:
        raise RuntimeError(f"Failed to open camera for hint '{camera_hint}' (result={result})")

    result, name = pyvizionsdk.VxGetDeviceName(camera)
    if result != 0:
        closeCamera(camera)
        raise RuntimeError(f"Failed to read camera name for hint '{camera_hint}' (result={result})")

    pyvizionsdk.VxGetDeviceInterfaceType(camera)
    pyvizionsdk.VxGetUSBDeviceSpeed(camera)

    result, format_list = pyvizionsdk.VxGetFormatList(camera)
    if result != 0:
        closeCamera(camera)
        raise RuntimeError(f"Failed to get format list for camera '{name}' (result={result})")

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
    if target_format is None:
        closeCamera(camera)
        raise RuntimeError(f"No supported UYVY or MJPG format found for camera '{name}'")
    result = pyvizionsdk.VxSetFormat(camera, target_format)
    if result != 0:
        closeCamera(camera)
        raise RuntimeError(f"Failed to set format for camera '{name}' (result={result})")

    last_result = None
    for _ in range(STREAM_START_RETRIES):
        result = pyvizionsdk.VxStartStreaming(camera)
        if result == 0:
            return camera, target_format
        last_result = result
        time.sleep(RETRY_DELAY_SECONDS)

    closeCamera(camera)
    raise RuntimeError(f"Failed to start streaming for camera '{name}' (result={last_result})")

def takePhoto(camera, target_format, save_photo = False, destination = "/home/collector/Documents/deltaArmControl/Mechatronic-Design/cameraCode/photos/", name = "default.jpg"):
    import numpy as np
    import cv2

    for _ in range(FRAME_CAPTURE_RETRIES):
        result, image_data = pyvizionsdk.VxGetImage(camera, 1000, target_format)
        if result == pyvizionsdk.VX_CAPTURE_RESULT.VX_SUCCESS and image_data.size > 0:
            w = target_format.width
            h = target_format.height

            uyvy = np.frombuffer(image_data, dtype=np.uint8).reshape((h, w, 2))
            bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)

            if save_photo:
                cv2.imwrite(destination + name, bgr)
            return bgr
        time.sleep(RETRY_DELAY_SECONDS)

    return None

def closeCamera(camera):
    try:
        pyvizionsdk.VxStopStreaming(camera)
    except Exception:
        pass
    try:
        pyvizionsdk.VxClose(camera)
    except Exception:
        pass

def main():
    camera, target_format = initialzeCamera()
    takePhoto(camera, target_format, save_photo=True)
    closeCamera(camera)

if __name__ == "__main__":
    main()
