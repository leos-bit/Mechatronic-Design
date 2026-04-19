# Pi Access Quick Guide

Pi IP:

```bash
172.26.241.192
```

Username:

```bash
collector
```

## Connect To The Pi

From your Mac terminal:

```bash
ssh collector@172.26.241.192

pswd collector
```

## Go To The Project

On the Pi:

```bash
cd ~/Documents/deltaArmControl/Mechatronic-Design
```

If needed, activate the virtual environment:

```bash
source .venv312/bin/activate
```

## Run Python Code On The Pi

Example:

```bash
python iTet.py
```

Or:

```bash
python xxxxx.py
```

## Copy A File From Mac To Pi

Run this on your Mac:

```bash
scp "~/Documents" \
"collector@172.26.241.192:/home/collector/Documents/deltaArmControl/Mechatronic-Design/"
```

## Copy A File From Pi To Mac

Run this on your Mac:

```bash
scp "collector@172.26.241.192:/home/collector/Documents/deltaArmControl/Mechatronic-Design/iTet.py" \
"~/Documents"
```

## Copy An Image Or Video From Pi To Mac

Example image:

```bash
scp "collector@172.26.241.192:/home/collector/Documents/deltaArmControl/Mechatronic-Design/cameraCode/photos/default.jpg" \
"~/Documents"
```

Example video:

```bash
scp "collector@172.26.241.192:/home/collector/Documents/deltaArmControl/Mechatronic-Design/pi_camera_annotated.mp4" \
"~/Downloads/"
```

## Recalibrate Homography

Use this whenever the camera moves. The Pi captures the image, but the calibration clicking should be done on the Mac because the Pi usually does not have a display for the OpenCV GUI.

On the Pi, capture a fresh calibration image:

```bash
cd ~/Documents/deltaArmControl/Mechatronic-Design
source .venv312/bin/activate
python homography_calibration_helper.py --capture
```

The Pi may crash when it tries to open the GUI. That is okay if it already saved this image:

```bash
/home/collector/Documents/deltaArmControl/Mechatronic-Design/cameraCode/photos/homography_calibration_points.jpg
```

On the Mac, pull the calibration image:

```bash
scp "collector@172.26.241.192:/home/collector/Documents/deltaArmControl/Mechatronic-Design/cameraCode/photos/homography_calibration_points.jpg" \
"/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/cameraCode/photos/"
```

On the Mac, run the calibration helper:

```bash
python3 "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/homography_calibration_helper.py" \
  --image "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/cameraCode/photos/homography_calibration_points.jpg"
```

In the image window:

```text
click calibration points in order
press c when done
enter the matching robot/world x,y coordinates in mm
```

The helper prints new values like:

```python
CV_HOMOGRAPHY_SRC = "..."
CV_HOMOGRAPHY_DST = "..."
```

Copy those two lines into `iTet.py` or `iTet2.py`, replacing the old homography values.

Then push the updated script back to the Pi:

```bash
scp "/Users/leoshaw/Documents/VSCode/VS_CMU_S26/MechatronicDesign/iTet.py" \
"collector@172.26.241.192:/home/collector/Documents/deltaArmControl/Mechatronic-Design/"
```

## Stop A Running Script On The Pi

In the same terminal:

```bash
Ctrl+C
```

If that does not work, from another Pi terminal:

```bash
pkill -f iTet.py
```

Or replace with another script name:

```bash
pkill -f isaacCode.py
```

## Check Connected Devices On The Pi

USB devices:

```bash
lsusb
```

Cameras:

```bash
v4l2-ctl --list-devices
```

Serial devices:

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```
