# Mechatronic-Design

Delta-arm + computer vision workspace for conveyor object detection and robot motion control.

## Repository Layout
- `testCode.py`: interactive manual control (`a`, `x`, `sq`, `ud`, `t`)
- `motorControl.py`: CV pipeline + target selection + servo motion
- `arm_server.py`: headless Pi arm control server (network target input)
- `cursor_control_web_client.py`: browser-based cursor controller for `arm_server.py`
- `cursor_control.py` / `cursor_control_client.py`: local Tk-based cursor tools
- `Computer Vision/`: detection, labeling, tracking, homography utilities
- `Inverse Kinematics/`: IK/FK math used by arm control

## 1) Manual Arm Test (Pi)
Run on Raspberry Pi from repo root:

```bash
python3 testCode.py
```

### Prompt Commands
- `t,on` / `t,off` / `t,on,3`: torque control
- `a,a3,a4,a5`: direct joint-angle input
- `x,x,y,z`: IK move
- `sq,cx,cy,z,side_len[,dwell_s]`: square path via IK (returns to P1)
- `ud[,cycles[,dwell_s]]`: up/down test at `x=0,y=0`, between `z=-600` and `z=-400`

## 2) CV Runtime (Pi or Mac)
From repo root:

```bash
python3 motorControl.py
```

Current default behavior:
- runs detector from `Computer Vision/trials/.../best.pt`
- applies homography to world coordinates (`mm`)
- selects one target and maps to arm commands

Important control settings are at top of `motorControl.py`:
- `CV_CONTROL_MODE` (`world_ik` or `centroid`)
- `CV_TARGET_Z_MM`
- `CV_WORLD_X_BIAS_MM`, `CV_WORLD_Y_BIAS_MM`

## 3) Cursor-to-Arm Control (Recommended Split)

### On Pi (server)
```bash
python3 arm_server.py --host 0.0.0.0 --port 8765
```

### On Mac (web bridge)
```bash
python3 cursor_control_web_client.py --pi-host <PI_IP> --pi-port 8765 --listen-host 127.0.0.1 --listen-port 8080
```

Open:
- `http://127.0.0.1:8080`

Controls:
- move cursor in canvas = `x,y`
- trackpad/mouse wheel = `z`
- `Torque ON/OFF` button
- `Home` button

## Calibration Notes
Servo mapping constants are currently tuned in `testCode.py` and `motorControl.py`:
- `SERVO_ZERO_OFFSETS_DEG`
- `SERVO_DIRECTIONS`
- `SERVO_ANGLE_SCALES`

Keep these values consistent between scripts when switching control modes.

## Existing CV README
Detailed CV detection/labeling commands remain in:
- `Computer Vision/README.md`

## 4) Roboflow Workflow Runtime (Pi)
Set the Roboflow environment variables on the Raspberry Pi:

```bash
export ROBOFLOW_API_URL="https://serverless.roboflow.com"
export ROBOFLOW_API_KEY="oPusoqJbAhSfo6zbicdc"
export ROBOFLOW_WORKSPACE="leos-workspace-qswhy"
export ROBOFLOW_WORKFLOW_ID="yolov11"
export ROBOFLOW_INPUT_NAME="image"
export ROBOFLOW_USE_CACHE="false"
```

### Live camera view with workflow detections
Use this to verify the camera feed, classes, polygons, and centroids:

```bash
python3 liveRoboflowView.py
```

### Robotic sorting with Roboflow workflow backend
`motorControl.py` already supports the Roboflow workflow backend. Run:

```bash
python3 motorControl.py
```

Notes:
- `liveRoboflowView.py` is the safest first test because it only visualizes detections.
- `motorControl.py` uses the same environment variables and can map detections into world coordinates for robot motion.
- If your workflow returns SAM 3 polygons, the code computes centroids from the mask geometry instead of the bounding box center.
