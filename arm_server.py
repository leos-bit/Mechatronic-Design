import argparse
import json
import os
import socket
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Motor Control", "board_demo"))
sys.path.insert(0, os.path.join(ROOT, "Inverse Kinematics"))

import ros_robot_controller_sdk as rrc
import inverseKinematics


SERVO_IDS = [3, 4, 5]
SERVO_ZERO_OFFSETS_DEG = {3: 87.0, 4: 90.0, 5: 90.0}
SERVO_DIRECTIONS = {3: -1.0, 4: -1.0, 5: -1.0}
SERVO_ANGLE_SCALES = {3: 1.0, 4: 1.0, 5: 1.0}
MOVE_DURATION_S = 0.30
SEND_HZ = 20.0
TARGET_DEADBAND_MM = 0.5

# Workspace limits (mm)
X_MIN, X_MAX = -180.0, 180.0
Y_MIN, Y_MAX = -180.0, 180.0
Z_MIN, Z_MAX = -650.0, -400.0

# XY calibration transform (commanded -> robot mm)
# [x_robot]   [A11 A12] [x_cmd] + [BX]
# [y_robot] = [A21 A22] [y_cmd] + [BY]
XY_CAL_ENABLED = True
XY_A11, XY_A12 = 1.0, 0.0
XY_A21, XY_A22 = 0.0, 1.0
XY_BX, XY_BY = 0.0, 0.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def apply_xy_calibration(x_cmd, y_cmd):
    if not XY_CAL_ENABLED:
        return x_cmd, y_cmd
    x_robot = (XY_A11 * x_cmd) + (XY_A12 * y_cmd) + XY_BX
    y_robot = (XY_A21 * x_cmd) + (XY_A22 * y_cmd) + XY_BY
    return x_robot, y_robot


class ArmServer:
    def __init__(self):
        self.running = True
        self.board = None
        self.lock = threading.Lock()
        self.motors_enabled = False
        self.target = {"x": 0.0, "y": 0.0, "z": -550.0}
        self.target_robot = {"x": 0.0, "y": 0.0, "z": -550.0}
        self.last_sent = {"x": None, "y": None, "z": None}

    def init_board(self):
        self.board = rrc.Board()
        self.board.enable_reception()
        print("[SERVER] Board initialized")

    def set_torque(self, on):
        if self.board is None:
            return
        for sid in SERVO_IDS:
            self.board.bus_servo_enable_torque(sid, 0 if on else 1)
            time.sleep(0.15)
        self.motors_enabled = on
        print(f"[SERVER] Torque {'ON' if on else 'OFF'}")

    def target_changed_enough(self):
        x, y, z = self.target["x"], self.target["y"], self.target["z"]
        lx, ly, lz = self.last_sent["x"], self.last_sent["y"], self.last_sent["z"]
        if lx is None:
            return True
        return (
            abs(x - lx) >= TARGET_DEADBAND_MM
            or abs(y - ly) >= TARGET_DEADBAND_MM
            or abs(z - lz) >= TARGET_DEADBAND_MM
        )

    def angles_to_servo_raw(self, angles):
        cmds = []
        for sid, ang in zip(SERVO_IDS, angles):
            logical = float(ang)
            direction = SERVO_DIRECTIONS.get(sid, 1.0)
            scale = SERVO_ANGLE_SCALES.get(sid, 1.0)
            physical = SERVO_ZERO_OFFSETS_DEG.get(sid, 0.0) + (direction * scale * logical)
            physical = clamp(physical, 0.0, 240.0)
            raw = int((physical / 240.0) * 1000)
            cmds.append([sid, raw])
        return cmds

    def motion_loop(self):
        period = 1.0 / SEND_HZ
        while self.running:
            t0 = time.time()
            try:
                with self.lock:
                    x = float(clamp(self.target["x"], X_MIN, X_MAX))
                    y = float(clamp(self.target["y"], Y_MIN, Y_MAX))
                    z = float(clamp(self.target["z"], Z_MIN, Z_MAX))
                    self.target["x"], self.target["y"], self.target["z"] = x, y, z
                    xr, yr = apply_xy_calibration(x, y)
                    xr = float(clamp(xr, X_MIN, X_MAX))
                    yr = float(clamp(yr, Y_MIN, Y_MAX))
                    self.target_robot["x"], self.target_robot["y"], self.target_robot["z"] = xr, yr, z

                    if self.board is not None and self.motors_enabled and self.target_changed_enough():
                        angles = inverseKinematics.getAngles(xr, yr, z)
                        if angles is None:
                            print(
                                f"[SERVER] IK invalid cmd=({x:.1f},{y:.1f},{z:.1f}) "
                                f"robot=({xr:.1f},{yr:.1f},{z:.1f})"
                            )
                        else:
                            cmds = self.angles_to_servo_raw(angles)
                            self.board.bus_servo_set_position(MOVE_DURATION_S * 3, cmds)
                            self.last_sent["x"], self.last_sent["y"], self.last_sent["z"] = x, y, z
            except Exception as e:
                print(f"[SERVER] motion loop error: {e}")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def handle_message(self, msg):
        cmd = msg.get("cmd")
        if cmd == "set_target":
            with self.lock:
                self.target["x"] = float(msg.get("x", self.target["x"]))
                self.target["y"] = float(msg.get("y", self.target["y"]))
                self.target["z"] = float(msg.get("z", self.target["z"]))
            return {"ok": True}
        if cmd == "set_torque":
            on = bool(msg.get("on", False))
            self.set_torque(on)
            return {"ok": True, "motors_enabled": self.motors_enabled}
        if cmd == "get_state":
            with self.lock:
                return {
                    "ok": True,
                    "motors_enabled": self.motors_enabled,
                    "target_cmd": dict(self.target),
                    "target_robot": dict(self.target_robot),
                }
        if cmd == "home":
            with self.lock:
                self.target = {"x": 0.0, "y": 0.0, "z": -550.0}
            return {"ok": True, "target": dict(self.target)}
        if cmd == "shutdown":
            self.running = False
            return {"ok": True}
        return {"ok": False, "error": "unknown cmd"}

    def serve(self, host, port):
        motion_thread = threading.Thread(target=self.motion_loop, daemon=True)
        motion_thread.start()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(1)
            print(f"[SERVER] Listening on {host}:{port}")
            while self.running:
                conn, addr = s.accept()
                print(f"[SERVER] Client connected: {addr}")
                with conn:
                    file = conn.makefile("rwb")
                    while self.running:
                        line = file.readline()
                        if not line:
                            break
                        try:
                            msg = json.loads(line.decode("utf-8"))
                            resp = self.handle_message(msg)
                        except Exception as e:
                            resp = {"ok": False, "error": str(e)}
                        file.write((json.dumps(resp) + "\n").encode("utf-8"))
                        file.flush()
                print("[SERVER] Client disconnected")

        if self.board is not None:
            try:
                self.set_torque(False)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Delta arm headless server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ArmServer()
    server.init_board()
    try:
        server.serve(args.host, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        server.running = False
        if server.board is not None:
            try:
                server.set_torque(False)
            except Exception:
                pass
        print("[SERVER] Shutdown")


if __name__ == "__main__":
    main()
