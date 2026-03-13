import os
import sys
import time
import threading
import tkinter as tk

# Local imports
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Motor Control", "board_demo"))
sys.path.insert(0, os.path.join(ROOT, "Inverse Kinematics"))

import ros_robot_controller_sdk as rrc
import inverseKinematics


# Servo calibration (matches current testCode.py)
SERVO_IDS = [3, 4, 5]
SERVO_ZERO_OFFSETS_DEG = {3: 87.0, 4: 90.0, 5: 90.0}
SERVO_DIRECTIONS = {3: -1.0, 4: -1.0, 5: -1.0}
SERVO_ANGLE_SCALES = {3: 1.0, 4: 1.0, 5: 1.0}
MOVE_DURATION_S = 0.20

# Workspace limits (mm)
X_MIN, X_MAX = -180.0, 180.0
Y_MIN, Y_MAX = -180.0, 180.0
Z_MIN, Z_MAX = -650.0, -450.0

# UI
W, H = 900, 700
FPS_MS = 50
SEND_DEADBAND_MM = 1.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class CursorController:
    def __init__(self):
        self.running = True
        self.board = None
        self.motors_enabled = False
        self.lock = threading.Lock()

        # Target state
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = -550.0
        self.last_sent = (None, None, None)
        self.last_angles = None
        self.last_error = ""

        self._init_board()
        self._init_ui()

    def _init_board(self):
        try:
            self.board = rrc.Board()
            self.board.enable_reception()
            self.last_error = "Board connected. Press 't' to torque ON."
        except Exception as e:
            self.board = None
            self.last_error = f"Board unavailable (preview mode): {e}"

    def _init_ui(self):
        self.root = tk.Tk()
        self.root.title("Delta Arm Cursor Control")
        self.root.geometry(f"{W}x{H}")
        self.root.configure(bg="#121418")

        self.canvas = tk.Canvas(self.root, width=W, height=H - 120, bg="#0f1318", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.info = tk.Label(
            self.root,
            text="",
            justify=tk.LEFT,
            anchor="w",
            bg="#121418",
            fg="#d7e0ea",
            font=("Menlo", 11),
        )
        self.info.pack(fill=tk.X, padx=10, pady=8)

        self.root.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)    # mac/windows
        self.root.bind("<Button-4>", self.on_mouse_wheel_up)   # linux
        self.root.bind("<Button-5>", self.on_mouse_wheel_down) # linux
        self.root.bind("<KeyPress-t>", self.on_toggle_torque)
        self.root.bind("<KeyPress-r>", self.on_reset_center)
        self.root.bind("<KeyPress-q>", self.on_quit)
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        self.root.after(FPS_MS, self.update_loop)

    def canvas_to_world(self, px, py):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        nx = (px / float(cw))
        ny = (py / float(ch))
        x = X_MIN + nx * (X_MAX - X_MIN)
        y_screen = Y_MIN + ny * (Y_MAX - Y_MIN)
        y = -y_screen  # make screen-up be +Y
        return clamp(x, X_MIN, X_MAX), clamp(y, Y_MIN, Y_MAX)

    def world_to_canvas(self, x, y):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        nx = (x - X_MIN) / (X_MAX - X_MIN)
        ny = ((-y) - Y_MIN) / (Y_MAX - Y_MIN)
        return int(nx * cw), int(ny * ch)

    def on_mouse_move(self, event):
        with self.lock:
            self.target_x, self.target_y = self.canvas_to_world(event.x, event.y)

    def on_mouse_wheel(self, event):
        dz = -10.0 if event.delta > 0 else 10.0
        with self.lock:
            self.target_z = clamp(self.target_z + dz, Z_MIN, Z_MAX)

    def on_mouse_wheel_up(self, _event):
        with self.lock:
            self.target_z = clamp(self.target_z - 10.0, Z_MIN, Z_MAX)

    def on_mouse_wheel_down(self, _event):
        with self.lock:
            self.target_z = clamp(self.target_z + 10.0, Z_MIN, Z_MAX)

    def on_toggle_torque(self, _event):
        if self.board is None:
            self.last_error = "Board not connected (preview mode)."
            return
        try:
            if not self.motors_enabled:
                for sid in SERVO_IDS:
                    self.board.bus_servo_enable_torque(sid, 0)
                    time.sleep(0.2)
                self.motors_enabled = True
                self.last_error = "Torque ON"
            else:
                for sid in SERVO_IDS:
                    self.board.bus_servo_enable_torque(sid, 1)
                self.motors_enabled = False
                self.last_error = "Torque OFF"
        except Exception as e:
            self.last_error = f"Torque toggle error: {e}"

    def on_reset_center(self, _event):
        with self.lock:
            self.target_x = 0.0
            self.target_y = 0.0
            self.target_z = -550.0

    def on_quit(self, _event=None):
        self.running = False
        try:
            if self.board is not None and self.motors_enabled:
                for sid in SERVO_IDS:
                    self.board.bus_servo_enable_torque(sid, 1)
        except Exception:
            pass
        self.root.destroy()

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

    def maybe_send_command(self, x, y, z):
        lx, ly, lz = self.last_sent
        if lx is not None:
            if abs(x - lx) < SEND_DEADBAND_MM and abs(y - ly) < SEND_DEADBAND_MM and abs(z - lz) < SEND_DEADBAND_MM:
                return

        angles = inverseKinematics.getAngles(x, y, z)
        self.last_angles = angles
        if angles is None:
            self.last_error = "IK invalid for current target"
            return

        cmds = self.angles_to_servo_raw(angles)
        self.last_sent = (x, y, z)

        if self.board is not None and self.motors_enabled:
            try:
                self.board.bus_servo_set_position(MOVE_DURATION_S * 3, cmds)
                self.last_error = ""
            except Exception as e:
                self.last_error = f"Send error: {e}"

    def draw_overlay(self, x, y):
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        # grid
        for i in range(1, 6):
            gx = int(i * cw / 6)
            gy = int(i * ch / 6)
            self.canvas.create_line(gx, 0, gx, ch, fill="#1f2b36")
            self.canvas.create_line(0, gy, cw, gy, fill="#1f2b36")

        # axes
        cx, cy = self.world_to_canvas(0.0, 0.0)
        self.canvas.create_line(cx, 0, cx, ch, fill="#2e8b57", width=2)
        self.canvas.create_line(0, cy, cw, cy, fill="#2e8b57", width=2)

        # target
        tx, ty = self.world_to_canvas(x, y)
        self.canvas.create_oval(tx - 7, ty - 7, tx + 7, ty + 7, fill="#ffcc00", outline="")

    def update_loop(self):
        if not self.running:
            return
        with self.lock:
            x, y, z = self.target_x, self.target_y, self.target_z

        self.maybe_send_command(x, y, z)
        self.draw_overlay(x, y)

        mode = "LIVE" if (self.board is not None and self.motors_enabled) else "PREVIEW"
        angles_str = str(tuple(round(a, 2) for a in self.last_angles)) if self.last_angles is not None else "None"
        self.info.config(
            text=(
                f"Mode: {mode}    Keys: t=torque toggle, r=reset center, q=quit, mouse wheel=z\n"
                f"Target (mm): x={x:.1f}, y={y:.1f}, z={z:.1f}\n"
                f"IK angles: {angles_str}\n"
                f"{self.last_error}"
            )
        )
        self.root.after(FPS_MS, self.update_loop)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = CursorController()
    app.run()
