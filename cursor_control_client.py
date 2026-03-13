import argparse
import json
import socket
import threading
import time
import tkinter as tk


X_MIN, X_MAX = -180.0, 180.0
Y_MIN, Y_MAX = -180.0, 180.0
Z_MIN, Z_MAX = -650.0, -400.0

W, H = 900, 700
SEND_MS = 50
SEND_DEADBAND_MM = 1.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ArmClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.file = None
        self.lock = threading.Lock()
        self.connected = False
        self.error = ""
        self.motors_enabled = False
        self.last_sent = (None, None, None)

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = -550.0

        self.root = tk.Tk()
        self.root.title("Delta Arm Cursor Client")
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
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)
        self.root.bind("<Button-4>", self.on_mouse_wheel_up)
        self.root.bind("<Button-5>", self.on_mouse_wheel_down)
        self.root.bind("<KeyPress-t>", self.on_toggle_torque)
        self.root.bind("<KeyPress-r>", self.on_home)
        self.root.bind("<KeyPress-q>", self.on_quit)
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        self.root.after(50, self.ensure_connection)
        self.root.after(SEND_MS, self.update_loop)

    def send_cmd(self, msg):
        if not self.connected or self.file is None:
            return None
        try:
            self.file.write((json.dumps(msg) + "\n").encode("utf-8"))
            self.file.flush()
            line = self.file.readline()
            if not line:
                self.connected = False
                return None
            return json.loads(line.decode("utf-8"))
        except Exception as e:
            self.connected = False
            self.error = f"Send error: {e}"
            return None

    def ensure_connection(self):
        if not self.connected:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=2.0)
                self.file = self.sock.makefile("rwb")
                self.connected = True
                self.error = ""
            except Exception as e:
                self.connected = False
                self.error = f"Disconnected: {e}"
        self.root.after(1000, self.ensure_connection)

    def canvas_to_world(self, px, py):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        nx = (px / float(cw))
        ny = (py / float(ch))
        x = X_MIN + nx * (X_MAX - X_MIN)
        y_screen = Y_MIN + ny * (Y_MAX - Y_MIN)
        y = -y_screen
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
        target_state = not self.motors_enabled
        resp = self.send_cmd({"cmd": "set_torque", "on": target_state})
        if resp and resp.get("ok"):
            self.motors_enabled = bool(resp.get("motors_enabled", target_state))
        elif resp and not resp.get("ok"):
            self.error = resp.get("error", "torque command failed")

    def on_home(self, _event):
        with self.lock:
            self.target_x, self.target_y, self.target_z = 0.0, 0.0, -550.0
        self.send_cmd({"cmd": "home"})

    def on_quit(self, _event=None):
        try:
            self.send_cmd({"cmd": "set_torque", "on": False})
        except Exception:
            pass
        try:
            if self.file:
                self.file.close()
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.root.destroy()

    def draw_overlay(self, x, y):
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        for i in range(1, 6):
            gx = int(i * cw / 6)
            gy = int(i * ch / 6)
            self.canvas.create_line(gx, 0, gx, ch, fill="#1f2b36")
            self.canvas.create_line(0, gy, cw, gy, fill="#1f2b36")

        cx, cy = self.world_to_canvas(0.0, 0.0)
        self.canvas.create_line(cx, 0, cx, ch, fill="#2e8b57", width=2)
        self.canvas.create_line(0, cy, cw, cy, fill="#2e8b57", width=2)

        tx, ty = self.world_to_canvas(x, y)
        self.canvas.create_oval(tx - 7, ty - 7, tx + 7, ty + 7, fill="#ffcc00", outline="")

    def maybe_send_target(self, x, y, z):
        lx, ly, lz = self.last_sent
        if lx is not None:
            if abs(x - lx) < SEND_DEADBAND_MM and abs(y - ly) < SEND_DEADBAND_MM and abs(z - lz) < SEND_DEADBAND_MM:
                return
        resp = self.send_cmd({"cmd": "set_target", "x": x, "y": y, "z": z})
        if resp and resp.get("ok"):
            self.last_sent = (x, y, z)
        elif resp and not resp.get("ok"):
            self.error = resp.get("error", "set_target failed")

    def update_loop(self):
        with self.lock:
            x, y, z = self.target_x, self.target_y, self.target_z

        self.draw_overlay(x, y)
        if self.connected:
            self.maybe_send_target(x, y, z)

        mode = "LIVE" if self.connected else "DISCONNECTED"
        self.info.config(
            text=(
                f"Server: {self.host}:{self.port}  Mode: {mode}\n"
                f"Keys: t=torque toggle, r=home, q=quit, mouse wheel=z\n"
                f"Target (mm): x={x:.1f}, y={y:.1f}, z={z:.1f}    Torque: {'ON' if self.motors_enabled else 'OFF'}\n"
                f"{self.error}"
            )
        )
        self.root.after(SEND_MS, self.update_loop)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Cursor client for delta arm server")
    parser.add_argument("--host", required=True, help="Pi IP or hostname")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    app = ArmClient(args.host, args.port)
    app.run()


if __name__ == "__main__":
    main()
