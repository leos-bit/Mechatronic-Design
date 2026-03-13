import argparse
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Delta Arm Web Cursor Client</title>
  <style>
    body { margin: 0; font-family: ui-monospace, Menlo, monospace; background: #11161d; color: #dbe6f3; }
    .wrap { display: grid; grid-template-rows: auto 1fr auto; height: 100vh; }
    .top { padding: 10px 14px; background: #0d1218; border-bottom: 1px solid #223040; display: flex; gap: 10px; align-items: center; }
    button { background: #1f3145; border: 1px solid #37516d; color: #dbe6f3; padding: 6px 10px; cursor: pointer; }
    .ok { color: #79d996; } .bad { color: #ff8a8a; }
    #c { width: 100%; height: 100%; display: block; }
    .foot { padding: 10px 14px; border-top: 1px solid #223040; background: #0d1218; white-space: pre-wrap; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <button id="torqueBtn">Torque OFF</button>
      <button id="homeBtn">Home</button>
      <span id="conn" class="bad">Disconnected</span>
      <span>Use mouse for x/y, wheel for z</span>
    </div>
    <canvas id="c"></canvas>
    <div class="foot" id="info">starting...</div>
  </div>
<script>
const X_MIN=-180, X_MAX=180, Y_MIN=-180, Y_MAX=180, Z_MIN=-650, Z_MAX=-400;
let target = {x:0, y:0, z:-550};
let lastSent = {x:null,y:null,z:null};
let motorsEnabled = false;
let connected = false;
let lastError = "";

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const info = document.getElementById("info");
const conn = document.getElementById("conn");
const torqueBtn = document.getElementById("torqueBtn");
const homeBtn = document.getElementById("homeBtn");

function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function worldToCanvas(x,y){
  const w = canvas.width, h = canvas.height;
  const nx = (x - X_MIN)/(X_MAX-X_MIN);
  const ny = ((-y)-Y_MIN)/(Y_MAX-Y_MIN);
  return [nx*w, ny*h];
}
function canvasToWorld(px,py){
  const w = canvas.width, h = canvas.height;
  const nx = px/w, ny = py/h;
  const x = X_MIN + nx*(X_MAX-X_MIN);
  const yScreen = Y_MIN + ny*(Y_MAX-Y_MIN);
  const y = -yScreen;
  return [clamp(x,X_MIN,X_MAX), clamp(y,Y_MIN,Y_MAX)];
}
function resize(){
  canvas.width = canvas.clientWidth * window.devicePixelRatio;
  canvas.height = canvas.clientHeight * window.devicePixelRatio;
  ctx.setTransform(window.devicePixelRatio,0,0,window.devicePixelRatio,0,0);
}
window.addEventListener("resize", resize);
resize();

canvas.addEventListener("mousemove", (e)=>{
  const r = canvas.getBoundingClientRect();
  const [x,y] = canvasToWorld((e.clientX-r.left)*window.devicePixelRatio, (e.clientY-r.top)*window.devicePixelRatio);
  target.x=x; target.y=y;
});
canvas.addEventListener("wheel", (e)=>{
  e.preventDefault();
  target.z = clamp(target.z + (e.deltaY < 0 ? -10 : 10), Z_MIN, Z_MAX);
}, {passive:false});

document.addEventListener("keydown", (e)=>{
  if(e.key==="t"){ toggleTorque(); }
  if(e.key==="r"){ home(); }
});
torqueBtn.onclick = toggleTorque;
homeBtn.onclick = home;

async function api(path, body){
  const resp = await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});
  return await resp.json();
}
async function toggleTorque(){
  try{
    const d = await api("/torque",{on:!motorsEnabled});
    if(d.ok){ motorsEnabled = !!d.motors_enabled; lastError=""; }
    else{ lastError = d.error || "torque failed"; }
  }catch(e){ lastError = String(e); }
}
async function home(){
  target = {x:0,y:0,z:-550};
  try{
    const d = await api("/home",{});
    if(!d.ok){ lastError = d.error || "home failed"; }
  }catch(e){ lastError = String(e); }
}
async function tickSend(){
  try{
    const d = await api("/set_target", target);
    connected = !!d.ok;
    if(!connected){ lastError = d.error || "set_target failed"; }
  }catch(e){
    connected = false;
    lastError = String(e);
  }
}
setInterval(tickSend, 50);

async function pollState(){
  try{
    const d = await api("/state",{});
    connected = !!d.ok;
    if(d.ok){
      motorsEnabled = !!d.motors_enabled;
      conn.textContent = "Connected";
      conn.className = "ok";
      lastError = "";
    }else{
      conn.textContent = "Disconnected";
      conn.className = "bad";
    }
  }catch(e){
    connected = false;
    conn.textContent = "Disconnected";
    conn.className = "bad";
  }
}
setInterval(pollState, 1000);

function draw(){
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.fillStyle = "#0f1318"; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = "#1f2b36"; ctx.lineWidth = 1;
  for(let i=1;i<=5;i++){
    const gx = i*w/6, gy = i*h/6;
    ctx.beginPath(); ctx.moveTo(gx,0); ctx.lineTo(gx,h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,gy); ctx.lineTo(w,gy); ctx.stroke();
  }
  const [cx,cy] = worldToCanvas(0,0);
  ctx.strokeStyle = "#2e8b57"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cx,0); ctx.lineTo(cx,h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,cy); ctx.lineTo(w,cy); ctx.stroke();
  const [tx,ty] = worldToCanvas(target.x,target.y);
  ctx.fillStyle = "#ffcc00";
  ctx.beginPath(); ctx.arc(tx,ty,7,0,Math.PI*2); ctx.fill();

  torqueBtn.textContent = motorsEnabled ? "Torque ON" : "Torque OFF";
  info.textContent =
    `Target(mm): x=${target.x.toFixed(1)}, y=${target.y.toFixed(1)}, z=${target.z.toFixed(1)}\\n` +
    `Torque: ${motorsEnabled ? "ON" : "OFF"}\\n` +
    `${lastError}`;
  requestAnimationFrame(draw);
}
draw();
</script>
</body>
</html>
"""


class PiBridge:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.file = None
        self.lock = threading.Lock()

    def _connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=2.0)
        self.file = self.sock.makefile("rwb")

    def call(self, payload):
        with self.lock:
            try:
                if self.file is None:
                    self._connect()
                self.file.write((json.dumps(payload) + "\n").encode("utf-8"))
                self.file.flush()
                line = self.file.readline()
                if not line:
                    raise RuntimeError("server closed connection")
                return json.loads(line.decode("utf-8"))
            except Exception as e:
                try:
                    if self.file:
                        self.file.close()
                    if self.sock:
                        self.sock.close()
                except Exception:
                    pass
                self.file = None
                self.sock = None
                return {"ok": False, "error": str(e)}


def make_handler(bridge):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj, ctype="application/json"):
            body = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {}

            if self.path == "/set_target":
                resp = bridge.call({
                    "cmd": "set_target",
                    "x": payload.get("x", 0.0),
                    "y": payload.get("y", 0.0),
                    "z": payload.get("z", -550.0),
                })
                self._send(200, resp)
                return
            if self.path == "/torque":
                resp = bridge.call({"cmd": "set_torque", "on": bool(payload.get("on", False))})
                self._send(200, resp)
                return
            if self.path == "/home":
                resp = bridge.call({"cmd": "home"})
                self._send(200, resp)
                return
            if self.path == "/state":
                resp = bridge.call({"cmd": "get_state"})
                self._send(200, resp)
                return
            self._send(404, {"ok": False, "error": "not found"})

        def log_message(self, fmt, *args):
            return

    return Handler


def main():
    p = argparse.ArgumentParser(description="Web cursor client for arm_server")
    p.add_argument("--pi-host", required=True, help="Pi IP/hostname running arm_server.py")
    p.add_argument("--pi-port", type=int, default=8765)
    p.add_argument("--listen-host", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, default=8080)
    args = p.parse_args()

    bridge = PiBridge(args.pi_host, args.pi_port)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), make_handler(bridge))
    print(f"[WEB] Open http://{args.listen_host}:{args.listen_port} in your browser")
    print(f"[WEB] Bridging to Pi {args.pi_host}:{args.pi_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
