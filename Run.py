#!/usr/bin/env python3
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading hand_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, HAND_MODEL_PATH)
    print("Downloaded.")
"""
Hand Gesture 3D Tree Visualizer — Flask/SocketIO backend for Render deployment.
Browser captures webcam → sends base64 frames → server runs MediaPipe → returns rendered frame.
"""

import os, math, random, time, collections, base64, io
import numpy as np
import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# --- NEW MEDIAPIPE TASKS API IMPORTS ---
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode as VisionRunningMode
from mediapipe import Image as MPImage
from mediapipe import ImageFormat

HAND_MODEL_PATH = "hand_landmarker.task"  # Download this model and place in your project root

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "gesture-tree-secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent",
                    max_http_buffer_size=5 * 1024 * 1024)

# ── Layout ────────────────────────────────────────────────────────────────────
W      = 540
VIZ_H  = 480
CAM_H  = 300
BAR_H  = 30
TOTAL_H = VIZ_H + CAM_H + BAR_H

# HAND CONNECTIONS (from mediapipe docs)
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),      # Thumb
    (0,5),(5,6),(6,7),(7,8),      # Index
    (5,9),(9,10),(10,11),(11,12), # Middle
    (9,13),(13,14),(14,15),(15,16), # Ring
    (13,17),(17,18),(18,19),(19,20), # Pinky
    (0,17)
]

CX, CY   = 270, 240
SCALE    = 195
EYE_Z    = 3.5
RX_FIXED = -0.18

# ── Projection ────────────────────────────────────────────────────────────────
def proj_batch(pts_3d, ry):
    x, y, z = pts_3d[:,0], pts_3d[:,1], pts_3d[:,2]
    x2 =  x*math.cos(ry) + z*math.sin(ry)
    z2 = -x*math.sin(ry) + z*math.cos(ry)
    rx = RX_FIXED
    y3 =  y*math.cos(rx) - z2*math.sin(rx)
    z3 =  y*math.sin(rx) + z2*math.cos(rx)
    dz = np.maximum(EYE_Z - z3, 0.01)
    px = (CX + x2/dz*SCALE).astype(np.int32)
    py = (CY - y3/dz*SCALE).astype(np.int32)
    return np.stack([px, py], axis=1)

def proj_single(x, y, z, ry):
    x2 =  x*math.cos(ry) + z*math.sin(ry)
    z2 = -x*math.sin(ry) + z*math.cos(ry)
    rx = RX_FIXED
    y3 =  y*math.cos(rx) - z2*math.sin(rx)
    z3 =  y*math.sin(rx) + z2*math.cos(rx)
    dz = max(EYE_Z - z3, 0.01)
    return int(CX + x2/dz*SCALE), int(CY - y3/dz*SCALE)

# ── Cube ──────────────────────────────────────────────────────────────────────
CUBE_V = np.array([[-1,-1,-1],[+1,-1,-1],[+1,+1,-1],[-1,+1,-1],
                   [-1,-1,+1],[+1,-1,+1],[+1,+1,+1],[-1,+1,+1]], dtype=np.float32)*0.95
CUBE_E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def draw_cube(canvas, ry):
    pts = proj_batch(CUBE_V, ry)
    for a, b in CUBE_E:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (200,200,200), 1, cv2.LINE_AA)
    for p in pts:
        cv2.rectangle(canvas,(p[0]-5,p[1]-5),(p[0]+5,p[1]+5),(255,255,255),-1)

def draw_grid_floor(canvas, ry, n=8):
    sc = 0.95; step = 2*sc/n; y = -sc
    for i in range(n+1):
        xi = -sc + i*step
        p0 = proj_single(xi, y, -sc, ry)
        p1 = proj_single(xi, y,  sc, ry)
        cv2.line(canvas, p0, p1, (40,40,40), 1, cv2.LINE_AA)
        p0 = proj_single(-sc, y, xi, ry)
        p1 = proj_single( sc, y, xi, ry)
        cv2.line(canvas, p0, p1, (40,40,40), 1, cv2.LINE_AA)

# ── Tree ──────────────────────────────────────────────────────────────────────
def _grow(pos, dirn, length, depth, maxd, pts, rng):
    if depth > maxd or length < 0.008: return
    steps  = max(6, int(length * 80))
    spread = 0.014*(1 - depth/max(maxd,1))
    for s in range(steps+1):
        t = s/max(steps,1)
        pts.append([pos[0]+dirn[0]*length*t + rng.gauss(0,spread),
                    pos[1]+dirn[1]*length*t + rng.gauss(0,spread),
                    pos[2]+dirn[2]*length*t + rng.gauss(0,spread),
                    float(depth), float(maxd)])
    end = [pos[i]+dirn[i]*length for i in range(3)]
    if depth < maxd:
        nc = 5 if depth==0 else (4 if depth<3 else 3)
        for _ in range(nc):
            ax = rng.uniform(-0.65,0.65); az = rng.uniform(-0.65,0.65)
            d = dirn[:]
            c2,s2 = math.cos(ax),math.sin(ax)
            d[1],d[2] = d[1]*c2-d[2]*s2, d[1]*s2+d[2]*c2
            c2,s2 = math.cos(az),math.sin(az)
            d[0],d[1] = d[0]*c2-d[1]*s2, d[0]*s2+d[1]*c2
            n = math.sqrt(sum(v*v for v in d))+1e-9; d=[v/n for v in d]
            _grow(end[:], d, length*rng.uniform(0.60,0.74), depth+1, maxd, pts, rng)

def make_tree(maxd=9, seed=7):
    rng = random.Random(seed); pts = []
    _grow([0,-0.93,0],[0,1,0],0.58,0,maxd,pts,rng)
    arr = np.array(pts, dtype=np.float32)
    order = np.argsort(arr[:,1])
    return arr[order]

def draw_tree_fast(canvas, tree_arr, reveal, ry):
    n = int(len(tree_arr)*min(reveal,1.0))
    if n <= 0: return
    pts  = tree_arr[:n]
    xy   = proj_batch(pts[:,:3], ry)
    mask = (xy[:,0]>=0)&(xy[:,0]<W)&(xy[:,1]>=0)&(xy[:,1]<VIZ_H)
    xy   = xy[mask]; pts = pts[mask]
    if len(xy)==0: return
    depth = pts[:,3]; maxd = pts[:,4]
    t     = depth / np.maximum(maxd, 1)
    R = np.where(t<0.50, 255.0, 200.0 - t*150.0).clip(0,255)
    G = np.full(len(t), 255.0)
    B = np.where(t<0.20, 230.0,
        np.where(t<0.50, 230.0 - (t-0.20)/0.30*110.0,
                          80.0 - (t-0.50)/0.50*40.0)).clip(0,255)
    glow = np.zeros((VIZ_H, W, 3), dtype=np.float32)
    px, py = xy[:,0], xy[:,1]
    np.add.at(glow, (py, px, 0), B)
    np.add.at(glow, (py, px, 1), G)
    np.add.at(glow, (py, px, 2), R)
    glow = np.clip(glow, 0, 255).astype(np.uint8)
    b1 = cv2.GaussianBlur(glow, (9,9),  0)
    b2 = cv2.GaussianBlur(glow, (21,21), 0)
    b3 = cv2.GaussianBlur(glow, (41,41), 0)
    composite = np.clip(
        glow.astype(np.float32)*1.0 + b1.astype(np.float32)*0.9 +
        b2.astype(np.float32)*0.6  + b3.astype(np.float32)*0.3,
        0, 255).astype(np.uint8)
    canvas[:] = np.clip(canvas.astype(np.uint16) + composite, 0, 255).astype(np.uint8)

# ── Particles ─────────────────────────────────────────────────────────────────
class P:
    __slots__=("pos","vel","color","life","maxlife")
    def __init__(self,pos,vel,color,life):
        self.pos=list(pos);self.vel=list(vel);self.color=color;self.life=self.maxlife=float(life)
    def step(self,dt):
        for i in range(3): self.pos[i]+=self.vel[i]*dt
        self.vel[1]-=2.8*dt; self.life-=dt

def spawn_particles(tree_arr):
    ps=[]; step=max(1,len(tree_arr)//400)
    for row in tree_arr[::step]:
        px,py,pz,depth,maxd=row; t=depth/max(maxd,1)
        vel=[random.uniform(-4,4),random.uniform(0,5),random.uniform(-4,4)]
        R=255 if t<0.5 else int(60+100*(1-t)); G=255; B=int(200*(1-t)+40*t)
        ps.append(P([px,py,pz],vel,(B,G,R),random.uniform(2,5)))
    return ps

def draw_particles(canvas,ps,ry):
    for p in ps:
        fade=max(0.0,p.life/p.maxlife); c=tuple(int(v*fade) for v in p.color)
        sx,sy=proj_single(p.pos[0],p.pos[1],p.pos[2],ry); s=max(1,int(fade*7))
        if 0<=sx<W and 0<=sy<VIZ_H: cv2.circle(canvas,(sx,sy),s,c,-1)

# ── Hand drawing ──────────────────────────────────────────────────────────────
TIP_IDS={4,8,12,16,20}
GREEN=(0,220,80);CYAN=(0,220,220);YELLOW=(0,200,255);WHITE=(255,255,255);GRAY=(100,100,100)

def draw_landmarks(frame, lms, fw, fh, dot_color, label, label_color):
    pts={}
    for i,lm in enumerate(lms):
        px, py = int(lm.x*fw), int(lm.y*fh)
        pts[i]=(px, py)
    for a,b in CONNECTIONS:
        if a in pts and b in pts: cv2.line(frame,pts[a],pts[b],(180,180,180),1,cv2.LINE_AA)
    for i,(px,py) in pts.items():
        s=9 if i in TIP_IDS else 6
        cv2.rectangle(frame,(px-s,py-s),(px+s,py+s),dot_color if i==0 else (255,255,255),-1)
    wx,wy=pts[0]; lines=label.split("\n"); lw=120; lh=14+len(lines)*16
    lx=max(0,min(wx-lw//2,fw-lw-4)); ly=wy+15
    if ly+lh>fh: ly=wy-lh-15
    ov=frame.copy()
    cv2.rectangle(ov,(lx-4,ly-4),(lx+lw,ly+lh),(0,0,0),-1)
    cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
    cv2.rectangle(frame,(lx-4,ly-4),(lx+lw,ly+lh),label_color,2)
    for j,line in enumerate(lines):
        cv2.putText(frame,line,(lx,ly+12+j*16),cv2.FONT_HERSHEY_SIMPLEX,0.45,label_color,1,cv2.LINE_AA)

def draw_viz_ui(viz, branches, points, fps):
    items=[
        ("GESTURE CONTROLS", YELLOW, 0.50, 2),
        ("[LEFT]  Grow / Rotate Tree", GREEN, 0.40, 1),
        ("[RIGHT] Add Branches",        CYAN,  0.40, 1),
        ("[BOTH]  Rotate Tree",         WHITE, 0.40, 1),
        ("[SPREAD] Explode!",           GRAY,  0.40, 1),
    ]
    for i,(txt,col,sc,th) in enumerate(items):
        cv2.putText(viz,txt,(10,22+i*21),cv2.FONT_HERSHEY_SIMPLEX,sc,col,th,cv2.LINE_AA)
    stats=[f"BRANCHES: {branches}", f"POINTS: {points:,}", f"FPS: {fps:.0f}"]
    by=VIZ_H-len(stats)*21-8
    for i,s in enumerate(stats):
        cv2.putText(viz,s,(10,by+i*21),cv2.FONT_HERSHEY_SIMPLEX,0.45,GREEN,1,cv2.LINE_AA)

# ── Per-session state ─────────────────────────────────────────────────────────
sessions = {}

def create_hand_detector():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2
    )
    return HandLandmarker.create_from_options(options)

def new_state(seed=7, maxd=9):
    print(f" Building tree seed={seed} maxd={maxd}...")
    tree_arr = make_tree(maxd=maxd, seed=seed)
    print(f" Tree: {len(tree_arr):,} pts")
    return {
        "state": 0,          # 0=empty 1=tree 2=explode
        "seed": seed, "maxd": maxd,
        "tree_arr": tree_arr,
        "tree_reveal": 0.0,
        "cube_ry": 0.0,
        "particles": [],
        "exploded": False,
        "left_prev": None,
        "both_prev_x": None,
        "fps_ring": collections.deque(maxlen=30),
        "prev_t": time.time(),
        "detector": create_hand_detector(),
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Socket events ─────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    from flask_socketio import request as sreq
    sid = sreq.sid
    sessions[sid] = new_state()
    print(f"[+] Client connected: {sid}")

@socketio.on("disconnect")
def on_disconnect():
    from flask_socketio import request as sreq
    sid = sreq.sid
    if sid in sessions:
        del sessions[sid]
    print(f"[-] Client disconnected: {sid}")

@socketio.on("action")
def on_action(data):
    from flask_socketio import request as sreq
    sid = sreq.sid
    if sid not in sessions: return
    s = sessions[sid]
    act = data.get("type","")
    if act == "reset":
        s.update({"state":0,"tree_reveal":0.0,"maxd":9,"seed":7,
                  "exploded":False,"particles":[]})
        s["tree_arr"] = make_tree(maxd=9, seed=7)
    elif act == "randomize":
        s["seed"] = random.randint(0,9999)
        s["tree_arr"] = make_tree(maxd=s["maxd"], seed=s["seed"])
        s["tree_reveal"] = 1.0; s["state"] = 1
    elif act == "clear":
        s["state"] = 0; s["tree_reveal"] = 0.0

@socketio.on("frame")
def on_frame(data):
    from flask_socketio import request as sreq
    sid = sreq.sid
    if sid not in sessions: return
    s = sessions[sid]

    # FPS
    now = time.time(); dt = min(now - s["prev_t"], 0.1); s["prev_t"] = now
    s["fps_ring"].append(1.0/max(dt,0.001))
    fps = sum(s["fps_ring"])/len(s["fps_ring"])

    # Decode incoming frame
    try:
        img_bytes = base64.b64decode(data.split(",")[1] if "," in data else data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return
    except Exception as e:
        print("decode err:", e); return

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (W, CAM_H))
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# --- NEW HAND DETECTION ---
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
    results = s["detector"].detect(mp_image)

    # Hand parsing
    left_lm = right_lm = None
    handedness = results.handedness if hasattr(results, "handedness") else []
    hand_landmarks = results.hand_landmarks if hasattr(results, "hand_landmarks") else []

    for i, lm in enumerate(hand_landmarks):
        if i < len(handedness):
            label = handedness[i][0].category_name
            if label == "Left":
                left_lm = lm
            elif label == "Right":
                right_lm = lm

    # Gesture logic (same as original)
    if left_lm and right_lm:
        mid_x = (left_lm[0].x + right_lm[0].x)/2
        if s["both_prev_x"] is not None:
            s["cube_ry"] += (mid_x - s["both_prev_x"])*6.0
        s["both_prev_x"] = mid_x; s["left_prev"] = None
        spread = math.hypot(left_lm[0].x - right_lm[0].x,
                            left_lm[0].y - right_lm[0].y)
        if spread > 0.38 and not s["exploded"] and s["state"]==1:
            s["state"] = 2; s["exploded"] = True
            s["particles"] = spawn_particles(s["tree_arr"])
    else:
        s["both_prev_x"] = None
        if left_lm and not right_lm:
            lx = left_lm[0].x; ly = left_lm[0].y
            if s["state"] == 0: s["state"] = 1; s["exploded"] = False
            if s["left_prev"]:
                dy = ly - s["left_prev"][1]; dx = lx - s["left_prev"][0]
                if dy > 0.003: s["tree_reveal"] = min(1.0, s["tree_reveal"] + dy*5.0)
                s["cube_ry"] += dx*4.0
            s["left_prev"] = (lx, ly)
        else:
            s["left_prev"] = None
            if not left_lm: s["cube_ry"] += dt*0.25
        if right_lm and s["state"]==1:
            new_d = min(s["maxd"]+1, 11)
            if new_d != s["maxd"]:
                s["maxd"] = new_d
                s["tree_arr"] = make_tree(maxd=s["maxd"], seed=s["seed"])

    # Update particles
    s["particles"] = [p for p in s["particles"] if p.life > 0]
    for p in s["particles"]: p.step(dt)
    if s["state"]==2 and not s["particles"]:
        s["state"] = 0; s["tree_reveal"] = 0.0; s["exploded"] = False

    # ── Render viz panel ─────────────────────────────────────────────────────
    viz = np.zeros((VIZ_H, W, 3), dtype=np.uint8)
    draw_grid_floor(viz, s["cube_ry"])
    draw_cube(viz, s["cube_ry"])

    if s["state"]==1:
        draw_tree_fast(viz, s["tree_arr"], s["tree_reveal"], s["cube_ry"])
    elif s["state"]==2 and s["particles"]:
        draw_particles(viz, s["particles"], s["cube_ry"])
        blurred = cv2.GaussianBlur(viz,(15,15),0)
        cv2.addWeighted(viz,1.0,blurred,0.5,0,viz)

    branches_n = max(1,int(len(s["tree_arr"])/900))
    draw_viz_ui(viz, branches_n, len(s["tree_arr"]), fps)

    if s["state"]==1 and s["tree_reveal"]>0:
        bw = int((W-20)*s["tree_reveal"])
        cv2.rectangle(viz,(10,VIZ_H-4),(10+bw,VIZ_H-1),(0,220,80),-1)

    # Landmarks on cam frame
    for i, lm in enumerate(hand_landmarks):
        if i < len(handedness):
            label = handedness[i][0].category_name
            if label == "Left":
                draw_landmarks(frame, lm, W, CAM_H, (0,220,80), "GRAB\nMOVE", (0,220,80))
            else:
                draw_landmarks(frame, lm, W, CAM_H, (255,160,0), "ADD\nBRANCHES", (255,160,0))

    # Status bar
    num_hands = (1 if left_lm else 0)+(1 if right_lm else 0)
    bar = np.zeros((BAR_H, W, 3), dtype=np.uint8); bar[:] = 18
    cv2.line(bar,(0,0),(W,0),(55,55,55),1)
    if num_hands==2:   mode="BOTH HANDS: rotate | SPREAD: explode"
    elif left_lm:      mode="LEFT: swipe down to grow | L/R to rotate"
    elif right_lm:     mode="RIGHT: add branches"
    else:              mode="Raise LEFT hand to grow tree"
    cv2.putText(bar,mode,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.38,WHITE,1,cv2.LINE_AA)

    final = np.vstack([viz, frame, bar])

    # Encode and emit
    _, buf = cv2.imencode(".jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf).decode("utf-8")
    emit("frame_out", {"img": "data:image/jpeg;base64," + b64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)