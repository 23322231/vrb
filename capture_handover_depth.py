#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 C：深度量測用的錄影網頁（ROS 版）。在 VRB container 執行，**手臂 container 的 EtH 相機 launch 要開著**。

錄的是和實際 handover 一模一樣的資料來源：A 的 dual_amm_eth_camera.launch.py（固定曝光 33000 µs、gain 16、
1280×720@15）發出的 color／infra1／infra2／depth（同一個時間戳），加上兩隻手臂當下的 tool_pose（法蘭位姿）。
之後用 eval_handover_depth.py 離線比較三種深度：FoundationStereo、D405 原生深度、遞針手指尖平面。

    source /opt/ros/humble/setup.bash && cd /workspaces/vrb
    python capture_handover_depth.py --session captures/20261118
    # 瀏覽器開 http://localhost:8767

網頁上的流程（每個「情況」一個資料夾：right_holds = 右手夾針（夾在 head 1/5）、left_holds = 左手夾針（夾在 tail_1_3））：
  1. 選情況。遞針手夾著針停在拍照位附近、停穩（頁面顯示「靜止」）。
  2. 「記錄這個位置」：收下一批 N 組影像（預設 20）+ 兩手位姿 → <case>/pos_00/。
  3. 把遞針手移動 1–2 cm（TMflow 點動即可，方向隨意，最好包含靠近／遠離相機），停穩後再記一次 → pos_01、pos_02 …
     位移的真值由錄到的 tool_pose 算，不需要精確的移動量。
  4. （選做，最有參考價值）接針手慢慢移到「它要夾的那個點」正上方，指尖對準（不要碰動針），按「記錄接針手對點」。
     存在最後一個 pos 資料夾的 touch.json。這個點就是真正的夾取目標，三種深度都和它比。

畫面上的十字：用 A 的校正 + 目前 tool_pose + 指尖偏移推算的「指尖」位置（黃 = 右手、青 = 左手）。
十字落在白色指尖上，就表示位姿、校正、指尖偏移三者在這一刻是一致的；差很多先不要錄，回報給 C。

輸出：
  <session>/meta.json                 各 stream 的 camera_info、realsense extrinsics、校正檔與指尖偏移（錄影當下的內容）
  <session>/<case>/pos_XX/NNN_color.png、_ir1.png、_ir2.png、_depth.png（16-bit 原始值）
  <session>/<case>/pos_XX/frames.json  每組的時間戳與兩手位姿（[x, y, z, qx, qy, qz, qw]，公尺）
  <session>/<case>/pos_XX/touch.json   （選做）接針手對點時的位姿
"""

import argparse
import datetime
import json
import math
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import yaml

CAM = "arm/camera_eth/realsense_camera_eth/"
STREAMS = {"color": CAM + "color/image_raw", "ir1": CAM + "infra1/image_rect_raw",
           "ir2": CAM + "infra2/image_rect_raw", "depth": CAM + "depth/image_rect_raw"}
INFOS = {"color": CAM + "color/camera_info", "ir1": CAM + "infra1/camera_info",
         "ir2": CAM + "infra2/camera_info", "depth": CAM + "depth/camera_info"}
EXTR = {"depth_to_color": CAM + "extrinsics/depth_to_color", "depth_to_infra2": CAM + "extrinsics/depth_to_infra2"}
POSE = {"left": "arm/tm12_left/tm_driver/tool_pose", "right": "arm/tm12_right/tm_driver/tool_pose"}
CALIB_DIR = "/data/calibration"                 # 手臂 container 的 AI_Robot_ws/data（VRB container 以唯讀掛載）
DEFAULT_CALIB = {"right": (f"{CALIB_DIR}/20261115_070456/calibration_camera_eth_right.yaml",
                           f"{CALIB_DIR}/20261115_070456/fingertip_offset_right.yaml"),
                 "left": (f"{CALIB_DIR}/20261115_074833/calibration_camera_eth_left.yaml",
                          f"{CALIB_DIR}/20261115_074833/fingertip_offset_left.yaml")}
GRIP = {"right": "head_1_5", "left": "tail_1_3"}     # 計畫 §2.0.2：各手固定夾的位置
CASES = {"right_holds": "right", "left_holds": "left"}


# =============================================================================
# 位姿與校正（不依賴 ROS，eval_handover_depth.py 也用）
# =============================================================================
def quat_to_R(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_tm_euler(R):
    """TM 的 [rx, ry, rz]（R = Rz·Ry·Rx，和 tm_driver 用 setRPY 產生四元數相同）。"""
    ry = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    return [math.atan2(R[2, 1], R[2, 2]), ry, math.atan2(R[1, 0], R[0, 0])]


def pose7_to_T(p):
    """[x, y, z, qx, qy, qz, qw] → 4x4（法蘭在 base 的位姿）。"""
    T = np.eye(4)
    T[:3, :3] = quat_to_R(p[3:7])
    T[:3, 3] = p[:3]
    return T


def load_calib(calib_yaml, offset_yaml):
    with open(calib_yaml, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    with open(offset_yaml, "r", encoding="utf-8") as f:
        o = yaml.safe_load(f)
    T = np.eye(4)
    T[:3, :3] = np.asarray(c["hand_eye"]["rotation"], float).reshape(3, 3)
    T[:3, 3] = np.asarray(c["hand_eye"]["translation"], float)
    return {"calib_yaml": os.path.abspath(calib_yaml), "offset_yaml": os.path.abspath(offset_yaml),
            "calibration_date": str(c.get("calibration_date", "")),
            "T_cam2base": T.tolist(), "camera_matrix": np.asarray(c["camera"]["camera_matrix"], float).tolist(),
            "dist_coeffs": list(map(float, c["camera"].get("dist_coeffs", [0] * 5))),
            "fingertip_offset": list(map(float, o["fingertip_offset"]))}


def fingertip_cam(pose7, cal):
    """指尖在相機 color optical frame 的位置（公尺）。"""
    p_base = pose7_to_T(pose7) @ np.r_[cal["fingertip_offset"], 1.0]
    return (np.linalg.inv(np.asarray(cal["T_cam2base"])) @ p_base)[:3]


# =============================================================================
# ROS
# =============================================================================
def msg_to_np(m):
    enc = m.encoding.lower()
    if enc in ("16uc1", "mono16"):
        a = np.frombuffer(bytes(m.data), np.uint16).reshape(m.height, m.step // 2)[:, :m.width]
        return a.byteswap() if m.is_bigendian else a
    ch = {"mono8": 1, "8uc1": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[enc]
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width * ch]
    a = a.reshape(m.height, m.width, ch) if ch > 1 else a
    if enc == "rgb8":
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(a, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    return a.copy()


def info_to_dict(m):
    return {"width": m.width, "height": m.height, "distortion_model": m.distortion_model, "d": list(m.d),
            "k": list(m.k), "r": list(m.r), "p": list(m.p), "frame_id": m.header.frame_id}


class Recorder:
    def __init__(self, args):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.args = args
        self.lock = threading.RLock()
        self.partial = {}                                     # stamp → {stream: ndarray}
        self.complete = deque(maxlen=4)                       # (stamp, {stream: ndarray})
        self.counts = {k: deque(maxlen=60) for k in list(STREAMS) + list(POSE)}
        self.infos, self.extr = {}, {}
        self.poses = {k: deque(maxlen=200) for k in POSE}      # (t_recv, stamp, [x,y,z,qx,qy,qz,qw])
        self.log = deque(maxlen=40)
        self.job = None                                       # 錄影中的工作
        self.preview = None
        self.calib = {}
        for side, (cy, oy) in DEFAULT_CALIB.items():
            cy = getattr(args, f"calib_{side}") or cy
            oy = getattr(args, f"offset_{side}") or oy
            try:
                self.calib[side] = load_calib(cy, oy)
            except Exception as e:  # noqa: BLE001
                self._log(f"讀不到{side}手的校正檔／指尖偏移（{e}）；指尖十字與平面備案會缺這隻手")

        rclpy.init()
        self.node = Node("handover_depth_capture")
        for k, tp in STREAMS.items():
            self.node.create_subscription(Image, tp, lambda m, k=k: self._on_img(k, m), qos_profile_sensor_data)
        for k, tp in INFOS.items():
            self.node.create_subscription(CameraInfo, tp, lambda m, k=k: self.infos.__setitem__(k, info_to_dict(m)),
                                          qos_profile_sensor_data)
        for side, tp in POSE.items():
            self.node.create_subscription(PoseStamped, tp, lambda m, s=side: self._on_pose(s, m), qos_profile_sensor_data)
        try:                                                  # realsense2_camera 的外參（latched）；沒有這個套件就跳過
            from realsense2_camera_msgs.msg import Extrinsics
            latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            for k, tp in EXTR.items():
                self.node.create_subscription(Extrinsics, tp, lambda m, k=k: self.extr.__setitem__(
                    k, {"rotation": list(m.rotation), "translation": list(m.translation)}), latched)
        except ImportError:
            self._log("沒有 realsense2_camera_msgs，不記錄 extrinsics（分析時當成 infra1 與 color 同一個視點）")
        self.ros = rclpy
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()
        self._log(f"訂閱 {CAM}…（color／infra1／infra2／depth）與兩手 tool_pose")

    # ------------------------------------------------------------------ callbacks
    def _log(self, s):
        line = time.strftime("%H:%M:%S ") + s
        print(line, flush=True)
        self.log.appendleft(line)

    def _on_img(self, k, m):
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        a = msg_to_np(m)
        done = None
        with self.lock:
            self.counts[k].append(time.time())
            d = self.partial.setdefault(round(st, 4), {})
            d[k] = a
            if len(d) == len(STREAMS):
                done = (st, self.partial.pop(round(st, 4)))
                self.complete.append(done)
            if len(self.partial) > 12:                       # 丟掉湊不齊的舊時間戳
                for key in sorted(self.partial)[:-8]:
                    self.partial.pop(key, None)
        if done is not None and self.job is not None:         # 寫檔在鎖外（rclpy.spin 是單執行緒，寫檔期間的影像會跳過）
            self._job_add(*done)

    def _on_pose(self, side, m):
        p = m.pose.position
        q = m.pose.orientation
        v = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        if max(abs(p.x), abs(p.y), abs(p.z)) > 5.0:            # 看起來是 mm
            v[:3] = [p.x / 1000, p.y / 1000, p.z / 1000]
        with self.lock:
            self.counts[side].append(time.time())
            self.poses[side].append((time.time(), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, v))

    # ------------------------------------------------------------------ 狀態
    def pose_now(self, side, window=1.0):
        """最新位姿、是否新鮮、過去 window 秒的最大位移（mm）與旋轉（deg）。"""
        with self.lock:
            d = list(self.poses[side])
        if not d:
            return None
        now = time.time()
        last = d[-1]
        recent = [x for x in d if now - x[0] <= window] or [last]
        P = np.array([x[2][:3] for x in recent])
        move_mm = float(np.max(np.linalg.norm(P - P[-1], axis=1)) * 1000)
        R0 = quat_to_R(last[2][3:])
        rot_deg = max(math.degrees(math.acos(max(-1, min(1, (np.trace(R0.T @ quat_to_R(x[2][3:])) - 1) / 2))))
                      for x in recent)
        return {"pose7": last[2], "age_s": now - last[0], "move_mm": move_mm, "rot_deg": rot_deg,
                "static": (now - last[0] < 0.5) and move_mm < 0.2 and rot_deg < 0.05}

    def hz(self, k):
        with self.lock:
            t = list(self.counts[k])
        t = [x for x in t if time.time() - x < 2.0]
        return round(len(t) / 2.0, 1)

    def status(self):
        st = {"streams": {k: self.hz(k) for k in STREAMS}, "pose_hz": {k: self.hz(k) for k in POSE},
              "infos": sorted(self.infos), "extrinsics": sorted(self.extr), "arms": {}, "session": self.args.session,
              "cases": {}, "job": None, "log": list(self.log)}
        for side in POSE:
            p = self.pose_now(side)
            if p is None:
                st["arms"][side] = None
                continue
            R = quat_to_R(p["pose7"][3:])
            a = {"xyz_mm": [round(v * 1000, 2) for v in p["pose7"][:3]],
                 "rxyz_deg": [round(math.degrees(v), 2) for v in R_to_tm_euler(R)],
                 "age_s": round(p["age_s"], 2), "move_mm": round(p["move_mm"], 3), "static": p["static"]}
            if side in self.calib:
                tip = fingertip_cam(p["pose7"], self.calib[side])
                a["tip_cam_mm"] = [round(v * 1000, 1) for v in tip]
            st["arms"][side] = a
        for case in CASES:
            d = os.path.join(self.args.session, case)
            pos = sorted(x for x in os.listdir(d) if x.startswith("pos_")) if os.path.isdir(d) else []
            st["cases"][case] = [{"name": x, "n": len([f for f in os.listdir(os.path.join(d, x)) if f.endswith("_ir1.png")]),
                                  "touch": os.path.isfile(os.path.join(d, x, "touch.json"))} for x in pos]
        j = self.job
        if j is not None:
            st["job"] = {"case": j["case"], "pos": os.path.basename(j["dir"]), "got": len(j["frames"]), "n": j["n"]}
        return st

    # ------------------------------------------------------------------ 錄影
    def _write_meta(self):
        meta_path = os.path.join(self.args.session, "meta.json")
        meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "topics": {**STREAMS, **POSE},
                "camera_info": self.infos, "extrinsics": self.extr, "calib": self.calib, "grip_point": GRIP}
        if os.path.isfile(meta_path):                         # 已有的不覆蓋（只補缺的 extrinsics）
            with open(meta_path) as f:
                old = json.load(f)
            old.setdefault("extrinsics", {}).update({k: v for k, v in self.extr.items() if k not in old["extrinsics"]})
            meta = old
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)

    def start_record(self, case, n):
        if case not in CASES:
            return False, f"未知的情況 {case}"
        if self.job is not None:
            return False, "正在錄影"
        missing = [k for k in INFOS if k not in self.infos]
        if missing:
            return False, f"還沒收到 camera_info：{missing}"
        holder = CASES[case]
        p = self.pose_now(holder)
        if p is None or p["age_s"] > 0.5:
            return False, f"收不到{holder}手的 tool_pose（{POSE[holder]}）"
        if not p["static"]:
            return False, f"{holder}手還在動（1 秒內移動 {p['move_mm']:.2f} mm）；停穩再錄"
        d = os.path.join(self.args.session, case)
        os.makedirs(d, exist_ok=True)
        k = len([x for x in os.listdir(d) if x.startswith("pos_")])
        out = os.path.join(d, f"pos_{k:02d}")
        os.makedirs(out)
        self._write_meta()
        self.job = {"case": case, "holder": holder, "dir": out, "n": int(n), "frames": [], "t0": time.time()}
        self._log(f"開始錄 {case}/{os.path.basename(out)}：{n} 組")
        return True, out

    def _job_add(self, st, imgs):
        j = self.job
        poses = {}
        for side in POSE:
            q = self.pose_now(side, window=0.3)
            poses[side] = None if q is None else {"pose7": q["pose7"], "age_s": q["age_s"], "move_mm": q["move_mm"]}
        i = len(j["frames"])
        stem = os.path.join(j["dir"], f"{i:03d}")
        cv2.imwrite(stem + "_color.png", imgs["color"])
        cv2.imwrite(stem + "_ir1.png", imgs["ir1"])
        cv2.imwrite(stem + "_ir2.png", imgs["ir2"])
        cv2.imwrite(stem + "_depth.png", imgs["depth"])
        j["frames"].append({"i": i, "stamp": st, "poses": poses})
        if len(j["frames"]) >= j["n"]:
            self._finish_job()

    def _finish_job(self):
        j, self.job = self.job, None
        with open(os.path.join(j["dir"], "frames.json"), "w") as f:
            json.dump({"case": j["case"], "holder": j["holder"], "frames": j["frames"]}, f, indent=1)
        moves = [fr["poses"][j["holder"]]["move_mm"] for fr in j["frames"] if fr["poses"].get(j["holder"])]
        warn = "（錄影中遞針手有移動，這個位置不可信）" if moves and max(moves) > 0.3 else ""
        self._log(f"完成 {j['case']}/{os.path.basename(j['dir'])}：{len(j['frames'])} 組，"
                  f"{time.time() - j['t0']:.1f} s{warn}")

    def record_touch(self, case):
        if case not in CASES:
            return False, f"未知的情況 {case}"
        holder = CASES[case]
        receiver = "left" if holder == "right" else "right"
        d = os.path.join(self.args.session, case)
        pos = sorted(x for x in os.listdir(d) if x.startswith("pos_")) if os.path.isdir(d) else []
        if not pos:
            return False, "這個情況還沒有任何位置；先錄位置，再讓接針手對點"
        p = self.pose_now(receiver)
        if p is None or not p["static"]:
            return False, f"{receiver}手（接針手）沒有位姿或還在動"
        h = self.pose_now(holder)
        out = {"receiver": receiver, "point": GRIP[receiver], "pose7": p["pose7"],
               "holder_pose7": None if h is None else h["pose7"], "time": time.time()}
        if receiver in self.calib:
            out["fingertip_cam"] = fingertip_cam(p["pose7"], self.calib[receiver]).tolist()
        with open(os.path.join(d, pos[-1], "touch.json"), "w") as f:
            json.dump(out, f, indent=1)
        self._log(f"記錄接針手對點：{case}/{pos[-1]}，{receiver}手指尖 → {out['point']}"
                  + ("" if "fingertip_cam" not in out else f"，相機座標 z = {out['fingertip_cam'][2] * 1000:.1f} mm"))
        return True, pos[-1]

    # ------------------------------------------------------------------ 預覽
    def jpeg(self):
        with self.lock:
            fr = self.complete[-1][1] if self.complete else None
        if fr is None:
            img = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(img, "no frames yet", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            img = fr["color"].copy()
            for side, col in (("right", (0, 220, 255)), ("left", (255, 220, 0))):
                p = self.pose_now(side)
                if p is None or side not in self.calib:
                    continue
                c = self.calib[side]
                tip = fingertip_cam(p["pose7"], c)
                if tip[2] <= 0.02:
                    continue
                uv = cv2.projectPoints(tip.reshape(1, 1, 3), np.zeros(3), np.zeros(3), np.asarray(c["camera_matrix"]),
                                       np.asarray(c["dist_coeffs"]))[0].ravel()
                u, v = int(round(uv[0])), int(round(uv[1]))
                cv2.drawMarker(img, (u, v), col, cv2.MARKER_CROSS, 28, 2)
                cv2.putText(img, f"{side} tip z={tip[2] * 1000:.0f}mm", (u + 12, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>Handover 深度量測錄影</title>
<style>
body{font-family:sans-serif;margin:12px;background:#f6f6f4;color:#222}
.row{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin-bottom:10px}
img{border-radius:6px;max-width:960px;width:100%}
button{font-size:15px;padding:8px 14px;margin:4px 4px 4px 0;border-radius:6px;border:1px solid #888;background:#fff;cursor:pointer}
button.main{background:#2b6cb0;color:#fff;border-color:#2b6cb0}
.ok{color:#18794e;font-weight:bold}.bad{color:#c0392b;font-weight:bold}
table{border-collapse:collapse}td,th{padding:2px 8px;border-bottom:1px solid #eee;text-align:left;font-size:14px}
#log{font-family:monospace;font-size:12px;white-space:pre;max-height:260px;overflow:auto}
</style></head><body>
<h2>Handover 深度量測錄影</h2>
<div class="row">
 <div><img src="/stream.mjpg"><div style="font-size:13px">十字 = 由校正 + tool_pose + 指尖偏移推算的指尖（黃 右手、青 左手）。十字要落在白色指尖上才錄。</div></div>
 <div style="min-width:380px">
  <div class="card"><b>情況</b><br>
   <label><input type="radio" name="case" value="right_holds" checked> 右手夾針（夾 head 1/5；左手之後夾 tail_1_3）</label><br>
   <label><input type="radio" name="case" value="left_holds"> 左手夾針（夾 tail_1_3；右手之後夾 head 1/5）</label><br>
   每個位置錄 <input id="n" type="number" value="20" min="1" max="200" style="width:60px"> 組<br>
   <button class="main" onclick="act('record')">記錄這個位置</button>
   <button onclick="act('touch')">記錄接針手對點</button>
   <div id="msg"></div></div>
  <div class="card"><b>相機與手臂</b><div id="st"></div></div>
  <div class="card"><b>已錄</b><div id="cases"></div></div>
 </div>
</div>
<div class="card"><b>日誌</b><div id="log"></div></div>
<script>
function cs(){return document.querySelector('input[name=case]:checked').value}
async function act(a){
 const r=await fetch('/api/'+a+'?case='+cs()+'&n='+document.getElementById('n').value,{method:'POST'});
 const j=await r.json(); document.getElementById('msg').innerHTML=(j.ok?'<span class=ok>':'<span class=bad>')+j.msg+'</span>';}
function f(x){return x==null?'-':x}
async function tick(){
 try{const s=await (await fetch('/status')).json();
 let h='<table><tr><th>stream</th><th>Hz</th></tr>';
 for(const k in s.streams){h+=`<tr><td>${k}</td><td class="${s.streams[k]>10?'ok':'bad'}">${s.streams[k]}</td></tr>`}
 h+=`</table>camera_info：${s.infos.join(', ')||'<span class=bad>無</span>'}<br>extrinsics：${s.extrinsics.join(', ')||'無'}<br><table>`;
 for(const a of ['right','left']){const x=s.arms[a];
  if(!x){h+=`<tr><td>${a}</td><td colspan=3 class=bad>沒有 tool_pose</td></tr>`;continue}
  h+=`<tr><td>${a}</td><td>${x.xyz_mm.join(', ')} mm<br>${x.rxyz_deg.join(', ')} °</td><td class="${x.static?'ok':'bad'}">${x.static?'靜止':'移動 '+x.move_mm+' mm'}</td><td>指尖 z ${x.tip_cam_mm?x.tip_cam_mm[2]:'-'} mm</td></tr>`}
 h+='</table>'+(s.job?`<span class=bad>錄影中 ${s.job.case}/${s.job.pos}：${s.job.got}/${s.job.n}</span>`:'');
 document.getElementById('st').innerHTML=h;
 let c='';for(const k in s.cases){c+=`<b>${k}</b>：`+(s.cases[k].map(p=>`${p.name}（${p.n}${p.touch?'，有對點':''}）`).join('、')||'無')+'<br>'}
 document.getElementById('cases').innerHTML=c+`<span style="font-size:12px">${s.session}</span>`;
 document.getElementById('log').textContent=s.log.join('\\n');}catch(e){}
 setTimeout(tick,500)}
tick();
</script></body></html>"""


def make_handler(rec):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                b = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/status":
                self._json(rec.status())
            elif u.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        j = rec.jpeg()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        time.sleep(0.2)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/api/record":
                ok, msg = rec.start_record(q.get("case", ""), int(q.get("n", rec.args.frames)))
            elif u.path == "/api/touch":
                ok, msg = rec.record_touch(q.get("case", ""))
            else:
                return self.send_error(404)
            if not ok:
                rec._log("✗ " + msg)
            self._json({"ok": ok, "msg": msg})
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=os.path.join("captures", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--frames", type=int, default=20)
    for s in ("right", "left"):
        ap.add_argument(f"--calib_{s}", help=f"預設 {DEFAULT_CALIB[s][0]}")
        ap.add_argument(f"--offset_{s}", help=f"預設 {DEFAULT_CALIB[s][1]}")
    args = ap.parse_args()
    os.makedirs(args.session, exist_ok=True)
    rec = Recorder(args)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(rec))
    rec._log(f"網頁：http://localhost:{args.port}　資料夾：{os.path.abspath(args.session)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        rec.node.destroy_node()
        rec.ros.shutdown()


if __name__ == "__main__":
    main()
