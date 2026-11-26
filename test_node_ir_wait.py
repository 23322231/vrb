#!/usr/bin/env python3
"""不需要 ROS 與 GPU：用假的 rclpy / sensor_msgs / handover_perception 測 handover_perception_node.py 的
input_reliable（QoS 選擇）與「收到 color 後等 IR」的深度配對。"""
import sys
import threading
import time
import types

import numpy as np

# ---------------------------------------------------------------- 假模組
subs = []


class FakeNode:
    def __init__(self, name):
        self.params = FakeNode.PARAMS

    def declare_parameter(self, n, d):
        return types.SimpleNamespace(value=self.params.get(n, d))

    def get_logger(self):
        return types.SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None, error=print)

    def create_subscription(self, typ, topic, cb, qos, callback_group=None):
        subs.append((typ.__name__, topic, qos, cb))

    def create_publisher(self, *a):
        return types.SimpleNamespace(publish=lambda m: None)

    def create_service(self, *a, **k):
        pass

    def create_timer(self, *a, **k):
        pass


class QoSProfile:
    def __init__(self, **k):
        self.__dict__.update(k)


Enum = types.SimpleNamespace
qos = types.ModuleType("rclpy.qos")
qos.QoSProfile = QoSProfile
qos.ReliabilityPolicy = Enum(RELIABLE="RELIABLE", BEST_EFFORT="BEST_EFFORT")
qos.DurabilityPolicy = Enum(VOLATILE="VOLATILE")
qos.HistoryPolicy = Enum(KEEP_LAST="KEEP_LAST")
qos.qos_profile_sensor_data = "SENSOR_DATA"
mods = {"rclpy": types.ModuleType("rclpy"), "rclpy.qos": qos,
        "rclpy.node": types.SimpleNamespace(Node=FakeNode),
        "rclpy.callback_groups": types.SimpleNamespace(MutuallyExclusiveCallbackGroup=lambda: None),
        "rclpy.executors": types.SimpleNamespace(MultiThreadedExecutor=None)}


class Msg:
    pass


class Image(Msg):
    pass


class CameraInfo(Msg):
    pass


mods["sensor_msgs.msg"] = types.SimpleNamespace(Image=Image, CameraInfo=CameraInfo)
mods["std_msgs.msg"] = types.SimpleNamespace(String=lambda data="": types.SimpleNamespace(data=data))
mods["std_srvs.srv"] = types.SimpleNamespace(Trigger=None)


class FakePerc:
    names = ["head"]

    def __init__(self, *a, **k):
        pass

    def predict(self, img, stamp=None, frame_id="", depth_fn=None):
        time.sleep(0.07)                                        # 推論時間
        res = {"ok": True, "error": "", "stamp": stamp, "points": [{"name": "tail_1_3", "uv": [1, 2], "conf": 0.9}]}
        if depth_fn is not None:
            res["points"][0]["xyz_cam"] = depth_fn([[1, 2]], None, None)[0]
        return res, {}

    def point(self, res, name="tail_1_3"):
        return res["points"][0] if res.get("points") else None


hp = types.ModuleType("handover_perception")
hp.DEFAULT_YOLO, hp.DEFAULT_RUN, hp.DEFAULT_VRB_WEIGHTS = "y", "r", "w"
hp.NeedlePerception = FakePerc
hp.gpu_mem_report = lambda: ""
hp.save_debug = lambda *a, **k: None
hp.draw_overlay = lambda *a, **k: None
mods["handover_perception"] = hp
hd = types.ModuleType("handover_depth")
hd.StereoDepth = lambda backend, calib: None
hd.make_backend = lambda name: name
hd.D405Calib = types.SimpleNamespace(from_camera_info=lambda *a: None)
mods["handover_depth"] = hd
import cv2  # noqa: E402,F401  先 import（node 裡是第一次轉影像時才 import，會多 0.3 s）
sys.modules.update(mods)
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import handover_perception_node as N  # noqa: E402


def img_msg(ts, enc="mono8"):
    m = Image()
    m.header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=int(ts), nanosec=int(round((ts % 1) * 1e9))),
                                     frame_id="color")
    ch = 3 if enc == "rgb8" else 1
    m.height, m.width, m.step, m.encoding = 4, 6, 6 * ch, enc
    m.data = np.zeros(4 * 6 * ch, np.uint8).tobytes()
    return m


def make_node(**params):
    subs.clear()
    FakeNode.PARAMS = params
    n = N.HandoverPerceptionNode()
    return n


class FakeDepth:
    def locate(self, left, right, uv_color, box_color, mask_color):
        return [{"ok": True, "xyz_cam": [0.0, 0.0, 0.25]} for _ in uv_color]


def run_request(n, color_ts, ir_ts=None, ir_delay=None):
    """模擬：請求 → 0.02 s 後 color 到 → ir_delay 秒後 infra1、infra2 到（handover_node 先發 IR 的情況 ir_delay<0）"""
    out = {}
    resp = types.SimpleNamespace(success=None, message=None)

    def call():
        out["resp"] = n.on_predict(None, resp)
    t = threading.Thread(target=call)
    if ir_delay is not None and ir_delay < 0:
        n._on_ir(0, img_msg(ir_ts)); n._on_ir(1, img_msg(ir_ts))
    t.start()
    time.sleep(0.02)
    t0 = time.time()
    n.on_image(img_msg(color_ts, "rgb8"))
    if ir_delay is not None and ir_delay >= 0:
        time.sleep(ir_delay)
        n._on_ir(0, img_msg(ir_ts)); n._on_ir(1, img_msg(ir_ts))
    t.join()
    import json
    return json.loads(resp.message), time.time() - t0


fails = 0


def check(name, cond, info=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("  " + info if info else ""))
    fails += 0 if cond else 1


# 1) input_reliable 預設 false：影像與 IR 都是 sensor data（行為不變）
n = make_node(depth_backend="")
check("預設 false：color 訂閱 = qos_profile_sensor_data", subs[0][2] == "SENSOR_DATA")
r, dt = run_request(n, 100.0)
check("沒開深度：不等 IR、沒有 depth_error", "depth_error" not in r and dt < 0.15, f"{dt * 1e3:.0f} ms")

n = make_node(depth_backend="fs")
check("預設 false + 深度：IR 訂閱仍是 sensor data", all(x[2] == "SENSOR_DATA" for x in subs))

# 2) input_reliable=true：image/infra1/infra2 reliable depth 5，camera_info 維持 sensor data
n = make_node(input_reliable=True, depth_backend="fs", image_topic="handover/predict_image",
              infra1_topic="handover/predict_infra1", infra2_topic="handover/predict_infra2")
img_subs = [s for s in subs if s[0] == "Image"]
info_subs = [s for s in subs if s[0] == "CameraInfo"]
check("true：3 個影像訂閱都是 RELIABLE/VOLATILE/KEEP_LAST 5",
      len(img_subs) == 3 and all(getattr(s[2], "reliability", None) == "RELIABLE" and s[2].durability == "VOLATILE"
                                 and s[2].history == "KEEP_LAST" and s[2].depth == 5 for s in img_subs),
      str([s[1] for s in img_subs]))
check("true：camera_info 3 個仍是 sensor data", len(info_subs) == 3 and all(s[2] == "SENSOR_DATA" for s in info_subs))
n.depth = FakeDepth()

# 3) IR 先到（handover_node 先發 IR）
r, dt = run_request(n, 200.0, ir_ts=200.005, ir_delay=-1)
check("IR 先到：算得出深度", r["points"][0].get("xyz_cam") == [0.0, 0.0, 0.25] and "depth_error" not in r, f"{dt * 1e3:.0f} ms")

# 4) IR 比 color 晚 0.1 s 到
r, dt = run_request(n, 300.0, ir_ts=300.0, ir_delay=0.1)
check("IR 晚 0.1 s：算得出深度", r["points"][0].get("xyz_cam") == [0.0, 0.0, 0.25] and "depth_error" not in r,
      f"總時間 {dt * 1e3:.0f} ms（推論 70 ms 期間已在等）")

# 5) IR 一直沒來：0.3 s 後 depth_error（清空緩衝區 = 開機後從沒收到 IR）
n.ir = {0: [], 1: []}
r, dt = run_request(n, 400.0)
check("IR 沒來：0.3 s 後回 depth_error", r["points"][0].get("xyz_cam") is None and r.get("depth_error")
      and 0.28 <= dt <= 0.45, f"{dt * 1e3:.0f} ms：{r.get('depth_error')}")

# 6) 緩衝區只有時間差 66 ms 的 IR：等滿 0.3 s 後回時間差錯誤
r, dt = run_request(n, 500.0, ir_ts=500.066, ir_delay=-1)
check("IR 差 66 ms：回時間差錯誤", "66 ms" in r.get("depth_error", "") and dt >= 0.28, f"{dt * 1e3:.0f} ms：{r.get('depth_error')}")

# 7) 66 ms 的舊 IR 在緩衝區，正確的 IR 0.15 s 後才到：選到正確的
r, dt = run_request(n, 600.0, ir_ts=600.0, ir_delay=0.15)
n2_ok = r["points"][0].get("xyz_cam") == [0.0, 0.0, 0.25] and "depth_error" not in r
check("IR 晚 0.15 s（超過推論時間）：仍算得出", n2_ok, f"{dt * 1e3:.0f} ms")

# 8) infra1、infra2 時間戳不同：不配對
n.ir = {0: [], 1: []}
n._on_ir(0, img_msg(700.0)); n._on_ir(1, img_msg(700.01))
with n.cv:
    check("infra1／infra2 差 10 ms 不配對", n._ir_match(700.0) is None)

# 9) stamp 仍是 color header
check("stamp = color header 時間戳", abs(r["stamp"] - 600.0) < 1e-6, str(r["stamp"]))

# 10) camera_info 還沒到（depth=None）：不等
n.depth = None
r, dt = run_request(n, 800.0)
check("深度模組還沒就緒：不等 IR", dt < 0.15, f"{dt * 1e3:.0f} ms")

print("\n全部通過" if not fails else f"\n{fails} 項失敗")
sys.exit(1 if fails else 0)
