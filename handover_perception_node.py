#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 B：感知服務 ROS 2 node（VRB container）。介面照 claude/handover_plan.md §3.1。

    服務   handover/predict_needle   std_srvs/srv/Trigger
           收到請求後，等「請求之後才收到」的下一張 color 影像 → YOLO → 224 → VRB → message = JSON 字串
           success=false：no_image（逾時沒有新影像）、no_detection / no_valid_detection（YOLO 沒找到 / 都被 ROI 濾掉）
    訂閱   <image_topic>（sensor_msgs/Image，rgb8 / bgr8 / rgba8 / bgra8 / mono8）
    發布   handover/needle_result（std_msgs/String，每次結果的 JSON，給 rqt / 錄 bag 看）
           handover/needle_debug （sensor_msgs/Image bgr8，疊圖；publish_debug:=true 時）
    不用 cv_bridge（照 AI_Robot_ws 的規則），影像直接由 msg.data 轉 numpy。

執行（VRB container 內，一定要用 /opt/venv 的 python：系統 python 的 torch 2.1 不支援 RTX 50 系列）：

    source /opt/ros/humble/setup.bash
    cd /workspaces/vrb
    python handover_perception_node.py                                  # 預設就是 A 的 EtH 相機
    # 例：-p roi:="[300.0,100.0,1000.0,650.0]"

    # 另一個終端機測試
    ros2 service call handover/predict_needle std_srvs/srv/Trigger
    python handover_call_predict.py --n 20                     # 量延遲與重複性

參數（都可用 -p 改）：
    image_topic         EtH color 影像 topic（預設 arm/camera_eth/realsense_camera_eth/color/image_raw，
                        A 的 dual_amm_eth_camera.launch.py；計畫 §2.0、§3.1.1）
    yolo_weights / vrb_run / vrb_weights / yolo_imgsz / yolo_conf / box_src / crop_mode    見 handover_perception.py
    decode              "vrb_anchor"（預設，計畫 v1.7 §3.1.1a：VRB 3 點投影到 YOLO 中心線、夾取點由 VRB 決定，6 點含 head_1_5，
                        另附 geom_only 純幾何結果與 vrb_agree）/ "geom"（YOLO mask 幾何 + 定方向）/ "vrb"（VRB 直接預測）
                        預設模型：yolo_weights = runs/needle_seg/best_20261116_225446/weights/best.pt（imgsz 1280、conf 0.1），
                        vrb_run = runs/handover_v3/needle5_20261116_205018
    detector            "yolo"（預設）/ "sam3"（SAM 3 文字提示；sam3_weights、sam3_prompts）
    orient              geom 時怎麼定方向："left"（預設，針頭固定在畫面左邊）/ "heat" / "ends" / "all"（vrb_anchor 不用這個參數）
    roi                 [x1, y1, x2, y2]，只接受框中心在裡面的偵測；空 list = 不限
    wait_timeout        等新影像的秒數（預設 2.0）
    skip_frames         請求之後再多丟幾張（預設 0；手臂剛停下來還在晃時可設 1~2）
    debug_root          每次推論存 input / yolo_overlay / crop224 / overlay / result.json 的位置（空字串 = 不存）；
                        YOLO 失敗時也存 input、yolo_overlay、result.json；路徑印在每次的 log（debug_dir=…）
    publish_debug       是否發布疊圖 topic
    depth_backend       ""（預設，xyz_cam 一律 null）/ "fs" / "ffs" / "sgbm"：接 C 的 handover_depth.StereoDepth
                        需要 infra1_topic、infra2_topic、infra1_info_topic、infra2_info_topic、color_info_topic
                        ※ 深度這段還沒在實機測過，接合由統籌與 C 確認
"""

import array
import datetime
import json
import os
import sys
import threading
import time

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handover_perception as HP  # noqa: E402


# =============================================================================
# sensor_msgs/Image <-> numpy（不用 cv_bridge）
# =============================================================================
_CH = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8UC1": 1, "8UC3": 3}


def image_to_bgr(msg):
    enc = msg.encoding.lower() if msg.encoding.lower() in _CH else msg.encoding
    if enc not in _CH:
        raise ValueError(f"不支援的 encoding：{msg.encoding}（要 rgb8 / bgr8 / rgba8 / bgra8 / mono8）")
    ch = _CH[enc]
    buf = np.frombuffer(msg.data, np.uint8)                     # rclpy 給 array.array('B')，可直接當 buffer
    a = buf.reshape(msg.height, msg.step)[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    import cv2
    if enc == "rgb8":
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(a, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    if ch == 1:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    return a.copy()


def image_to_mono(msg):
    """infra1 / infra2（mono8）-> HxW uint8"""
    if msg.encoding not in ("mono8", "8UC1"):
        raise ValueError(f"IR 影像要 mono8，收到 {msg.encoding}")
    buf = np.frombuffer(msg.data, np.uint8)
    return buf.reshape(msg.height, msg.step)[:, :msg.width].copy()


def bgr_to_image(img, header):
    m = Image()
    m.header = header
    m.height, m.width = int(img.shape[0]), int(img.shape[1])
    m.encoding, m.is_bigendian, m.step = "bgr8", 0, int(img.shape[1] * 3)
    m.data = array.array("B", np.ascontiguousarray(img).tobytes())   # 直接給 bytes 時 rclpy 會逐元素檢查，很慢
    return m


def stamp_sec(st):
    return st.sec + st.nanosec * 1e-9


# =============================================================================
class HandoverPerceptionNode(Node):
    def __init__(self):
        super().__init__("handover_perception")
        p = self.declare_parameter
        here = os.path.dirname(os.path.abspath(__file__))
        self.image_topic = p("image_topic", "arm/camera_eth/realsense_camera_eth/color/image_raw").value
        yolo_w = p("yolo_weights", HP.DEFAULT_YOLO).value
        run = p("vrb_run", HP.DEFAULT_RUN).value
        vw = p("vrb_weights", HP.DEFAULT_VRB_WEIGHTS).value
        imgsz = p("yolo_imgsz", 1280).value
        conf = p("yolo_conf", 0.1).value
        box_src = p("box_src", "box").value
        crop_mode = p("crop_mode", "train").value
        decode = p("decode", "vrb_anchor").value               # vrb_anchor（handover 預設）/ geom / vrb
        orient = p("orient", "left").value
        detector = p("detector", "yolo").value                 # yolo / sam3
        sam3_w = p("sam3_weights", os.path.join(here, "sam3.pt")).value
        sam3_prompts = list(p("sam3_prompts", ["needle"]).value)
        roi = list(p("roi", [0.0]).value)                      # 預設 [0.0] 只是讓型別是 double array；長度不是 4 = 不限
        self.wait_timeout = float(p("wait_timeout", 2.0).value)
        self.skip_frames = int(p("skip_frames", 0).value)
        self.debug_root = p("debug_root", os.path.join(here, "pred_handoff_output")).value
        self.publish_debug = bool(p("publish_debug", True).value)
        depth_backend = p("depth_backend", "").value
        self.ir_topics = [p("infra1_topic", "arm/camera_eth/realsense_camera_eth/infra1/image_rect_raw").value,
                          p("infra2_topic", "arm/camera_eth/realsense_camera_eth/infra2/image_rect_raw").value]
        self.info_topics = [p("infra1_info_topic", "arm/camera_eth/realsense_camera_eth/infra1/camera_info").value,
                            p("infra2_info_topic", "arm/camera_eth/realsense_camera_eth/infra2/camera_info").value,
                            p("color_info_topic", "arm/camera_eth/realsense_camera_eth/color/camera_info").value]
        self.ir_max_dt = float(p("ir_max_dt", 0.02).value)

        t = time.perf_counter()
        self.perc = HP.NeedlePerception(yolo_w, run, vw, imgsz, conf, box_src=box_src, crop_mode=crop_mode,
                                        roi_xyxy=roi if len(roi) == 4 else None, decode=decode, orient_mode=orient,
                                        detector=detector, sam3_weights=sam3_w, sam3_prompts=sam3_prompts)
        self.get_logger().info(f"模型載入 + 暖機 {time.perf_counter() - t:.1f}s；YOLO={yolo_w}；VRB={run}；"
                               f"detector={detector} decode={decode}({orient}) crop={crop_mode} box={box_src} imgsz={imgsz}；GPU {HP.gpu_mem_report()}")

        self.cv = threading.Condition()
        self.frame = None          # (msg, recv_time, seq)
        self.seq = 0
        self.ir = {0: [], 1: []}   # 最近幾張 IR (stamp, msg)
        self.infos = [None, None, None]
        self.depth = None
        self.depth_err = ""

        g_img, g_srv = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, self.image_topic, self.on_image, qos_profile_sensor_data, callback_group=g_img)
        self.pub_res = self.create_publisher(String, "handover/needle_result", 10)
        self.pub_dbg = self.create_publisher(Image, "handover/needle_debug", 1) if self.publish_debug else None
        if depth_backend:
            self._setup_depth(depth_backend, g_img)
        self.create_service(Trigger, "handover/predict_needle", self.on_predict, callback_group=g_srv)
        self.get_logger().info(f"服務 handover/predict_needle 就緒；訂閱 {self.image_topic}"
                               + (f"；深度 {depth_backend}" if depth_backend else "；xyz_cam = null（沒開深度）"))
        self.create_timer(5.0, self._warn_no_image, callback_group=g_img)

    # ------------------------------------------------------------------ 影像
    def on_image(self, msg):
        with self.cv:
            self.seq += 1
            self.frame = (msg, time.time(), self.seq)
            self.cv.notify_all()

    def _warn_no_image(self):
        if self.frame is None:
            self.get_logger().warn(f"還沒收到任何影像：{self.image_topic}（檢查 topic 名稱、ROS_DOMAIN_ID、"
                                   "FastDDS SHM 權限，見 check_handover_env.py）", throttle_duration_sec=30.0)

    def wait_new_frame(self):
        """等「請求之後」收到的第 (1 + skip_frames) 張影像；逾時回傳 None。"""
        with self.cv:
            target = self.seq + 1 + self.skip_frames
            ok = self.cv.wait_for(lambda: self.seq >= target, timeout=self.wait_timeout)
            return self.frame if ok else None

    # ------------------------------------------------------------------ 深度（C 的模組，預設關閉）
    def _setup_depth(self, backend, group):
        try:
            from handover_depth import StereoDepth, make_backend
            self._StereoDepth, self._backend = StereoDepth, make_backend(backend)
        except Exception as e:  # noqa: BLE001
            self.depth_err = f"depth backend 載入失敗：{type(e).__name__}: {e}"
            self.get_logger().error(self.depth_err)
            return
        for i, tp in enumerate(self.ir_topics):
            self.create_subscription(Image, tp, lambda m, i=i: self._on_ir(i, m), qos_profile_sensor_data, callback_group=group)
        for i, tp in enumerate(self.info_topics):
            self.create_subscription(CameraInfo, tp, lambda m, i=i: self._on_info(i, m), qos_profile_sensor_data, callback_group=group)

    def _on_ir(self, i, msg):
        with self.cv:
            self.ir[i] = (self.ir[i] + [(stamp_sec(msg.header.stamp), msg)])[-10:]

    def _on_info(self, i, msg):
        self.infos[i] = msg
        if self.depth is None and self.infos[0] is not None and self.infos[1] is not None and self.infos[2] is not None \
                and hasattr(self, "_StereoDepth"):
            from handover_depth import D405Calib
            # infra1→color 外參沒給：D405 的 color 與 infra1 同一個 imager，平移 ≈ 0（見 handover_depth.py 說明）
            self.depth = self._StereoDepth(self._backend, D405Calib.from_camera_info(self.infos[0], self.infos[1], self.infos[2]))
            self.get_logger().info("深度模組就緒（camera_info 已收到）")

    def _depth_fn_for(self, color_msg):
        if self.depth is None:
            return None
        ts = stamp_sec(color_msg.header.stamp)
        with self.cv:
            pairs = {round(s, 4): m for s, m in self.ir[0]}
            cands = [(abs(s - ts), s, pairs[round(s, 4)], m) for s, m in self.ir[1] if round(s, 4) in pairs]
        if not cands:
            self.depth_err = "沒有同時間戳的 infra1 / infra2"
            return None
        dt, _, m1, m2 = min(cands, key=lambda x: x[0])
        if dt > self.ir_max_dt:
            self.depth_err = f"IR 與 color 時間差 {dt * 1e3:.0f} ms > {self.ir_max_dt * 1e3:.0f} ms"
            return None
        left, right = image_to_mono(m1), image_to_mono(m2)

        def fn(uv_list, box, mask):
            out = self.depth.locate(left, right, uv_color=uv_list, box_color=box, mask_color=mask)
            return [o.get("xyz_cam") if o.get("ok", True) else None for o in out]
        return fn

    # ------------------------------------------------------------------ 服務
    def on_predict(self, req, resp):
        t_req = time.time()
        fr = self.wait_new_frame()
        if fr is None:
            res = {"ok": False, "error": "no_image", "stamp": t_req, "frame_id": "", "image_size": None, "yolo": None,
                   "crop_xyxy": None, "points": [], "debug_dir": "",
                   "detail": f"{self.wait_timeout}s 內沒有收到新的影像（{self.image_topic}）"}
            return self._reply(resp, res, None, None, None)
        msg, t_recv, _ = fr
        try:
            img = image_to_bgr(msg)
            self.depth_err = ""
            res, extra = self.perc.predict(img, stamp=stamp_sec(msg.header.stamp), frame_id=msg.header.frame_id,
                                           depth_fn=self._depth_fn_for(msg))
            if self.depth_err:
                res["depth_error"] = self.depth_err
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"推論失敗：{type(e).__name__}: {e}")
            res = {"ok": False, "error": f"inference_error: {type(e).__name__}: {e}", "stamp": stamp_sec(msg.header.stamp),
                   "frame_id": msg.header.frame_id, "image_size": [msg.width, msg.height], "yolo": None,
                   "crop_xyxy": None, "points": [], "debug_dir": ""}
            return self._reply(resp, res, img if "img" in locals() else None, {}, msg)
        res["latency_ms"] = {"wait_frame": round((t_recv - t_req) * 1e3, 1), "total": round((time.time() - t_req) * 1e3, 1)}
        return self._reply(resp, res, img, extra, msg)

    def _reply(self, resp, res, img, extra, msg):
        if img is not None and self.debug_root:
            d = os.path.join(self.debug_root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3])
            HP.save_debug(d, img, res, extra or {}, self.perc.names, async_write=True)   # 會填 res["debug_dir"]
        resp.success = bool(res.get("ok"))
        resp.message = json.dumps(res, ensure_ascii=False)
        self.pub_res.publish(String(data=resp.message))
        if self.pub_dbg is not None and img is not None and msg is not None:
            def _pub(img=img, res=json.loads(resp.message), extra=extra or {}, header=msg.header):
                try:
                    self.pub_dbg.publish(bgr_to_image(HP.draw_overlay(img, res, extra, self.perc.names), header))
                except Exception as e:  # noqa: BLE001
                    self.get_logger().warn(f"疊圖失敗：{e}")
            threading.Thread(target=_pub, daemon=True).start()      # 不拖慢服務回應
        q, h = self.perc.point(res), self.perc.point(res, "head_1_5")
        a = res.get("vrb_anchor") or {}
        self.get_logger().info(f"predict_needle: ok={res.get('ok')} {res.get('error')} "
                               f"tail_1_3={None if not q else q['uv']} head_1_5={None if not h else h['uv']} "
                               f"conf={None if not q else q['conf']} vrb_agree={res.get('vrb_agree')} "
                               f"span={a.get('span_ratio')} diff_vs_geom={a.get('grasp_diff_px')} "
                               f"yolo={(res.get('yolo') or {}).get('conf')} {res.get('latency_ms')} "
                               f"stamp={res.get('stamp')} debug_dir={res.get('debug_dir') or '(沒存)'}")
        return resp


def main():
    rclpy.init()
    node = HandoverPerceptionNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
