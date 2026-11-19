#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 C：分辨「FoundationStereo 在針上的深度」和「推算的指尖深度」差 1 cm 是誰的問題。

在同一組影像上，對三個位置各用三種方法給深度並排印出：
  位置：遞針手指尖（推算位置投影到影像上；那裡是白色指頭，面積大、紋理清楚）、針的夾持點、針的接針點
  方法：FoundationStereo、RealSense 原生深度、推算（A 的校正 + tool_pose + 指尖偏移；只有指尖有）

判讀：
  指尖上 FS ≈ 原生 ≈ 推算        → 推算的指尖沒問題，差異在針上（FS 在針上偏，或夾持點不在指尖上）
  指尖上 FS ≈ 原生，但都和推算差很多 → 校正／指尖偏移的問題；FS 本身是對的
  指尖上 FS 和原生差很多          → FS 在這個場景有整體偏差

前提：手臂 container 的 EtH 相機 launch + 遞針手的 tm_driver 開著；遞針手夾著針停住。
B 的感知 node 可以開著也可以關掉（這支自己載入 YOLO／VRB／FoundationStereo，GPU 約 4.5 GB，兩個都開也放得下）。

    source /opt/ros/humble/setup.bash && cd /workspaces/vrb
    python check_depth_tip.py --holder right --n 3

另外存一張對照圖到 pred_handoff_output/depth_tip_<時間>.png（左：彩色影像放大 + 標記，右：FS 深度圖）。
"""

import argparse
import datetime
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from capture_handover_depth import (CAM, DEFAULT_CALIB, GRIP, INFOS, POSE, STREAMS, fingertip_cam,  # noqa: E402
                                    info_to_dict, load_calib, msg_to_np)
from eval_handover_depth import build_perception, guess_depth_scale  # noqa: E402
from handover_depth import D405Calib, StereoDepth, lookup_disparity, make_backend  # noqa: E402


class Grabber:
    def __init__(self, holder):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node("check_depth_tip")
        self.partial, self.done, self.infos, self.pose = {}, None, {}, None
        for k, tp in STREAMS.items():
            self.node.create_subscription(Image, tp, lambda m, k=k: self._img(k, m), qos_profile_sensor_data)
        for k, tp in INFOS.items():
            self.node.create_subscription(CameraInfo, tp, lambda m, k=k: self.infos.__setitem__(k, info_to_dict(m)),
                                          qos_profile_sensor_data)
        self.node.create_subscription(PoseStamped, POSE[holder], self._pose, qos_profile_sensor_data)

    def _img(self, k, m):
        st = round(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, 4)
        d = self.partial.setdefault(st, {})
        d[k] = msg_to_np(m)
        if len(d) == len(STREAMS):
            self.done = self.partial.pop(st)
        if len(self.partial) > 12:
            for key in sorted(self.partial)[:-8]:
                self.partial.pop(key, None)

    def _pose(self, m):
        p, q = m.pose.position, m.pose.orientation
        self.pose = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]

    def grab(self, timeout=10.0):
        self.done = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.done is not None and self.pose is not None and len(self.infos) == len(INFOS):
                return self.done, list(self.pose)
        raise SystemExit(f"{timeout:.0f} 秒內湊不齊一組影像／位姿／camera_info（{CAM}…、{POSE}）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holder", choices=["right", "left"], default="right")
    ap.add_argument("--n", type=int, default=3, help="取幾組影像")
    ap.add_argument("--radius", type=float, default=6.0, help="查深度的半徑（px）")
    args = ap.parse_args()
    holder = args.holder
    receiver = "left" if holder == "right" else "right"
    cal = load_calib(*DEFAULT_CALIB[holder])
    K, dist = np.asarray(cal["camera_matrix"], float), np.asarray(cal["dist_coeffs"], float)

    g = Grabber(holder)
    print("載入 YOLO／VRB／FoundationStereo …")
    perc, _ = build_perception()
    frames, pose = g.grab()
    ci = g.infos
    calib = D405Calib.from_camera_info(ci["ir1"], ci["ir2"], ci["color"])
    sd = StereoDepth(make_backend("fs"), calib)
    scale = guess_depth_scale(frames["depth"])
    print(f"IR fx {calib.fx:.1f}、baseline {calib.baseline * 1000:.2f} mm、原生深度單位 {scale} m")

    rows_all = []
    for it in range(args.n):
        frames, pose = g.grab()
        color, ir1, ir2, raw = frames["color"], frames["ir1"], frames["ir2"], frames["depth"]
        tip = fingertip_cam(pose, cal)
        tip_uv = cv2.projectPoints(tip.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist)[0].ravel()
        res, extra = perc.predict(color)
        if not res.get("ok"):
            print(f"第 {it + 1} 組：感知失敗 {res.get('error')}")
            continue
        pts = {p["name"]: p["uv"] for p in res["points"]}
        mask = (extra.get("det") or {}).get("mask")
        box = extra.get("box")
        # ROI 要同時包住針和指尖
        xs = [box[0], box[2], tip_uv[0] - 40, tip_uv[0] + 40]
        ys = [box[1], box[3], tip_uv[1] - 40, tip_uv[1] + 40]
        box_all = [min(xs), min(ys), max(xs), max(ys)]
        res_fs = sd.disparity(ir1, ir2, sd.roi_from_box(calib.color_uv_to_left(np.reshape(box_all, (2, 2))).ravel()))
        z_native = raw.astype(np.float32) * scale
        with np.errstate(divide="ignore"):
            disp_native = np.where(z_native > 0, calib.fx * calib.baseline / z_native, np.nan).astype(np.float32)
        mask_l = None if mask is None else calib.color_mask_to_left(mask)

        def depth_at(uv, use_mask):
            # 指尖沒有 mask，而且推算位置可能偏幾 px（校正誤差），所以半徑放大一倍，再靠「取較近的那一群」挑出指頭
            r = args.radius if use_mask else 2 * args.radius
            ul = calib.color_uv_to_left([uv])[0]
            q = lookup_disparity(res_fs.disp, res_fs.to_local([ul])[0],
                                 None if not use_mask or mask_l is None else
                                 cv2.resize(mask_l[res_fs.y0:, res_fs.x0:][:int(round(res_fs.disp.shape[0] / res_fs.scale)),
                                                                             :int(round(res_fs.disp.shape[1] / res_fs.scale))],
                                            res_fs.disp.shape[::-1], interpolation=cv2.INTER_NEAREST),
                                 radius=r * res_fs.scale)
            qn = lookup_disparity(disp_native, ul, mask_l if use_mask else None, radius=r)
            zf = None if q["disp"] is None else calib.disp_to_z(q["disp"]) * 1000
            zn = None if qn["disp"] is None else calib.disp_to_z(qn["disp"]) * 1000
            return zf, zn, q.get("sep_px"), qn.get("sep_px")

        rows = [("指尖（白色指頭）", tip_uv, False, tip[2] * 1000),
                (f"針 {GRIP[holder]}（夾持點）", pts.get(GRIP[holder]), True, None),
                (f"針 {GRIP[receiver]}（接針點）", pts.get(GRIP[receiver]), True, None)]
        print(f"\n第 {it + 1} 組（FS {res_fs.time_s * 1000:.0f} ms，ROI {res_fs.disp.shape[1]}×{res_fs.disp.shape[0]}）")
        print(f"{'位置':<22}{'像素 (u, v)':>16}{'FS z':>9}{'原生 z':>9}{'推算 z':>9}   FS−推算  原生−推算  FS−原生")
        for name, uv, use_mask, z_calc in rows:
            if uv is None:
                print(f"{name:<22}（沒有這個點）")
                continue
            zf, zn, _, _ = depth_at(uv, use_mask)
            f = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
            d = lambda a, b: "-" if a is None or b is None else f"{a - b:+.1f}"  # noqa: E731
            print(f"{name:<22}{f'({uv[0]:.0f}, {uv[1]:.0f})':>16}{f(zf):>9}{f(zn):>9}{f(z_calc):>9}"
                  f"   {d(zf, z_calc):>7}  {d(zn, z_calc):>9}  {d(zf, zn):>7}")
            rows_all.append((name, zf, zn, z_calc))

        if it == 0:                                           # 存一張對照圖
            x1, y1, x2, y2 = [int(v) for v in box_all]
            x1, y1, x2, y2 = max(0, x1 - 40), max(0, y1 - 40), min(1280, x2 + 40), min(720, y2 + 40)
            crop = color[y1:y2, x1:x2].copy()
            full = res_fs.full()[y1:y2, x1:x2]
            with np.errstate(divide="ignore", invalid="ignore"):
                zmap = calib.fx * calib.baseline / full * 1000
            fin = np.isfinite(zmap)
            lo, hi = (np.percentile(zmap[fin], [2, 98]) if fin.any() else (0, 1))
            vis = cv2.applyColorMap(np.uint8(np.clip((np.nan_to_num(zmap, nan=hi) - lo) / max(hi - lo, 1e-6), 0, 1) * 255),
                                    cv2.COLORMAP_TURBO)
            for name, uv, _, _ in rows:
                if uv is None:
                    continue
                p = (int(uv[0] - x1), int(uv[1] - y1))
                for im in (crop, vis):
                    cv2.drawMarker(im, p, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
                    cv2.drawMarker(im, p, (0, 0, 0), cv2.MARKER_CROSS, 18, 1)
            out = np.hstack([crop, vis])
            s = 900 / out.shape[1] if out.shape[1] < 900 else 1.0
            out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            cv2.putText(out, f"FS depth {lo:.0f}-{hi:.0f} mm (blue=near)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            path = os.path.join(_HERE, "pred_handoff_output", "depth_tip_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, out)
            print(f"對照圖：{path}")

    print("\n單位 mm。z = 相機光軸方向的距離。")
    for name in dict.fromkeys(r[0] for r in rows_all):
        zs = [r for r in rows_all if r[0] == name]
        mf = [r[1] for r in zs if r[1] is not None]
        mn = [r[2] for r in zs if r[2] is not None]
        print(f"{name}：FS 平均 {np.mean(mf):.1f}" if mf else f"{name}：FS 無", end="")
        print(f"、原生平均 {np.mean(mn):.1f}（{len(mn)}/{len(zs)} 次有值）" if mn else "、原生無值", end="")
        print(f"、推算 {zs[0][3]:.1f}" if zs[0][3] is not None else "")
    g.node.destroy_node()
    g.rclpy.shutdown()


if __name__ == "__main__":
    main()
