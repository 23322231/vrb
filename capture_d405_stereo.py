#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 C：用 pyrealsense2 直接從 EtH D405 錄「同一個 frameset」的 color / infra1 / infra2 / 原生深度，
並把立體參數存成 calib.json（handover_depth.D405Calib.from_json 讀的格式）。不需要 ROS、不需要 A 的 launch。

    pip install pyrealsense2            # 裝在 /opt/venv（只需要一次）
    python capture_d405_stereo.py --out captures/z350_a            # 預覽視窗：空白鍵存一組、q 離開
    python capture_d405_stereo.py --out captures/z350_a --n 30     # 不開視窗，暖機後連續存 30 組（量重複性用）

注意：
  - 相機同一時間只能被一個程式開啟：先關掉手臂 container 裡的 camera_recorder / realsense node。
  - 一定要 USB 3（1280x720 四路同時開）。開不起來時先試 --fps 15，再不行用 --no_depth。
  - D405 沒有紅外線投射器也沒有 IR 濾鏡以外的東西，「IR」影像其實就是可見光的灰階影像，所以現場的燈光要和實際 handover 時一樣。

輸出：<out>/000000_color.png、_ir1.png、_ir2.png、_depth.png（16-bit，乘上 calib.json 的 depth_scale 是公尺）、
      <out>/calib.json、<out>/frames.json（每組的時間戳）
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

DEFAULT_SERIAL = "260422273204"        # EtH D405


def _intr(p):
    i = p.as_video_stream_profile().get_intrinsics()
    return {"K": [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], "dist": list(i.coeffs),
            "model": str(i.model).split(".")[-1].lower(), "width": i.width, "height": i.height}


def _extr(a, b):
    """a → b：X_b = R @ X_a + t。librealsense 的 rotation 是 column-major。"""
    e = a.get_extrinsics_to(b)
    return {"R": np.array(e.rotation).reshape(3, 3).T.tolist(), "t": list(e.translation)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--serial", default=DEFAULT_SERIAL)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--n", type=int, default=0, help="> 0：不開視窗，自動連續存 n 組")
    ap.add_argument("--warmup", type=int, default=30, help="開始前丟掉幾張（等自動曝光穩定）")
    ap.add_argument("--no_depth", action="store_true", help="不錄原生深度（頻寬不夠時）")
    args = ap.parse_args()

    import pyrealsense2 as rs
    os.makedirs(args.out, exist_ok=True)
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    W, H, F = args.width, args.height, args.fps
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, F)
    cfg.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, F)
    cfg.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, F)
    if not args.no_depth:
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, F)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    try:
        ir1 = prof.get_stream(rs.stream.infrared, 1)
        ir2 = prof.get_stream(rs.stream.infrared, 2)
        col = prof.get_stream(rs.stream.color)
        e12 = _extr(ir1, ir2)
        calib = {"serial": args.serial, "width": W, "height": H, "fps": F,
                 "baseline_m": abs(e12["t"][0]),
                 "left": _intr(ir1), "right": _intr(ir2), "color": _intr(col),
                 "left_to_right": e12, "left_to_color": _extr(ir1, col),
                 "depth_scale": None if args.no_depth else prof.get_device().first_depth_sensor().get_depth_scale(),
                 "captured": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(args.out, "calib.json"), "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
        K = calib["left"]["K"]
        print(f"baseline = {calib['baseline_m'] * 1000:.3f} mm, infra1 fx = {K[0][0]:.2f}, cx = {K[0][2]:.2f}, cy = {K[1][2]:.2f}")
        print(f"color fx = {calib['color']['K'][0][0]:.2f}, 畸變模型 = {calib['color']['model']}, 係數 = {calib['color']['dist']}")
        print(f"infra1→color 平移 = {calib['left_to_color']['t']}（應該 ≈ 0）")

        for _ in range(args.warmup):
            pipe.wait_for_frames()
        frames_meta, idx = [], 0
        while True:
            fs = pipe.wait_for_frames()
            c, l, r = fs.get_color_frame(), fs.get_infrared_frame(1), fs.get_infrared_frame(2)
            d = None if args.no_depth else fs.get_depth_frame()
            if not (c and l and r):
                continue
            c_img, l_img, r_img = (np.asanyarray(x.get_data()) for x in (c, l, r))
            save = args.n > 0
            if args.n == 0:
                vis = np.hstack([cv2.resize(c_img, (640, 360)), cv2.cvtColor(cv2.resize(l_img, (640, 360)), cv2.COLOR_GRAY2BGR)])
                cv2.putText(vis, f"saved {idx}   SPACE = save, q = quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("D405  color | infra1", vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                save = k == ord(" ")
            if save:
                stem = os.path.join(args.out, f"{idx:06d}")
                cv2.imwrite(stem + "_color.png", c_img)
                cv2.imwrite(stem + "_ir1.png", l_img)
                cv2.imwrite(stem + "_ir2.png", r_img)
                if d:
                    cv2.imwrite(stem + "_depth.png", np.asanyarray(d.get_data()))
                frames_meta.append({"idx": idx, "t_color_ms": c.get_timestamp(), "t_ir1_ms": l.get_timestamp(), "t_ir2_ms": r.get_timestamp()})
                idx += 1
                print(f"存第 {idx} 組；color 與 infra1 的時間差 = {c.get_timestamp() - l.get_timestamp():+.2f} ms")
                if args.n and idx >= args.n:
                    break
        with open(os.path.join(args.out, "frames.json"), "w", encoding="utf-8") as f:
            json.dump(frames_meta, f, indent=1)
        print(f"共 {idx} 組 → {args.out}")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
