#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 B：把資料夾裡的影像當成「假相機」發布成 sensor_msgs/Image（rgb8，同 realsense2_camera），不用 cv_bridge。

用途：
  - 沒有相機時測 handover_perception_node.py 的線上流程（ROS 傳輸、編碼、等新影像、回傳 JSON）
  - 在「手臂 container」跑這支、在 VRB container 跑 check_handover_env.py --topic，測跨 container 收不收得到（SHM / UID）

    source /opt/ros/humble/setup.bash
    python handover_pub_images.py --images my_needle_thread/Images/cam_260422273204_002901_1792135221.635050781 --fps 10
    python handover_pub_images.py --images one.png --topic handover/test_image --fps 30      # 一張圖重複發
手臂 container 的系統 python 只需要 numpy + cv2 + rclpy（都有）。
"""

import argparse
import array
import glob
import os
import time

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image

EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--topic", default="handover/test_image")
    ap.add_argument("--frame_id", default="eth_camera_color_optical_frame")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--hold", type=int, default=1, help="每張圖重複發幾次（模擬手臂靜止）")
    ap.add_argument("--loop", type=int, default=1, help="1 = 播完從頭再來")
    ap.add_argument("--encoding", default="rgb8", choices=["rgb8", "bgr8"])
    a = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(a.images, "**", "*"), recursive=True) if p.lower().endswith(EXTS)) \
        if os.path.isdir(a.images) else [a.images]
    if not paths:
        raise SystemExit(f"{a.images} 沒有影像")
    rclpy.init()
    node = rclpy.create_node("handover_fake_camera")
    pub = node.create_publisher(Image, a.topic, 5)
    node.get_logger().info(f"{len(paths)} 張 → {a.topic} @ {a.fps} Hz（{a.encoding}）")
    dt = 1.0 / a.fps
    try:
        while rclpy.ok():
            for p in paths:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if a.encoding == "rgb8":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for _ in range(a.hold):
                    m = Image()
                    m.header.stamp = node.get_clock().now().to_msg()
                    m.header.frame_id = a.frame_id
                    m.height, m.width = img.shape[:2]
                    m.encoding, m.is_bigendian, m.step = a.encoding, 0, img.shape[1] * 3
                    m.data = array.array("B", np.ascontiguousarray(img).tobytes())
                    pub.publish(m)
                    rclpy.spin_once(node, timeout_sec=0)
                    time.sleep(dt)
                print(f"\r{os.path.basename(p)}", end="", flush=True)
            if not a.loop:
                break
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
