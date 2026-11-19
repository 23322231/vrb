#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 B（任務書 §5-B-4）：檢查 VRB container 能不能跑感知 node。

    source /opt/ros/humble/setup.bash
    python check_handover_env.py                                   # 只檢查 Python 環境
    python check_handover_env.py --topic <EtH color topic>         # 再實際訂閱 5 秒，看收不收得到、多快
    python check_handover_env.py --topic handover/test_image       # 搭配 handover_pub_images.py 測跨 container

檢查：
  1. 同一個 python 能不能同時 import rclpy / sensor_msgs / std_srvs 與 torch(CUDA) / ultralytics
     （ROS 的訊息 C 擴充是用 NumPy 1.x 編的，venv 的 numpy 變成 2.x 會 import 失敗）
  2. ROS_DOMAIN_ID、RMW_IMPLEMENTATION、ROS_LOCALHOST_ONLY、FASTRTPS_DEFAULT_PROFILES_FILE
  3. 目前的 UID，以及 /dev/shm 裡 FastDDS 共享記憶體檔案的擁有者（UID 不同 = 可能收不到 SHM 傳的影像）
  4. （--topic）publisher 是誰、QoS、實際收到幾張、頻率、encoding、解析度
兩個 container 都跑一次（手臂 container 用系統 python 跑也可以，只有第 1 項會不同），把輸出貼給統籌。
"""

import argparse
import collections
import glob
import os
import pwd
import sys
import time


def section(t):
    print(f"\n=== {t} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default=None)
    ap.add_argument("--secs", type=float, default=5.0)
    a = ap.parse_args()

    section("1. Python 環境")
    print("python:", sys.executable, sys.version.split()[0])
    ok = True
    for mod in ["numpy", "cv2", "torch", "torchvision", "ultralytics", "rclpy", "sensor_msgs.msg", "std_srvs.srv", "std_msgs.msg"]:
        try:
            m = __import__(mod, fromlist=["_"])
            print(f"  {mod:16s} OK  {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {mod:16s} 失敗：{type(e).__name__}: {e}")
    try:
        import numpy as np
        if int(np.__version__.split(".")[0]) >= 2:
            print("  [!] numpy >= 2：ROS 2 Humble 的訊息 C 擴充是 NumPy 1.x 編的，可能 import 失敗或 CameraInfo 出錯 → pip install 'numpy<2'")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        print(f"  CUDA available={torch.cuda.is_available()}", end="")
        if torch.cuda.is_available():
            print(f"  {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))} "
                  f"arch_list={torch.cuda.get_arch_list()[-3:]}  free/total="
                  f"{[round(x / 2 ** 30, 2) for x in torch.cuda.mem_get_info()]} GB")
            torch.zeros(1, device="cuda")
        else:
            print()
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  CUDA 測試失敗：{e}")
    try:
        from sensor_msgs.msg import CameraInfo, Image
        ci = CameraInfo()
        ci.k = [1.0] * 9
        Image()
        print("  ROS 訊息（含 numpy 欄位）建立 OK")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  ROS 訊息建立失敗：{e}")

    section("2. ROS 環境變數")
    for k in ["ROS_DISTRO", "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "ROS_LOCALHOST_ONLY", "FASTRTPS_DEFAULT_PROFILES_FILE",
              "FASTDDS_DEFAULT_PROFILES_FILE", "RMW_FASTRTPS_USE_QOS_FROM_XML", "CYCLONEDDS_URI"]:
        print(f"  {k:32s} = {os.environ.get(k, '（未設定）')}")
    print("  ※ 兩個 container 的 ROS_DOMAIN_ID（未設定 = 0）與 RMW 要一致")

    section("3. UID 與 FastDDS 共享記憶體")
    uid = os.getuid()
    print(f"  uid={uid} ({pwd.getpwuid(uid).pw_name})  gid={os.getgid()}")
    owners = collections.Counter()
    for p in glob.glob("/dev/shm/fastrtps_*") + glob.glob("/dev/shm/fast_datasharing*") + glob.glob("/dev/shm/sem.fastrtps_*"):
        try:
            st = os.stat(p)
            owners[(st.st_uid, oct(st.st_mode & 0o777))] += 1
        except OSError:
            pass
    if owners:
        for (u, mode), n in owners.items():
            try:
                name = pwd.getpwuid(u).pw_name
            except KeyError:
                name = "?"
            print(f"  /dev/shm FastDDS 檔案：uid {u} ({name}) 權限 {mode} × {n}" + ("" if u == uid else "   ← 和本 container UID 不同"))
        if any(u != uid for (u, _) in owners):
            print("  [!] 有其他 UID 建的 SHM 檔：若訂閱收不到影像，兩邊都改用 UDP（fastdds_udp_only.xml），或讓兩個 container 用相同 UID")
    else:
        print("  /dev/shm 沒有 FastDDS 檔案（沒有 node 在跑，或 /dev/shm 沒共用）")

    if not a.topic:
        print("\n" + ("環境 OK" if ok else "環境有問題，見上面"))
        return

    section(f"4. 訂閱 {a.topic} {a.secs}s")
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    rclpy.init()
    node = rclpy.create_node("handover_env_check")
    t_end = time.time() + 2.0
    while time.time() < t_end:                      # 等 discovery
        rclpy.spin_once(node, timeout_sec=0.1)
    names = dict(node.get_topic_names_and_types())
    key = a.topic if a.topic.startswith("/") else "/" + a.topic
    print(f"  topic 存在：{key in names}  型別：{names.get(key)}")
    if key not in names:
        cams = [t for t in names if "image" in t or "camera" in t]
        print(f"  看得到的影像相關 topic：{cams[:30]}")
    for info in node.get_publishers_info_by_topic(key):
        q = info.qos_profile
        print(f"  publisher: {info.node_namespace}/{info.node_name}  reliability={q.reliability.name} "
              f"durability={q.durability.name} depth={q.depth}")
    stamps, meta = [], {}

    def cb(m):
        stamps.append(time.time())
        meta.update(encoding=m.encoding, size=(m.width, m.height), frame_id=m.header.frame_id, step=m.step)
    node.create_subscription(Image, a.topic, cb, qos_profile_sensor_data)
    t_end = time.time() + a.secs
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    n = len(stamps)
    hz = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else 0
    print(f"  收到 {n} 張，{hz:.1f} Hz，{meta}")
    if n == 0:
        print("  [!] 收不到：檢查 (a) topic 名稱 (b) ROS_DOMAIN_ID (c) 第 3 項的 UID / SHM；"
              "可試 FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/fastdds_udp_only.xml 再跑一次")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
