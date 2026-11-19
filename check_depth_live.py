#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 C（精簡驗證）：FoundationStereo 算出的針深度，和夾著針的那隻手的指尖深度是否相同。不存任何檔案。

前提：
  - 手臂 container：EtH 相機 launch + 兩手 tm_driver 開著；遞針手夾著針停在縫合墊上方（針頭朝畫面左邊）。
  - VRB container 另一個終端機：B 的感知 node 開 FoundationStereo
        python handover_perception_node.py --ros-args -p depth_backend:=fs

    source /opt/ros/humble/setup.bash && cd /workspaces/vrb
    python check_depth_live.py --holder right          # 右手夾針（夾在 head 1/5）
    python check_depth_live.py --holder left           # 左手夾針（夾在 tail_1_3）

每次呼叫 handover/predict_needle，印：
  夾持點（右手 → head_1_5、左手 → tail_1_3）的 FoundationStereo 深度 z、同一時刻遞針手指尖的 z（A 的校正 + tool_pose
  + 指尖偏移）、兩者的差，以及影像上兩者差幾個 px（夾持點本來就該在指尖上，px 差很大代表點找錯，不是深度的問題）。
  另外印接針手要夾的那一點的深度（只是參考，它本來就不一定和指尖一樣深）。
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_handover_depth import DEFAULT_CALIB, GRIP, POSE, fingertip_cam, load_calib  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holder", choices=["right", "left"], default="right", help="哪隻手夾著針")
    ap.add_argument("--n", type=int, default=10, help="呼叫幾次")
    ap.add_argument("--interval", type=float, default=1.0, help="每次之間停幾秒")
    ap.add_argument("--service", default="handover/predict_needle")
    ap.add_argument("--calib", help=f"遞針手的校正檔（預設右 {DEFAULT_CALIB['right'][0]}、左 {DEFAULT_CALIB['left'][0]}）")
    ap.add_argument("--offset", help="遞針手的指尖偏移檔（預設同資料夾的 fingertip_offset_*.yaml）")
    args = ap.parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from std_srvs.srv import Trigger

    holder = args.holder
    receiver = "left" if holder == "right" else "right"
    cal = load_calib(args.calib or DEFAULT_CALIB[holder][0], args.offset or DEFAULT_CALIB[holder][1])
    K = np.asarray(cal["camera_matrix"], float)
    print(f"遞針手 = {holder}（夾 {GRIP[holder]}），接針手 = {receiver}（要夾 {GRIP[receiver]}）")
    print(f"校正 {cal['calib_yaml']}（{cal['calibration_date']}），指尖偏移 {np.round(np.array(cal['fingertip_offset']) * 1000, 1).tolist()} mm")

    rclpy.init()
    node = rclpy.create_node("check_depth_live")
    last = {}

    def on_pose(m):
        p, q = m.pose.position, m.pose.orientation
        last["pose7"] = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        last["t"] = time.time()
    node.create_subscription(PoseStamped, POSE[holder], on_pose, qos_profile_sensor_data)
    cli = node.create_client(Trigger, args.service)
    print(f"等服務 {args.service} …")
    if not cli.wait_for_service(timeout_sec=20.0):
        raise SystemExit("等不到感知服務：B 的 node 有開嗎？")
    t0 = time.time()
    while "pose7" not in last and time.time() - t0 < 5:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "pose7" not in last:
        raise SystemExit(f"收不到 {POSE[holder]}：{holder}手的 tm_driver 有開嗎？")

    hdr = f"{'#':>2} {'夾持點 z':>9} {'指尖 z':>8} {'差 z':>7} {'3D 差':>7} {'影像差':>7} │ {'接針點 z':>9} {'深度 ms':>7}  備註"
    print(hdr)
    dz_all, d3_all, t_all = [], [], []
    for i in range(args.n):
        fut = cli.call_async(Trigger.Request())
        t_call = time.time()
        rclpy.spin_until_future_complete(node, fut, timeout_sec=30.0)
        rtt = time.time() - t_call
        rclpy.spin_once(node, timeout_sec=0.05)
        if not fut.done() or fut.result() is None:
            print(f"{i + 1:>2} 服務沒有回應（30 s）")
            continue
        try:
            res = json.loads(fut.result().message)
        except ValueError:
            print(f"{i + 1:>2} 回傳不是 JSON：{fut.result().message[:200]}")
            continue
        if not res.get("ok"):
            print(f"{i + 1:>2} 感知失敗：{res.get('error')} {res.get('detail', '')}")
            time.sleep(args.interval)
            continue
        pts = {p["name"]: p for p in res.get("points", [])}
        ph, pr = pts.get(GRIP[holder]), pts.get(GRIP[receiver])
        tip = fingertip_cam(last["pose7"], cal)
        note = []
        if time.time() - last.get("t", 0) > 0.5:
            note.append("tool_pose 不新鮮")
        if res.get("vrb_agree") is False:
            note.append("VRB 方向和「針頭朝左」不一致")
        if res.get("depth_error"):
            note.append(f"depth_error: {res['depth_error']}")
        tip_uv = K @ tip
        tip_uv = tip_uv[:2] / tip_uv[2]
        duv = None if ph is None else math.hypot(ph["uv"][0] - tip_uv[0], ph["uv"][1] - tip_uv[1])
        xh = None if ph is None else ph.get("xyz_cam")
        xr = None if pr is None else pr.get("xyz_cam")
        tdep = (res.get("timing_ms") or {}).get("depth")
        if xh is None:
            note.append("夾持點沒有深度（FoundationStereo 判為無效，或 node 沒開 depth_backend）")
            print(f"{i + 1:>2} {'-':>9} {tip[2] * 1000:>8.1f} {'-':>7} {'-':>7} {duv if duv is None else round(duv):>7} │ "
                  f"{'-' if xr is None else f'{xr[2] * 1000:.1f}':>9} {tdep if tdep is not None else '-':>7}  {'；'.join(note)}")
        else:
            dz = (xh[2] - tip[2]) * 1000
            d3 = float(np.linalg.norm(np.array(xh) - tip) * 1000)
            dz_all.append(dz)
            d3_all.append(d3)
            print(f"{i + 1:>2} {xh[2] * 1000:>9.1f} {tip[2] * 1000:>8.1f} {dz:>+7.1f} {d3:>7.1f} {round(duv):>5}px │ "
                  f"{'-' if xr is None else f'{xr[2] * 1000:.1f}':>9} {tdep if tdep is not None else '-':>7}  {'；'.join(note)}")
        if tdep is not None:
            t_all.append(tdep)
        if i == 0 and res.get("debug_dir"):
            print(f"   疊圖：{res['debug_dir']}（往返 {rtt:.2f} s）")
        time.sleep(args.interval)

    print("單位：mm（z = 相機光軸方向的距離）。差 z = FoundationStereo − 指尖。")
    if dz_all:
        dz, d3 = np.array(dz_all), np.array(d3_all)
        print(f"有深度 {len(dz)}/{args.n} 次；差 z 平均 {dz.mean():+.1f}、標準差 {dz.std():.1f}、最大 |差| {np.abs(dz).max():.1f}；"
              f"3D 差平均 {d3.mean():.1f}")
        print("判讀：差 z 在幾 mm 內（指尖偏移與夾的位置本身就有 2–4 mm 不確定）→ FoundationStereo 在針上可用；"
              "差到 1 cm 以上，或常常沒有深度 → 不能直接用，貼結果給 C。")
    if t_all:
        print(f"深度計算時間中位數 {np.median(t_all):.0f} ms")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
