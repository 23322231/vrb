#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 B：呼叫 handover/predict_needle（std_srvs/Trigger）N 次，量延遲、成功率，以及 tail_1_3 的重複性。
針與手臂都靜止時，tail_1_3 的標準差就是「線上」的抖動（相機雜訊 + YOLO 框變動 + VRB）。

    source /opt/ros/humble/setup.bash
    python handover_call_predict.py --n 20 --interval 0.5
    python handover_call_predict.py --n 1 --print_json           # 看一次完整 JSON（給 D 對介面）

結果另存 pred_handoff_output/calls_<時間>.jsonl（每行一個回傳的 JSON）。
不需要 torch；手臂 container 的系統 python 也能跑（D 可以拿來測連線）。
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import rclpy
from std_srvs.srv import Trigger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default="handover/predict_needle")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--print_json", action="store_true")
    ap.add_argument("--out", default="pred_handoff_output")
    a = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("handover_predict_client")
    cli = node.create_client(Trigger, a.service)
    if not cli.wait_for_service(timeout_sec=10.0):
        raise SystemExit(f"10 秒內找不到服務 {a.service}")
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "calls_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl")
    lat, uv, conf, ok, server_ms = [], [], [], 0, []
    with open(log, "w") as f:
        for i in range(a.n):
            t = time.perf_counter()
            fut = cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, fut, timeout_sec=a.timeout)
            dt = (time.perf_counter() - t) * 1e3
            if not fut.done() or fut.result() is None:
                print(f"[{i + 1}] 逾時")
                continue
            r = fut.result()
            try:
                res = json.loads(r.message)
            except json.JSONDecodeError:
                print(f"[{i + 1}] message 不是 JSON：{r.message[:200]}")
                continue
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            lat.append(dt)
            q = next((p for p in res.get("points", []) if p["name"] == "tail_1_3"), None)
            if r.success and q:
                ok += 1
                uv.append(q["uv"])
                conf.append(q["conf"])
                server_ms.append((res.get("timing_ms") or {}).get("total", np.nan))
            print(f"[{i + 1}] success={r.success} {res.get('error')} 往返 {dt:.0f} ms  server={res.get('timing_ms')} "
                  f"wait={(res.get('latency_ms') or {}).get('wait_frame')} tail_1_3={q and q['uv']} conf={q and q['conf']} "
                  f"yolo={(res.get('yolo') or {}).get('conf')}")
            if a.print_json:
                print(json.dumps(res, indent=1, ensure_ascii=False))
            time.sleep(a.interval)
    print(f"\n成功 {ok}/{a.n}；往返延遲 中位數 {np.median(lat):.0f} ms、最大 {np.max(lat):.0f} ms" if lat else "沒有任何回應")
    if uv:
        u = np.asarray(uv)
        print(f"推論時間（server 端 YOLO+VRB）中位數 {np.nanmedian(server_ms):.0f} ms")
        print(f"tail_1_3 重複性（靜止時）：std u={u[:, 0].std():.2f} px, v={u[:, 1].std():.2f} px，"
              f"離平均最遠 {np.linalg.norm(u - u.mean(0), axis=1).max():.2f} px；信心 {np.min(conf):.2f}~{np.max(conf):.2f}")
    print(f"→ {log}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
