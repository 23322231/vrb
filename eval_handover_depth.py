#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 C：比較三種深度（FoundationStereo／D405 原生深度／遞針手指尖平面），資料由 capture_handover_depth.py 錄。

    cd /workspaces/vrb
    python eval_handover_depth.py --selftest                                  # 只測幾何（不需要 GPU）
    python eval_handover_depth.py --session captures/20261118                 # 全部情況
    python eval_handover_depth.py --session captures/20261118 --needle_arc_mm 26 --needle_circle 3/8   # 加弦長檢查

每一組影像：B 的感知（和 node 相同的預設：vrb_anchor、YOLO imgsz 1280 conf 0.1）找出兩個夾取點的像素
（遞針手自己夾的點 + 接針手要夾的點），再用三種方法給 3D：
  fs      FoundationStereo（handover_depth.StereoDepth，ROI + 放大 + mask 前景中位數）
  native  D405 原生深度（depth/image_rect_raw，和 fs 用同一個查法）
  plane   D 的深度備案：像素射線與「通過遞針手指尖、垂直光軸」的平面相交（和 vrb_handover/geometry.py 相同的算法）

和三種真值比：
  T1 遞針手自己夾的點 ≈ 遞針手指尖（校正 + tool_pose + 指尖偏移）。看 fs／native 有沒有整體偏差。plane 在這點上結構上必然對，不列。
  T2 位移：同一個情況的 pos_01、pos_02… 相對 pos_00，用 tool_pose 的剛體變換預測點的新位置，比預測和量到的差。
     不受校正的平移誤差影響。plane 對「純平移」結構上必然一致，所以 T2 只看 fs／native。
  T3 （有 touch.json 時）接針手要夾的點 ≈ 接針手對點時的指尖。這是實際要去的位置，三種方法都比。最有參考價值。
  另外：弦長（兩個夾取點的 3D 距離 vs 針的實際弦長，需要 --needle_arc_mm／--needle_circle 或 --chord_mm），三種方法都比。

輸出：<session>/depth_eval_<時間>/ 的 frames.csv、summary.json、summary.md、vis/*.png
"""

import argparse
import csv
import datetime
import glob
import json
import math
import os
import sys
import time
from fractions import Fraction

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from capture_handover_depth import CASES, GRIP, fingertip_cam, pose7_to_T  # noqa: E402
from handover_depth import D405Calib, StereoDepth, lookup_disparity, make_backend  # noqa: E402

METHODS = ("fs", "native", "plane")
POINT_FRAC = {"head_1_5": 0.2, "tail_1_3": 2 / 3}


# =============================================================================
# 幾何（selftest 測這些）
# =============================================================================
def plane_point(uv, K, z):
    """D 的深度備案（vrb_handover/geometry.pixel_to_point_at_depth）：不去畸變的針孔射線 ∩ z = 常數。"""
    K = np.asarray(K, float).reshape(3, 3)
    return np.array([(uv[0] - K[0, 2]) / K[0, 0] * z, (uv[1] - K[1, 2]) / K[1, 1] * z, z])


def rigid_predict(p0_cam, pose_a, pose_b, T_cam2base):
    """點跟著法蘭剛體移動：法蘭從 pose_a 到 pose_b 時，相機座標的點 p0 會到哪裡。"""
    T = np.asarray(T_cam2base)
    M = np.linalg.inv(T) @ pose7_to_T(pose_b) @ np.linalg.inv(pose7_to_T(pose_a)) @ T
    return (M @ np.r_[p0_cam, 1.0])[:3]


def needle_chord(arc_mm, circle, f1=0.2, f2=2 / 3):
    """圓弧針（弧長 arc_mm、佔圓的 circle，例如 3/8）上，沿弧長比例 f1、f2 兩點的直線距離。"""
    theta = 2 * math.pi * float(Fraction(circle))
    R = arc_mm / theta
    return 2 * R * math.sin(abs(f2 - f1) * theta / 2)


def guess_depth_scale(raw):
    """realsense 原始深度值的單位：D405 的 depth_units 可能是 0.0001 m 或 0.001 m。手臂工作距離 0.07–0.6 m。"""
    v = raw[raw > 0]
    if v.size < 100:
        return None
    med = float(np.median(v))
    return 1e-4 if med > 650 else 1e-3


def ray_split(err, p_true):
    """誤差分成「沿視線（深度方向）」與「垂直視線」兩部分（mm）。"""
    r = np.asarray(p_true, float) / np.linalg.norm(p_true)
    along = float(np.dot(err, r))
    return along, float(np.linalg.norm(np.asarray(err) - along * r))


# =============================================================================
# 讀資料
# =============================================================================
def load_session(session):
    with open(os.path.join(session, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cases = {}
    for case in CASES:
        pos = sorted(glob.glob(os.path.join(session, case, "pos_*")))
        items = []
        for p in pos:
            fj = os.path.join(p, "frames.json")
            if not os.path.isfile(fj):
                print(f"略過 {p}（沒有 frames.json，錄影沒有完成）")
                continue
            with open(fj) as f:
                fr = json.load(f)
            touch = None
            if os.path.isfile(os.path.join(p, "touch.json")):
                with open(os.path.join(p, "touch.json")) as f:
                    touch = json.load(f)
            items.append({"dir": p, "name": os.path.basename(p), "frames": fr["frames"], "touch": touch})
        if items:
            cases[case] = items
    return meta, cases


def make_calib(meta, baseline=None):
    ci = meta["camera_info"]
    R, t = None, None
    ex = meta.get("extrinsics", {}).get("depth_to_color")
    if ex:                                                    # librealsense 的 rotation 是 column-major；depth frame = infra1
        R = np.asarray(ex["rotation"], float).reshape(3, 3).T
        t = np.asarray(ex["translation"], float)
    return D405Calib.from_camera_info(ci["ir1"], ci["ir2"], ci["color"], R_c_l=R, t_c_l=t, baseline=baseline)


# =============================================================================
# 主流程
# =============================================================================
def build_perception():
    import handover_perception as HP
    ap = argparse.ArgumentParser()
    HP.add_common_args(ap)
    a = ap.parse_args([])                                     # 用 B 的 CLI 預設 = node 的部署設定
    return HP.build_from_args(a), a


def run(args):
    torch = None
    if "fs" in args.methods and args.backend in ("fs", "ffs"):
        import torch
    meta, cases = load_session(args.session)
    if not cases:
        raise SystemExit(f"{args.session} 裡沒有錄好的位置")
    calib = make_calib(meta, args.baseline_m)
    cal_arm = meta["calib"]
    print(f"infra1 fx = {calib.fx:.2f}、baseline = {calib.baseline * 1000:.3f} mm、"
          f"infra1→color 平移 = {np.round(calib.t_c_l * 1000, 3).tolist()} mm（{'有' if meta.get('extrinsics', {}).get('depth_to_color') else '沒有'} extrinsics）")
    out = os.path.join(args.session, "depth_eval_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(out, "vis"), exist_ok=True)

    perc, pa = build_perception()
    print(f"感知：decode={pa.decode}、YOLO imgsz {pa.yolo_imgsz} conf {pa.yolo_conf}")
    sd = None
    if "fs" in args.methods:
        kw = {"valid_iters": args.iters} if args.backend in ("fs", "ffs") and args.iters else {}
        sd = StereoDepth(make_backend(args.backend, **kw), calib, upscale=args.upscale, radius=args.radius)
    depth_scale = args.depth_scale
    chord_true = args.chord_mm
    if chord_true is None and args.needle_arc_mm:
        chord_true = needle_chord(args.needle_arc_mm, args.needle_circle)
    if chord_true:
        print(f"兩個夾取點（弧長 0.2 與 2/3）的真實弦長 = {chord_true:.2f} mm")

    rows, times, mem = [], [], []
    for case, items in cases.items():
        holder = CASES[case]
        receiver = "left" if holder == "right" else "right"
        pt_hold, pt_grasp = GRIP[holder], GRIP[receiver]
        cal_h = cal_arm.get(holder)
        if cal_h is None:
            print(f"⚠ meta 裡沒有{holder}手的校正，{case} 的 plane 與 T1 會缺")
        for it in items:
            for k, fr in enumerate(it["frames"]):
                if args.every > 1 and k % args.every:
                    continue
                stem = os.path.join(it["dir"], f"{fr['i']:03d}")
                color = cv2.imread(stem + "_color.png")
                ir1, ir2 = cv2.imread(stem + "_ir1.png", 0), cv2.imread(stem + "_ir2.png", 0)
                raw = cv2.imread(stem + "_depth.png", cv2.IMREAD_UNCHANGED)
                if depth_scale is None and raw is not None:
                    depth_scale = guess_depth_scale(raw)
                    print(f"原生深度單位推測 = {depth_scale} m（不對的話用 --depth_scale 指定）")
                res, extra = perc.predict(color)
                base = {"case": case, "pos": it["name"], "frame": fr["i"], "perc_ok": bool(res.get("ok")),
                        "error": res.get("error", ""), "vrb_agree": res.get("vrb_agree"),
                        "grasp_conf": (res.get("vrb_anchor") or {}).get("grasp_conf"),
                        "span_ratio": (res.get("vrb_anchor") or {}).get("span_ratio")}
                hp = fr["poses"].get(holder)
                tip_h = None if (hp is None or cal_h is None) else fingertip_cam(hp["pose7"], cal_h)
                pts = {p["name"]: p["uv"] for p in res.get("points", [])}
                if not res.get("ok") or pt_hold not in pts or pt_grasp not in pts:
                    rows.append({**base, "point": None, "method": None})
                    continue
                det = extra.get("det") or {}
                mask, box = det.get("mask"), extra.get("box")
                uvs = [pts[pt_hold], pts[pt_grasp]]
                xyz = {m: [None, None] for m in METHODS}
                info = {m: [{}, {}] for m in METHODS}
                if sd is not None:
                    o = sd.locate(ir1, ir2, uvs, box_color=box, mask_color=mask)
                    for j in range(2):
                        xyz["fs"][j] = o[j]["xyz_cam"] if o[j]["ok"] else None
                        info["fs"][j] = {"ok": o[j]["ok"], "raw_xyz": o[j]["xyz_cam"], "sep_px": o[j]["sep_px"],
                                         "n": o[j]["n"], "mad_px": o[j]["mad_px"]}
                    times.append(sd.backend.last_time)
                    if torch is not None:
                        mem.append(torch.cuda.max_memory_allocated() / 2 ** 20)
                if "native" in args.methods and raw is not None:
                    z = raw.astype(np.float32) * depth_scale
                    with np.errstate(divide="ignore"):
                        disp = np.where(z > 0, calib.fx * calib.baseline / z, np.nan).astype(np.float32)
                    m_l = None if mask is None else calib.color_mask_to_left(mask)
                    uvl = calib.color_uv_to_left(uvs)
                    for j in range(2):
                        q = lookup_disparity(disp, uvl[j], m_l, radius=args.radius)
                        ok = q["disp"] is not None and (q["sep_px"] is None or q["sep_px"] >= 0.75)
                        p = None if q["disp"] is None else calib.left_uvz_to_color_xyz(uvl[j], calib.disp_to_z(q["disp"]))[0].tolist()
                        xyz["native"][j] = p if ok else None
                        info["native"][j] = {"ok": ok, "raw_xyz": p, "sep_px": q["sep_px"], "n": q["n"], "mad_px": q["mad_px"]}
                if "plane" in args.methods and tip_h is not None:
                    for j in range(2):
                        xyz["plane"][j] = plane_point(uvs[j], cal_h["camera_matrix"], tip_h[2]).tolist()
                        info["plane"][j] = {"ok": True}
                for j, (pname, role) in enumerate(((pt_hold, "held"), (pt_grasp, "grasp"))):
                    for m in METHODS:
                        if m not in args.methods:
                            continue
                        r = {**base, "point": pname, "role": role, "method": m, "u": uvs[j][0], "v": uvs[j][1],
                             "xyz": xyz[m][j], **{k2: v2 for k2, v2 in info[m][j].items()},
                             "tip_holder": None if tip_h is None else tip_h.tolist(),
                             "pose_holder": None if hp is None else hp["pose7"]}
                        rows.append(r)
                if k == 0:
                    save_vis(out, case, it["name"], color, uvs, (pt_hold, pt_grasp), xyz, sd, ir1, ir2, box)
            print(f"{case}/{it['name']}：{len(it['frames'])} 組完成")

    summ = summarize(rows, cases, meta, chord_true)
    if times:
        summ["fs_time_ms"] = {"first": round(times[0] * 1000, 1), "median_rest": round(float(np.median(times[1:] or times)) * 1000, 1)}
    if mem:
        summ["gpu_peak_mb_torch"] = round(max(mem), 0)
        free, total = torch.cuda.mem_get_info()
        summ["gpu_used_now_mb_device"] = round((total - free) / 2 ** 20, 0)
    summ.update(session=args.session, backend=args.backend, upscale=args.upscale, depth_scale=depth_scale,
                baseline_mm=calib.baseline * 1000, perception={"decode": pa.decode, "yolo_imgsz": pa.yolo_imgsz, "yolo_conf": pa.yolo_conf})
    with open(os.path.join(out, "frames.csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, ensure_ascii=False)
    md = to_markdown(summ)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"→ {out}")


def _stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": round(float(v.mean()), 2), "std": round(float(v.std()), 2),
            "rms": round(float(np.sqrt((v ** 2).mean())), 2), "max_abs": round(float(np.abs(v).max()), 2)}


def summarize(rows, cases, meta, chord_true):
    S = {"cases": {}}
    for case, items in cases.items():
        holder = CASES[case]
        receiver = "left" if holder == "right" else "right"
        cal_h, cal_r = meta["calib"].get(holder), meta["calib"].get(receiver)
        R = [r for r in rows if r["case"] == case]
        nfr = len({(r["pos"], r["frame"]) for r in R})
        nperc = len({(r["pos"], r["frame"]) for r in R if r["perc_ok"] and r.get("point")})
        C = {"holder": holder, "frames": nfr, "perception_ok": nperc, "points": {}}
        for role in ("held", "grasp"):
            pname = GRIP[holder] if role == "held" else GRIP[receiver]
            P = {}
            for m in METHODS:
                rr = [r for r in R if r.get("role") == role and r.get("method") == m]
                if not rr:
                    continue
                okr = [r for r in rr if r.get("xyz") is not None]
                d = {"valid_rate": round(len(okr) / len(rr), 3)}
                # 重複性：每個位置內的 std（mm），取各位置的中位數
                sd_z, sd_3 = [], []
                means = {}
                for it in items:
                    X = np.array([r["xyz"] for r in okr if r["pos"] == it["name"]], float)
                    n_all = sum(1 for r in rr if r["pos"] == it["name"])
                    if len(X) >= 3:
                        sd_z.append(X[:, 2].std() * 1000)
                        sd_3.append(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()) * 1000)
                    if len(X) >= max(3, 0.5 * n_all):              # 一半以上有效才拿來做位移檢查
                        means[it["name"]] = (X.mean(0), np.mean([r["pose_holder"] for r in okr if r["pos"] == it["name"]], 0))
                d["repeat_std_z_mm"] = round(float(np.median(sd_z)), 3) if sd_z else None
                d["repeat_rms_3d_mm"] = round(float(np.median(sd_3)), 3) if sd_3 else None
                # T1：遞針手自己夾的點 vs 遞針手指尖
                if role == "held" and m != "plane":
                    e = [(np.array(r["xyz"]) - np.array(r["tip_holder"])) * 1000 for r in okr if r.get("tip_holder")]
                    if e:
                        sp = [ray_split(x, r["tip_holder"]) for x, r in zip(e, [r for r in okr if r.get("tip_holder")])]
                        d["T1_vs_holder_tip"] = {"dz_mm": _stats([x[2] for x in e]), "along_ray_mm": _stats([a for a, _ in sp]),
                                                 "lateral_mm": _stats([b for _, b in sp]), "dist_mm": _stats([np.linalg.norm(x) for x in e])}
                # T2：位移（相對 pos_00）
                if m != "plane" and cal_h is not None and len(means) >= 2:
                    names = sorted(means)
                    p0, q0 = means[names[0]]
                    e3, ez, step = [], [], []
                    for n in names[1:]:
                        p, q = means[n]
                        pred = rigid_predict(p0, q0, q, cal_h["T_cam2base"])
                        err = (p - pred) * 1000
                        e3.append(float(np.linalg.norm(err)))
                        ez.append(float(err[2]))
                        step.append(float(np.linalg.norm(pred - p0) * 1000))
                    d["T2_displacement"] = {"n_moves": len(e3), "move_mm": [round(x, 1) for x in step],
                                            "err_3d_mm": [round(x, 2) for x in e3], "err_z_mm": [round(x, 2) for x in ez],
                                            "rms_3d_mm": round(float(np.sqrt(np.mean(np.square(e3)))), 2)}
                # T3：接針手對點
                if role == "grasp" and cal_r is not None:
                    e, sp = [], []
                    for it in items:
                        if not it["touch"]:
                            continue
                        X = [r["xyz"] for r in okr if r["pos"] == it["name"]]
                        if not X:
                            continue
                        truth = np.asarray(it["touch"].get("fingertip_cam") or fingertip_cam(it["touch"]["pose7"], cal_r))
                        x = (np.mean(X, 0) - truth) * 1000
                        e.append(x)
                        sp.append(ray_split(x, truth))
                    if e:
                        d["T3_vs_receiver_touch"] = {"n_touch": len(e), "dz_mm": [round(x[2], 2) for x in e],
                                                     "along_ray_mm": [round(a, 2) for a, _ in sp],
                                                     "lateral_mm": [round(b, 2) for _, b in sp],
                                                     "dist_mm": [round(float(np.linalg.norm(x)), 2) for x in e]}
                P[m] = d
            C["points"][f"{role}:{pname}"] = P
        # 弦長：同一組影像兩點都有值才算
        C["chord_mm"] = {}
        for m in METHODS:
            byf = {}
            for r in R:
                if r.get("method") == m and r.get("xyz") is not None:
                    byf.setdefault((r["pos"], r["frame"]), {})[r["role"]] = np.array(r["xyz"])
            ch = [np.linalg.norm(v["held"] - v["grasp"]) * 1000 for v in byf.values() if len(v) == 2]
            if ch:
                st = _stats(ch)
                if chord_true:
                    st["err_mean_mm"] = round(st["mean"] - chord_true, 2)
                C["chord_mm"][m] = st
        S["cases"][case] = C
    S["chord_true_mm"] = None if chord_true is None else round(chord_true, 2)
    return S


def to_markdown(S):
    L = [f"# 深度比較（{S.get('session', '')}）", ""]
    L.append(f"backend {S.get('backend')}、upscale {S.get('upscale')}、原生深度單位 {S.get('depth_scale')} m、baseline {S.get('baseline_mm', 0):.3f} mm")
    if "fs_time_ms" in S:
        L.append(f"立體匹配時間：第一次 {S['fs_time_ms']['first']} ms、之後中位數 {S['fs_time_ms']['median_rest']} ms")
    if "gpu_peak_mb_torch" in S:
        L.append(f"GPU 尖峰（torch，含 YOLO + VRB + 立體匹配）{S['gpu_peak_mb_torch']} MB、整張卡目前已用 {S['gpu_used_now_mb_device']} MB")
    L.append("")
    for case, C in S["cases"].items():
        L += [f"## {case}（遞針手 = {C['holder']}）", f"影像 {C['frames']} 組，感知成功 {C['perception_ok']} 組", ""]
        for key, P in C["points"].items():
            role, pname = key.split(":")
            L.append(f"### {pname}（{'遞針手自己夾的點' if role == 'held' else '接針手要夾的點'}）")
            L.append("| 方法 | 有效率 | 靜止 std z (mm) | T1 沿視線偏差 平均±std (mm) | T2 位移誤差 RMS (mm) | T3 對點：沿視線／垂直 (mm) |")
            L.append("|---|---|---|---|---|---|")
            for m, d in P.items():
                t1 = d.get("T1_vs_holder_tip", {}).get("along_ray_mm")
                t2 = d.get("T2_displacement")
                t3 = d.get("T3_vs_receiver_touch")
                L.append(f"| {m} | {d['valid_rate']:.0%} | {d['repeat_std_z_mm']} | "
                         f"{'-' if not t1 else str(t1['mean']) + '±' + str(t1['std'])} | "
                         f"{'-' if not t2 else t2['rms_3d_mm']} | "
                         f"{'-' if not t3 else ' / '.join(f'{a}/{b}' for a, b in zip(t3['along_ray_mm'], t3['lateral_mm']))} |")
            L.append("")
        if C["chord_mm"]:
            L.append(f"弦長（兩個夾取點的 3D 距離；真值 {S.get('chord_true_mm') or '未給'} mm）：" + "、".join(
                f"{m} {d['mean']}±{d['std']}" + (f"（差 {d['err_mean_mm']:+}）" if "err_mean_mm" in d else "") for m, d in C["chord_mm"].items()))
            L.append("")
    L += ["說明：plane 在「遞針手自己夾的點」與「純平移的位移檢查」上結構上必然正確，所以那兩欄不列；"
          "plane 的真正考驗是 T3（接針手要夾的點）與弦長。", ""]
    return "\n".join(L)


def save_vis(out, case, pos, color, uvs, names, xyz, sd, ir1, ir2, box):
    """第一組影像：needle 附近放大 + 每個點三種方法的 z。"""
    if box is None:
        return
    x1, y1, x2, y2 = [int(v) for v in box]
    pad = 60
    H, W = color.shape[:2]
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    crop = color[y1:y2, x1:x2].copy()
    s = 3.0 if max(x2 - x1, y2 - y1) < 320 else 2.0
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    for j, (uv, n) in enumerate(zip(uvs, names)):
        p = (int((uv[0] - x1) * s), int((uv[1] - y1) * s))
        cv2.drawMarker(crop, p, (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
        txt = "  ".join(f"{m}:{'-' if xyz[m][j] is None else f'{xyz[m][j][2] * 1000:.1f}'}" for m in METHODS)
        cv2.putText(crop, f"{n} z(mm) {txt}", (8, 24 + 24 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(crop, f"{n} z(mm) {txt}", (8, 24 + 24 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    panels = [crop]
    if sd is not None:
        res = sd.disparity(ir1, ir2, sd.roi_from_box(sd.calib.color_uv_to_left(np.reshape(box, (2, 2))).ravel()))
        full = res.full()[y1:y2, x1:x2]
        fin = np.isfinite(full)
        if fin.any():
            lo, hi = np.percentile(full[fin], [2, 98])
            v = cv2.applyColorMap(np.uint8(np.clip((np.nan_to_num(full, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1) * 255), cv2.COLORMAP_TURBO)
            panels.append(cv2.resize(v, crop.shape[1::-1], interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(os.path.join(out, "vis", f"{case}_{pos}.png"), np.hstack(panels))


# =============================================================================
# selftest
# =============================================================================
def selftest():
    from capture_handover_depth import R_to_tm_euler, quat_to_R
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'OK' if cond else 'FAIL'}] {name} {detail}")

    def tm_R(rx, ry, rz):                                     # TM：R = Rz·Ry·Rx
        cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
        return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))

    def R_to_q(R):                                            # 測試用：矩陣 → 四元數 (x, y, z, w)
        w = math.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        return [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]

    e = [-2.5588, -0.3428, 1.5401]                            # A 的樣本 calibration_01_pose 的姿態
    R = tm_R(*e)
    check("四元數 → 矩陣 → TM 歐拉角", np.allclose(R_to_tm_euler(quat_to_R(R_to_q(R))), e, atol=1e-9))

    # 指尖：A 的右手校正與指尖偏移、樣本姿態
    T_cb = np.eye(4)
    T_cb[:3, :3] = np.array([[-0.9550105386553671, 0.19269759608488451, -0.22543847834895486],
                             [0.29577928374879414, 0.6743965088995911, -0.6765382207156474],
                             [0.021667633978720957, -0.7127811622420659, -0.7010517301815993]])
    T_cb[:3, 3] = [-0.1233963633386425, 0.8388364946443314, 0.11270592586173903]
    cal = {"T_cam2base": T_cb.tolist(), "fingertip_offset": [0.00015, 0.02764, 0.21473]}
    pose = [-0.0678, 0.6456, 0.1896, *R_to_q(R)]
    tip = fingertip_cam(pose, cal)
    flange = pose7_to_T(pose)
    manual = np.linalg.inv(T_cb) @ (flange @ np.r_[0.00015, 0.02764, 0.21473, 1])
    check("指尖 = inv(T_cam2base)·T_flange·offset", np.allclose(tip, manual[:3]), f"z = {tip[2] * 1000:.1f} mm")
    check("指尖在相機前方、距離合理", 0.05 < tip[2] < 0.6, f"{np.round(tip * 1000, 1).tolist()} mm")

    # plane：和 D 的 pixel_to_point_at_depth 相同
    K = [[656.26, 0, 631.74], [0, 655.50, 362.46], [0, 0, 1]]
    p = plane_point([700, 300], K, 0.25)
    check("plane = D 的算法", np.allclose(p, [(700 - 631.74) / 656.26 * 0.25, (300 - 362.46) / 655.50 * 0.25, 0.25]))

    # 剛體預測：法蘭平移 + 旋轉，針上的點跟著動
    P_base0 = flange @ np.r_[0.01, 0.03, 0.22, 1]
    pose_b = [-0.0578, 0.6356, 0.2096, *R_to_q(tm_R(e[0] + 0.05, e[1] - 0.03, e[2] + 0.1))]
    P_base1 = pose7_to_T(pose_b) @ np.r_[0.01, 0.03, 0.22, 1]
    cam = lambda Pb: (np.linalg.inv(T_cb) @ Pb)[:3]  # noqa: E731
    pred = rigid_predict(cam(P_base0), pose, pose_b, T_cb)
    check("位移的剛體預測", np.allclose(pred, cam(P_base1), atol=1e-12))

    # 深度單位推測
    check("深度單位 0.1 mm", guess_depth_scale(np.full((10, 20), 2500, np.uint16)) == 1e-4)
    check("深度單位 1 mm", guess_depth_scale(np.full((10, 20), 250, np.uint16)) == 1e-3)

    # 弦長：半圓針，0 與 1 兩端 = 直徑
    check("弦長公式（半圓兩端 = 直徑）", abs(needle_chord(math.pi * 10, "1/2", 0, 1) - 20) < 1e-9)
    print(f"  例：弧長 26 mm、3/8 圓的針，0.2 與 2/3 兩點弦長 = {needle_chord(26, '3/8'):.2f} mm")

    a, b = ray_split(np.array([0, 0, 3.0]), [0, 0, 0.3])
    check("誤差拆成沿視線／垂直", abs(a - 3) < 1e-12 and abs(b) < 1e-12)
    print("selftest", "全部通過" if ok else "有失敗")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--session")
    ap.add_argument("--methods", default="fs,native,plane")
    ap.add_argument("--backend", default="fs", choices=["fs", "ffs", "sgbm"])
    ap.add_argument("--iters", type=int, default=0, help="refine 次數（預設 fs 32、ffs 8）")
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--every", type=int, default=1, help="每個位置每幾組取一組")
    ap.add_argument("--depth_scale", type=float, help="原生深度的單位（公尺），不給就自動推測")
    ap.add_argument("--baseline_m", type=float, help="infra2 camera_info 沒有 baseline 時用")
    ap.add_argument("--chord_mm", type=float, help="兩個夾取點的實際直線距離（量的）")
    ap.add_argument("--needle_arc_mm", type=float, help="針的弧長（包裝上的長度）")
    ap.add_argument("--needle_circle", default="3/8", help="針佔圓的比例，例如 3/8、1/2")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.session:
        ap.error("要給 --session（或 --selftest）")
    args.methods = [m.strip() for m in args.methods.split(",")]
    run(args)


if __name__ == "__main__":
    main()
