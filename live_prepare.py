#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 handover 實機測試時感知服務存下的影像（pred_handoff_output/<時間>/input.png）整理成 VRB／YOLO 的補訓資料。
原本的訓練資料（handover_train_masks/、needle5_handover_train/ …）完全不動；pred_handoff_output 只讀、只「複製」。

流程（每張）：
    1. input.png 複製到 handover_live_frames/<clip>/<6 位數序號>_<影像 header 時間戳>.png（和原本錄影影格的命名規則相同）
    2. 用新 YOLO（imgsz 1280、conf 0.1，和感知服務相同的過濾）預測針的 mask，當作 ground truth：
       所有通過過濾的偵測取聯集 → 依信心最高的偵測選出「那一根針」（和感知服務的幾何解碼相同）→
       只保留屬於這根針的碎片，寫成 sam2_video_label.py 的格式：
           handover_live_masks/Images/<clip>/<檔名>.png        原圖（複製）
           handover_live_masks/Labels/<clip>/<檔名>.png        mask：針 = BGR (0,0,255)，其餘黑色
           handover_live_masks/overlay/<clip>/<檔名>.png       疊圖（檢查用）
           handover_live_masks/sessions/<clip>.json            讓 sam2_video_label.py 認得這段 clip（之後想手修 mask 用）
    3. 之後用 mask_to_needle5.py（--suggest_rule left / --auto_rule left）從 mask 算出頭、尾、中間等點

clip 名稱：handover_<日期>_<train|test>_<left|right>（left／right = 遞針手，來自手臂端 rounds.jsonl）
測試集：「頭尾判斷失敗」的場景（vrb_agree = false 方向錯誤，或 span_ratio < 0.8 頭尾塌縮）。
    和失敗那張「完全相同的畫面」（同一隻手、前後兩張在針附近的差異 < --scene_px 像素，預設 100 = 連按看針、畫面沒變）
    一起放進測試集，避免一模一樣的影像同時出現在訓練集和測試集。
不去重：所有影像都加入（使用者決定）。manifest.csv 仍會標出「和前一張完全相同」的影像（dup_of），之後想去重可以用。

用法（VRB container 內，/workspaces/vrb）：
    python live_prepare.py                     # 只列清單（dry run）：寫 handover_live_frames/manifest_preview.csv，不複製
    python live_prepare.py --run               # 真的複製並產生 YOLO mask
    python live_prepare.py --run --dates 20261123   # 只處理某幾天
"""

import argparse
import csv
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEEDLE_BGR = (0, 0, 255)
THREAD_BGR = (255, 0, 255)


def load_plan(path, dates):
    plan = json.load(open(path, encoding="utf-8"))
    return [p for p in plan if p["date"] in dates]


def gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if img is None else cv2.GaussianBlur(img.astype(np.float32), (5, 5), 0)


def box_of(res):
    y = res.get("yolo") or {}
    return y.get("box_xyxy")


def diff_px(ga, gb, ba, bb, pad=60):
    """兩張圖在針附近（兩個 YOLO 框的聯集外擴 pad）差異 > 35 灰階的像素數。"""
    bs = [b for b in (ba, bb) if b]
    if not bs:
        x1, y1, x2, y2 = 0, 0, ga.shape[1], ga.shape[0]
    else:
        x1 = max(0, int(min(b[0] for b in bs)) - pad)
        y1 = max(0, int(min(b[1] for b in bs)) - pad)
        x2 = min(ga.shape[1], int(max(b[2] for b in bs)) + pad)
        y2 = min(ga.shape[0], int(max(b[3] for b in bs)) + pad)
    A, B = ga[y1:y2, x1:x2], gb[y1:y2, x1:x2]
    B = B * (A.mean() / max(B.mean(), 1.0))
    return int((np.abs(A - B) > 35).sum())


def needle_mask_from_yolo(perc, G, img_bgr):
    """和感知服務相同：通過過濾的 YOLO mask 取聯集，選出信心最高偵測所在的那根針，只保留那根針的碎片。"""
    dets = perc.detect(img_bgr)
    det, _ = perc.select(dets)
    if det is None or det.get("mask") is None:
        return None, None, len(dets)
    union = np.zeros(img_bgr.shape[:2], bool)
    for d in perc.valid(dets):
        if d.get("mask") is not None:
            union |= d["mask"]
    nd = G.centerline(union, det["mask"])
    if nd is None:
        return None, None, len(dets)
    line = np.zeros(union.shape, np.uint8)
    cv2.polylines(line, [np.round(nd["poly"]).astype(np.int32)], False, 255, 1)
    near = cv2.distanceTransform(255 - line, cv2.DIST_L2, 3) <= 12           # 中心線 12 px 內
    n, lab = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
    keep = np.zeros(union.shape, bool)
    for i in range(1, n):
        comp = lab == i
        if (comp & near).sum() >= 0.5 * comp.sum():
            keep |= comp
    thr = perc.thread_mask(dets)
    if thr is not None:
        thr = thr & ~keep
    return (keep if keep.any() else None), thr, len(dets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(HERE, "handover_live_plan.json"),
                    help="每張的遞針手、頭尾判斷是否失敗（由 rounds.jsonl 與 result.json 整理）")
    ap.add_argument("--pred_root", default=os.path.join(HERE, "pred_handoff_output"))
    ap.add_argument("--frames_out", default=os.path.join(HERE, "handover_live_frames"))
    ap.add_argument("--masks_out", default=os.path.join(HERE, "handover_live_masks"))
    ap.add_argument("--color_txt", default=os.path.join(HERE, "handover_train_masks", "Color_code.txt"))
    ap.add_argument("--dates", nargs="+", default=["20261119", "20261120", "20261123"])
    ap.add_argument("--yolo", default=None, help="YOLO 權重（預設 = 感知服務的 DEFAULT_YOLO）")
    ap.add_argument("--yolo_imgsz", type=int, default=1280)
    ap.add_argument("--yolo_conf", type=float, default=0.1)
    ap.add_argument("--scene_px", type=int, default=100,
                    help="前後兩張差異 < 這個像素數 = 同一組（預設 100 = 只把『完全相同的畫面』和失敗那張一起放進測試集；"
                         "改 3000 = 連『只動了一點針／線』的也一起放進測試集，比較嚴格但測試集會變 103 張）")
    ap.add_argument("--dup_px", type=int, default=100, help="差異 < 這個像素數 = 完全相同的畫面（只標記，不刪）")
    ap.add_argument("--run", action="store_true", help="真的複製與產生 mask（不加 = 只列清單）")
    ap.add_argument("--overwrite", action="store_true", help="目的地已有同名檔時覆蓋（預設停止）")
    a = ap.parse_args()

    plan = load_plan(a.plan, set(a.dates))
    if not plan:
        sys.exit("plan 裡沒有這些日期的資料")
    plan.sort(key=lambda p: p["dir"])
    for p in plan:
        p["src"] = os.path.join(a.pred_root, p["dir"], "input.png")
        p["res"] = json.load(open(os.path.join(a.pred_root, p["dir"], "result.json"), encoding="utf-8"))
        if not os.path.isfile(p["src"]):
            sys.exit(f"找不到 {p['src']}")

    # 1) 場景分組（同一天、同一隻遞針手、前後差異 < scene_px）
    print(f"計算前後影像差異（{len(plan)} 張）…")
    prev_g = None
    scene, dup_of = 0, None
    for i, p in enumerate(plan):
        g = gray(p["src"])
        if i == 0 or p["date"] != plan[i - 1]["date"] or p["giver"] != plan[i - 1]["giver"]:
            d = None
        else:
            d = diff_px(prev_g, g, box_of(plan[i - 1]["res"]), box_of(p["res"]))
        if d is None or d >= a.scene_px:
            scene += 1
        p["scene"], p["diff_prev"] = scene, d
        p["dup_of"] = plan[i - 1]["dir"] if (d is not None and d < a.dup_px) else ""
        prev_g = g
    fail_scenes = {p["scene"] for p in plan if p["headtail_fail"]}
    for p in plan:
        p["split"] = "test" if p["scene"] in fail_scenes else "train"

    # 2) clip 與檔名
    idx = {}
    for p in plan:
        clip = f"handover_{p['date']}_{p['split']}_{p['giver']}"
        idx[clip] = idx.get(clip, 0) + 1
        stamp = float(p["res"].get("stamp") or 0.0)
        p["clip"], p["name"] = clip, f"{idx[clip]:06d}_{stamp:.9f}.png"

    os.makedirs(a.frames_out, exist_ok=True)
    fields = ["source_dir", "dest", "clip", "split", "giver", "giver_inferred", "scene", "headtail_fail",
              "vrb_agree", "span_ratio", "grasp_conf", "diff_prev_px", "dup_of", "round", "round_reason",
              "yolo_mask", "n_yolo_det"]

    def row(p, ymask="", ndet=""):
        return {"source_dir": p["dir"], "dest": f"{p['clip']}/{p['name']}", "clip": p["clip"], "split": p["split"],
                "giver": p["giver"], "giver_inferred": p["giver_inferred"], "scene": p["scene"],
                "headtail_fail": p["headtail_fail"], "vrb_agree": p["vrb_agree"], "span_ratio": p["span_ratio"],
                "grasp_conf": p["grasp_conf"], "diff_prev_px": p["diff_prev"], "dup_of": p["dup_of"],
                "round": p["round"], "round_reason": p["round_reason"], "yolo_mask": ymask, "n_yolo_det": ndet}

    from collections import Counter
    cnt = Counter((p["date"], p["split"], p["giver"]) for p in plan)
    print("\n日期      split  遞針手  張數")
    for k in sorted(cnt):
        print(f"{k[0]}  {k[1]:5s}  {k[2]:5s}  {cnt[k]}")
    print(f"共 {len(plan)} 張；場景 {scene} 個，其中 {len(fail_scenes)} 個有頭尾判斷失敗 → 測試集；"
          f"完全相同的畫面 {sum(1 for p in plan if p['dup_of'])} 張（只標記在 dup_of，不刪）")

    if not a.run:
        out = os.path.join(a.frames_out, "manifest_preview.csv")
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(row(p) for p in plan)
        print(f"\n[dry run] 清單：{out}\n確認後加 --run 才會複製與產生 mask")
        return

    # 3) 複製 + YOLO mask
    conflicts = [p for p in plan if os.path.exists(os.path.join(a.frames_out, p["clip"], p["name"]))]
    if conflicts and not a.overwrite:
        sys.exit(f"[停止] 目的地已有 {len(conflicts)} 個同名檔（例如 {conflicts[0]['clip']}/{conflicts[0]['name']}）。"
                 f"確定要覆蓋請加 --overwrite")
    sys.path.insert(0, HERE)
    import handover_perception as HP
    import needle_geom as G
    kw = {"yolo_weights": a.yolo} if a.yolo else {}
    perc = HP.NeedlePerception(**kw, yolo_imgsz=a.yolo_imgsz, yolo_conf=a.yolo_conf,
                               decode="geom", orient_mode="left", warmup=False)   # 和感知服務相同的 YOLO 設定
    os.makedirs(os.path.join(a.masks_out, "sessions"), exist_ok=True)
    cc = os.path.join(a.masks_out, "Color_code.txt")
    if not os.path.isfile(cc):
        shutil.copy2(a.color_txt, cc)
    rows, n_ok = [], 0
    for i, p in enumerate(plan):
        fdir = os.path.join(a.frames_out, p["clip"])
        os.makedirs(fdir, exist_ok=True)
        dst = os.path.join(fdir, p["name"])
        shutil.copy2(p["src"], dst)                                  # 複製，不移動
        img = cv2.imread(dst)
        mask, thr, ndet = needle_mask_from_yolo(perc, G, img)
        if mask is not None:
            stem = os.path.splitext(p["name"])[0]
            for sub in ("Images", "Labels", "overlay"):
                os.makedirs(os.path.join(a.masks_out, sub, p["clip"]), exist_ok=True)
            lab = np.zeros_like(img)
            if thr is not None:
                lab[thr] = THREAD_BGR                                  # 線也一起存（和原本標註格式相同）
            lab[mask] = NEEDLE_BGR
            cv2.imwrite(os.path.join(a.masks_out, "Labels", p["clip"], stem + ".png"), lab)
            shutil.copy2(dst, os.path.join(a.masks_out, "Images", p["clip"], p["name"]))
            ov = img.copy()
            ov[mask] = (0.4 * ov[mask] + 0.6 * np.array(NEEDLE_BGR)).astype(np.uint8)
            cv2.imwrite(os.path.join(a.masks_out, "overlay", p["clip"], stem + ".png"), ov)
            n_ok += 1
        rows.append(row(p, "yes" if mask is not None else "NO（YOLO 沒抓到針，這張不會進訓練）", ndet))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(plan)}（有 mask {n_ok}）")

    # sessions：讓 sam2_video_label.py 認得每段 clip、從 Labels 讀回 mask（之後想手修 mask 時用）
    for clip in sorted({p["clip"] for p in plan}):
        frames = sorted(f for f in os.listdir(os.path.join(a.frames_out, clip)) if f.endswith(".png"))
        with open(os.path.join(a.masks_out, "sessions", f"{clip}.json"), "w") as fs:
            json.dump({"frames_dir": os.path.abspath(os.path.join(a.frames_out, clip)), "frames": frames, "stride": 1,
                       "obj_ids": [1], "obj_cls": {"1": "needle"}, "excluded": [], "points": {}, "boxes": {},
                       "source": "live_prepare.py（YOLO 預測當 ground truth）"}, fs, indent=1)

    out = os.path.join(a.frames_out, "manifest.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n完成：複製 {len(plan)} 張 → {a.frames_out}；YOLO mask {n_ok} 張 → {a.masks_out}")
    print(f"清單（來源 → 目的地、遞針手、測試集原因）：{out}")
    print("\n下一步：從 mask 算點（左端 = 針頭當預設，空白鍵接受；按 1 / 2 改、b 排除）")
    print(f"  python mask_to_needle5.py --images_dir {a.masks_out}/Images --labels_dir {a.masks_out}/Labels "
          f"--color_txt {cc} --out_dir needle5_handover_live --one_needle --no_thread --suggest_rule left")


if __name__ == "__main__":
    main()
