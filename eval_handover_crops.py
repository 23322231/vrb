#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 B（任務書 §5-B-2）：有 GT 的影像上，比較不同「框 × 裁切」組合下 VRB 5 點的誤差（原圖 pixel）。

方法（--methods，預設全部能跑的）：
    gt_train         GT 針 mask 外接框 + 訓練裁切           ← 應該重現 run 的 test_metrics（test 7.3 px），用來驗證這支程式
    yolo_train       YOLO box + 訓練裁切                    ← handover node 的預設
    yolomask_train   YOLO mask 外接框 + 訓練裁切
    yolo_official    YOLO box + 官方 inference.py 裁切（predict_needle5_langsam.py 的切法）
    langsam_official LangSAM 框 + 官方裁切                  ← predict_needle5_langsam.py 原本的流程（要 --langsam）
    langsam_train    LangSAM 框 + 訓練裁切                  （要 --langsam）
    gt_geom          GT mask 的幾何中心線 + 定方向（線 → VRB）  ← 只看「方向（哪端是針尖）」判斷得準不準
    yolo_geom        YOLO mask（聯集）的幾何中心線 + 定方向（--orient，handover 用 left）
    gt_anchor        GT mask 的中心線 + VRB 3 點投影（decode=vrb_anchor）← 只看 VRB 決定的點準不準
    yolo_anchor      YOLO mask 的中心線 + VRB 3 點投影          ← decode=vrb_anchor 的完整流程（handover 預設）
    *_geom / *_anchor 會多量 head_1_5（GT 由 GT mask 中心線依弧長 0.2 取，與 GT 5 點同一套）；
    *_anchor 另外記同一條中心線上「純幾何 + 針頭朝左」（geom_only）的誤差，並做 heatmap 信心門檻的掃描。
    ※ 加 --detector sam3 時，名稱有 yolo 的方法改用 SAM 3 的偵測（方便和 YOLO 用同一張表比較）

資料：mask_to_needle5.py 的 annotations.json（points_px、needle_bbox），split 依 build_needle5_aug 的 split.json。
    預設：needle5_thread（VRB val = clip 002101、test = clip 002901）。
    注意：目前的 YOLO best.pt 是用 002901 訓練、002101 驗證的（needle_yolo_seg_needle），
          所以 YOLO 框在這兩段上偏樂觀；handover 新拍的影像標好之後用 --ann 換成那份才是真正的測試。

    python eval_handover_crops.py                                   # val + test，全部方法（沒有 lang_sam 就跳過 LangSAM）
    python eval_handover_crops.py --splits test --langsam 1
    python eval_handover_crops.py --ann needle5_handover/annotations.json --splits all --split_file none

輸出 pred_handoff_output/eval_crops_<時間>/：
    per_sample.csv     每張 × 每個方法：找到沒、框 IoU、5 點誤差、頭尾翻轉、信心、時間
    summary.json/.md   每個方法：找到率、tail_1_3 與 5 點平均 / 中位數 / p90 誤差、翻轉率、時間、GPU 記憶體
    worst/<方法>/      tail_1_3 誤差最大的幾張疊圖（GT 實心點、預測空心圈）
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handover_perception import add_common_args, box_iou, build_from_args, gpu_mem_report, to_orig  # noqa: E402

ALL_METHODS = ["gt_train", "yolo_train", "yolomask_train", "yolo_official", "gt_geom", "yolo_geom",
               "gt_anchor", "yolo_anchor", "langsam_official", "langsam_train"]
GRASP = ("tail_1_3", "head_1_5")
THRESHOLDS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BAD_PX = 10.0                     # 夾取點誤差 > 10 px（約 3 mm）算「不能夾」


def load_gt_mask(s):
    if not (s.get("mask_path") and os.path.isfile(s["mask_path"])):
        return None, None
    lab = cv2.imread(s["mask_path"], cv2.IMREAD_COLOR)
    gmask = (lab[..., 2] > 245) & (lab[..., 1] < 10) & (lab[..., 0] < 10)      # 針 = BGR (0,0,255)
    seed = np.zeros_like(gmask)
    x1, y1, x2, y2 = [int(round(v)) for v in s["gt_box"]]
    seed[y1:y2 + 1, x1:x2 + 1] = True
    return gmask, seed


def gt_six(s, gmask, seed, anchor_names):
    """GT 6 點：GT 5 點 + head_1_5（GT mask 中心線、GT head 那端在前，依弧長取；和產生 GT 5 點同一套）。
    回傳 (6×2 或 None, 和 GT 5 點的最大差 px)。"""
    import needle_geom as G
    if gmask is None:
        return None, None
    nd = G.centerline(gmask, seed)
    if nd is None:
        return None, None
    poly = nd["poly"]
    if np.linalg.norm(poly[-1] - s["gt"][0]) < np.linalg.norm(poly[0] - s["gt"][0]):
        poly = poly[::-1]
    p6, _ = G.points_along(poly, [G.POINT_FRACS[n] for n in anchor_names])
    return p6, float(np.max(np.linalg.norm(p6[:5] - s["gt"][:5], axis=1)))


def load_samples(ann_path, images_root, split_file, splits):
    ann = json.load(open(ann_path))
    root = images_root or ann.get("images_root")
    sp = json.load(open(split_file))["image_split"] if split_file and os.path.isfile(split_file) else {}
    labels_root = ann.get("labels_root")
    out = []
    for s in ann["samples"]:
        if not s.get("head_confirmed", True):
            continue
        split = sp.get(s["image"], s.get("split", "?"))
        if "all" not in splits and split not in splits:
            continue
        out.append({"path": os.path.join(root, s["image"]), "image": s["image"], "split": split,
                    "needle_id": s.get("needle_id", 0), "gt": np.asarray(s["points_px"], np.float64),
                    "gt_box": [float(v) for v in s["needle_bbox"]], "clip": os.path.dirname(s["image"]),
                    "mask_path": os.path.join(labels_root, s["mask"]) if labels_root and s.get("mask") else None})
    return out, root


def point_errors(pred, gt):
    e = np.linalg.norm(pred - gt, axis=-1)
    flip = (np.linalg.norm(pred[0] - gt[1]) + np.linalg.norm(pred[1] - gt[0])) < (e[0] + e[1])
    return e, bool(flip)


def stats(v, px=True):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], np.float64)
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
            "p90": round(float(np.percentile(v, 90)), 2), "max": round(float(v.max()), 2),
            **({"le5px": round(float((v <= 5).mean()), 3), "le10px": round(float((v <= 10).mean()), 3)} if px else {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--ann", default="needle5_thread/annotations.json")
    ap.add_argument("--images_root", default=None, help="預設用 annotations.json 的 images_root")
    ap.add_argument("--split_file", default="needle5_thread_aug/split.json")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], help="val / test / train / all")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--langsam", type=int, default=0, help="1 = 也跑 LangSAM（要 lang_sam，會多用不少 GPU 記憶體）")
    ap.add_argument("--n_worst", type=int, default=8)
    ap.add_argument("--out", default="pred_handoff_output")
    a = ap.parse_args()

    samples, root = load_samples(a.ann, a.images_root, a.split_file, a.splits)
    if not samples:
        sys.exit("[ERROR] 沒有樣本（檢查 --ann / --splits）")
    methods = a.methods or [m for m in ALL_METHODS if a.langsam or not m.startswith("langsam")]
    print(f"{len(samples)} 張（{a.splits}），方法：{methods}")

    a.decode = "vrb"                  # 這支程式自己逐一跑各方法，不用 predict() 的 decode
    perc = build_from_args(a)
    vnames, gnames = perc.vrb_names, perc.geom_names      # VRB 輸出的點（3 或 5）/ 幾何解碼一律 5 點
    names = gnames
    anames = perc.anchor_names                             # 6 點（5 點 + head_1_5）
    t13 = lambda nm, e: float(e[nm.index("tail_1_3")]) if "tail_1_3" in nm else float(e.mean())
    print(f"VRB 輸出 {len(vnames)} 點：{vnames}" + ("（*_train 方法沒有 tail_1_3，排序與列印改用各點平均）" if "tail_1_3" not in vnames else ""))
    finder = None
    if any(m.startswith("langsam") for m in methods):
        from predict_needle5_langsam import BoxFinder
        finder = BoxFinder(0.3, 0.25)

    out_dir = os.path.join(a.out, "eval_crops_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    rows, keep = [], {m: [] for m in methods}
    t_yolo, t_vrb = [], []
    for j, s in enumerate(samples, 1):
        img = cv2.imread(s["path"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 讀不到 {s['path']}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = {"gt": s["gt_box"]}
        t = time.perf_counter()
        dets = perc.detect(img)
        t_yolo.append((time.perf_counter() - t) * 1e3)
        det, _ = perc.select(dets)
        if det is not None:
            boxes["yolo"] = det["box_xyxy"]
            boxes["yolomask"] = det["mask_box_xyxy"] or det["box_xyxy"]
        if finder is not None:
            from PIL import Image
            lb, _, _ = finder(Image.fromarray(img_rgb), "needle")
            if lb is not None:
                boxes["langsam"] = lb
        line = []
        gmask, gseed = load_gt_mask(s)
        gt6, gt6_chk = gt_six(s, gmask, gseed, anames)
        if gt6 is None:
            gt6 = np.vstack([s["gt"][:5], np.full((1, 2), np.nan)])
        for m in methods:
            src, mode = m.split("_")
            if mode in ("geom", "anchor"):
                row = {"image": s["image"], "split": s["split"], "clip": s["clip"], "method": m, "found": False}
                t = time.perf_counter()
                g = None
                gm = "vrb_anchor" if mode == "anchor" else "geom"
                if src == "gt" and gmask is not None:
                    g = perc.geom_decode(img_rgb, gmask, gseed, perc.thread_mask(dets), mode=gm)  # 線用 YOLO 的（若有線類別）
                elif src == "yolo" and det is not None and det.get("mask") is not None:
                    union = np.zeros(img.shape[:2], bool)
                    for d in perc.valid(dets):
                        if d.get("mask") is not None:
                            union |= d["mask"]
                    g = perc.geom_decode(img_rgb, union, det["mask"], perc.thread_mask(dets), mode=gm)
                row.update(gt_head_left=bool(s["gt"][0][0] < s["gt"][1][0]), gt6_check_px=gt6_chk)
                if g is not None:
                    t_vrb.append((time.perf_counter() - t) * 1e3)
                    e6 = np.linalg.norm(g["pts6"] - gt6, axis=1)
                    _, flip = point_errors(g["pts6"], gt6)
                    vc = dict(zip(vnames, g["vconf"]))
                    row.update({"found": True, "box_iou_gt": round(box_iou(g["box"], s["gt_box"]), 3),
                                "crop_side": g["crop"][2] - g["crop"][0], "flip": flip,
                                "err_mean": round(float(np.nanmean(e6[:5])), 2),
                                "orient_conf": round(g["orient_conf"], 3), "n_fragments": g["needle"]["n_fragments"],
                                "orient_source": g["orient"].get("source"), "thread": g["orient"].get("thread"),
                                **{f"err_{n}": (None if not np.isfinite(v) else round(float(v), 2)) for n, v in zip(anames, e6)},
                                **{f"conf_{n}": round(float(v), 3) for n, v in vc.items()},
                                "grasp_conf": round(float(min(vc["head"], vc["tail"])), 3)})
                    if mode == "anchor":
                        ai = g["anchor"]
                        eg = np.linalg.norm(g["geom_only"]["pts"] - gt6, axis=1)
                        row.update({"vrb_agree": ai["vrb_agree"], "span_ratio": ai["span_ratio"],
                                    "middle_consistency_px": ai.get("middle_consistency_px"),
                                    "head_off_px": ai["head_off_px"], "tail_off_px": ai["tail_off_px"],
                                    **{f"err_geomonly_{n}": (None if not np.isfinite(v) else round(float(v), 2))
                                       for n, v in zip(anames, eg) if n in GRASP},
                                    **{f"diff_geom_{n}": v for n, v in ai["grasp_diff_px"].items()}})
                    keep[m].append((float(np.nan_to_num(e6[anames.index("tail_1_3")])), s, g["box"], g["crop"], g["crop224"],
                                    g["pn"], g["conf"], g["hm"], g["seg"], g["pts"], g["names"], gt6))
                    line.append(f"{m}={e6[anames.index('tail_1_3')]:.1f}/{e6[anames.index('head_1_5')]:.1f}")
                rows.append(row)
                continue
            row = {"image": s["image"], "split": s["split"], "clip": s["clip"], "method": m, "found": src in boxes}
            if src in boxes:
                box = boxes[src]
                row["box_iou_gt"] = round(box_iou(box, s["gt_box"]), 3)
                t = time.perf_counter()
                c224, crop = perc.crop(img_rgb, box, mode)
                if c224 is None:
                    row["found"] = False
                else:
                    pn, conf, hm, seg = perc.keypoints(c224)
                    t_vrb.append((time.perf_counter() - t) * 1e3)
                    pts = to_orig(pn, crop)
                    e, flip = point_errors(pts, s["gt"][:len(vnames)])
                    row.update({"crop_side": crop[2] - crop[0], "flip": flip, "err_mean": round(float(e.mean()), 2),
                                **{f"err_{n}": round(float(v), 2) for n, v in zip(vnames, e)},
                                **{f"conf_{n}": round(float(v), 3) for n, v in zip(vnames, conf)}})
                    if src == "yolo" or src == "yolomask":
                        row["yolo_conf"] = round(det["conf"], 3)
                    keep[m].append((t13(vnames, e), s, box, crop, c224, pn, conf, hm, seg, pts, vnames, s["gt"]))
                    line.append(f"{m}={t13(vnames, e):.1f}")
            rows.append(row)
        print(f"[{j}/{len(samples)}] {s['image']} det={len(dets)} tail_1_3（/head_1_5）誤差: " + " ".join(line))

    # ---------------------------------------------------------------- 彙總
    summary = {"ann": a.ann, "splits": a.splits, "n_images": len(samples), "settings": {
        k: getattr(a, k) for k in ("detector", "yolo", "sam3", "sam3_prompt", "run", "weights", "yolo_imgsz", "yolo_conf", "orient", "roi")}, "methods": {}}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        ok = [r for r in rs if "err_mean" in r]
        summary["methods"][m] = {
            "found_rate": round(len(ok) / max(len(rs), 1), 3),
            "n_points": {"geom": len(gnames), "anchor": len(anames)}.get(m.split("_")[1], len(vnames)),
            "tail_1_3_px": stats([r.get("err_tail_1_3") for r in ok]),
            "head_1_5_px": stats([r.get("err_head_1_5") for r in ok]),
            "all5_mean_px": stats([r["err_mean"] for r in ok]),
            **{f"{n}_px_mean": (stats([r.get(f"err_{n}") for r in ok]) or {}).get("mean") for n in names},
            "flip_rate": round(float(np.mean([r["flip"] for r in ok])), 3) if ok else None,
            "box_iou_gt_median": (stats([r.get("box_iou_gt") for r in rs]) or {}).get("median"),
            "by_split": {sp: stats([r.get("err_tail_1_3") for r in ok if r["split"] == sp]) for sp in sorted({r["split"] for r in ok})},
        }
    for m in methods:                                         # geom：依「方向由誰決定」分開看翻轉率
        ok = [r for r in rows if r["method"] == m and r.get("orient_source")]
        if ok:
            by = {}
            for src in sorted({r["orient_source"] for r in ok}):
                rs = [r for r in ok if r["orient_source"] == src]
                by[src] = {"n": len(rs), "flip_rate": round(float(np.mean([r["flip"] for r in rs])), 3)}
            summary["methods"][m]["orient_by_source"] = by
            summary["methods"][m]["thread_reasons"] = {k: sum(1 for r in ok if str(r.get("thread")) == k)
                                                       for k in sorted({str(r.get("thread")) for r in ok})}
    for m in methods:                                         # geom / anchor：分「針頭在左（流程內）／在右」看
        ok = [r for r in rows if r["method"] == m and r.get("gt_head_left") is not None and "err_mean" in r]
        if not ok:
            continue
        grp = {}
        for name, sel in (("head_left", [r for r in ok if r["gt_head_left"]]), ("head_right", [r for r in ok if not r["gt_head_left"]])):
            if sel:
                grp[name] = {"n": len(sel), "flip_rate": round(float(np.mean([r["flip"] for r in sel])), 3),
                             **{f"{n}_px": stats([r.get(f"err_{n}") for r in sel]) for n in GRASP}}
                if m.endswith("_anchor"):
                    grp[name]["geom_only"] = {f"{n}_px": stats([r.get(f"err_geomonly_{n}") for r in sel]) for n in GRASP}
                    grp[name]["vrb_agree_rate"] = round(float(np.mean([bool(r.get("vrb_agree")) for r in sel])), 3)
        summary["methods"][m]["by_head_side"] = grp
    for m in methods:                                         # anchor：heatmap 信心門檻掃描
        if not m.endswith("_anchor"):
            continue
        ok = [r for r in rows if r["method"] == m and "err_mean" in r]
        sweep = []
        for gate in ("conf", "conf+agree"):
            for th in THRESHOLDS:
                kept = [r for r in ok if r["grasp_conf"] >= th and (gate == "conf" or r.get("vrb_agree"))]
                rej = [r for r in ok if r not in kept]
                bad = lambda rs, n: sum(1 for r in rs if (r.get(f"err_{n}") or 0) > BAD_PX)
                sweep.append({"gate": gate, "th": th, "kept": len(kept), "rejected": len(rej),
                              **{f"kept_bad_{n}": bad(kept, n) for n in GRASP},
                              **{f"rejected_bad_{n}": bad(rej, n) for n in GRASP},
                              **{f"kept_{n}_median": (stats([r.get(f"err_{n}") for r in kept]) or {}).get("median") for n in GRASP}})
        summary["methods"][m]["threshold_sweep"] = sweep
    summary["timing_ms"] = {"yolo": stats(t_yolo, False), "vrb_crop_plus_forward": stats(t_vrb, False)}
    summary["gpu"] = gpu_mem_report()

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(out_dir, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, ensure_ascii=False)

    md = ["| 方法 | 找到率 | tail_1_3 平均 | 中位數 | p90 | ≤5px | head_1_5 平均 | 中位數 | p90 | ≤5px | 點數 | head | tail | middle | 翻轉率 | 框 IoU 中位數 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, v in summary["methods"].items():
        t3, h5 = v["tail_1_3_px"] or {}, v["head_1_5_px"] or {}
        md.append(f"| {m} | {v['found_rate']} | {t3.get('mean')} | {t3.get('median')} | {t3.get('p90')} | {t3.get('le5px')} | "
                  f"{h5.get('mean')} | {h5.get('median')} | {h5.get('p90')} | {h5.get('le5px')} | "
                  f"{v['n_points']} | {v.get('head_px_mean')} | {v.get('tail_px_mean')} | {v.get('middle_px_mean')} | "
                  f"{v['flip_rate']} | {v['box_iou_gt_median']} |")
    for m, v in summary["methods"].items():
        for side, gv in (v.get("by_head_side") or {}).items():
            f = lambda d: "—" if not d else f"中位數 {d['median']}／平均 {d['mean']}／≤5px {d['le5px']}"
            line = (f"\n{m}［{side}，{gv['n']} 張］翻轉 {gv['flip_rate']}；tail_1_3 {f(gv['tail_1_3_px'])}；"
                    f"head_1_5 {f(gv['head_1_5_px'])}")
            if "geom_only" in gv:
                line += (f"；vrb_agree {gv['vrb_agree_rate']}；同條中心線的純幾何：tail_1_3 {f(gv['geom_only']['tail_1_3_px'])}，"
                         f"head_1_5 {f(gv['geom_only']['head_1_5_px'])}")
            md.append(line)
    for m, v in summary["methods"].items():
        if v.get("threshold_sweep"):
            md.append(f"\n{m} 信心門檻掃描（grasp_conf = min(head, tail) heatmap 尖峰；bad = 誤差 > {BAD_PX:g} px）")
            md.append("| 條件 | 門檻 | 保留 | 擋掉 | 保留中 tail_1_3 bad | 保留中 head_1_5 bad | 擋掉中 tail_1_3 bad | 擋掉中 head_1_5 bad | 保留 tail_1_3 中位數 | 保留 head_1_5 中位數 |")
            md.append("|---|---|---|---|---|---|---|---|---|---|")
            for r in v["threshold_sweep"]:
                md.append(f"| {r['gate']} | {r['th']} | {r['kept']} | {r['rejected']} | {r['kept_bad_tail_1_3']} | {r['kept_bad_head_1_5']} | "
                          f"{r['rejected_bad_tail_1_3']} | {r['rejected_bad_head_1_5']} | {r['kept_tail_1_3_median']} | {r['kept_head_1_5_median']} |")
    for m, v in summary["methods"].items():
        if v.get("orient_by_source"):
            md.append(f"\n{m} 方向由誰決定：{v['orient_by_source']}；線的判斷結果：{v['thread_reasons']}")
    md.append("")
    md.append(f"時間（ms）：YOLO {summary['timing_ms']['yolo']}；VRB（裁切 + forward）{summary['timing_ms']['vrb_crop_plus_forward']}")
    md.append(f"GPU：{summary['gpu']}")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---------------------------------------------------------------- 最差的幾張
    for m in methods:
        d = os.path.join(out_dir, "worst", m)
        os.makedirs(d, exist_ok=True)
        for e, s, box, crop, c224, pn, conf, hm, seg, pts, nm, gtp in sorted(keep[m], key=lambda x: -x[0])[:a.n_worst]:
            img = cv2.imread(s["path"], cv2.IMREAD_COLOR)
            from predict_needle5_langsam import draw
            vis = draw(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), box, crop, pts, conf, nm, c224, pn, hm, seg, 0.5,
                       np.nan_to_num(np.asarray(gtp)[:len(pts)]))
            cv2.imwrite(os.path.join(d, f"{e:06.1f}px__{s['image'].replace('/', '__')}"), vis)
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
