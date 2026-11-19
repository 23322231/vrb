#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 B（任務書 §5-B-1）：測 YOLO26-seg（best.pt）找針的能力。只看「框夠不夠準拿來切圖」，mask 不必完美。

兩種輸入：
  (1) 有 GT：mask_to_needle5.py 的 annotations.json（每根針的 needle_bbox = GT mask 外接框）
      回報（依 clip 分組，並標出這段 clip 是不是 YOLO 訓練 / 驗證用過的）：
        - 偵測率：有任一偵測與 GT 框 IoU ≥ --iou_match
        - top-1 正確率：信心最高的那個偵測就是針（handover node 只取最高的，所以這個最重要）
        - 誤偵測：每張圖沒對到 GT 的偵測數（在多個信心門檻下）
        - 框品質：對到的框與 GT 的 IoU、四邊最大誤差（px）；box 與 mask 外接框兩種都算
  (2) 沒有 GT（--images 資料夾，例如 handover 現場新拍的 1280×720）：
        每張的偵測數、信心，疊圖存起來人工看誤偵測（縫合墊圖樣、縫線）

    python eval_handover_yolo.py                                              # needle5_thread 全部 clip
    python eval_handover_yolo.py --imgsz 640 1280                             # 比較推論解析度
    python eval_handover_yolo.py --images handover_samples/ --no_gt           # 只跑沒有 GT 的資料夾
    python eval_handover_yolo.py --images /data/calibration/20261115_084810/images --glob '*_rgb.png' --no_gt
    python eval_handover_yolo.py --yolo runs/needle_seg/<新的 run>/weights/best.pt

輸出 pred_handoff_output/eval_yolo_<時間>/：summary.json / summary.md、per_image.csv、
     vis/<分類>/ 疊圖（miss = 沒偵測到、top1_wrong = 最高分不是針、fp = 有多餘偵測、nogt = 沒 GT 的資料夾）
"""

import argparse
import csv
import datetime
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handover_perception import DEFAULT_YOLO, IMG_EXTS, box_iou, gpu_mem_report, mask_bbox  # noqa: E402


def yolo_split_of(yolo_data_dir):
    """needle_yolo_seg_needle/images/{train,val,test}/<clip>__<frame>.png -> {(clip, frame 檔名): split}"""
    m = {}
    for sp in ("train", "val", "test"):
        for p in glob.glob(os.path.join(yolo_data_dir, "images", sp, "*")):
            b = os.path.basename(p)
            if "__" in b:
                clip, frame = b.split("__", 1)
                m[(clip, os.path.splitext(frame)[0])] = sp
    return m


def run_yolo(model, img, imgsz, conf):
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.7, retina_masks=True, verbose=False)[0]
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    b = r.boxes.xyxy.cpu().numpy()
    c = r.boxes.conf.cpu().numpy()
    ms = (r.masks.data.cpu().numpy() > 0.5) if getattr(r, "masks", None) is not None else None
    for i in np.argsort(-c):
        m = ms[i] if ms is not None else None
        if m is not None and m.shape != img.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        out.append({"conf": float(c[i]), "box": b[i].tolist(), "mbox": mask_bbox(m) if m is not None else None, "mask": m})
    return out


def edge_err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def draw(img, dets, gts, title):
    vis = img.copy()
    for g in gts:
        x1, y1, x2, y2 = [int(round(v)) for v in g]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for k, d in enumerate(dets):
        x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
        col = (0, 200, 255) if k == 0 else (0, 0, 255)
        if d["mask"] is not None:
            vis[d["mask"]] = (0.5 * vis[d["mask"]] + 0.5 * np.array(col)).astype(np.uint8)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 1)
        cv2.putText(vis, f"{d['conf']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(vis, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def summarize(rows, conf_ths, iou_match):
    out = {"n_images": len(rows)}
    for th in conf_ths:
        det = [r for r in rows]
        hit = np.mean([any(d["iou"] >= iou_match and d["conf"] >= th for d in r["dets"]) for r in det]) if det else None
        top = []
        for r in det:
            ds = [d for d in r["dets"] if d["conf"] >= th]
            top.append(bool(ds) and ds[0]["iou"] >= iou_match)
        fp = [sum(1 for d in r["dets"] if d["conf"] >= th and d["iou"] < iou_match) for r in det]
        out[f"conf>={th}"] = {"recall": round(float(hit), 3) if hit is not None else None,
                              "top1_correct": round(float(np.mean(top)), 3) if top else None,
                              "fp_per_image": round(float(np.mean(fp)), 3) if fp else None,
                              "images_with_fp": round(float(np.mean([f > 0 for f in fp])), 3) if fp else None}
    ious = [r["dets"][r["best"]]["iou"] for r in rows if r["best"] is not None]
    mious = [r["dets"][r["best"]]["miou"] for r in rows if r["best"] is not None and r["dets"][r["best"]]["miou"] is not None]
    ee = [r["dets"][r["best"]]["edge"] for r in rows if r["best"] is not None]
    mee = [r["dets"][r["best"]]["medge"] for r in rows if r["best"] is not None and r["dets"][r["best"]]["medge"] is not None]
    q = lambda v: None if not v else {"mean": round(float(np.mean(v)), 3), "median": round(float(np.median(v)), 3),
                                      "p10": round(float(np.percentile(v, 10)), 3), "p90": round(float(np.percentile(v, 90)), 3)}
    out["matched_box_iou"] = q(ious)
    out["matched_maskbox_iou"] = q(mious)
    out["matched_box_max_edge_err_px"] = q(ee)
    out["matched_maskbox_max_edge_err_px"] = q(mee)
    out["matched_conf"] = q([r["dets"][r["best"]]["conf"] for r in rows if r["best"] is not None])
    out["top_fp_conf"] = q([max(d["conf"] for d in r["dets"] if d["iou"] < iou_match) for r in rows
                            if any(d["iou"] < iou_match for d in r["dets"])])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo", default=DEFAULT_YOLO)
    ap.add_argument("--detector", default="yolo", choices=["yolo", "sam3"], help="sam3：改測 SAM 3 的文字提示分割（--imgsz 無作用）")
    ap.add_argument("--sam3", default="sam3.pt")
    ap.add_argument("--sam3_prompt", nargs="+", default=["needle"])
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640], help="可給多個比較，例如 640 1280")
    ap.add_argument("--ann", default="needle5_thread/annotations.json")
    ap.add_argument("--images_root", default=None)
    ap.add_argument("--yolo_data", default="needle_yolo_seg_needle", help="YOLO 的資料夾，用來標出 clip 是否被 YOLO 看過")
    ap.add_argument("--images", default=None, help="沒有 GT 的影像資料夾（例如 handover 現場）")
    ap.add_argument("--glob", default="*", help="--images 裡要用的檔名樣式，例如 '*_rgb.png'")
    ap.add_argument("--no_gt", action="store_true", help="只跑 --images，不跑 --ann")
    ap.add_argument("--every", type=int, default=1, help="有 GT 的資料每 N 張取 1 張")
    ap.add_argument("--min_conf", type=float, default=0.05, help="推論時的最低信心（之後再用 --conf_ths 篩）")
    ap.add_argument("--conf_ths", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--iou_match", type=float, default=0.5)
    ap.add_argument("--n_vis", type=int, default=30, help="每一類最多存幾張疊圖")
    ap.add_argument("--out", default="pred_handoff_output")
    a = ap.parse_args()

    import torch
    if a.detector == "sam3":
        from handover_perception import Sam3Detector
        sam3 = Sam3Detector(a.sam3, a.sam3_prompt, a.min_conf)

        class _M:                                          # 讓 run_yolo 照用
            def predict(self, img, imgsz=None, conf=None, **k):
                return [sam3(img, conf)]
        model = _M()
        a.imgsz = a.imgsz[:1]
    else:
        from ultralytics import YOLO
        model = YOLO(a.yolo)
    out_dir = os.path.join(a.out, "eval_yolo_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    summary = {"detector": a.detector, "yolo": os.path.abspath(a.yolo), "sam3": a.sam3, "sam3_prompt": a.sam3_prompt,
               "iou_match": a.iou_match, "by_imgsz": {}}
    csv_rows = []

    gt_items = []
    if not a.no_gt and a.ann and os.path.isfile(a.ann):
        ann = json.load(open(a.ann))
        root = a.images_root or ann["images_root"]
        per_img = {}
        for s in ann["samples"]:
            per_img.setdefault(s["image"], []).append([float(v) for v in s["needle_bbox"]])
        ysplit = yolo_split_of(a.yolo_data) if os.path.isdir(a.yolo_data) else {}
        for rel in sorted(per_img)[::max(1, a.every)]:
            clip, frame = os.path.dirname(rel), os.path.splitext(os.path.basename(rel))[0]
            gt_items.append({"path": os.path.join(root, rel), "rel": rel, "clip": clip, "gts": per_img[rel],
                             "yolo_split": ysplit.get((clip, frame), "unseen" if ysplit else "?")})
        clip_seen = {}
        for it in gt_items:
            clip_seen.setdefault(it["clip"], set()).add(it["yolo_split"])
        summary["clips"] = {c: sorted(v) for c, v in clip_seen.items()}
    nogt_items = []
    if a.images:
        nogt_items = [p for p in sorted(glob.glob(os.path.join(a.images, "**", a.glob), recursive=True)) if p.lower().endswith(IMG_EXTS)]
        print(f"沒有 GT 的資料夾：{len(nogt_items)} 張（{a.images}/**/{a.glob}）")

    for imgsz in a.imgsz:
        rows, times = [], []
        nvis = {}
        for j, it in enumerate(gt_items, 1):
            img = cv2.imread(it["path"], cv2.IMREAD_COLOR)
            if img is None:
                continue
            t = time.perf_counter()
            dets = run_yolo(model, img, imgsz, a.min_conf)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1e3)
            for d in dets:
                ious = [box_iou(d["box"], g) for g in it["gts"]]
                gi = int(np.argmax(ious))
                d["iou"], d["gt"] = ious[gi], gi
                d["edge"] = edge_err(d["box"], it["gts"][gi])
                d["miou"] = box_iou(d["mbox"], it["gts"][gi]) if d["mbox"] else None
                d["medge"] = edge_err(d["mbox"], it["gts"][gi]) if d["mbox"] else None
            best = next((k for k, d in enumerate(dets) if d["iou"] >= a.iou_match), None)
            r = {"rel": it["rel"], "clip": it["clip"], "yolo_split": it["yolo_split"], "dets": dets, "best": best}
            rows.append(r)
            th = 0.25
            ds = [d for d in dets if d["conf"] >= th]
            cat = None
            if best is None or dets[best]["conf"] < th:
                cat = "miss"
            elif ds and ds[0]["iou"] < a.iou_match:
                cat = "top1_wrong"
            elif any(d["iou"] < a.iou_match for d in ds):
                cat = "fp"
            if cat and nvis.get(cat, 0) < a.n_vis:
                nvis[cat] = nvis.get(cat, 0) + 1
                d = os.path.join(out_dir, f"vis_{imgsz}", cat)
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, it["rel"].replace("/", "__")),
                            draw(img, ds, it["gts"], f"{cat} imgsz={imgsz} {it['yolo_split']}"))
            csv_rows.append({"imgsz": imgsz, "image": it["rel"], "clip": it["clip"], "yolo_split": it["yolo_split"],
                             "n_det@0.25": len(ds), "matched": best is not None,
                             "matched_conf": round(dets[best]["conf"], 3) if best is not None else None,
                             "matched_iou": round(dets[best]["iou"], 3) if best is not None else None,
                             "matched_maskbox_iou": round(dets[best]["miou"], 3) if best is not None and dets[best]["miou"] is not None else None,
                             "top1_iou@0.25": round(ds[0]["iou"], 3) if ds else None, "category": cat or "ok"})
            if j % 50 == 0:
                print(f"[imgsz {imgsz}] {j}/{len(gt_items)}")
        res = {"timing_ms_median": round(float(np.median(times)), 1) if times else None,
               "timing_ms_p90": round(float(np.percentile(times, 90)), 1) if times else None}
        if rows:
            res["all"] = summarize(rows, a.conf_ths, a.iou_match)
            for g in ("unseen", "val", "train"):
                rs = [r for r in rows if r["yolo_split"] == g]
                if rs:
                    res[f"yolo_{g}"] = summarize(rs, a.conf_ths, a.iou_match)
            res["by_clip"] = {c: summarize([r for r in rows if r["clip"] == c], [0.25], a.iou_match)
                              for c in sorted({r["clip"] for r in rows})}
        # ------------------------- 沒有 GT 的資料夾
        if nogt_items:
            cnt, confs = [], []
            for p in nogt_items:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                dets = run_yolo(model, img, imgsz, a.min_conf)
                ds = [d for d in dets if d["conf"] >= 0.25]
                cnt.append(len(ds))
                confs.append(ds[0]["conf"] if ds else 0.0)
                d = os.path.join(out_dir, f"vis_{imgsz}", "nogt")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, os.path.relpath(p, a.images).replace(os.sep, "__")),
                            draw(img, [x for x in dets if x["conf"] >= a.conf_ths[0]], [], f"imgsz={imgsz} n@0.25={len(ds)}"))
                csv_rows.append({"imgsz": imgsz, "image": p, "clip": "nogt", "n_det@0.25": len(ds),
                                 "top1_conf": round(ds[0]["conf"], 3) if ds else None,
                                 "all_confs": " ".join(f"{x['conf']:.2f}" for x in dets)})
            res["nogt"] = {"n": len(cnt), "zero_det@0.25": int(sum(c == 0 for c in cnt)),
                           "one_det@0.25": int(sum(c == 1 for c in cnt)), "multi_det@0.25": int(sum(c > 1 for c in cnt)),
                           "top1_conf_median": round(float(np.median(confs)), 3) if confs else None,
                           "note": "沒有 GT：多偵測不一定是誤偵測（可能真的有兩根針），要看 vis_*/nogt 的疊圖"}
        summary["by_imgsz"][imgsz] = res
        print(json.dumps({k: v for k, v in res.items() if k != "by_clip"}, indent=1, ensure_ascii=False))

    summary["gpu"] = gpu_mem_report()
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, ensure_ascii=False)
    keys = []
    for r in csv_rows:
        keys += [k for k in r if k not in keys]
    if csv_rows:
        with open(os.path.join(out_dir, "per_image.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(csv_rows)
    md = []
    for imgsz, res in summary["by_imgsz"].items():
        md.append(f"## imgsz {imgsz}（YOLO 單張 {res['timing_ms_median']} ms 中位數）\n")
        md.append("| 資料 | 張數 | 偵測率@0.25 | top-1 正確@0.25 | 誤偵測/張@0.25 | 有誤偵測的張數比例 | 框 IoU 中位數 | mask 框 IoU 中位數 | 框四邊最大誤差中位數 px |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for g in ["all", "yolo_unseen", "yolo_val", "yolo_train"] + [f"clip:{c}" for c in res.get("by_clip", {})]:
            s = res["by_clip"][g[5:]] if g.startswith("clip:") else res.get(g)
            if not s:
                continue
            c = s.get("conf>=0.25", {})
            md.append(f"| {g} | {s['n_images']} | {c.get('recall')} | {c.get('top1_correct')} | {c.get('fp_per_image')} | "
                      f"{c.get('images_with_fp')} | {(s['matched_box_iou'] or {}).get('median')} | "
                      f"{(s['matched_maskbox_iou'] or {}).get('median')} | {(s['matched_box_max_edge_err_px'] or {}).get('median')} |")
        if "nogt" in res:
            md.append(f"\n沒有 GT 的資料夾：{res['nogt']}")
        md.append("")
    md.append(f"clip 與 YOLO 的關係：{summary.get('clips')}")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
