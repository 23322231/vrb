#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把「vis 資料夾裡留下來（= 已人工確認）的影像」轉成 Ultralytics YOLO segmentation 資料集，
用來訓練 YOLO 自動標出針 / 線的 mask，之後就不用再一張一張標。

資料來源（都是現有的輸出，不需要另外準備）:
    <needle5_dir>/vis/<split>/<影像名>.png     你檢查後留下來的圖 -> 決定用哪些影像
    <needle5_dir>/annotations.json            每張影像對應的原圖與 mask 路徑（images_root / labels_root）
    <Labels>/<clip>/<影像名>.png               sam2_video_label.py 的彩色 mask（針 (0,0,255)、線 (255,0,255)）
    Color_code.txt                             類別顏色（預設在 labels_root 的上一層）

輸出:
    <out_dir>/images/{train,val,test}/<clip>__<檔名>
    <out_dir>/labels/{train,val,test}/<clip>__<檔名>.txt   每行：class x1 y1 x2 y2 ...（polygon，normalized）
    <out_dir>/data.yaml
    <out_dir>/check/<split>/...                            polygon 檢查圖（每組前 --n_vis 張）

切分：以 clip 為單位（同一段影片不會同時在 train 和 val），用 --val_clips / --test_clips 指定 clip 名稱的
一部分（例如 002101），其餘都是 train。

用法:
    python vis_to_yolo_seg.py --needle5_dir needle5_thread --out_dir needle_yolo_seg --val_clips 002101

    # 訓練（YOLO26 segmentation；第一次會自動下載 yolo26s-seg.pt）
    yolo segment train model=yolo26s-seg.pt data=needle_yolo_seg/data.yaml imgsz=640 epochs=100 batch=8
"""

import argparse
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np


def parse_color_txt(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*(.+?)\s*:\s*(\d+)\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", line)
            if m:
                out[m.group(1).strip().lower()] = (int(m.group(2)),
                                                    (int(m.group(3)), int(m.group(4)), int(m.group(5))))
    return out


def class_mask(mask, cid, color, tol):
    if mask.ndim == 2:
        return mask == cid
    return np.all(np.abs(mask[:, :, :3].astype(np.int16) - np.array(color, np.int16)) <= tol, axis=2)


def mask_to_polygons(m, min_area, eps):
    """每個連通區塊一個 polygon（外輪廓）。針被夾爪擋住斷成兩段會變成兩個 instance，
    之後 YOLO 預測的 mask 交給 mask_to_needle5.py 時會再把碎片接回同一根針。"""
    polys = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        if eps > 0:
            c = cv2.approxPolyDP(c, eps, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1, 2).astype(np.float64))
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--needle5_dir", required=True, help="mask_to_needle5.py 的 --out_dir（有 vis/ 與 annotations.json）")
    ap.add_argument("--out_dir", default="needle_yolo_seg")
    ap.add_argument("--val_clips", nargs="+", required=True, help="當 val 的 clip（名稱的一部分即可，例如 002101）")
    ap.add_argument("--test_clips", nargs="*", default=[], help="當 test 的 clip（可省略）")
    ap.add_argument("--images_root", default=None, help="預設用 annotations.json 的 images_root")
    ap.add_argument("--labels_root", default=None, help="預設用 annotations.json 的 labels_root")
    ap.add_argument("--color_txt", default=None, help="預設 <labels_root>/../Color_code.txt")
    ap.add_argument("--classes", nargs="+", default=["Needle", "Thread"], help="Color_code.txt 裡的類別名稱")
    ap.add_argument("--tol", type=int, default=10)
    ap.add_argument("--min_area", type=int, default=10, help="小於此面積的碎片忽略 (px)")
    ap.add_argument("--eps", type=float, default=0.7, help="polygon 簡化程度 (px)；細的針 / 線請保持很小")
    ap.add_argument("--n_vis", type=int, default=20)
    a = ap.parse_args()

    ann_path = os.path.join(a.needle5_dir, "annotations.json")
    if not os.path.isfile(ann_path):
        sys.exit(f"找不到 {ann_path}")
    ann = json.load(open(ann_path, encoding="utf-8"))
    images_root = a.images_root or ann.get("images_root")
    labels_root = a.labels_root or ann.get("labels_root")
    color_txt = a.color_txt or os.path.join(os.path.dirname(os.path.normpath(labels_root)), "Color_code.txt")
    for p in (images_root, labels_root, color_txt):
        if not p or not os.path.exists(p):
            sys.exit(f"找不到 {p}（可用 --images_root / --labels_root / --color_txt 指定）")

    colors = parse_color_txt(color_txt)
    cls = []
    for name in a.classes:
        if name.lower() not in colors:
            sys.exit(f"{color_txt} 裡找不到 '{name}'，現有：{list(colors)}")
        cls.append((name, *colors[name.lower()]))
    print("類別：", ", ".join(f"{k}={n}(BGR {c})" for k, (n, _, c) in enumerate(cls)))

    # ---- vis 裡留下來的影像 = 已確認 ----
    vis_root = os.path.join(a.needle5_dir, "vis")
    kept = set()
    for sp in os.listdir(vis_root) if os.path.isdir(vis_root) else []:
        d = os.path.join(vis_root, sp)
        if os.path.isdir(d):
            kept |= {os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(".png")}
    if not kept:
        sys.exit(f"{vis_root} 裡沒有圖")

    images = {}                                              # rel -> mask rel（同一張影像可能有多根針，只取一次）
    for s in ann["samples"]:
        images.setdefault(s["image"], s["mask"])
    stem = lambda rel: os.path.splitext(os.path.basename(rel))[0]
    by_stem = {}
    for rel in images:
        by_stem.setdefault(stem(rel), []).append(rel)
    chosen = sorted(rel for rel in images if stem(rel) in kept and len(by_stem[stem(rel)]) == 1)
    amb = sorted(rel for rel in images if stem(rel) in kept and len(by_stem[stem(rel)]) > 1)
    print(f"vis 裡 {len(kept)} 張；annotations.json 裡 {len(images)} 張影像；採用 {len(chosen)} 張")
    if amb:
        print(f"  [WARN] {len(amb)} 張在不同 clip 有同名檔案，無法從 vis 判斷是否確認過，略過：{amb[:4]}")

    def split_of(rel):
        clip = rel.replace("\\", "/").split("/")[0]
        if any(c in clip for c in a.test_clips):
            return "test"
        if any(c in clip for c in a.val_clips):
            return "val"
        return "train"

    stats, n_vis, clips = {}, {}, {}
    for rel in chosen:
        sp = split_of(rel)
        clip = rel.replace("\\", "/").split("/")[0] if "/" in rel.replace("\\", "/") else "root"
        clips.setdefault(sp, set()).add(clip)
        flat = rel.replace("\\", "/").replace("/", "__")
        img_src = os.path.join(images_root, rel)
        mask = cv2.imread(os.path.join(labels_root, images[rel]), cv2.IMREAD_UNCHANGED)
        if mask is None or not os.path.isfile(img_src):
            print(f"  [WARN] 讀不到影像或 mask：{rel}")
            continue
        H, W = mask.shape[:2]
        lines, polys_all = [], []
        for k, (_, cid, color) in enumerate(cls):
            for poly in mask_to_polygons(class_mask(mask, cid, color, a.tol), a.min_area, a.eps):
                xy = (poly / [W, H]).clip(0, 1).reshape(-1)
                lines.append(f"{k} " + " ".join(f"{v:.6f}" for v in xy))
                polys_all.append((k, poly))

        img_dst = os.path.join(a.out_dir, "images", sp, flat)
        lab_dst = os.path.join(a.out_dir, "labels", sp, os.path.splitext(flat)[0] + ".txt")
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(lab_dst), exist_ok=True)
        shutil.copy2(img_src, img_dst)
        with open(lab_dst, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        s = stats.setdefault(sp, {"images": 0, **{n: 0 for n, _, _ in cls}})
        s["images"] += 1
        for k, _ in polys_all:
            s[cls[k][0]] += 1
        if n_vis.get(sp, 0) < a.n_vis:
            n_vis[sp] = n_vis.get(sp, 0) + 1
            img = cv2.imread(img_src)
            for k, poly in polys_all:
                cv2.polylines(img, [poly.round().astype(np.int32)], True,
                              (0, 255, 255) if k == 0 else (255, 0, 255), 1, cv2.LINE_AA)
            vdir = os.path.join(a.out_dir, "check", sp)
            os.makedirs(vdir, exist_ok=True)
            cv2.imwrite(os.path.join(vdir, flat), img)

    if "val" not in stats:
        print(f"  [WARN] 沒有任何影像被分到 val：請確認 --val_clips {a.val_clips} 有對到 clip 名稱")
    yaml_path = os.path.join(a.out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(a.out_dir)}\n")
        for sp in ("train", "val", "test"):
            if sp in stats:
                f.write(f"{sp}: images/{sp}\n")
        f.write("names:\n")
        for k, (n, _, _) in enumerate(cls):
            f.write(f"  {k}: {n.lower().replace(' ', '_')}\n")

    print("\n完成：")
    for sp in ("train", "val", "test"):
        if sp in stats:
            s = stats[sp]
            print(f"  {sp:5s}: {s['images']} 張（clip {len(clips[sp])} 段：{', '.join(sorted(clips[sp]))[:120]}）；"
                  + "、".join(f"{n} {s[n]} 個 instance" for n, _, _ in cls))
    print(f"  → {yaml_path}；polygon 檢查圖在 {os.path.join(a.out_dir, 'check')}/")


if __name__ == "__main__":
    main()
