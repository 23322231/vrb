#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把學長的彩色 mask（依 Color_code.txt，BGR 順序）轉成 Ultralytics YOLO segmentation 格式。

    DATA/Images/{train,val,test}/xxx.png   +   DATA/Labels/{train,val,test}/xxx.png
        ↓
    needle_seg_yolo/
        images/{train,val,test}/xxx.png      （預設複製；主機與 container 路徑不同，symlink 會失效）
        labels/{train,val,test}/xxx.txt      每行：class x1 y1 x2 y2 ...（polygon，座標 normalized）
        data.yaml
        vis/{split}/xxx.png                  polygon 檢查圖（前 --n_vis 張）

說明:
    - 每個「連通區塊」當成一個 instance。針被夾爪擋住斷成兩段時會變成兩個 instance，
      這沒關係：預測完輸出成 mask 後，mask_to_needle5.py 會再把相鄰碎片接回同一根針。
    - 預設只轉 Needle；可用 --classes 加入其他類別（例如 "Tool end-effector"），之後也能偵測夾爪。
    - 沒有任何物體的影像也會保留（空的 txt），讓模型學會「這裡沒有針」。
    - 資料夾第一層是 train/val/test 時沿用（學長資料）；
      不是的話（例如 sam2_video_label.py 的 Images/<clip>/...），用 --clip_split 依「clip 資料夾」切分，
      同一段影片不會同時出現在 train 與 test。
    - 可以對同一個 --out_dir 執行多次來合併資料集（例如先轉學長的，再轉自己的），data.yaml 會自動更新。

    # 合併範例
    python masks_to_yolo_seg.py --images_dir DATA/Images --labels_dir DATA/Labels \
        --color_txt DATA/Labels/Color_code.txt --out_dir needle_seg_yolo
    python masks_to_yolo_seg.py --images_dir my_needle_masks/Images --labels_dir my_needle_masks/Labels \
        --color_txt my_needle_masks/Color_code.txt --out_dir needle_seg_yolo --clip_split 8 1 1

用法:
    python masks_to_yolo_seg.py --images_dir DATA/Images --labels_dir DATA/Labels \
        --color_txt DATA/Labels/Color_code.txt --out_dir needle_seg_yolo
"""

import argparse
import os
import random
import re
import shutil

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


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
    """二值 mask -> 每個連通區塊一個 polygon（外輪廓）。細線物體用很小的 eps 簡化，避免變形。"""
    polys = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (lab == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        if eps > 0:
            c = cv2.approxPolyDP(c, eps, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1, 2).astype(np.float64))
    return polys


def find_pairs(images_dir, labels_dir):
    labels = {}
    for root, _, files in os.walk(labels_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                rel = os.path.relpath(os.path.join(root, f), labels_dir)
                labels[os.path.splitext(rel)[0]] = os.path.join(root, f)
    pairs = []
    for root, _, files in os.walk(images_dir):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTS):
                rel = os.path.relpath(os.path.join(root, f), images_dir)
                if os.path.splitext(rel)[0] in labels:
                    pairs.append((rel, labels[os.path.splitext(rel)[0]]))
    return sorted(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--color_txt", required=True)
    ap.add_argument("--out_dir", default="needle_seg_yolo")
    ap.add_argument("--classes", nargs="+", default=["Needle"], help="要轉的類別名稱（Color_code.txt 裡的名稱）")
    ap.add_argument("--tol", type=int, default=10)
    ap.add_argument("--min_area", type=int, default=10, help="小於此面積的碎片忽略 (px)")
    ap.add_argument("--eps", type=float, default=0.7, help="polygon 簡化程度 (px)，細針請保持很小")
    ap.add_argument("--symlink", action="store_true",
                    help="改用 symlink（只在同一個環境轉換與訓練時用；主機轉、container 訓練會讀不到）")
    ap.add_argument("--n_vis", type=int, default=20)
    ap.add_argument("--clip_split", type=float, nargs=3, default=None, metavar=("TRAIN", "VAL", "TEST"),
                    help="第一層資料夾不是 train/val/test 時，依資料夾（clip）切分的比例，例如 8 1 1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    colors = parse_color_txt(a.color_txt)
    cls = []
    for name in a.classes:
        if name.lower() not in colors:
            raise SystemExit(f"Color_code.txt 裡找不到 '{name}'，現有：{list(colors)}")
        cls.append((name, *colors[name.lower()]))
    print("類別：", ", ".join(f"{k}={n}(BGR {c})" for k, (n, _, c) in enumerate(cls)))

    pairs = find_pairs(a.images_dir, a.labels_dir)
    print(f"找到 {len(pairs)} 組 影像/mask")

    # 第一層資料夾不是 train/val/test 時：依 clip 資料夾切分
    folder_split = {}
    if a.clip_split:
        folders = sorted({rel.replace("\\", "/").split("/")[0] for rel, _ in pairs
                          if rel.replace("\\", "/").split("/")[0].lower() not in SPLIT_ALIASES
                          and "/" in rel.replace("\\", "/")})
        random.Random(a.seed).shuffle(folders)
        r = np.array(a.clip_split, float) / sum(a.clip_split)
        n_tr = int(round(len(folders) * r[0]))
        n_va = int(round(len(folders) * r[1]))
        if len(folders) >= 3:                  # clip 很少時，至少各留 1 段給 val / test
            n_va = max(n_va, 1)
            n_tr = min(n_tr, len(folders) - n_va - 1)
        for i, fd in enumerate(folders):
            folder_split[fd] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        print("clip 切分：", {sp: [f for f, v in folder_split.items() if v == sp] for sp in ("train", "val", "test")})
    stats = {}
    n_vis = {}
    for rel, mpath in pairs:
        first_raw = rel.replace("\\", "/").split("/")[0]
        first = first_raw.lower()
        split = SPLIT_ALIASES.get(first) or folder_split.get(first_raw, "train")
        name = os.path.basename(rel)
        if first in SPLIT_ALIASES:
            sub = os.path.relpath(rel, rel.replace("\\", "/").split("/")[0])
        else:
            sub = rel
        img_dst = os.path.join(a.out_dir, "images", split, sub)
        lab_dst = os.path.join(a.out_dir, "labels", split, os.path.splitext(sub)[0] + ".txt")
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(lab_dst), exist_ok=True)

        src_img = os.path.abspath(os.path.join(a.images_dir, rel))
        if not os.path.exists(img_dst):
            if a.symlink:
                os.symlink(src_img, img_dst)
            else:
                shutil.copy2(src_img, img_dst)

        mask = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
        H, W = mask.shape[:2]
        lines, polys_all = [], []
        for k, (_, cid, color) in enumerate(cls):
            for poly in mask_to_polygons(class_mask(mask, cid, color, a.tol), a.min_area, a.eps):
                xy = (poly / [W, H]).clip(0, 1).reshape(-1)
                lines.append(f"{k} " + " ".join(f"{v:.6f}" for v in xy))
                polys_all.append((k, poly))
        with open(lab_dst, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        s = stats.setdefault(split, {"images": 0, "instances": 0, "empty": 0})
        s["images"] += 1
        s["instances"] += len(lines)
        s["empty"] += int(not lines)

        if n_vis.get(split, 0) < a.n_vis and polys_all:
            n_vis[split] = n_vis.get(split, 0) + 1
            img = cv2.imread(src_img)
            for k, poly in polys_all:
                cv2.polylines(img, [poly.round().astype(np.int32)], True,
                              (0, 255, 255) if k == 0 else (255, 255, 0), 1, cv2.LINE_AA)
            vdir = os.path.join(a.out_dir, "vis", split)
            os.makedirs(vdir, exist_ok=True)
            cv2.imwrite(os.path.join(vdir, name), img)

    yaml_path = os.path.join(a.out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(a.out_dir)}\n")
        for sp in ("train", "val", "test"):
            if os.path.isdir(os.path.join(a.out_dir, "images", sp)):
                f.write(f"{sp}: images/{sp}\n")
        f.write("names:\n")
        for k, (n, _, _) in enumerate(cls):
            f.write(f"  {k}: {n.lower().replace(' ', '_')}\n")

    print("\n完成：")
    for sp, s in stats.items():
        print(f"  {sp:5s}: {s['images']} 張影像，{s['instances']} 個 instance，{s['empty']} 張沒有物體")
    print(f"  → {yaml_path}")


if __name__ == "__main__":
    main()
