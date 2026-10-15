#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 mask_to_needle5.py 的標註，做成「離線增強」的 VRB 訓練資料集：

    1. 依 8:1:1 重新切分 train / val / test：
       - 自己的影片（Images/<clip>/...）：以 clip 為單位，同一段影片的連續影格一定在同一組
       - 學長資料（Images/train|val|test/...）：以影像為單位，left_/right_ 立體影像視為同一組
    2. train：每根針存 1 張「無增強」裁切 + n_aug 張隨機增強（裁切縮放/位移、水平/上下翻轉、旋轉、色彩）
       val / test：每根針只存 1 張固定裁切（不做任何隨機增強，才能反映真實表現）
    3. 每張都是 224x224 的裁切圖，5 點座標直接是裁切圖內的 normalized (x, y)
       另外記錄裁切邊長 side_px（原圖 pixel），訓練時用來把誤差換算回原圖 pixel

這支腳本只用 numpy / PIL / OpenCV，不需要 torch，在主機或 container 都能跑。

用法:
    python build_needle5_aug.py \
        --ann needle5_mine/annotations.json \
        --images_root DATA/Images \
        --out_dir needle5_aug --n_aug 10

輸出:
    needle5_aug/
        train/images/*.png   val/images/*.png   test/images/*.png
        vis/{train,val,test}/*.png        5 點檢查圖（2px 圓；放大 2 倍方便看）
        annotations.json                  "format": "needle5_crops"（train_vrb_needle5.py 會自動辨識）
        split.json                        每張原圖分到哪一組（之後重做可沿用：--split_file）
"""

import argparse
import json
import math
import os
import random
import re

import cv2
import numpy as np
from PIL import Image, ImageEnhance

POINT_NAMES = ["head", "tail", "middle", "head_1_3", "tail_1_3"]
POINT_COLORS = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 140, 255), (255, 255, 0)]  # BGR
OUT_SIZE = 224


# -----------------------------------------------------------------------------
# 裁切 / 增強（與 train_vrb_needle5.py 的線上版本相同規則）
# -----------------------------------------------------------------------------
def needle_crop(bbox, W, H, a, rng=None):
    x1, y1, x2, y2 = bbox
    long_side = max(x2 - x1, y2 - y1)
    side = long_side + 2 * max(a.box_pad, a.box_pad_ratio * long_side)
    side = max(side, a.min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if rng is not None:
        side *= rng.uniform(1 - a.scale_jitter, 1 + a.scale_jitter)
        cx += rng.uniform(-a.center_jitter, a.center_jitter) * side
        cy += rng.uniform(-a.center_jitter, a.center_jitter) * side
    side = int(round(min(side, W, H)))
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return x0, y0, side


def plain_crop(img, pts, x0, y0, side):
    crop = img.crop((x0, y0, x0 + side, y0 + side))
    return crop, (pts - np.array([x0, y0])) / side


def rotated_crop(img, pts, x0, y0, side, deg):
    """以裁切中心逆時針旋轉 deg 度；先裁 1.5 倍大的區域旋轉再取中間（不會有黑角）。
    旋轉後有點跑出範圍回傳 (None, None)。"""
    cx, cy = x0 + side / 2, y0 + side / 2
    big = int(math.ceil(side * 1.5))
    bx0, by0 = int(round(cx - big / 2)), int(round(cy - big / 2))
    region = img.crop((bx0, by0, bx0 + big, by0 + big)).rotate(deg, resample=Image.BILINEAR)
    o = int(round((big - side) / 2))
    crop = region.crop((o, o, o + side, o + side))
    th = math.radians(deg)
    rc = np.array([bx0 + big / 2, by0 + big / 2]) - 0.5        # PIL 旋轉中心（像素索引座標）
    d = pts - rc
    rd = np.stack([d[:, 0] * math.cos(th) + d[:, 1] * math.sin(th),
                   -d[:, 0] * math.sin(th) + d[:, 1] * math.cos(th)], axis=1)
    pn = (rd + rc - np.array([bx0 + o, by0 + o])) / side
    if (pn < 0).any() or (pn > 1).any():
        return None, None
    return crop, pn


def color_jitter(img, rng, a):
    """亮度 / 對比 / 飽和度 / 色相（與 torchvision ColorJitter 同參數意義）+ 小機率灰階。"""
    ops = [("b", a.brightness), ("c", a.contrast), ("s", a.saturation), ("h", a.hue)]
    rng.shuffle(ops)
    for k, v in ops:
        if v <= 0:
            continue
        if k == "b":
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(1 - v, 1 + v))
        elif k == "c":
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(1 - v, 1 + v))
        elif k == "s":
            img = ImageEnhance.Color(img).enhance(rng.uniform(1 - v, 1 + v))
        else:
            shift = rng.uniform(-v, v)
            hsv = np.asarray(img.convert("HSV")).copy()
            hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + int(round(shift * 255))) % 256).astype(np.uint8)
            img = Image.fromarray(hsv, "HSV").convert("RGB")
    if rng.random() < a.grayscale:
        img = img.convert("L").convert("RGB")
    return img


def make_aug(img, pts, bbox, a, rng):
    """回傳 (224 圖, 5 點 normalized, side_px, 參數紀錄)。"""
    W, H = img.size
    x0, y0, side = needle_crop(bbox, W, H, a, rng)
    deg = rng.uniform(-a.rot_deg, a.rot_deg) if a.rot_deg > 0 else 0.0
    crop, pn = (rotated_crop(img, pts, x0, y0, side, deg) if deg else (None, None))
    if crop is None:
        deg = 0.0
        crop, pn = plain_crop(img, pts, x0, y0, side)
    hf = rng.random() < a.hflip
    vf = rng.random() < a.vflip
    if hf:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        pn[:, 0] = 1.0 - pn[:, 0]
    if vf:
        crop = crop.transpose(Image.FLIP_TOP_BOTTOM)
        pn[:, 1] = 1.0 - pn[:, 1]
    crop = crop.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
    if a.color_aug:
        crop = color_jitter(crop, rng, a)
    return crop, pn, side, {"crop": [x0, y0, side], "rot_deg": round(deg, 2), "hflip": hf, "vflip": vf}


# -----------------------------------------------------------------------------
# 切分
# -----------------------------------------------------------------------------
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}


def group_key(rel, pattern, group_by):
    """image：同一張圖（left_/right_ 合併）為一組；folder：第一層資料夾（= 一段 clip）為一組。"""
    parts = rel.replace("\\", "/").split("/")
    if group_by == "auto":
        group_by = "folder" if len(parts) > 1 and parts[0].lower() not in SPLIT_NAMES else "image"
    if group_by == "folder":
        return "clip:" + parts[0]
    stem = os.path.splitext(parts[-1])[0]
    return re.sub(pattern, "", stem, flags=re.IGNORECASE) if pattern else stem


def split_groups(keys, ratios, seed):
    keys = sorted(set(keys))
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_tr = int(round(n * ratios[0] / sum(ratios)))
    n_va = int(round(n * ratios[1] / sum(ratios)))
    if n >= 3 and ratios[1] > 0 and ratios[2] > 0:    # 組數很少（例如只有幾段 clip）時，val / test 至少各 1 組
        n_va = max(n_va, 1)
        n_tr = min(n_tr, n - n_va - 1)
    out = {}
    for i, k in enumerate(keys):
        out[k] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return out


def draw_check(img_pil, pn, path, scale=2, r=2):
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    S = OUT_SIZE * scale
    for p, c in zip(pn, POINT_COLORS):
        cv2.circle(img, (int(round(p[0] * S)), int(round(p[1] * S))), r, c, -1, cv2.LINE_AA)
    cv2.imwrite(path, img)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="mask_to_needle5.py 的 annotations.json")
    ap.add_argument("--images_root", default=None, help="原始影像根目錄（預設用 annotations.json 裡的）")
    ap.add_argument("--out_dir", default="needle5_aug")
    ap.add_argument("--ratios", type=float, nargs=3, default=[8, 1, 1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--group_by", choices=["auto", "image", "folder"], default="auto",
                    help="切分單位：auto（第一層資料夾是 train/val/test 時以影像為單位，否則以資料夾 = clip 為單位）")
    ap.add_argument("--chunks_per_clip", type=int, default=0,
                    help="把每段 clip 依時間切成 K 個連續區段分別當一組（只有 1~2 段影片時用，例如 10）")
    ap.add_argument("--chunk_gap", type=int, default=2, help="區段交界丟掉的影格數（避免相鄰影格跨組）")
    ap.add_argument("--group_regex", default=r"^(left|right)[_-]?",
                    help="從檔名移除這段後相同的影像視為同一組（預設合併 left_/right_ 立體影像）；'' = 不合併")
    ap.add_argument("--split_file", default=None, help="沿用之前的 split.json（重做增強時切分保持不變）")
    ap.add_argument("--n_aug", type=int, default=10, help="train 每根針額外產生幾張增強圖")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vis", choices=["all", "some", "none"], default="some",
                    help="檢查圖：all 全部；some 每根針只畫無增強 + 前 2 張增強；none 不畫")
    # 裁切
    ap.add_argument("--box_pad", type=float, default=20)
    ap.add_argument("--box_pad_ratio", type=float, default=0.15)
    ap.add_argument("--min_side", type=float, default=96)
    # 增強
    ap.add_argument("--scale_jitter", type=float, default=0.15)
    ap.add_argument("--center_jitter", type=float, default=0.08)
    ap.add_argument("--rot_deg", type=float, default=180)
    ap.add_argument("--hflip", type=float, default=0.5)
    ap.add_argument("--vflip", type=float, default=0.5)
    ap.add_argument("--color_aug", type=int, default=1)
    ap.add_argument("--brightness", type=float, default=0.4)
    ap.add_argument("--contrast", type=float, default=0.3)
    ap.add_argument("--saturation", type=float, default=0.3)
    ap.add_argument("--hue", type=float, default=0.05)
    ap.add_argument("--grayscale", type=float, default=0.05)
    a = ap.parse_args()

    with open(a.ann, "r", encoding="utf-8") as f:
        ann = json.load(f)
    root = a.images_root or ann.get("images_root")
    if not root or not os.path.isdir(root):
        raise SystemExit(f"找不到影像資料夾 {root}，請用 --images_root 指定")
    src = [s for s in ann["samples"] if s.get("head_confirmed", True)]
    print(f"讀入 {len(src)} 根針（{len(set(s['image'] for s in src))} 張影像）")

    # ---- 切分 ----
    if a.split_file:
        with open(a.split_file) as f:
            img_split = json.load(f)["image_split"]
    else:
        keys = {s["image"]: group_key(s["image"], a.group_regex, a.group_by) for s in src}
        chunk_info = {}
        if a.chunks_per_clip > 1:
            # 同一段 clip 依時間（檔名順序）切成 K 個連續區段，每段當一組
            by_clip = {}
            for rel, k in keys.items():
                if k.startswith("clip:"):
                    by_clip.setdefault(k, []).append(rel)
            for k, rels in by_clip.items():
                rels.sort()
                n = len(rels)
                for r_i, rel in enumerate(rels):
                    c = min(r_i * a.chunks_per_clip // n, a.chunks_per_clip - 1)
                    keys[rel] = f"{k}#t{c:02d}"
                    chunk_info[rel] = (k, r_i, rels)
        gsplit = split_groups(keys.values(), a.ratios, a.seed)
        img_split = {rel: gsplit[k] for rel, k in keys.items()}
        if chunk_info and a.chunk_gap > 0:
            # 相鄰區段被分到不同組時，交界附近 chunk_gap 張不使用（避免幾乎一樣的影格跨組）
            dropped = set()
            for rel, (k, r_i, rels) in chunk_info.items():
                for d in range(1, a.chunk_gap + 1):
                    for nb in (r_i - d, r_i + d):
                        if 0 <= nb < len(rels) and img_split[rels[nb]] != img_split[rel]:
                            dropped.add(rel)
            if dropped:
                print(f"區段交界丟掉 {len(dropped)} 張（--chunk_gap {a.chunk_gap}）")
                src = [x for x in src if x["image"] not in dropped]
                img_split = {r: v for r, v in img_split.items() if r not in dropped}
        n_groups = len(set(keys.values()))
        print(f"切分單位：{'clip（資料夾）' if any(k.startswith('clip:') for k in keys.values()) else '影像'}，共 {n_groups} 組")
        for sp in ("train", "val", "test"):
            gs = sorted({k for k in keys.values() if gsplit[k] == sp})
            print(f"  {sp:5s}: {len(gs)} 組" + (f"  {[g[5:] for g in gs]}" if gs and gs[0].startswith("clip:") else ""))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "split.json"), "w") as f:
        json.dump({"ratios": a.ratios, "group_regex": a.group_regex, "seed": a.seed,
                   "image_split": img_split}, f, indent=1)

    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(a.out_dir, sp, "images"), exist_ok=True)
        if a.vis != "none":
            os.makedirs(os.path.join(a.out_dir, "vis", sp), exist_ok=True)

    # 輸出檔名用原圖檔名；不同資料夾有同名檔案時才把資料夾名稱加上去
    base_count = {}
    for s in src:
        b = os.path.basename(s["image"])
        base_count.setdefault(b, set()).add(s["image"])

    rng = random.Random(a.seed)
    samples, cache = [], {}
    for idx, s in enumerate(src):
        sp = img_split.get(s["image"])
        if sp is None:
            print(f"  [WARN] split.json 裡沒有 {s['image']}，略過")
            continue
        if s["image"] not in cache:
            cache.clear()
            cache[s["image"]] = Image.open(os.path.join(root, s["image"])).convert("RGB")
        img = cache[s["image"]]
        W, H = img.size
        pts = np.asarray(s["points_px"], dtype=np.float64)
        b = os.path.basename(s["image"])
        base = b if len(base_count[b]) == 1 else s["image"].replace("/", "_")
        stem = f"{os.path.splitext(base)[0]}_n{s['needle_id']}"

        # 無增強版本（train / val / test 都有）
        x0, y0, side = needle_crop(s["needle_bbox"], W, H, a, None)
        crop, pn = plain_crop(img, pts, x0, y0, side)
        versions = [(crop.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR), pn, side,
                     {"crop": [x0, y0, side], "rot_deg": 0.0, "hflip": False, "vflip": False}, "orig")]
        if sp == "train":
            for k in range(a.n_aug):
                c, p, sd, info = make_aug(img, pts, s["needle_bbox"], a, rng)
                versions.append((c, p, sd, info, f"aug{k:02d}"))

        for j, (c, p, sd, info, tag) in enumerate(versions):
            name = f"{stem}_{tag}.png"
            rel = f"{sp}/images/{name}"
            c.save(os.path.join(a.out_dir, rel))
            if a.vis == "all" or (a.vis == "some" and j < 3):
                draw_check(c, p, os.path.join(a.out_dir, "vis", sp, name))
            samples.append({
                "image": rel,
                "split": sp,
                "source_image": s["image"],
                "needle_id": s["needle_id"],
                "augmented": tag != "orig",
                "points": np.round(p, 6).tolist(),
                "side_px": int(sd),
                "aug": info,
            })
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(src)} 根針")

    out = {
        "format": "needle5_crops",
        "point_names": POINT_NAMES,
        "image_size": OUT_SIZE,
        "coordinate_convention": "points: 224x224 裁切圖內的 normalized (x, y)；誤差 × side_px = 原圖 pixel",
        "source_ann": os.path.abspath(a.ann),
        "build_args": vars(a),
        "samples": samples,
    }
    with open(os.path.join(a.out_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print("\n完成：")
    for sp in ("train", "val", "test"):
        ss = [x for x in samples if x["split"] == sp]
        n_img = len(set(x["source_image"] for x in ss))
        n_needle = len(set((x["source_image"], x["needle_id"]) for x in ss))
        print(f"  {sp:5s}: 原圖 {n_img:3d} 張 / 針 {n_needle:3d} 根 / 訓練圖 {len(ss):4d} 張")
    print(f"  → {os.path.join(a.out_dir, 'annotations.json')}")


if __name__ == "__main__":
    main()