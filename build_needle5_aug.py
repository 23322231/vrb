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
    4. 針的 segmentation mask 用「完全相同」的裁切 / 旋轉 / 翻轉，存成對齊的 224x224 灰階圖
       （<split>/masks/，0~255 = 這個像素有多少比例是針；訓練時給 segmentation 輔助頭用）
       mask 來源：annotations.json 每筆的 "mask"（相對於 labels_root），顏色依 --color_txt；找不到就不存 mask

幾何增強（2026-10-15 改版）:
    - 裁切、位移、縮放、旋轉合成一個仿射變換（cv2.warpAffine），影像 / mask / 5 點用同一個矩陣，保證對齊
    - 旋轉 ±rot_deg 一律套用：舊版旋轉後點跑出裁切範圍就退回「不旋轉」，針斜放時大多數旋轉都失敗；
      新版改成自動把裁切範圍放大到剛好裝得下（終端機會印出被放大的比例）
    - 位移（center_jitter）也一樣：點會跑出範圍時自動放大裁切，針不會被切掉
    - 裁切範圍超出原圖的部分補黑（mask 為 0）

這支腳本只用 numpy / PIL / OpenCV，不需要 torch，在主機或 container 都能跑。

用法:
    python build_needle5_aug.py \
        --ann needle5_mine/annotations.json \
        --images_root DATA/Images \
        --out_dir needle5_aug --n_aug 10

    # 建議（減少「針一定在正中央」的捷徑）：
        --center_jitter 0.25 --scale_jitter 0.3 --rot_deg 180

輸出:
    needle5_aug/
        train/images/*.png   val/images/*.png   test/images/*.png
        train/masks/*.png    ...                  對齊的針 mask（有 mask 來源時）
        vis/{train,val,test}/*.png        5 點檢查圖（2px 圓；放大 2 倍方便看；綠色輪廓 = mask）
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
    """bbox -> 正方形裁切 (x0, y0, side)，原圖 pixel；rng 不為 None 時加隨機縮放 / 位移（不保證點在範圍內，見 warp_crop）。"""
    x1, y1, x2, y2 = bbox
    long_side = max(x2 - x1, y2 - y1)
    side = long_side + 2 * max(a.box_pad, a.box_pad_ratio * long_side)
    side = max(side, a.min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if rng is not None:
        side *= rng.uniform(1 - a.scale_jitter, 1 + a.scale_jitter)
        cx += rng.uniform(-a.center_jitter, a.center_jitter) * side
        cy += rng.uniform(-a.center_jitter, a.center_jitter) * side
        return cx - side / 2, cy - side / 2, side                     # 增強：可以超出原圖（補黑）
    side = int(round(min(side, W, H)))
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return x0, y0, side


def warp_crop(img, mask, pts, x0, y0, side, deg=0.0, margin=0.04, hflip=False, vflip=False):
    """以裁切中心 (x0+side/2, y0+side/2) 逆時針旋轉 deg 度後裁出 side x side，再縮成 224。
    影像、mask、5 點全部用同一個仿射矩陣。點會跑出範圍（含 margin）時自動把 side 放大到剛好裝得下。
    img: HxWx3 uint8 (RGB)；mask: HxW uint8 (0/255) 或 None；pts: (5, 2) 原圖 pixel（像素中心 = 整數座標）
    回傳 (crop 224x224x3, mask 224x224 或 None, pn (5,2) normalized, side_px, 放大倍率)"""
    c = np.array([x0 + side / 2, y0 + side / 2], dtype=np.float64)
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), float(deg), 1.0)     # 2x3，以 c 為中心旋轉
    q = (pts - c) @ M[:, :2].T                                                   # 旋轉後相對中心的位置
    need = 2 * (np.abs(q).max() / (1 - 2 * margin)) if len(q) else 0
    grow = max(1.0, need / side)
    side = side * grow
    S = int(math.ceil(side))
    M = M.copy()
    M[:, 2] += S / 2 - c                                                         # 中心移到輸出 (S/2, S/2)
    crop = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = None
    if mask is not None:
        m = cv2.warpAffine(mask, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    p = pts @ M[:, :2].T + M[:, 2]
    if hflip:
        crop = crop[:, ::-1]
        m = m[:, ::-1] if m is not None else None
        p[:, 0] = S - 1 - p[:, 0]
    if vflip:
        crop = crop[::-1]
        m = m[::-1] if m is not None else None
        p[:, 1] = S - 1 - p[:, 1]
    crop = cv2.resize(np.ascontiguousarray(crop), (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    if m is not None:
        m = cv2.resize(np.ascontiguousarray(m), (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    pn = (p + 0.5) / S                                                           # 像素中心座標 -> 0~1
    return crop, m, pn, S, grow


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


def make_aug(img, mask, pts, bbox, a, rng):
    """回傳 (224 PIL 圖, 224 mask 或 None, 5 點 normalized, side_px, 參數紀錄)。"""
    H, W = img.shape[:2]
    x0, y0, side = needle_crop(bbox, W, H, a, rng)
    deg = rng.uniform(-a.rot_deg, a.rot_deg) if a.rot_deg > 0 else 0.0
    hf = rng.random() < a.hflip
    vf = rng.random() < a.vflip
    crop, m, pn, S, grow = warp_crop(img, mask, pts, x0, y0, side, deg, a.keep_margin, hf, vf)
    crop = Image.fromarray(crop)
    if a.color_aug:
        crop = color_jitter(crop, rng, a)
    return crop, m, pn, S, {"crop": [round(x0, 1), round(y0, 1), round(side, 1)], "rot_deg": round(deg, 2),
                            "hflip": hf, "vflip": vf, "grow": round(grow, 3)}


# -----------------------------------------------------------------------------
# mask 來源
# -----------------------------------------------------------------------------
def parse_color_txt(path):
    """'Needle: 2, [0,0,255]' -> {'needle': (2, (0,0,255))}（與 mask_to_needle5.py 相同）"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*(.+?)\s*:\s*(\d+)\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", line)
            if m:
                out[m.group(1).strip().lower()] = (int(m.group(2)),
                                                    (int(m.group(3)), int(m.group(4)), int(m.group(5))))
    return out


def find_color_txt(labels_root):
    """在 labels_root 與它的上一層找 color.txt / Color_code.txt（不分大小寫）。"""
    for d in (labels_root, os.path.dirname(os.path.normpath(labels_root))):
        if not d or not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower() in ("color.txt", "color_code.txt", "colors.txt"):
                return os.path.join(d, f)
    return None


def load_needle_mask(path, class_id, color, tol):
    """讀 mask 檔 -> HxW uint8（針 = 255）。彩色 mask 依 color（BGR）比對；單通道 mask 依 class_id。"""
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 2:
        sel = m == class_id
    else:
        sel = np.all(np.abs(m[:, :, :3].astype(np.int16) - np.array(color, np.int16)) <= tol, axis=2)
    return sel.astype(np.uint8) * 255


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


def draw_check(img_pil, pn, path, mask=None, scale=2, r=2):
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    S = OUT_SIZE * scale
    if mask is not None:
        mb = cv2.resize((mask >= 128).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, (0, 255, 0), 1, cv2.LINE_AA)
    for p, c in zip(pn, POINT_COLORS):
        cv2.circle(img, (int(round(p[0] * S)), int(round(p[1] * S))), r, c, -1, cv2.LINE_AA)
    cv2.imwrite(path, img)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="mask_to_needle5.py 的 annotations.json")
    ap.add_argument("--images_root", default=None, help="原始影像根目錄（預設用 annotations.json 裡的）")
    ap.add_argument("--labels_root", default=None, help="mask 根目錄（預設用 annotations.json 裡的 labels_root）")
    ap.add_argument("--color_txt", default=None, help="mask 顏色表（預設在 labels_root 或上一層找 color.txt / Color_code.txt）")
    ap.add_argument("--needle_name", default="needle", help="color.txt 裡針的類別名稱")
    ap.add_argument("--mask_tol", type=int, default=10, help="mask 顏色比對容許誤差")
    ap.add_argument("--no_mask", action="store_true", help="不輸出 mask")
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
    ap.add_argument("--rot_deg", type=float, default=180, help="隨機旋轉 ±rot_deg 度（一律套用，必要時自動放大裁切）")
    ap.add_argument("--keep_margin", type=float, default=0.04,
                    help="5 點離裁切邊緣至少保留的比例；不夠時自動放大裁切範圍")
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

    # ---- mask 來源 ----
    labels_root = None if a.no_mask else (a.labels_root or ann.get("labels_root"))
    mask_cfg = None
    if labels_root and os.path.isdir(labels_root) and any("mask" in x for x in src):
        ctxt = a.color_txt or find_color_txt(labels_root)
        if ctxt and os.path.isfile(ctxt):
            colors = parse_color_txt(ctxt)
            if a.needle_name.lower() not in colors:
                raise SystemExit(f"{ctxt} 裡找不到 '{a.needle_name}'，現有類別：{list(colors)}")
            cid, ccol = colors[a.needle_name.lower()]
        else:
            cid, ccol = 2, (0, 0, 255)
            print("  [NOTE] 找不到 color.txt，假設 Needle = id 2 / BGR (0,0,255)（與 mask_to_needle5.py 範例相同）")
        mask_cfg = (labels_root, cid, ccol)
        print(f"mask：{labels_root}  needle id={cid} color(BGR)={ccol}" + (f"（{ctxt}）" if ctxt else ""))
    elif not a.no_mask:
        print(f"  [NOTE] 沒有 mask 來源（labels_root={labels_root}），不輸出 mask；可用 --labels_root 指定")

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
        if mask_cfg:
            os.makedirs(os.path.join(a.out_dir, sp, "masks"), exist_ok=True)
        if a.vis != "none":
            os.makedirs(os.path.join(a.out_dir, "vis", sp), exist_ok=True)

    # 輸出檔名用原圖檔名；不同資料夾有同名檔案時才把資料夾名稱加上去
    base_count = {}
    for s in src:
        b = os.path.basename(s["image"])
        base_count.setdefault(b, set()).add(s["image"])

    rng = random.Random(a.seed)
    samples, cache = [], {}
    grows, n_mask_missing = [], 0
    for idx, s in enumerate(src):
        sp = img_split.get(s["image"])
        if sp is None:
            print(f"  [WARN] split.json 裡沒有 {s['image']}，略過")
            continue
        if s["image"] not in cache:
            cache.clear()
            im = np.asarray(Image.open(os.path.join(root, s["image"])).convert("RGB"))
            mk = None
            if mask_cfg and s.get("mask"):
                mk = load_needle_mask(os.path.join(mask_cfg[0], s["mask"]), mask_cfg[1], mask_cfg[2], a.mask_tol)
                if mk is not None and mk.shape != im.shape[:2]:
                    print(f"  [WARN] mask 尺寸和影像不同，略過 mask：{s['mask']}")
                    mk = None
            cache[s["image"]] = (im, mk)
        img, mask = cache[s["image"]]
        if mask_cfg and mask is None:
            n_mask_missing += 1
        H, W = img.shape[:2]
        pts = np.asarray(s["points_px"], dtype=np.float64)
        b = os.path.basename(s["image"])
        base = b if len(base_count[b]) == 1 else s["image"].replace("/", "_")
        stem = f"{os.path.splitext(base)[0]}_n{s['needle_id']}"

        # 無增強版本（train / val / test 都有）
        x0, y0, side = needle_crop(s["needle_bbox"], W, H, a, None)
        c, m, pn, S, _ = warp_crop(img, mask, pts, x0, y0, side, 0.0, 0.0)
        versions = [(Image.fromarray(c), m, pn, S,
                     {"crop": [x0, y0, side], "rot_deg": 0.0, "hflip": False, "vflip": False, "grow": 1.0}, "orig")]
        if sp == "train":
            for k in range(a.n_aug):
                c, m, p, sd, info = make_aug(img, mask, pts, s["needle_bbox"], a, rng)
                grows.append(info["grow"])
                versions.append((c, m, p, sd, info, f"aug{k:02d}"))

        for j, (c, m, p, sd, info, tag) in enumerate(versions):
            name = f"{stem}_{tag}.png"
            rel = f"{sp}/images/{name}"
            c.save(os.path.join(a.out_dir, rel))
            mrel = None
            if m is not None:
                mrel = f"{sp}/masks/{name}"
                cv2.imwrite(os.path.join(a.out_dir, mrel), m)
            if a.vis == "all" or (a.vis == "some" and j < 3):
                draw_check(c, p, os.path.join(a.out_dir, "vis", sp, name), m)
            rec = {
                "image": rel,
                "split": sp,
                "source_image": s["image"],
                "needle_id": s["needle_id"],
                "augmented": tag != "orig",
                "points": np.round(p, 6).tolist(),
                "side_px": int(sd),
                "aug": info,
            }
            if mrel:
                rec["mask"] = mrel
            samples.append(rec)
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(src)} 根針")

    out = {
        "format": "needle5_crops",
        "point_names": POINT_NAMES,
        "image_size": OUT_SIZE,
        "coordinate_convention": "points: 224x224 裁切圖內的 normalized (x, y)；誤差 × side_px = 原圖 pixel",
        "mask_convention": "mask: 224x224 灰階，0~255 = 該像素是針的比例（與 image 逐像素對齊）",
        "source_ann": os.path.abspath(a.ann),
        "build_args": vars(a),
        "samples": samples,
    }
    with open(os.path.join(a.out_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    if grows:
        g = np.array(grows)
        print(f"\n增強：{len(g)} 張；為了裝下旋轉 / 位移後的針而放大裁切的比例 {np.mean(g > 1.001):.0%}"
              f"（放大倍率中位數 {np.median(g):.2f}，最大 {g.max():.2f}）")
    if mask_cfg:
        print(f"mask：{sum('mask' in x for x in samples)}/{len(samples)} 張有 mask"
              + (f"（{n_mask_missing} 根針讀不到 mask）" if n_mask_missing else ""))
    print("\n完成：")
    for sp in ("train", "val", "test"):
        ss = [x for x in samples if x["split"] == sp]
        n_img = len(set(x["source_image"] for x in ss))
        n_needle = len(set((x["source_image"], x["needle_id"]) for x in ss))
        print(f"  {sp:5s}: 原圖 {n_img:3d} 張 / 針 {n_needle:3d} 根 / 訓練圖 {len(ss):4d} 張")
    print(f"  → {os.path.join(a.out_dir, 'annotations.json')}")


if __name__ == "__main__":
    main()