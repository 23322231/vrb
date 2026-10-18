#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 EndoVis 2018 Robotic Scene Segmentation（MICCAI 2018）挑出「畫面中真的有縫針」的影格，
轉成和 sam2_video_label.py 相同的格式，可以直接接 mask_to_needle5.py。

輸入（自動往下找，每個同時有 left_frames/ 與 labels/ 的資料夾 = 一段序列）:
    <root>/.../seq_X/left_frames/frameXXX.png      左眼影像（只有左眼有標註）
    <root>/.../seq_X/labels/frameXXX.png           彩色標註（RGB，顏色依 labels.json）
    labels.json：先找 seq 資料夾，再往上找；都沒有就用內建的 EndoVis 2018 顏色表
    --repairs：repairs.zip 解壓的資料夾；有同一個「seq_X/labels/檔名」的修正標註時優先使用

輸出（--out_dir，預設 endovis_needle）:
    Images/<序列>/<檔名>      左眼影像（預設複製；--link 改成符號連結，不佔空間）
    Labels/<序列>/<檔名>      mask：針 BGR (0,0,255)、線 BGR (255,0,255)、其餘黑色
    overlay/<序列>/<檔名>     檢查用疊圖（--overlay 0 不輸出）
    Color_code.txt            與 sam2_video_label.py 相同
    frames.csv                每張影格：針 / 線像素數、針的碎片數、線有沒有碰到針、是否入選
    序列名稱 = <release 資料夾>_<seq 資料夾>（不同 release 都有 seq_1，避免撞名）

篩選（全部可調）:
    針像素 >= --min_needle_px（預設 40）
    --require_thread 1：還要有線、且線離針 <= --touch_px（之後用線自動判斷頭尾才有用）

用法:
    python endovis_needles.py --roots miccai_challenge_2018_release_1 miccai_challenge_release_2 \\
        miccai_challenge_release_3 miccai_challenge_release_4 --repairs repairs --out_dir endovis_needle --link
    # 之後（序列資料夾就是 clip）:
    python mask_to_needle5.py --images_dir endovis_needle/Images --labels_dir endovis_needle/Labels \\
        --color_txt endovis_needle/Color_code.txt --out_dir needle5_endovis --mode interactive
"""

import argparse
import csv
import json
import os
import shutil
import sys

import cv2
import numpy as np

# EndoVis 2018 labels.json（RGB）；找不到檔案時使用
DEFAULT_CLASSES = {
    "background-tissue": (0, 0, 0), "instrument-shaft": (0, 255, 0), "instrument-clasper": (0, 255, 255),
    "instrument-wrist": (125, 255, 12), "kidney-parenchyma": (255, 55, 0), "covered-kidney": (24, 55, 125),
    "thread": (187, 155, 25), "clamps": (0, 255, 125), "suturing-needle": (255, 255, 125),
    "suction-instrument": (123, 15, 175), "small-intestine": (124, 155, 5),
}
OUT_NEEDLE_BGR = (0, 0, 255)
OUT_THREAD_BGR = (255, 0, 255)
COLOR_CODE = """Background: 0, [0,0,0]

Tool shaft: 1, [255,0,0]

Needle: 2, [0,0,255]

Suture pad: 3, [0,255,0]

Tool end-effector: 4, [55,250,250]

Insertion points: 5, [209,125,42]

Thread: 6, [255,0,255]
"""
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def load_classes(seq_dir, stop_dir):
    d = os.path.abspath(seq_dir)
    stop = os.path.abspath(os.path.dirname(stop_dir))
    while True:
        p = os.path.join(d, "labels.json")
        if os.path.isfile(p):
            data = json.load(open(p))
            return {c["name"]: tuple(c["color"]) for c in data}, p
        if d == stop or os.path.dirname(d) == d:
            return dict(DEFAULT_CLASSES), "(內建 EndoVis 2018 顏色表)"
        d = os.path.dirname(d)


def find_sequences(root):
    out = []
    for d, subdirs, _ in os.walk(root):
        if "left_frames" in subdirs and "labels" in subdirs:
            out.append(d)
            subdirs[:] = []                                 # 不再往下
    return sorted(out)


def index_repairs(rep_root):
    """{(seq 資料夾名, 檔名): 路徑}，只收 .../<seq>/labels/<檔名> 這種結構。"""
    idx = {}
    if not rep_root:
        return idx
    for d, _, files in os.walk(rep_root):
        if os.path.basename(d) != "labels":
            continue
        seq = os.path.basename(os.path.dirname(d))
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                idx.setdefault((seq, f), []).append(os.path.join(d, f))
    return idx


def color_mask(rgb, color, tol):
    return np.all(np.abs(rgb.astype(np.int16) - np.array(color, np.int16)) <= tol, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="解壓後的 release 資料夾（可以給多個）")
    ap.add_argument("--repairs", default=None, help="repairs.zip 解壓後的資料夾（修正過的標註）")
    ap.add_argument("--out_dir", default="endovis_needle")
    ap.add_argument("--min_needle_px", type=int, default=40, help="針的像素數至少要這麼多才算「有針」")
    ap.add_argument("--require_thread", type=int, default=0,
                    help="1 = 只留「有線、且線碰到針」的影格（之後用線自動判斷頭尾）")
    ap.add_argument("--touch_px", type=float, default=15, help="線離針 <= 此值算碰到 (px)")
    ap.add_argument("--tol", type=int, default=3, help="顏色比對容許誤差")
    ap.add_argument("--link", action="store_true", help="影像用符號連結而不是複製（不佔空間）")
    ap.add_argument("--overlay", type=int, default=1, help="輸出檢查用疊圖")
    ap.add_argument("--dry_run", action="store_true", help="只統計，不輸出任何檔案")
    a = ap.parse_args()

    seqs = []
    for r in a.roots:
        if not os.path.isdir(r):
            sys.exit(f"找不到資料夾 {r}")
        found = find_sequences(r)
        if not found:
            print(f"[WARN] {r} 底下找不到同時有 left_frames/ 和 labels/ 的資料夾；請把 `find {r} -maxdepth 3 -type d` 的結果貼給我")
        seqs += [(r, s) for s in found]
    print(f"找到 {len(seqs)} 段序列")
    repairs = index_repairs(a.repairs)
    seq_count = {}
    for _, sd in seqs:
        seq_count[os.path.basename(sd)] = seq_count.get(os.path.basename(sd), 0) + 1
    if a.repairs:
        print(f"repairs：{sum(len(v) for v in repairs.values())} 張修正標註（{a.repairs}）")

    if not a.dry_run:
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, "Color_code.txt"), "w") as f:
            f.write(COLOR_CODE)
    rows, total_keep, used_rep, amb_rep = [], 0, 0, 0
    shown_json = set()
    for root, sd in seqs:
        classes, src = load_classes(sd, root)
        if src not in shown_json:
            print(f"  顏色表：{src}")
            shown_json.add(src)
        if "suturing-needle" not in classes:
            print(f"  [WARN] {sd} 的顏色表沒有 suturing-needle，跳過")
            continue
        c_needle, c_thread = classes["suturing-needle"], classes.get("thread")
        seq_name = os.path.basename(sd)
        clip = f"{os.path.basename(os.path.normpath(root))}_{seq_name}"
        lab_dir, img_dir = os.path.join(sd, "labels"), os.path.join(sd, "left_frames")
        imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)}
        n_keep = 0
        for lf in sorted(os.listdir(lab_dir)):
            if not lf.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(lf)[0]
            if stem not in imgs:
                continue
            lpath = os.path.join(lab_dir, lf)
            rep = repairs.get((seq_name, lf), [])
            if seq_count[seq_name] > 1:                         # 不同 release 都有這個 seq 名稱：repairs 路徑要含 release 名稱
                rel = os.path.basename(os.path.normpath(root))
                rep2 = [r for r in rep if rel in r.replace("\\", "/").split("/")]
                roots_n = {os.path.basename(os.path.normpath(r)) for r in a.roots}
                if rep and not rep2 and not any(set(r.replace("\\", "/").split("/")) & roots_n for r in rep):
                    amb_rep += 1                                # repairs 路徑沒有任何 release 名稱：分不出是哪一個
                rep = rep2
            if len(rep) == 1:
                lpath, used_rep = rep[0], used_rep + 1
            elif len(rep) > 1:
                amb_rep += 1                                    # 不同 release 有同名 seq：無法確定，不用 repair
            lab = cv2.imread(lpath, cv2.IMREAD_COLOR)
            if lab is None:
                continue
            rgb = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)
            nm = color_mask(rgb, c_needle, a.tol)
            tm = color_mask(rgb, c_thread, a.tol) if c_thread else np.zeros_like(nm)
            n_px, t_px = int(nm.sum()), int(tm.sum())
            n_comp = cv2.connectedComponents(nm.astype(np.uint8))[0] - 1 if n_px else 0
            touch = False
            if n_px and t_px:
                dt = cv2.distanceTransform((~tm).astype(np.uint8), cv2.DIST_L2, 5)
                touch = bool(dt[nm].min() <= a.touch_px)
            keep = n_px >= a.min_needle_px and (not a.require_thread or touch)
            rows.append({"clip": clip, "frame": imgs[stem], "needle_px": n_px, "thread_px": t_px,
                         "needle_fragments": n_comp, "thread_touches_needle": int(touch), "kept": int(keep),
                         "label_src": "repairs" if lpath != os.path.join(lab_dir, lf) else "release"})
            if not keep:
                continue
            n_keep += 1
            if a.dry_run:
                continue
            for sub in ("Images", "Labels") + (("overlay",) if a.overlay else ()):
                os.makedirs(os.path.join(a.out_dir, sub, clip), exist_ok=True)
            src_img = os.path.join(img_dir, imgs[stem])
            dst_img = os.path.join(a.out_dir, "Images", clip, imgs[stem])
            if not os.path.exists(dst_img):
                if a.link:
                    os.symlink(os.path.abspath(src_img), dst_img)
                else:
                    shutil.copy2(src_img, dst_img)
            out = np.zeros(lab.shape, np.uint8)
            out[tm] = OUT_THREAD_BGR
            out[nm] = OUT_NEEDLE_BGR                            # 針優先
            cv2.imwrite(os.path.join(a.out_dir, "Labels", clip, stem + ".png"), out)
            if a.overlay:
                im = cv2.imread(src_img)
                if im is not None and im.shape[:2] == out.shape[:2]:
                    sel = out.any(axis=2)
                    im[sel] = (0.35 * im[sel] + 0.65 * out[sel]).astype(np.uint8)
                    cv2.imwrite(os.path.join(a.out_dir, "overlay", clip, stem + ".png"), im)
        total_keep += n_keep
        n_all = sum(1 for r in rows if r["clip"] == clip)
        n_needle_any = sum(1 for r in rows if r["clip"] == clip and r["needle_px"] > 0)
        print(f"  {clip:45s} 影格 {n_all:4d}｜有針像素 {n_needle_any:4d}｜入選 {n_keep:4d}")

    if a.repairs:
        print(f"使用 repairs 標註 {used_rep} 張" + (f"；{amb_rep} 張因 seq 名稱重複而沒用（請告訴我 repairs 的資料夾結構）"
                                                   if amb_rep else ""))
    if rows and not a.dry_run:
        with open(os.path.join(a.out_dir, "frames.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    kept = [r for r in rows if r["kept"]]
    if kept:
        npx = np.array([r["needle_px"] for r in kept])
        print(f"\n入選 {len(kept)}/{len(rows)} 張；針像素 中位數 {int(np.median(npx))}（最小 {npx.min()}、最大 {npx.max()}）；"
              f"有線碰到針 {sum(r['thread_touches_needle'] for r in kept)} 張；"
              f"針斷成多段 {sum(r['needle_fragments'] > 1 for r in kept)} 張")
    else:
        print("\n沒有入選的影格：請確認顏色表（labels.json）和標註圖的顏色是否一致，或降低 --min_needle_px")
    if not a.dry_run:
        print(f"→ {a.out_dir}/（Images、Labels、overlay、frames.csv）")


if __name__ == "__main__":
    main()
