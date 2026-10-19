#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用訓練好的 YOLO segmentation 模型自動標註一段 clip 的「針」，輸出成 sam2_video_label.py 相同的格式，
之後直接用 sam2_video_label.py（同樣的 --frames_dir / --start / --num / --stride）開啟檢查、修改。

    <out_dir>/Images/<clip>/<原檔名>              原始影格（複製）
    <out_dir>/Labels/<clip>/<原檔名去副檔名>.png   mask：針 = BGR (0,0,255)，其餘黑色
    <out_dir>/overlay/<clip>/...                   疊圖（快速瀏覽用）
    <out_dir>/sessions/<clip>.json                 讓 sam2_video_label.py 認得這段 clip、從 Labels 讀回 mask
    <out_dir>/Color_code.txt                       （沒有才寫）

clip 名稱 / 影格挑選規則與 sam2_video_label.py 完全相同：
    clip = 排序後的影格[start : start + num*stride : stride]，名稱 = <frames_dir 上一層資料夾名>_<第一張檔名>

說明:
    - 只保留 confidence >= --conf 的偵測；同一張影格的多個偵測合併成一張針的 mask
    - 沒有偵測到針的影格不輸出 Labels / Images（和 sam2_video_label.py 一樣），但仍在 clip 裡，開啟後可以手動補標
    - YOLO 只偵測針、不偵測線：要用線判斷頭尾的話，在 sam2_video_label.py 裡選 Thread 補標後 Track
    - 這段 clip 已經有 Labels 或 session 時會停止（避免蓋掉你修過的結果），確定要覆蓋加 --overwrite

用法:
    python yolo_autolabel.py --weights runs/segment/train/weights/best.pt \\
        --frames_dir /data/camera_recordings/20261016_071708/cam_260422273204/rgb \\
        --start 005136_1792135370.605606445 --stride 15 --out_dir my_needle_thread
"""

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
NEEDLE_BGR = (0, 0, 255)
COLOR_CODE = "Needle: 1, [0, 0, 255]\nThread: 6, [255, 0, 255]\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="YOLO segmentation 權重（best.pt）")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", default="my_needle_thread")
    ap.add_argument("--start", default="0", help="起始影格：index（整數）或檔名的一部分（包含這一張）")
    ap.add_argument("--num", type=int, default=0, help="取幾張；0 = 一直取到資料夾最後")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--clip_name", default=None, help="預設與 sam2_video_label.py 相同：<上一層資料夾名>_<第一張檔名>")
    ap.add_argument("--conf", type=float, default=0.5, help="只保留 confidence >= 這個值的偵測")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="例如 0 或 cpu；預設自動")
    ap.add_argument("--overwrite", action="store_true", help="這段 clip 已有 Labels / session 時仍然覆蓋")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.frames_dir) if f.lower().endswith(IMG_EXTS))
    if not files:
        sys.exit(f"{a.frames_dir} 沒有影像")
    if a.start.isdigit():
        s0 = int(a.start)
    else:
        s0 = next((i for i, f in enumerate(files) if a.start in f), None)
        if s0 is None:
            sys.exit(f"找不到檔名包含 '{a.start}' 的影格")
    end = len(files) if a.num <= 0 else s0 + a.num * a.stride
    clip = files[s0:end:a.stride]
    parent = os.path.basename(os.path.normpath(os.path.dirname(os.path.abspath(a.frames_dir))))
    clip_name = a.clip_name or f"{parent}_{os.path.splitext(clip[0])[0]}"
    print(f"clip {clip_name}：{len(clip)} 張（{clip[0]} ~ {clip[-1]}，stride {a.stride}）")

    dirs = {sub: os.path.join(a.out_dir, sub, clip_name) for sub in ("Images", "Labels", "overlay")}
    sess_path = os.path.join(a.out_dir, "sessions", f"{clip_name}.json")
    edits_path = os.path.join(a.out_dir, "sessions", f"{clip_name}_edits.npz")
    has_old = os.path.isfile(sess_path) or (os.path.isdir(dirs["Labels"]) and os.listdir(dirs["Labels"]))
    if has_old and not a.overwrite:
        sys.exit(f"[停止] {clip_name} 已經有 Labels 或 session（可能是你修過的結果）。確定要覆蓋請加 --overwrite")
    if a.overwrite:
        for d in dirs.values():
            shutil.rmtree(d, ignore_errors=True)
        for p in (sess_path, edits_path):
            if os.path.exists(p):
                os.remove(p)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(sess_path), exist_ok=True)
    cc = os.path.join(a.out_dir, "Color_code.txt")
    if not os.path.isfile(cc):
        with open(cc, "w") as f:
            f.write(COLOR_CODE)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    print(f"模型類別：{model.names}；conf >= {a.conf}")

    n_pos, n_det = 0, 0
    for i, f in enumerate(clip):
        path = os.path.join(a.frames_dir, f)
        img = cv2.imread(path)
        if img is None:
            print(f"  [WARN] 讀不到 {path}")
            continue
        H, W = img.shape[:2]
        r = model.predict(img, conf=a.conf, imgsz=a.imgsz, retina_masks=True, device=a.device, verbose=False)[0]
        sel = np.zeros((H, W), bool)
        if r.masks is not None and len(r.masks):
            ms = r.masks.data.cpu().numpy() > 0.5                          # retina_masks：原圖解析度 (N, H, W)
            confs = r.boxes.conf.cpu().numpy()
            for m, c in zip(ms, confs):
                if c >= a.conf:
                    if m.shape != (H, W):
                        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                    sel |= m
                    n_det += 1
        stem = os.path.splitext(f)[0]
        if not sel.any():
            continue
        n_pos += 1
        lab = np.zeros((H, W, 3), np.uint8)
        lab[sel] = NEEDLE_BGR
        cv2.imwrite(os.path.join(dirs["Labels"], stem + ".png"), lab)
        shutil.copy2(path, os.path.join(dirs["Images"], f))
        ov = img.copy()
        ov[sel] = (0.4 * ov[sel] + 0.6 * np.array(NEEDLE_BGR)).astype(np.uint8)
        cv2.imwrite(os.path.join(dirs["overlay"], stem + ".png"), ov)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(clip)}（有針 {n_pos} 張）")

    # session：讓 sam2_video_label.py 認得這段 clip，並從 Labels 讀回 mask（物件 1 = 針）
    with open(sess_path, "w") as fs:
        json.dump({"frames_dir": os.path.abspath(a.frames_dir), "frames": clip, "stride": a.stride,
                   "obj_ids": [1], "obj_cls": {"1": "needle"}, "excluded": [], "points": {}, "boxes": {},
                   "source": f"yolo_autolabel.py ({os.path.abspath(a.weights)}, conf >= {a.conf})"},
                  fs, indent=1)

    print(f"\n完成：{len(clip)} 張中 {n_pos} 張偵測到針（共 {n_det} 個偵測），{len(clip) - n_pos} 張沒有")
    print(f"  → {dirs['Labels']}（疊圖：{dirs['overlay']}）")
    print("\n用 sam2_video_label.py 檢查（參數要一樣，才會是同一段 clip、讀回這些 mask）：")
    print(f"  python sam2_video_label.py --frames_dir {a.frames_dir} --out_dir {a.out_dir} "
          f"--start {clip[0]} --num {len(clip)} --stride {a.stride}")


if __name__ == "__main__":
    main()
