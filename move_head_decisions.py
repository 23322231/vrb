#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把「只對單一 clip 資料夾」跑 mask_to_needle5.py 所做的頭尾決定，搬到另一個 out_dir 的 head_decisions.json，
並把 key 從 "frame.png#0" 改成 "<clip>/frame.png#0"（也就是 --images_dir 指向上一層 Images/ 時的格式）。

    python move_head_decisions.py \
        --src needle5_dataset/head_decisions.json \
        --dst needle5_mine/head_decisions.json \
        --clip_dir my_needle_masks/Images/cam_241222062132_001001_1790759233.948209717

- 只搬 clip_dir 裡真的存在的影像的決定，其他 key 不動
- dst 的同名 key 會被覆蓋（以剛剛標的為準），dst 覆蓋前會先備份
- 加 --remove_from_src 會把搬走的 key 從 src 刪掉（src 也會先備份）
"""
import argparse
import json
import os
import shutil
import time

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--clip_dir", required=True, help="剛剛當作 --images_dir 的 clip 資料夾")
    ap.add_argument("--prefix", default=None, help="key 前綴，預設是 clip_dir 的資料夾名稱")
    ap.add_argument("--remove_from_src", action="store_true")
    a = ap.parse_args()

    prefix = a.prefix or os.path.basename(os.path.normpath(a.clip_dir))
    frames = set()
    for root, _, files in os.walk(a.clip_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                frames.add(os.path.relpath(os.path.join(root, f), a.clip_dir).replace("\\", "/"))

    src = json.load(open(a.src))
    dst = json.load(open(a.dst)) if os.path.isfile(a.dst) else {}
    stamp = time.strftime("%Y%m%d_%H%M%S")

    moved = {}
    for k, v in src.items():
        rel, _, nid = k.rpartition("#")
        if rel.replace("\\", "/") in frames:
            moved[k] = (f"{prefix}/{rel.replace(chr(92), '/')}#{nid}", v)
    if not moved:
        raise SystemExit(f"[ERROR] {a.src} 裡沒有屬於 {a.clip_dir} 的決定")

    changed = sum(1 for _, (nk, v) in moved.items() if nk in dst and dst[nk] != v)
    if os.path.isfile(a.dst):
        shutil.copy2(a.dst, f"{a.dst}.bak_{stamp}")
    for _, (nk, v) in moved.items():
        dst[nk] = v
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    json.dump(dst, open(a.dst, "w"), indent=1)
    print(f"搬了 {len(moved)} 個決定到 {a.dst}（其中 {changed} 個覆蓋了不同的舊決定）")
    print(f"  例：{next(iter(moved))}  ->  {next(iter(moved.values()))[0]}")

    if a.remove_from_src:
        shutil.copy2(a.src, f"{a.src}.bak_{stamp}")
        for k in moved:
            src.pop(k)
        json.dump(src, open(a.src, "w"), indent=1)
        print(f"已從 {a.src} 移除這些 key（備份：{a.src}.bak_{stamp}）")


if __name__ == "__main__":
    main()
