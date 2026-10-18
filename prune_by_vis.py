#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依 vis 資料夾裡「你留下來的檢查圖」決定哪些影像要拿去訓練：
vis 圖被你刪掉的影像，它的所有針都排除，不會進入訓練。

做的事（預設只列出清單，按 y 確認後才寫入；--yes 不詢問）:
    1. head_decisions.json：把這些影像的每根針標成 "bad"
       -> 之後重跑 mask_to_needle5.py（例如加了第二個 clip）也會一直排除，不會跑回來
    2. annotations.json：直接拿掉這些針（不用重跑 mask_to_needle5.py 就能 build_needle5_aug.py）
    3. （選用 --sam2_dir）sam2_video_label.py 的輸出：
       - sessions/<clip>.json 把這些影格標成「排除」-> 之後重開 clip、按 Save 也不會再輸出
       - Labels / Images / overlay 裡對應的檔案搬到 <sam2_dir>/removed/<clip>/（不是真的刪除，搬回去即可復原）
    寫入前都會先備份（*.bak_<時間>）。

注意:
    - vis 圖名稱 = 影像檔名（不含資料夾）。不同 clip 有同名影格時無法分辨，會列出來、不處理。
    - 只檢查 --splits 指定的組（預設 train）。其他組的影像不受影響。
    - 之後重跑 mask_to_needle5.py 會把 vis 圖重新畫出來（包含已排除的影像，那些圖上不會有點）；
      想再刪一輪就照常刪 vis 圖再跑一次本程式，之前排除過的會維持排除。

用法:
    python prune_by_vis.py --out_dir needle5_thread
    python prune_by_vis.py --out_dir needle5_thread --sam2_dir my_needle_thread      # 連 sam2 的 Labels 一起處理
"""

import argparse
import json
import os
import shutil
import sys
import time


def backup(path, stamp):
    if os.path.isfile(path):
        shutil.copy2(path, f"{path}.bak_{stamp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="mask_to_needle5.py 的 --out_dir（裡面有 annotations.json、vis/）")
    ap.add_argument("--splits", nargs="+", default=["train"], help="要檢查的 vis/<split> 資料夾")
    ap.add_argument("--sam2_dir", default=None,
                    help="sam2_video_label.py 的 --out_dir（有 Images/ Labels/ sessions/）；給了才會處理 sam2 的輸出")
    ap.add_argument("--yes", action="store_true", help="不詢問，直接寫入")
    ap.add_argument("--from_bad", action="store_true",
                    help="把 head_decisions.json 裡「所有針都是 bad」的影像也加進排除清單（救回之前刪過、又跑回來的圖）")
    a = ap.parse_args()

    ann_path = os.path.join(a.out_dir, "annotations.json")
    dec_path = os.path.join(a.out_dir, "head_decisions.json")
    if not os.path.isfile(ann_path):
        sys.exit(f"找不到 {ann_path}")
    ann = json.load(open(ann_path, encoding="utf-8"))
    decisions = json.load(open(dec_path)) if os.path.isfile(dec_path) else {}
    samples = ann["samples"]

    # ---- 哪些影像的 vis 圖還在 ----
    kept_vis, missing_dir = {}, []
    for sp in a.splits:
        d = os.path.join(a.out_dir, "vis", sp)
        if not os.path.isdir(d):
            missing_dir.append(d)
            continue
        kept_vis[sp] = {os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(".png")}
    if missing_dir:
        sys.exit(f"找不到 vis 資料夾：{missing_dir}")

    images = {}                                                 # rel -> (split, [needle_id])
    for s in samples:
        images.setdefault(s["image"], (s["split"], []))[1].append(s["needle_id"])
    stem = lambda rel: os.path.splitext(os.path.basename(rel))[0]

    # 不同 clip 同名影格：vis 圖會互相覆蓋，無法判斷
    by_stem = {}
    for rel, (sp, _) in images.items():
        if sp in kept_vis:
            by_stem.setdefault((sp, stem(rel)), []).append(rel)
    ambiguous = {r for v in by_stem.values() if len(v) > 1 for r in v}

    removed, kept = [], []
    for rel, (sp, ids) in sorted(images.items()):
        if sp not in kept_vis or rel in ambiguous:
            continue
        (kept if stem(rel) in kept_vis[sp] else removed).append((rel, ids))
    if a.from_bad:
        per_img = {}
        for k, v in decisions.items():
            rel, _, nid = k.rpartition("#")
            per_img.setdefault(rel, []).append(v)
        all_bad = {r for r, vs in per_img.items() if vs and all(v == "bad" for v in vs)}
        have = {r for r, _ in removed}
        extra = [(r, images.get(r, (None, []))[1]) for r in sorted(all_bad - have)]
        if extra:
            print(f"--from_bad：另外 {len(extra)} 張影像的所有針都標成 bad（之前刪過的），一併排除")
            removed += extra

    n_needles = sum(len(ids) for _, ids in removed)
    print(f"檢查 {', '.join(f'vis/{sp}' for sp in a.splits)}：保留 {len(kept)} 張、"
          f"排除 {len(removed)} 張影像（{n_needles} 根針）")
    by_clip = {}
    for rel, ids in removed:
        by_clip.setdefault(os.path.dirname(rel) or ".", []).append(os.path.basename(rel))
    for clip, fs in by_clip.items():
        print(f"  {clip}: {len(fs)} 張  例：{', '.join(fs[:3])}{' ...' if len(fs) > 3 else ''}")
    if ambiguous:
        print(f"  [WARN] {len(ambiguous)} 張影像在不同 clip 有同名檔案，無法從 vis 判斷，未處理：{sorted(ambiguous)[:4]}")
    if not removed:
        print("沒有需要排除的影像。")
        return

    if not a.yes:
        ans = input("確定排除這些影像嗎？[y/N] ").strip().lower()
        if ans != "y":
            print("已取消，沒有修改任何檔案。")
            return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    removed_set = {rel for rel, _ in removed}

    # 0. excluded_images.json：以「影像」為單位記錄（不依賴針的編號；mask_to_needle5.py 會整張跳過）
    exc_path = os.path.join(a.out_dir, "excluded_images.json")
    exc = set(json.load(open(exc_path))) if os.path.isfile(exc_path) else set()
    backup(exc_path, stamp)
    with open(exc_path, "w") as f:
        json.dump(sorted(exc | removed_set), f, indent=1)

    # 1. head_decisions.json：標成 bad（之後重跑 mask_to_needle5.py 也會排除）
    backup(dec_path, stamp)
    for rel, ids in removed:
        for nid in ids:
            decisions[f"{rel}#{nid}"] = "bad"
    with open(dec_path, "w") as f:
        json.dump(decisions, f, indent=1)

    # 2. annotations.json：拿掉
    backup(ann_path, stamp)
    ann["samples"] = [s for s in samples if s["image"] not in removed_set]
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=1, ensure_ascii=False)
    print(f"已更新 {dec_path}（{n_needles} 根針標成 bad）與 {ann_path}（剩 {len(ann['samples'])} 根針）"
          f"；備份 *.bak_{stamp}")

    # 3. sam2_video_label.py 的輸出
    if a.sam2_dir:
        n_moved, n_sess = 0, 0
        for clip, files in by_clip.items():
            sess_path = os.path.join(a.sam2_dir, "sessions", f"{clip}.json")
            if os.path.isfile(sess_path):
                sj = json.load(open(sess_path))
                idx = {f: i for i, f in enumerate(sj.get("frames", []))}
                exc = set(sj.get("excluded", []))
                add = {idx[f] for f in files if f in idx}
                if add - exc:
                    backup(sess_path, stamp)
                    sj["excluded"] = sorted(exc | add)
                    with open(sess_path, "w") as f:
                        json.dump(sj, f, indent=1)
                    n_sess += 1
            else:
                print(f"  [NOTE] 沒有 {sess_path}，只搬檔案")
            for fn in files:
                st = os.path.splitext(fn)[0]
                for sub, name in (("Labels", st + ".png"), ("Images", fn), ("overlay", st + ".png")):
                    src = os.path.join(a.sam2_dir, sub, clip, name)
                    if os.path.isfile(src):
                        dst = os.path.join(a.sam2_dir, "removed", clip, sub, name)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.move(src, dst)
                        n_moved += 1
        print(f"sam2 輸出：{n_sess} 個 session 已把影格標成排除；{n_moved} 個檔案搬到 "
              f"{os.path.join(a.sam2_dir, 'removed')}/")


if __name__ == "__main__":
    main()