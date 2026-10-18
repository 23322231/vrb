#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正：mask_to_needle5.py 的 --images_dir 誤指到「單一 clip 資料夾」時，head_decisions.json 裡
這個 clip 的 key 會少了 clip 名稱（例如 "001234.png#0"，正確應為 "<clip>/001234.png#0"）。
本程式把沒有 clip 前綴的 key 補上 --clip（寫入前備份 *.bak_<時間>）。

用法:
    python fix_clip_keys.py --out_dir needle5_thread --clip cam_260422273204_002901_1792135221.635050781
"""
import argparse
import json
import os
import shutil
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", required=True)
ap.add_argument("--clip", required=True, help="那次 --images_dir 指到的 clip 資料夾名稱")
a = ap.parse_args()

stamp = time.strftime("%Y%m%d_%H%M%S")
for name in ("head_decisions.json", "excluded_images.json"):
    p = os.path.join(a.out_dir, name)
    if not os.path.isfile(p):
        continue
    d = json.load(open(p))
    if isinstance(d, dict):
        bad = [k for k in d if "/" not in k.rpartition("#")[0]]
        new = {(f"{a.clip}/{k}" if k in bad else k): v for k, v in d.items()}
    else:
        bad = [k for k in d if "/" not in k]
        new = sorted({f"{a.clip}/{k}" if k in bad else k for k in d})
    if not bad:
        print(f"{name}: 不需要修正")
        continue
    shutil.copy2(p, f"{p}.bak_{stamp}")
    json.dump(new, open(p, "w"), indent=1)
    print(f"{name}: 補上 clip 名稱 {len(bad)} 筆（例：{bad[0]} -> {a.clip}/{bad[0]}）；備份 {p}.bak_{stamp}")
