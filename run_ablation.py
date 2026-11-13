#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次產生 / 執行 train_vrb_needle5.py 的各種組合，最後彙整成一張比較表。

組合的維度（每個維度可以選一個或多個值，全部做排列組合）:
    --points  3 | 5
        3 = head / tail / middle（同東京大學 Wang et al. 2025）；5 = 再加 head_1_3 / tail_1_3
    --loss    contact | mse | both
        contact = VRB 論文 Eq.2：spatial softmax 期望值座標 vs GT 的距離（官方 obj_loss 5·mean((μ−gt)²)），
                  評估用全圖期望值（--decode global，同論文）
        mse     = 東京大學論文：heatmap 直接回歸峰值 1 的高斯，逐像素 MSE；沒有座標 loss；
                  評估取 heatmap 尖峰（--decode local，同該論文取最大值的作法）
        both    = 兩個 loss 一起（contact 1 + MSE 0.1），評估用 local
    --seg     0 | 1          1 = 加 segmentation 輔助頭（--seg_weight 0.5）
    --arch    paper | attn
        paper = VRB 論文 Appendix B：deconv → cb1(64→10) → cb2(10→N)，沒有 attention、沒有 cb3（--attn_kp 0）
        attn  = 官方釋出 checkpoint 的架構：cb1、cb2(64) → 空間 attention → cb3（--attn_kp 1 --attn_mode spatial）
    --init    pretrained | scratch
    --order   ordered | perm
        ordered = 第 k 個 heatmap 固定對第 k 個部位（有順序性，原本的作法）
        perm    = 沒有順序性：K 個 heatmap 與 K 個 GT 取 loss 最小的配對；評估時預測與 GT 各自由左到右排序
                  （--order_by x，可用 -- --order_by chain 改成沿著針排）。組合名稱多一個 _unord

其餘設定一律照 VRB 論文（--paper 1）：Adam、lr 1e-4、cosine、不 warmup、不 grad clip、BN 照常更新。

預設只跑第一輪的 6 組（paper 架構、沒有 seg、用 pretrained；3 / 5 點 × 3 種 loss）。

用法:
    # 先看會跑哪些指令（不執行）
    python run_ablation.py --ann ./needle5_thread_aug/annotations.json --dry_run

    # 依序執行（已經跑完的組合會自動跳過，中斷後重下同一個指令即可接著跑）
    python run_ablation.py --ann ./needle5_thread_aug/annotations.json

    # 擴大：加上 seg、attention 架構、從零訓練（2×3×2×2×2 = 48 組）
    python run_ablation.py --ann ./needle5_thread_aug/annotations.json --seg 0 1 --arch paper attn --init pretrained scratch

    # 老師要的：沒有順序性的 p5 / p3（both、seg、paper 架構、pretrained）
    python run_ablation.py --ann ./needle5_thread_aug/annotations.json --exp unordered \
        --points 5 3 --loss both --seg 1 --arch paper --init pretrained --order perm

    # 只跑名稱包含某字串的組合
    python run_ablation.py --ann ... --only p3_mse

    # 彙整比較表（訓練中也可以看；同時存成 runs/<exp>/summary.csv）
    python run_ablation.py --summary
"""

import argparse
import csv
import glob
import itertools
import json
import os
import shlex
import subprocess
import sys

LOSS_ARGS = {
    "contact": [],                                                            # --paper 1 的預設
    "mse": ["--contact_weight", "0", "--hm_loss", "mse", "--hm_weight", "1", "--decode", "local"],
    "both": ["--contact_weight", "1", "--hm_loss", "mse", "--hm_weight", "0.1", "--decode", "local"],
}
ORDER_ARGS = {
    "ordered": [],
    "perm": ["--loss", "perm"],
}
ARCH_ARGS = {
    "paper": ["--attn_kp", "0"],
    "attn": ["--attn_kp", "1", "--attn_mode", "spatial"],
}


def combos(a):
    for pts, loss, seg, arch, init, order in itertools.product(a.points, a.loss, a.seg, a.arch, a.init, a.order):
        name = f"p{pts}_{loss}_seg{seg}_{arch}_{'pre' if init == 'pretrained' else 'scratch'}"
        if order == "perm":
            name += "_unord"
        cmd = [sys.executable, a.train_py, "--vrb_root", a.vrb_root, "--ann", a.ann,
               "--paper", "1", "--points", str(pts),
               "--pretrained", a.ckpt if init == "pretrained" else "none",
               "--exp", a.exp, "--out_dir", os.path.join("runs", name),
               "--epochs", str(a.epochs), "--batch_size", str(a.batch_size),
               *LOSS_ARGS[loss], *ARCH_ARGS[arch], *ORDER_ARGS[order],
               "--seg_weight", "0.5" if seg else "0"]
        cmd += a.extra
        yield name, cmd


def run_dirs(exp, name=None):
    pat = os.path.join("runs", exp, f"{name}_*" if name else "*")
    ds = sorted(d for d in glob.glob(pat) if os.path.isdir(d))
    if name:   # 名稱後面要直接接時間（避免 p5_..._pre 把 p5_..._pre_unord 也當成同一組）
        ds = [d for d in ds if os.path.basename(d)[len(name) + 1:][:1].isdigit()]
    return ds


def summarize(exp):
    rows = []
    for d in run_dirs(exp):
        log = os.path.join(d, "log.jsonl")
        if not os.path.isfile(log):
            continue
        recs = [json.loads(l) for l in open(log) if l.strip()]
        vals = [r for r in recs if "val" in r]
        if not vals:
            continue
        best = min(vals, key=lambda r: r["val"]["mean_px"])
        v = best["val"]
        base = json.load(open(os.path.join(d, "baseline.json"))) if os.path.isfile(os.path.join(d, "baseline.json")) else {}
        tm = json.load(open(os.path.join(d, "test_metrics.json"))) if os.path.isfile(os.path.join(d, "test_metrics.json")) else {}
        pts = [k[:-3] for k in v if k.endswith("_px") and k not in ("mean_px",) and not k.startswith("mean")]
        rows.append({
            "run": os.path.basename(d),
            "epochs": f"{recs[-1]['epoch']}",
            "best_ep": best["epoch"],
            "val_mean_px": round(v["mean_px"], 1),
            "baseline_px": round(base.get("mean_px", float("nan")), 1),
            "val_flip": f"{v.get('flip_rate', 0):.0%}",
            "val_matched_px": round(v["mean_px_matched"], 1) if "mean_px_matched" in v else "",
            "val_conf": round(v["conf_mean"], 2) if "conf_mean" in v else "",
            "val_softargmax_px": round(v.get("mean_px_softargmax", float("nan")), 1),
            "val_dead": "/".join(f"{x:.0%}" for x in v.get("hm_dead_frac", [])),
            "val_seg_iou": round(v["seg_iou"], 2) if "seg_iou" in v else "",
            "test_mean_px": round(tm["mean_px"], 1) if tm else "",
            "test_flip": f"{tm['flip_rate']:.0%}" if tm else "",
            "per_point_val": " ".join(f"{k}:{v[k + '_px']:.0f}" for k in pts),
            "train_mean_px": round(next((r["train_eval"]["mean_px"] for r in reversed(recs)
                                         if "train_eval" in r), float("nan")), 1),
        })
    if not rows:
        print(f"runs/{exp}/ 裡還沒有結果")
        return
    rows.sort(key=lambda r: r["val_mean_px"])
    cols = ["run", "best_ep", "val_mean_px", "baseline_px", "val_flip", "val_matched_px", "val_conf",
            "test_mean_px", "test_flip",
            "val_softargmax_px", "val_dead", "val_seg_iou", "per_point_val"]
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w[c]) for c in cols))
    out = os.path.join("runs", exp, "summary.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n依 val_mean_px 排序（越小越好）；baseline = 一律猜訓練集平均位置。val_mean_px 要明顯小於 baseline 才算有學到。")
    print("_unord（沒有順序性）的 run：val_mean_px = 預測與 GT 各自排序後逐點比；val_flip = 排序配錯的比例；"
          "val_matched_px = 最佳配對誤差（不受排序影響）。val_conf = 平均信心值。")
    print(f"→ {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", help="build_needle5_aug.py 的 annotations.json")
    ap.add_argument("--ckpt", default="./models/model_checkpoint_1249.pth.tar", help="VRB 官方 pretrained")
    ap.add_argument("--vrb_root", default=".")
    ap.add_argument("--train_py", default="train_vrb_needle5.py")
    ap.add_argument("--exp", default="ablation", help="結果存在 runs/<exp>/<組合名稱>_<時間>/")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--points", nargs="+", type=int, default=[3, 5], choices=[3, 5])
    ap.add_argument("--loss", nargs="+", default=["contact", "mse", "both"], choices=list(LOSS_ARGS))
    ap.add_argument("--seg", nargs="+", type=int, default=[0], choices=[0, 1])
    ap.add_argument("--arch", nargs="+", default=["paper"], choices=list(ARCH_ARGS))
    ap.add_argument("--init", nargs="+", default=["pretrained"], choices=["pretrained", "scratch"])
    ap.add_argument("--order", nargs="+", default=["ordered"], choices=list(ORDER_ARGS),
                    help="ordered = 有順序性（原本）；perm = 沒有順序性（老師的作法，名稱加 _unord）")
    ap.add_argument("--only", default=None, help="只跑名稱包含這個字串的組合")
    ap.add_argument("--dry_run", action="store_true", help="只印出指令")
    ap.add_argument("--rerun", action="store_true", help="已經跑完的組合也重跑")
    ap.add_argument("--summary", action="store_true", help="只彙整結果")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="其餘參數原封不動傳給 train_vrb_needle5.py（放在 -- 之後）")
    a = ap.parse_args()
    a.extra = [x for x in a.extra if x != "--"]

    if a.summary:
        summarize(a.exp)
        return
    if not a.ann:
        ap.error("需要 --ann")

    todo = [(n, c) for n, c in combos(a) if not a.only or a.only in n]
    print(f"共 {len(todo)} 組（每組 {a.epochs} epochs）")
    for i, (name, cmd) in enumerate(todo, 1):
        done = any(os.path.isfile(os.path.join(d, "test_metrics.json")) for d in run_dirs(a.exp, name))
        tag = "已完成，跳過" if done and not a.rerun else ""
        print(f"\n[{i}/{len(todo)}] {name} {tag}\n  " + " ".join(shlex.quote(x) for x in cmd))
        if a.dry_run or (done and not a.rerun):
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  [WARN] {name} 失敗（return code {r.returncode}），繼續下一組")
    if not a.dry_run:
        print("\n========== 結果 ==========")
        summarize(a.exp)


if __name__ == "__main__":
    main()