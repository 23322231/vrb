#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不用重新訓練：載入已經訓練好的 run，用 best 與 last 權重對 validation 畫注意力圖（Grad-CAM）。

每個 run 依它自己的 args.json 重建一模一樣的模型（點數、架構、seg 頭、softmax 溫度、解碼方式…），
所以舊的 run 也能畫。需要和 train_vrb_needle5.py（新版，有 GradCam）放在同一個資料夾。

輸出（放在各個 run 自己的資料夾裡）:
    <run>/vis/attention_best/*.png     model_best.pth.tar 的注意力圖
    <run>/vis/attention_last/*.png     model_last.pth.tar（沒有的話用 checkpoint_last.pth）的注意力圖
    <run>/cam_summary.json             每層「注意力在針上的比例」與「注意力峰值離 GT 的距離」
圖的格式見 train_vrb_needle5.py 的「注意力圖」說明：列 = 各層 + output（模型輸出的機率分布），欄 = 各個點。

用法:
    python cam_from_runs.py --runs_root runs/ablation                 # 底下所有 run，val 全部影像
    python cam_from_runs.py --runs runs/ablation/p3_both_seg0_paper_pre_20261018_164314
    python cam_from_runs.py --runs_root runs/ablation --n 16          # 每個 run 只畫前 16 張
    python cam_from_runs.py --runs_root runs/ablation --only best     # 只畫 best
"""

import argparse
import glob
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_vrb_needle5 as T  # noqa: E402


def load_args(run_dir):
    a = T.build_parser().parse_args([])                     # 先用目前版本的預設值（舊 run 沒有的新參數）
    saved = json.load(open(os.path.join(run_dir, "args.json")))
    for k, v in saved.items():
        setattr(a, k, v)
    if "points" not in saved:
        a.points = 5                                        # 舊版 run 都是 5 點
    return a


def load_state(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(st, dict) and "model" in st and isinstance(st["model"], dict):
        st = st["model"]                                    # checkpoint_last.pth
    return st


def build_net(args, state, device):
    """和 train_vrb_needle5.main() 相同的建構順序（不載入官方 pretrained，直接載入這個 run 的權重）"""
    T.configure_points(args.points)
    net = T.build_vrb(args)
    if args.attn_kp_fc:
        return None
    T.set_output_maps(net, T.NP())
    if args.kp_beta > 0:
        T.patch_get_coord(args.kp_beta)
    else:
        import networks.model as M
        orig = getattr(M, "_orig_get_coord", M.get_coord)
        M._orig_get_coord = orig

        def _capture(x, other_axis, axis_size):
            T.HM_CACHE["x"] = x
            return orig(x, other_axis, axis_size)
        M.get_coord = _capture
    T.fix_heatmap_head(net, remove_relu=args.hm_relu == 0, reinit=False)
    if args.attn_kp:
        T.patch_attention(net, args.attn_mode)
    if any(k.startswith("seg_head.") for k in state):
        T.add_seg_head(net)
    net.load_state_dict(state)
    return net.to(device).eval()


def val_loader(args):
    ann = T.load_annotations(args.ann if isinstance(args.ann, list) else [args.ann])
    if ann.get("format") == "needle5_crops":
        ds = T.Needle5CropDataset(ann, args.images_root, args.val_splits or ["val"])
    else:
        ds = T.Needle5Dataset(ann, args.images_root, args.val_splits or ["test"], args, train=False)
    return DataLoader(ds, batch_size=8, shuffle=False, num_workers=0), len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=None, help="底下每個有 args.json 的資料夾都當成一個 run")
    ap.add_argument("--runs", nargs="*", default=[], help="直接指定 run 資料夾")
    ap.add_argument("--only", choices=["best", "last", "both"], default="both")
    ap.add_argument("--n", type=int, default=0, help="每個 run 畫幾張 val（0 = 全部）")
    ap.add_argument("--overwrite", action="store_true", help="已經畫過的也重畫")
    a = ap.parse_args()

    runs = list(a.runs)
    if a.runs_root:
        runs += sorted(d for d in glob.glob(os.path.join(a.runs_root, "*")) if os.path.isfile(os.path.join(d, "args.json")))
    if not runs:
        sys.exit("沒有找到 run（--runs_root 或 --runs）")
    if not torch.cuda.is_available():
        sys.exit("[ERROR] 官方 VRBModel 需要 GPU")
    device = torch.device("cuda")
    print(f"共 {len(runs)} 個 run")

    for i, run in enumerate(runs, 1):
        print(f"\n========== [{i}/{len(runs)}] {os.path.basename(run)} ==========")
        args = load_args(run)
        if args.attn_kp_fc:
            print("  attn_kp_fc=1（fc 回歸，沒有 heatmap），跳過")
            continue
        weights = []
        if a.only in ("best", "both"):
            weights.append(("best", os.path.join(run, "model_best.pth.tar")))
        if a.only in ("last", "both"):
            last = os.path.join(run, "model_last.pth.tar")
            weights.append(("last", last if os.path.isfile(last) else os.path.join(run, T.CKPT_NAME)))
        try:
            loader, n_val = val_loader(args)
        except Exception as e:
            print(f"  [WARN] 讀不到 val 資料：{e}")
            continue
        summary_path = os.path.join(run, "cam_summary.json")
        summary = json.load(open(summary_path)) if os.path.isfile(summary_path) else {}
        for tag, path in weights:
            out_dir = os.path.join(run, "vis", f"attention_{tag}")
            if not os.path.isfile(path):
                print(f"  [{tag}] 沒有 {os.path.basename(path)}，跳過")
                continue
            if os.path.isdir(out_dir) and os.listdir(out_dir) and not a.overwrite:
                print(f"  [{tag}] 已經畫過（{out_dir}），跳過；要重畫加 --overwrite")
                continue
            try:
                net = build_net(args, load_state(path), device)
            except Exception as e:
                print(f"  [{tag}] [WARN] 模型載入失敗：{e}")
                continue
            cam = T.GradCam(net)
            print(f"  [{tag}] {os.path.basename(path)}，val {n_val} 張")
            res = T.attention_maps(net, cam, loader, device, args, out_dir, a.n or n_val)
            if res is not None:
                summary[tag] = {"weights": os.path.basename(path), "layers": res}
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
            del net, cam
            torch.cuda.empty_cache()
    print("\n完成。每個 run 的圖在 <run>/vis/attention_best/ 與 attention_last/，數字在 <run>/cam_summary.json")


if __name__ == "__main__":
    main()
