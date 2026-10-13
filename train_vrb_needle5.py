#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 mask_to_needle5.py 產生的針 5 點標註，fine-tune VRB 的 contact 預測（只訓練 contact，不看軌跡）。

5 個點的順序（VRB 的 5 個 contact map 依序對應，loss 逐點計算，順序固定）:
    0 head   1 tail   2 middle   3 head_1_3   4 tail_1_3

- 架構：直接 import 官方 networks/，建構參數與官方 demo.py 相同，從官方 pretrained 開始
- Loss：與官方 forward() 的 obj_loss 相同公式  5 * mean((mu - gt)^2)，但只呼叫 encode_img()，
        不跑軌跡分支（軌跡 loss 完全不計）
        --loss sym：每筆取「正序」與「頭尾互換」兩者較小的 loss（頭尾分不出來時的備案）
- --attn_kp_fc 1（官方預設）：全域池化 + 全連接層直接回歸座標
  --attn_kp_fc 0           ：官方原始碼的另一條路徑，每個點一張 heatmap 再取 soft-argmax，
                             空間定位通常較準。注意官方這條路徑輸出 (y, x)，本程式已自動轉換
- 裁切：以針的框為中心的正方形，邊長 = 框長邊 + 2*pad（pad = max(box_pad, box_pad_ratio*長邊)），
        再縮成 224x224。訓練時加隨機縮放 / 位移 / 翻轉
- 每次訓練存到 <out_dir>_<開始時間>/，可用 --resume 接續
- 學習曲線：每 --plot_every 個 epoch 更新 <run>/curves.png（train / val loss 與 val 各點 pixel 誤差）
  也可以隨時另外畫：python train_vrb_needle5.py --plot_only runs/needle5_fc1_20261012_181530

用法:
    python train_vrb_needle5.py --vrb_root . \
        --ann needle5_dataset/annotations.json \
        --images_root /path/to/DATA/Images \
        --pretrained ./models/model_checkpoint_1249.pth.tar \
        --out_dir runs/needle5

    python train_vrb_needle5.py --resume latest --out_dir runs/needle5
"""

import argparse
import datetime
import glob
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

POINT_NAMES = ["head", "tail", "middle", "head_1_3", "tail_1_3"]
SWAP_HEAD_TAIL = [1, 0, 2, 4, 3]                 # 頭尾互換後的對應順序
POINT_COLORS = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 140, 255), (255, 255, 0)]  # BGR
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CKPT_NAME = "checkpoint_last.pth"


# -----------------------------------------------------------------------------
# 裁切（predict 時要用同一套規則）
# -----------------------------------------------------------------------------
def needle_crop(bbox, W, H, args, rng=None):
    """bbox: (x1, y1, x2, y2) 原圖 pixel -> 正方形裁切 (x0, y0, side)。rng 不為 None 時加隨機擾動。"""
    x1, y1, x2, y2 = bbox
    long_side = max(x2 - x1, y2 - y1)
    side = long_side + 2 * max(args.box_pad, args.box_pad_ratio * long_side)
    side = max(side, args.min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if rng is not None:
        side *= rng.uniform(1 - args.scale_jitter, 1 + args.scale_jitter)
        cx += rng.uniform(-args.center_jitter, args.center_jitter) * side
        cy += rng.uniform(-args.center_jitter, args.center_jitter) * side
    side = int(round(min(side, W, H)))
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return x0, y0, side


class Needle5Dataset(Dataset):
    def __init__(self, ann, images_root, splits, args, train):
        self.samples = [s for s in ann["samples"] if s["split"] in splits and s.get("head_confirmed", True)]
        self.root = images_root
        self.args = args
        self.train = train
        t = []
        if train and args.color_aug:
            t += [transforms.RandomGrayscale(p=0.05),
                  transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=args.hue)]
        t += [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        self.tf = transforms.Compose(t)
        self.cache = {}

    def __len__(self):
        return len(self.samples)

    def _load(self, rel):
        if rel not in self.cache:
            self.cache[rel] = Image.open(os.path.join(self.root, rel)).convert("RGB")
        return self.cache[rel]

    def __getitem__(self, i):
        s = self.samples[i]
        img = self._load(s["image"])
        W, H = img.size
        pts = np.asarray(s["points_px"], dtype=np.float64)                   # (5, 2) x, y

        # train：全域 random（DataLoader 每個 worker / epoch 會重新 seed）；val：不加擾動
        x0, y0, side = needle_crop(s["needle_bbox"], W, H, self.args, random if self.train else None)
        crop = img.crop((x0, y0, x0 + side, y0 + side)).resize((224, 224), Image.BILINEAR)
        pn = (pts - np.array([x0, y0])) / side                               # crop 內 normalized (x, y)

        # 翻轉不改變點的語意（針頭還是針頭），所以順序不變
        if self.train and random.random() < self.args.hflip:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            pn[:, 0] = 1.0 - pn[:, 0]
        if self.train and random.random() < self.args.vflip:
            crop = crop.transpose(Image.FLIP_TOP_BOTTOM)
            pn[:, 1] = 1.0 - pn[:, 1]
        pn = pn.astype(np.float32)

        return {
            "img": self.tf(crop),
            "img_raw": torch.from_numpy(np.asarray(crop).copy()),
            "pts": torch.from_numpy(pn),                                     # (5, 2) x, y
            "crop": torch.tensor([x0, y0, side], dtype=torch.float32),
            "name": f"{os.path.splitext(os.path.basename(s['image']))[0]}_n{s['needle_id']}.png",
        }


# -----------------------------------------------------------------------------
# 模型（完全照官方 demo.py 建構）
# -----------------------------------------------------------------------------
def build_vrb(args):
    sys.path.insert(0, os.path.abspath(args.vrb_root))
    from networks.model import VRBModel
    from networks.traj import TrajAffCVAE

    hand_head = TrajAffCVAE(in_dim=2 * args.traj_len, hidden_dim=args.hidden_dim,
                            latent_dim=args.hand_latent_dim, condition_dim=args.cond_dim,
                            coord_dim=args.coord_dim, traj_len=args.traj_len)
    src_in_features = 2048 if args.resnet_type == "resnet50" else 512
    return VRBModel(src_in_features=src_in_features, num_patches=1, hidden_dim=args.hidden_dim,
                    hand_head=hand_head, encoder_time_embed_type=args.encoder_time_embed_type,
                    num_frames_input=10, resnet_type=args.resnet_type,
                    embed_dim=args.cond_dim, coord_dim=args.coord_dim,
                    num_heads=args.num_heads, enc_depth=args.enc_depth,
                    attn_kp=args.attn_kp, attn_kp_fc=args.attn_kp_fc, n_maps=5)


def load_pretrained(net, path):
    dt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(dt, dict) and "state_dict" in dt and isinstance(dt["state_dict"], dict):
        dt = dt["state_dict"]
    dt = {k[7:] if k.startswith("module.") else k: v for k, v in dt.items()}
    missing, unexpected = net.load_state_dict(dt, strict=False)
    print(f"載入 pretrained: {path}")
    print(f"  missing keys   : {len(missing)} {missing[:10]}")
    print(f"  unexpected keys: {len(unexpected)} {unexpected[:10]}")


def set_bn_eval(module):
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def predict_xy(net, img, args):
    """回傳 (B, 5, 2) crop 內 normalized 的 (x, y)。attn_kp_fc=0 時官方輸出是 (y, x)，在這裡轉回來。"""
    _, mu, _ = net.encode_img(img)
    if not args.attn_kp_fc:
        mu = mu.flip(-1)
    return mu


def contact_loss(pred, gt, mode):
    """與官方 obj_loss 相同：5 * mean((mu - gt)^2)。sym：每筆取正序 / 頭尾互換較小者。"""
    per = ((pred - gt) ** 2).mean(dim=(1, 2))                                 # (B,)
    if mode == "sym":
        per_sw = ((pred - gt[:, SWAP_HEAD_TAIL]) ** 2).mean(dim=(1, 2))
        per = torch.minimum(per, per_sw)
    return 5.0 * per.mean()


# -----------------------------------------------------------------------------
# 評估 / 視覺化
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, loader, device, args, vis_dir=None):
    net.eval()
    sum_err = torch.zeros(5)
    n, loss_sum, flips = 0, 0.0, 0
    for batch in loader:
        img = batch["img"].to(device)
        gt = batch["pts"].to(device)
        pred = predict_xy(net, img, args)
        loss_sum += contact_loss(pred, gt, args.loss).item() * img.shape[0]
        side = batch["crop"][:, 2].to(device)[:, None]
        err = (pred - gt).norm(dim=-1) * side                                # (B, 5) 原圖 pixel
        err_sw = (pred - gt[:, SWAP_HEAD_TAIL]).norm(dim=-1) * side
        flips += int((err_sw.mean(1) < err.mean(1)).sum())
        sum_err += err.sum(0).cpu()
        n += img.shape[0]
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            for b in range(img.shape[0]):
                draw(batch["img_raw"][b].numpy(), gt[b].cpu().numpy(), pred[b].cpu().numpy(),
                     os.path.join(vis_dir, batch["name"][b]))
    per = (sum_err / max(n, 1)).tolist()
    m = {"loss": loss_sum / max(n, 1), "mean_px": float(np.mean(per)), "flip_rate": flips / max(n, 1)}
    m.update({f"{k}_px": v for k, v in zip(POINT_NAMES, per)})
    return m


def draw(img_rgb, gt, pred, path, scale=3, r=2):
    """GT：實心 2px 圓；預測：同顏色空心圓（半徑 r+2），並以細線連到對應 GT。"""
    img = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_LINEAR)
    S = 224 * scale
    P = lambda p: (int(round(p[0] * S)), int(round(p[1] * S)))
    for g, p, c in zip(gt, pred, POINT_COLORS):
        cv2.line(img, P(g), P(p), c, 1, cv2.LINE_AA)
        cv2.circle(img, P(g), r, c, -1, cv2.LINE_AA)
        cv2.circle(img, P(p), r + 2, c, 1, cv2.LINE_AA)
    y = 16
    for name, c in zip(POINT_NAMES, POINT_COLORS):
        cv2.circle(img, (10, y - 4), 4, c, -1, cv2.LINE_AA)
        cv2.putText(img, name, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16
    cv2.putText(img, "filled = GT   ring = pred", (8, S - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


# -----------------------------------------------------------------------------
# 學習曲線
# -----------------------------------------------------------------------------
def plot_curves(run_dir):
    """讀 <run_dir>/log.jsonl 畫 curves.png。接續訓練造成的重複 epoch 以最後一筆為準。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(run_dir, "log.jsonl")
    if not os.path.isfile(path):
        return None
    recs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["epoch"]] = r
    if not recs:
        return None
    ep = sorted(recs)
    tr = [recs[e]["train_loss"] for e in ep]
    ve = [e for e in ep if "val" in recs[e]]
    vl = [recs[e]["val"]["loss"] for e in ve]

    ink, muted, grid = "#1f2328", "#59636e", "#e6e8eb"
    c_train, c_val = "#2a78d6", "#d9480f"
    pt_colors = ["#d62728", "#1f5bd6", "#2b9348", "#f08c00", "#0fa3b1"]   # 與 vis 圖的點同色系

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=120)
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9)
        ax.set_xlabel("epoch", color=muted)

    # (1) loss
    ax = axes[0]
    ax.plot(ep, tr, color=c_train, linewidth=2, label="train loss (with augmentation)")
    if ve:
        ax.plot(ve, vl, color=c_val, linewidth=2, label="val loss")
        b = int(np.argmin(vl))
        ax.scatter([ve[b]], [vl[b]], s=40, color=c_val, edgecolors="white", linewidths=1.5, zorder=3)
        right = ve[b] > ep[0] + 0.7 * (ep[-1] - ep[0])
        ax.annotate(f"best val {vl[b]:.4f} @ ep {ve[b]}", (ve[b], vl[b]), xytext=(-8 if right else 8, 12),
                    textcoords="offset points", fontsize=9, color=ink, ha="right" if right else "left")
    ax.set_yscale("log")
    ax.set_title("Contact loss  5·mean((pred−gt)²)", color=ink, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=ink)

    # (2) val pixel 誤差
    ax = axes[1]
    if ve:
        for name, c in zip(POINT_NAMES, pt_colors):
            ax.plot(ve, [recs[e]["val"][f"{name}_px"] for e in ve], color=c, linewidth=1.2,
                    alpha=0.85, label=name)
        vm = [recs[e]["val"]["mean_px"] for e in ve]
        ax.plot(ve, vm, color=ink, linewidth=2.4, label="val mean")
        te = [e for e in ep if "train_eval" in recs[e]]
        if te:
            ax.plot(te, [recs[e]["train_eval"]["mean_px"] for e in te], color=ink, linewidth=1.5,
                    linestyle="--", marker="o", markersize=4, label="train mean (no aug)")
        b = int(np.argmin(vm))
        right = ve[b] > ep[0] + 0.7 * (ep[-1] - ep[0])
        ax.scatter([ve[b]], [vm[b]], s=40, color=ink, edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(f"best {vm[b]:.1f}px @ ep {ve[b]}", (ve[b], vm[b]), xytext=(-8 if right else 8, -16),
                    textcoords="offset points", fontsize=9, color=ink, ha="right" if right else "left")
    ax.set_title("Point error in original-image pixels", color=ink, fontsize=11, loc="left")
    ax.set_ylabel("px", color=muted)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, ncol=2, labelcolor=ink)

    fig.suptitle(os.path.basename(os.path.normpath(run_dir)), color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out = os.path.join(run_dir, "curves.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# run 資料夾 / checkpoint（與 train_vrb.py 相同）
# -----------------------------------------------------------------------------
def make_run_dir(prefix):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{prefix.rstrip('/').rstrip(os.sep)}_{stamp}"
    run_dir, k = base, 1
    while os.path.exists(run_dir):
        run_dir = f"{base}_{k}"
        k += 1
    os.makedirs(run_dir)
    return run_dir


def resolve_resume(resume, out_prefix):
    if resume == "latest":
        cands = [d for d in glob.glob(f"{out_prefix.rstrip('/')}_*")
                 if os.path.isfile(os.path.join(d, CKPT_NAME))]
        if not cands:
            sys.exit(f"[ERROR] 找不到可以接續的 run：{out_prefix}_*/{CKPT_NAME}")
        d = max(cands, key=lambda d: os.path.getmtime(os.path.join(d, CKPT_NAME)))
        return d, os.path.join(d, CKPT_NAME)
    if os.path.isdir(resume):
        c = os.path.join(resume, CKPT_NAME)
        if not os.path.isfile(c):
            sys.exit(f"[ERROR] {resume} 裡沒有 {CKPT_NAME}")
        return resume, c
    if os.path.isfile(resume):
        return os.path.dirname(os.path.abspath(resume)), resume
    sys.exit(f"[ERROR] --resume 路徑不存在：{resume}")


def save_checkpoint(path, net, optim, epoch, step, best, args):
    state = {"model": net.state_dict(), "optim": optim.state_dict(), "epoch": epoch, "step": step,
             "best": best, "args": vars(args),
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                     "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}}
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


# -----------------------------------------------------------------------------
def main(args):
    if not torch.cuda.is_available():
        sys.exit("[ERROR] 官方 VRBModel 在 __init__ 內對 BatchNorm 呼叫了 .cuda()，需要 GPU")
    device = torch.device("cuda")

    resume_state = None
    if args.resume:
        run_dir, ckpt_path = resolve_resume(args.resume, args.out_dir)
        resume_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for k, v in resume_state["args"].items():
            if k != "resume":
                setattr(args, k, v)
        print(f"接續訓練：{run_dir}（已完成 epoch {resume_state['epoch']} / {args.epochs}）")
        if resume_state["epoch"] >= args.epochs:
            sys.exit("這個 run 已經訓練完成")
    else:
        run_dir = make_run_dir(args.out_dir)
        for k in ("vrb_root", "ann", "pretrained"):
            setattr(args, k, os.path.abspath(getattr(args, k)))
        print(f"新的訓練，結果存到：{run_dir}")

    with open(args.ann, "r", encoding="utf-8") as f:
        ann = json.load(f)
    images_root = args.images_root or ann.get("images_root")
    if not images_root or not os.path.isdir(images_root):
        sys.exit(f"[ERROR] 找不到影像資料夾 {images_root}，請用 --images_root 指定（container 內的路徑）")
    args.images_root = os.path.abspath(images_root)
    if not args.resume:
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_set = Needle5Dataset(ann, args.images_root, args.train_splits, args, train=True)
    val_set = Needle5Dataset(ann, args.images_root, args.val_splits, args, train=False)
    train_eval_set = Needle5Dataset(ann, args.images_root, args.train_splits, args, train=False)
    print(f"train {len(train_set)} 根針 ({args.train_splits}) / val {len(val_set)} 根針 ({args.val_splits})")
    if len(train_set) == 0:
        sys.exit("[ERROR] 沒有訓練資料")
    bs = min(args.batch_size, len(train_set))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=0) if len(val_set) else None
    train_eval_loader = DataLoader(train_eval_set, batch_size=bs, shuffle=False, num_workers=0)

    net = build_vrb(args)
    if resume_state is None:
        load_pretrained(net, args.pretrained)
    net = net.to(device)

    backbone = list(net.r3m.parameters())
    bids = {id(p) for p in backbone}
    others = [p for p in net.parameters() if id(p) not in bids]
    if args.freeze_backbone:
        for p in backbone:
            p.requires_grad = False
        groups = [{"params": others, "lr": args.lr}]
    else:
        groups = [{"params": others, "lr": args.lr},
                  {"params": backbone, "lr": args.lr * args.backbone_lr_mult}]
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in groups]
    steps_per_epoch = math.ceil(len(train_set) / bs)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(args.warmup_steps, total_steps // 10)

    def set_lr(step):
        f = (step + 1) / warmup if step < warmup else \
            0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))
        for g, b in zip(optim.param_groups, base_lrs):
            g["lr"] = b * f

    start_epoch, step, best = 1, 0, float("inf")
    if resume_state is not None:
        net.load_state_dict(resume_state["model"])
        optim.load_state_dict(resume_state["optim"])
        start_epoch, step, best = resume_state["epoch"] + 1, resume_state["step"], resume_state["best"]
        r = resume_state["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"])
        torch.cuda.set_rng_state_all(r["cuda"])
        del resume_state
    elif val_loader:
        m = evaluate(net, val_loader, device, args, os.path.join(run_dir, "vis", "epoch0000_val"))
        print("[pretrained] val: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    ckpt_path = os.path.join(run_dir, CKPT_NAME)
    log_f = open(os.path.join(run_dir, "log.jsonl"), "a")
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            net.train()
            if args.freeze_bn:
                set_bn_eval(net)
            t0 = time.time()
            tot, n = 0.0, 0
            for batch in train_loader:
                set_lr(step)
                img = batch["img"].to(device)
                gt = batch["pts"].to(device)
                loss = contact_loss(predict_xy(net, img, args), gt, args.loss)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optim.step()
                step += 1
                tot += loss.item() * img.shape[0]
                n += img.shape[0]

            rec = {"epoch": epoch, "lr": optim.param_groups[0]["lr"], "train_loss": tot / n}
            msg = f"ep {epoch:4d} | train loss {tot / n:.5f}"
            do_vis = epoch % args.vis_every == 0 or epoch == args.epochs
            if val_loader and (epoch % args.eval_every == 0 or do_vis):
                vm = evaluate(net, val_loader, device, args,
                              os.path.join(run_dir, "vis", f"epoch{epoch:04d}_val") if do_vis else None)
                rec["val"] = vm
                msg += (f" | val mean {vm['mean_px']:.1f}px (head {vm['head_px']:.1f} tail {vm['tail_px']:.1f} "
                        f"mid {vm['middle_px']:.1f} h1/3 {vm['head_1_3_px']:.1f} t1/3 {vm['tail_1_3_px']:.1f}) "
                        f"flip {vm['flip_rate']:.0%}")
                if vm["mean_px"] < best:
                    best = vm["mean_px"]
                    torch.save(net.state_dict(), os.path.join(run_dir, "model_best.pth.tar"))
                    msg += "  *best*"
            if do_vis:
                tm = evaluate(net, train_eval_loader, device, args,
                              os.path.join(run_dir, "vis", f"epoch{epoch:04d}_train"))
                rec["train_eval"] = tm
                msg += f" | train mean {tm['mean_px']:.1f}px"
            print(msg + f" | {time.time() - t0:.1f}s")
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if epoch % args.save_every == 0:
                torch.save(net.state_dict(), os.path.join(run_dir, f"model_ep{epoch:04d}.pth.tar"))
            if epoch % args.ckpt_every == 0 or epoch == args.epochs:
                save_checkpoint(ckpt_path, net, optim, epoch, step, best, args)
            if epoch % args.plot_every == 0 or epoch == args.epochs:
                plot_curves(run_dir)
    except KeyboardInterrupt:
        done = epoch - 1
        plot_curves(run_dir)
        print(f"\n[中斷] 已儲存進度到 epoch {done}，之後可用：python train_vrb_needle5.py --resume {run_dir}")
        if done >= start_epoch:
            save_checkpoint(ckpt_path, net, optim, done, done * steps_per_epoch, best, args)
        log_f.close()
        sys.exit(1)

    torch.save(net.state_dict(), os.path.join(run_dir, "model_last.pth.tar"))
    log_f.close()
    print(f"完成，best val mean = {best:.1f}px，結果在 {run_dir}（學習曲線：curves.png）")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vrb_root")
    p.add_argument("--ann", help="mask_to_needle5.py 輸出的 annotations.json")
    p.add_argument("--images_root", default=None, help="影像根目錄（預設用 annotations.json 裡的；container 路徑不同時要指定）")
    p.add_argument("--pretrained")
    p.add_argument("--out_dir", default="runs/needle5", help="實際存到 <out_dir>_<開始時間>/")
    p.add_argument("--resume", default=None, help="run 資料夾 / checkpoint_last.pth / latest")
    p.add_argument("--train_splits", nargs="+", default=["train", "val"])
    p.add_argument("--val_splits", nargs="+", default=["test"])

    # 架構（與官方 demo.py 相同；attn_kp_fc 可選 0 / 1）
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--enc_depth", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--hand_latent_dim", type=int, default=4)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--coord_dim", type=int, default=64)
    p.add_argument("--resnet_type", type=str, default="resnet18")
    p.add_argument("--attn_kp", type=int, default=1)
    p.add_argument("--attn_kp_fc", type=int, default=1, help="1: 全連接回歸（官方預設）；0: heatmap + soft-argmax")
    p.add_argument("--traj_len", type=int, default=5)
    p.add_argument("--encoder_time_embed_type", default="sin", choices=["sin", "param"])

    # 裁切 / augmentation
    p.add_argument("--box_pad", type=float, default=20, help="針的框四周最少留白 (px)")
    p.add_argument("--box_pad_ratio", type=float, default=0.15, help="留白至少為框長邊的此比例")
    p.add_argument("--min_side", type=float, default=96, help="裁切邊長下限 (px)")
    p.add_argument("--scale_jitter", type=float, default=0.15)
    p.add_argument("--center_jitter", type=float, default=0.08)
    p.add_argument("--hflip", type=float, default=0.5)
    p.add_argument("--vflip", type=float, default=0.0)
    p.add_argument("--color_aug", type=int, default=1)
    p.add_argument("--hue", type=float, default=0.05)

    # 訓練
    p.add_argument("--loss", choices=["ordered", "sym"], default="ordered")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1)
    p.add_argument("--freeze_backbone", type=int, default=0)
    p.add_argument("--freeze_bn", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--vis_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=10)
    p.add_argument("--plot_every", type=int, default=5, help="每幾個 epoch 更新一次 curves.png")
    p.add_argument("--plot_only", default=None, metavar="RUN_DIR",
                   help="不訓練，只依 RUN_DIR/log.jsonl 畫 curves.png（訓練中也可以用）")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.plot_only:
        out = plot_curves(a.plot_only)
        print(f"→ {out}" if out else f"[ERROR] {a.plot_only} 裡沒有 log.jsonl")
        sys.exit(0)
    if not a.resume and not (a.vrb_root and a.ann and a.pretrained):
        p.error("新的訓練需要 --vrb_root、--ann、--pretrained（接續只需要 --resume）")
    main(a)