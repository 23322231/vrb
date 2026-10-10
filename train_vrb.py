#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 VRB 官方 pretrained weight (model_checkpoint_1249.pth.tar) 開始，用自己 label 的 hand_off 資料 fine-tune。

- 網路架構：直接 import 官方 repo 的 networks/model.py、networks/traj.py，
  建構參數與官方 demo.py 完全相同（resnet18、cond_dim 256、hidden_dim 192、traj_len 5 ...）。
- Loss：直接用官方 VRBModel.forward() 回傳的 traj_loss / traj_kl_loss / obj_loss。
- 輸入前處理：沿用官方 inference.py 的 transform（Resize 224、RandomGrayscale、ColorJitter、Normalize）。
- 輸出的 checkpoint 是純 state_dict，格式和官方一樣，官方 demo.py 可以直接 load。

GT 格式（依官方 forward / inference 推得，官方沒有釋出訓練 code）：
    contact_point : (B, 5, 2)  裁切後影像中的 normalized (x, y)。n_maps = 5，我們只有 1 個 contact，所以重複 5 次
    future_hands  : (B, 5, 2)  每個軌跡點相對 contact 的位移，除以裁切邊長
                               y 預設「往上為正」，對應官方 inference.py 畫箭頭時用的 -dy（--traj_y_up 0 可關掉）

用法:
    # 新的訓練：結果存到 runs/handoff_ft_<開始時間>/（例如 runs/handoff_ft_20260926_134512）
    python train_vrb.py \
        --vrb_root /path/to/vrb \
        --data_dir vrb_dataset \
        --pretrained /path/to/vrb/models/model_checkpoint_1249.pth.tar \
        --out_dir runs/handoff_ft

    # 中斷後接續訓練（繼續寫進原本那個資料夾，所有設定沿用當初的 args.json）
    python train_vrb.py --resume runs/handoff_ft_20260926_134512
    python train_vrb.py --resume latest --out_dir runs/handoff_ft   # 自動找 runs/handoff_ft_* 裡最新的一個

每個 run 資料夾內容:
    args.json                 這次訓練的所有參數
    log.jsonl                 每個 epoch 的 loss / 評估結果
    checkpoint_last.pth       接續訓練用（模型 + optimizer + epoch + 亂數狀態）
    model_best.pth.tar        val loss 最低的權重（純 state_dict，官方 demo.py 可直接載入）
    model_epXXXX.pth.tar      每 save_every 個 epoch 的權重
    model_last.pth.tar        訓練結束時的權重
    vis/                      預測視覺化
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


# -----------------------------------------------------------------------------
# 資料
# -----------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_color_transform(train, args):
    """和官方 inference.py 的 transform 相同（Resize 在裁切時已經做了）。"""
    t = []
    if train and args.color_aug:
        t += [transforms.RandomGrayscale(p=0.05),
              transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=args.hue)]
    t += [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(t)


def compute_crop(cx, cy, W, H, train, args, rng):
    """
    回傳正方形裁切 (x0, y0, side)，單位為 pixel。
    官方 VRB 是在物體 bbox 周圍裁一個正方形再 resize 到 224。
      - 有給 --roi：以 ROI 為基準（固定相機時推論也用同一個 ROI）
      - 沒給：以 contact 為中心 ±center_jitter 的隨機偏移，避免模型學到「contact 永遠在正中間」
    """
    if args.roi is not None:
        x1, y1, x2, y2 = args.roi
        side = max(x2 - x1, y2 - y1)
        ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
        if train:
            side *= rng.uniform(1 - args.scale_jitter, 1 + args.scale_jitter)
            ccx += rng.uniform(-0.05, 0.05) * side
            ccy += rng.uniform(-0.05, 0.05) * side
    else:
        side = args.crop_size
        ccx, ccy = cx, cy
        if train:
            side *= rng.uniform(1 - args.scale_jitter, 1 + args.scale_jitter)
            ccx += rng.uniform(-args.center_jitter, args.center_jitter) * side
            ccy += rng.uniform(-args.center_jitter, args.center_jitter) * side
    side = int(round(min(side, W, H)))
    x0 = int(round(min(max(ccx - side / 2, 0), W - side)))
    y0 = int(round(min(max(ccy - side / 2, 0), H - side)))
    return x0, y0, side


class HandoffDataset(Dataset):
    def __init__(self, data_dir, split, args, train):
        with open(os.path.join(data_dir, "annotations.json"), "r", encoding="utf-8") as f:
            ann = json.load(f)
        self.samples = [s for s in ann["samples"] if s["split"] == split]
        self.data_dir = data_dir
        self.args = args
        self.train = train
        self.color_tf = build_color_transform(train, args)
        self.cache = {}

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        if path not in self.cache:
            self.cache[path] = Image.open(path).convert("RGB")
        return self.cache[path]

    def __getitem__(self, i):
        s = self.samples[i]
        img = self._load(os.path.join(self.data_dir, s["image"]))
        W, H = img.size
        cx, cy = s["contact"][0] * W, s["contact"][1] * H
        traj = np.asarray(s["traj"], dtype=np.float64) * np.array([W, H])

        # train 用全域 random（DataLoader 每個 worker / 每個 epoch 會重新 seed，augmentation 才會不同）
        # val 用固定 seed，每次裁切都一樣
        rng = random if self.train else random.Random(i)
        x0, y0, side = compute_crop(cx, cy, W, H, self.train, self.args, rng)
        crop = img.crop((x0, y0, x0 + side, y0 + side)).resize((224, 224), Image.BILINEAR)

        # contact：裁切後 normalized，重複 n_maps 次
        c_norm = np.array([(cx - x0) / side, (cy - y0) / side], dtype=np.float32)
        contact = np.repeat(c_norm[None], self.args.n_maps, axis=0)

        # 軌跡：相對 contact 的位移 / 裁切邊長
        off = (traj - np.array([cx, cy])) / side
        if self.args.traj_y_up:
            off[:, 1] = -off[:, 1]
        off = off.astype(np.float32)

        return {
            "img": self.color_tf(crop),
            "img_raw": torch.from_numpy(np.asarray(crop).copy()),  # 224x224x3 uint8 RGB，畫圖用
            "contact": torch.from_numpy(contact),
            "traj": torch.from_numpy(off),
            "crop": torch.tensor([x0, y0, side], dtype=torch.float32),
            "name": os.path.basename(s["image"]),
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
    net = VRBModel(src_in_features=src_in_features,
                   num_patches=1,
                   hidden_dim=args.hidden_dim,
                   hand_head=hand_head,
                   encoder_time_embed_type=args.encoder_time_embed_type,
                   num_frames_input=10,
                   resnet_type=args.resnet_type,
                   embed_dim=args.cond_dim, coord_dim=args.coord_dim,
                   num_heads=args.num_heads, enc_depth=args.enc_depth,
                   attn_kp=args.attn_kp, attn_kp_fc=args.attn_kp_fc, n_maps=args.n_maps)
    return net


def load_pretrained(net, path):
    dt = torch.load(path, map_location="cpu")
    if isinstance(dt, dict) and "state_dict" in dt and isinstance(dt["state_dict"], dict):
        dt = dt["state_dict"]
    dt = {k[7:] if k.startswith("module.") else k: v for k, v in dt.items()}
    missing, unexpected = net.load_state_dict(dt, strict=False)
    print(f"載入 pretrained: {path}")
    print(f"  missing keys   : {len(missing)} {missing[:10]}")
    print(f"  unexpected keys: {len(unexpected)} {unexpected[:10]}")
    if missing or unexpected:
        print("  [WARN] 權重與架構沒有完全對上，請確認 --vrb_root 是官方 repo、參數與 demo.py 一致")


def set_bn_eval(module):
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


# -----------------------------------------------------------------------------
# 評估 / 視覺化
# -----------------------------------------------------------------------------
def traj_offsets_to_points(anchor_xy, off, y_up):
    """anchor_xy: (2,) crop normalized；off: (T, 2) -> (T, 2) crop normalized 絕對座標"""
    off = off.copy()
    if y_up:
        off[:, 1] = -off[:, 1]
    return anchor_xy[None] + off


@torch.no_grad()
def evaluate(net, loader, device, args, vis_dir=None):
    net.eval()
    tot = {"loss": 0.0, "traj": 0.0, "kl": 0.0, "obj": 0.0, "contact_px": 0.0, "traj_ade_px": 0.0}
    n = 0
    for batch in loader:
        img = batch["img"].to(device)
        contact = batch["contact"].to(device)
        traj = batch["traj"].to(device)
        B = img.shape[0]
        fv = torch.ones(B, device=device)

        traj_loss, kl, obj_loss, _ = net(img, None, None, traj, contact, fv)
        loss = args.w_traj * traj_loss + args.w_kl * kl + args.w_obj * obj_loss

        pred_traj, pred_contact = net.inference(img, None, None)
        pred_mu = pred_contact[:, 0]                              # (B, n_maps, 2)
        side = batch["crop"][:, 2].to(device)                     # (B,)

        # 誤差換算回原始影像 pixel
        c_err = (pred_mu.mean(1) - contact[:, 0]).norm(dim=-1) * side
        t_err = (pred_traj - traj).norm(dim=-1).mean(-1) * side

        tot["loss"] += loss.item() * B
        tot["traj"] += traj_loss.item() * B
        tot["kl"] += kl.item()
        tot["obj"] += obj_loss.item() * B
        tot["contact_px"] += c_err.sum().item()
        tot["traj_ade_px"] += t_err.sum().item()
        n += B

        if vis_dir is not None:
            os.makedirs(vis_dir, exist_ok=True)
            for b in range(B):
                draw_prediction(batch["img_raw"][b].numpy(), contact[b, 0].cpu().numpy(),
                                traj[b].cpu().numpy(), pred_mu[b].cpu().numpy(),
                                pred_traj[b].cpu().numpy(), args,
                                os.path.join(vis_dir, batch["name"][b]))
    return {k: v / max(n, 1) for k, v in tot.items()}


def draw_prediction(img_rgb, gt_c, gt_off, pred_mu, pred_off, args, out_path, scale=3, r=2):
    """
    GT  : contact 紅、軌跡 綠
    Pred: contact(5 個 map) 藍、軌跡 黃（以預測 contact 平均為起點）
    影像放大 scale 倍方便看，點仍然是 2px 圓
    """
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    S = 224 * scale
    img = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR)

    def P(xy):
        return int(round(xy[0] * S)), int(round(xy[1] * S))

    for p in traj_offsets_to_points(gt_c, gt_off, args.traj_y_up):
        cv2.circle(img, P(p), r, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(img, P(gt_c), r, (0, 0, 255), -1, cv2.LINE_AA)

    pc = pred_mu.mean(0)
    for p in traj_offsets_to_points(pc, pred_off, args.traj_y_up):
        cv2.circle(img, P(p), r, (0, 255, 255), -1, cv2.LINE_AA)
    for m in pred_mu:
        cv2.circle(img, P(m), r, (255, 0, 0), -1, cv2.LINE_AA)

    cv2.putText(img, "GT: red/green   Pred: blue/yellow", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


# -----------------------------------------------------------------------------
# Run 資料夾 / checkpoint
# -----------------------------------------------------------------------------
CKPT_NAME = "checkpoint_last.pth"


def make_run_dir(prefix):
    """runs/handoff_ft -> runs/handoff_ft_20260926_134512（同一秒重複時加 _1、_2 ...）"""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{prefix.rstrip('/').rstrip(os.sep)}_{stamp}"
    run_dir, k = base, 1
    while os.path.exists(run_dir):
        run_dir = f"{base}_{k}"
        k += 1
    os.makedirs(run_dir)
    return run_dir


def resolve_resume(resume, out_prefix):
    """--resume 可以是 run 資料夾、checkpoint 檔，或 'latest'。回傳 (run_dir, ckpt_path)"""
    if resume == "latest":
        cands = [d for d in glob.glob(f"{out_prefix.rstrip('/')}_*")
                 if os.path.isfile(os.path.join(d, CKPT_NAME))]
        if not cands:
            sys.exit(f"[ERROR] 找不到可以接續的 run：{out_prefix}_*/{CKPT_NAME}")
        run_dir = max(cands, key=lambda d: os.path.getmtime(os.path.join(d, CKPT_NAME)))
        return run_dir, os.path.join(run_dir, CKPT_NAME)
    if os.path.isdir(resume):
        ckpt = os.path.join(resume, CKPT_NAME)
        if not os.path.isfile(ckpt):
            sys.exit(f"[ERROR] {resume} 裡沒有 {CKPT_NAME}，無法接續")
        return resume, ckpt
    if os.path.isfile(resume):
        return os.path.dirname(os.path.abspath(resume)), resume
    sys.exit(f"[ERROR] --resume 路徑不存在：{resume}")


def save_checkpoint(path, net, optim, epoch, step, best, args):
    state = {
        "model": net.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,          # 已經完成的 epoch
        "step": step,
        "best": best,
        "args": vars(args),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)        # 先寫暫存檔再改名，存到一半中斷也不會弄壞舊的 checkpoint


# -----------------------------------------------------------------------------
# 訓練
# -----------------------------------------------------------------------------
def main(args):
    if not torch.cuda.is_available():
        sys.exit("[ERROR] 官方 VRBModel 在 __init__ 內對 BatchNorm 呼叫了 .cuda()，需要 GPU 才能建構")
    device = torch.device("cuda")

    # ---- 決定 run 資料夾：新訓練 or 接續 ----
    resume_state = None
    if args.resume:
        run_dir, ckpt_path = resolve_resume(args.resume, args.out_dir)
        resume_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved = resume_state["args"]
        # 沿用當初的所有設定（資料、架構、lr 排程…），只保留這次指定的 resume 相關參數
        for k, v in saved.items():
            if k not in ("resume",):
                setattr(args, k, v)
        print(f"接續訓練：{run_dir}（已完成 epoch {resume_state['epoch']} / {args.epochs}）")
        if resume_state["epoch"] >= args.epochs:
            sys.exit("這個 run 已經訓練完成，不需要接續")
    else:
        run_dir = make_run_dir(args.out_dir)
        # 存成絕對路徑，之後從別的資料夾執行 --resume 也找得到
        for k in ("vrb_root", "data_dir", "pretrained"):
            setattr(args, k, os.path.abspath(getattr(args, k)))
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"新的訓練，結果存到：{run_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # data
    train_set = HandoffDataset(args.data_dir, "train", args, train=True)
    val_set = HandoffDataset(args.data_dir, "val", args, train=False)
    train_eval_set = HandoffDataset(args.data_dir, "train", args, train=False)
    print(f"train {len(train_set)} / val {len(val_set)}")
    if len(train_set) == 0:
        sys.exit("[ERROR] 沒有訓練資料")
    bs = min(args.batch_size, len(train_set))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=0) if len(val_set) else None
    train_eval_loader = DataLoader(train_eval_set, batch_size=bs, shuffle=False, num_workers=0)

    # model
    net = build_vrb(args)
    if resume_state is None:
        load_pretrained(net, args.pretrained)
    net = net.to(device)

    backbone = list(net.r3m.parameters())
    backbone_ids = {id(p) for p in backbone}
    others = [p for p in net.parameters() if id(p) not in backbone_ids]
    if args.freeze_backbone:
        for p in backbone:
            p.requires_grad = False
        groups = [{"params": others, "lr": args.lr}]
    else:
        groups = [{"params": others, "lr": args.lr},
                  {"params": backbone, "lr": args.lr * args.backbone_lr_mult}]
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in groups]

    total_steps = args.epochs * math.ceil(len(train_set) / bs)
    warmup = min(args.warmup_steps, total_steps // 10)

    def set_lr(step):
        if step < warmup:
            f = (step + 1) / warmup
        else:
            p = (step - warmup) / max(total_steps - warmup, 1)
            f = 0.5 * (1 + math.cos(math.pi * p))
        for g, b in zip(optim.param_groups, base_lrs):
            g["lr"] = b * f

    # ---- 還原狀態 ----
    start_epoch, step, best = 1, 0, float("inf")
    if resume_state is not None:
        net.load_state_dict(resume_state["model"])
        optim.load_state_dict(resume_state["optim"])
        start_epoch = resume_state["epoch"] + 1
        step = resume_state["step"]
        best = resume_state["best"]
        rng = resume_state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        del resume_state
    elif val_loader:
        # 新訓練：先評估一次 pretrained（baseline）
        m = evaluate(net, val_loader, device, args, os.path.join(run_dir, "vis", "epoch0000_val"))
        print(f"[pretrained] val: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    ckpt_path = os.path.join(run_dir, CKPT_NAME)
    log_f = open(os.path.join(run_dir, "log.jsonl"), "a")
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            net.train()
            if args.freeze_bn:
                set_bn_eval(net)
            t0 = time.time()
            agg = {"loss": 0.0, "traj": 0.0, "kl": 0.0, "obj": 0.0}
            n = 0
            for batch in train_loader:
                set_lr(step)
                img = batch["img"].to(device)
                contact = batch["contact"].to(device)
                traj = batch["traj"].to(device)
                B = img.shape[0]
                fv = torch.ones(B, device=device)

                traj_loss, kl, obj_loss, _ = net(img, None, None, traj, contact, fv)
                loss = args.w_traj * traj_loss + args.w_kl * kl + args.w_obj * obj_loss

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optim.step()
                step += 1

                agg["loss"] += loss.item() * B
                agg["traj"] += traj_loss.item() * B
                agg["kl"] += kl.item()
                agg["obj"] += obj_loss.item() * B
                n += B
            agg = {k: v / n for k, v in agg.items()}

            rec = {"epoch": epoch, "lr": optim.param_groups[0]["lr"], "train": agg}
            msg = f"ep {epoch:4d} | train loss {agg['loss']:.4f} (traj {agg['traj']:.4f} obj {agg['obj']:.4f} kl {agg['kl']:.3f})"

            do_vis = (epoch % args.vis_every == 0) or epoch == args.epochs
            if val_loader and (epoch % args.eval_every == 0 or do_vis):
                vis = os.path.join(run_dir, "vis", f"epoch{epoch:04d}_val") if do_vis else None
                vm = evaluate(net, val_loader, device, args, vis)
                rec["val"] = vm
                msg += f" | val loss {vm['loss']:.4f} contact {vm['contact_px']:.1f}px traj ADE {vm['traj_ade_px']:.1f}px"
                if vm["loss"] < best:
                    best = vm["loss"]
                    torch.save(net.state_dict(), os.path.join(run_dir, "model_best.pth.tar"))
                    msg += "  *best*"
            if do_vis:
                tm = evaluate(net, train_eval_loader, device, args,
                              os.path.join(run_dir, "vis", f"epoch{epoch:04d}_train"))
                rec["train_eval"] = tm
                msg += f" | train contact {tm['contact_px']:.1f}px ADE {tm['traj_ade_px']:.1f}px"

            msg += f" | {time.time() - t0:.1f}s"
            print(msg)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

            if epoch % args.save_every == 0:
                torch.save(net.state_dict(), os.path.join(run_dir, f"model_ep{epoch:04d}.pth.tar"))
            if epoch % args.ckpt_every == 0 or epoch == args.epochs:
                save_checkpoint(ckpt_path, net, optim, epoch, step, best, args)

    except KeyboardInterrupt:
        # Ctrl+C：把目前這個 epoch 之前完成的進度存起來
        # （若中斷在 epoch 中間，這個 epoch 會在接續時重新跑）
        done = epoch - 1
        print(f"\n[中斷] 已儲存進度到 epoch {done}，之後可用：python train_vrb.py --resume {run_dir}")
        if done >= start_epoch:
            # step 也要退回該 epoch 開始時的值，lr 排程才會對齊
            steps_per_epoch = math.ceil(len(train_set) / bs)
            save_checkpoint(ckpt_path, net, optim, done, done * steps_per_epoch, best, args)
        log_f.close()
        sys.exit(1)

    torch.save(net.state_dict(), os.path.join(run_dir, "model_last.pth.tar"))
    log_f.close()
    print(f"完成，checkpoint 與視覺化結果在 {run_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # 路徑
    p.add_argument("--vrb_root", help="官方 VRB repo 路徑（含 networks/）")
    p.add_argument("--data_dir", default="vrb_dataset", help="prepare_vrb_dataset.py 的輸出資料夾")
    p.add_argument("--pretrained", help="官方 model_checkpoint_1249.pth.tar")
    p.add_argument("--out_dir", default="runs/handoff_ft",
                   help="輸出資料夾前綴，實際會存到 <out_dir>_<開始時間>/")
    p.add_argument("--resume", default=None,
                   help="接續訓練：run 資料夾、checkpoint_last.pth 路徑，或 'latest'（找 <out_dir>_* 裡最新的）")
    p.add_argument("--ckpt_every", type=int, default=10, help="每幾個 epoch 更新一次接續用的 checkpoint_last.pth")
    # 架構參數（預設值與官方 demo.py 相同，不要改，否則 pretrained 對不上）
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--enc_depth", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--hand_latent_dim", type=int, default=4)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--coord_dim", type=int, default=64)
    p.add_argument("--resnet_type", type=str, default="resnet18")
    p.add_argument("--attn_kp", type=int, default=1)
    p.add_argument("--attn_kp_fc", type=int, default=1)
    p.add_argument("--traj_len", type=int, default=5)
    p.add_argument("--n_maps", type=int, default=5)
    p.add_argument("--encoder_time_embed_type", default="sin", choices=["sin", "param"])

    # 資料 / 裁切
    p.add_argument("--crop_size", type=float, default=480, help="正方形裁切邊長 (pixel, 原始影像)")
    p.add_argument("--roi", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                   help="固定 ROI (pixel)，給了就以 ROI 裁切，而不是以 contact 為中心")
    p.add_argument("--center_jitter", type=float, default=0.25, help="訓練時裁切中心隨機偏移（× 邊長）")
    p.add_argument("--scale_jitter", type=float, default=0.15, help="訓練時裁切邊長隨機縮放比例")
    p.add_argument("--color_aug", type=int, default=1, help="官方 transform 的 ColorJitter / RandomGrayscale")
    p.add_argument("--hue", type=float, default=0.3, help="ColorJitter hue（官方 0.3；手術場景可考慮調低）")
    p.add_argument("--traj_y_up", type=int, default=1, help="軌跡 y 往上為正（對應官方 inference 的 -dy）")

    # 訓練
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1, help="ResNet backbone 的 lr 倍率")
    p.add_argument("--freeze_backbone", type=int, default=0)
    p.add_argument("--freeze_bn", type=int, default=1, help="資料很少時固定 BN 統計量（用 pretrained 的）")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--w_traj", type=float, default=1.0)
    p.add_argument("--w_obj", type=float, default=1.0, help="contact loss 權重（官方 forward 內已乘 5）")
    p.add_argument("--w_kl", type=float, default=1e-3, help="CVAE KL 權重（官方推論時 z=0，只作正則化）")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--vis_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not args.resume and (not args.vrb_root or not args.pretrained):
        p.error("新的訓練需要 --vrb_root 與 --pretrained（接續訓練時只需要 --resume）")
    main(args)
