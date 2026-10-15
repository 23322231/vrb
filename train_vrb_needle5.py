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
  --attn_kp_fc 0           ：官方原始碼的另一條路徑，每個點一張 heatmap 再取 soft-argmax。
                             注意官方這條路徑輸出 (y, x)，本程式已自動轉換。

attn_kp_fc 0 的架構（與官方 networks/model.py 相同，分 5 段；--freeze_upto 依這個順序凍結）:
    backbone  ResNet18（R3M）conv1 ~ layer4                       官方 pretrained
    decoder   deconv1/2/3 + bn1/2/3 + ReLU（含 layer3、layer2 skip） 官方 pretrained   -> 56x56x64
    cb        cb1、cb2：Conv3x3-BN-ReLU x2                          官方 pretrained   -> 56x56x64
    attn      位置編碼 + nn.MultiheadAttention                       官方 pretrained
    head      cb3：Conv1x1-BN-ReLU                                  官方 ckpt 有這層，但官方訓練走 fc 路徑、
                                                                    cb3 從未被訓練 -> 預設重新初始化  -> 56x56x5 heatmap
    每次新訓練會在 run 資料夾寫 init_report.md：每一層的初始化來源、是否用到、是否訓練、學習率（寫報告用）

對官方程式的修改（全部是選項，預設值如下）:
    1. get_coord：官方是「先沿一軸取平均再 softmax」，分布幾乎均勻 -> 5 點縮到中心。
       --kp_beta > 0 改成 2D softmax(beta * heatmap) 的邊際分布（無參數函式，不影響權重）
    2. 高斯 heatmap KL 監督（--hm_weight），只是多一個 loss
    3. cb3 最後的 ReLU：預設保留（--hm_relu 1 = 官方原版）；--hm_relu 0 才拿掉
    4. --attn_mode：官方的 nn.MultiheadAttention 沒設 batch_first，輸入 (B, H*W, C) 會被當成
       (序列 = B, batch = H*W)，也就是「同一個位置、跨 batch 裡不同影像」做 attention，不是空間 attention。
       official（預設）= 照原版；spatial = 改成 56x56 個位置之間的 attention（權重形狀相同，可載入 pretrained）

segmentation 輔助頭（--seg_weight > 0，且資料有 mask 時才啟用；mask 由 build_needle5_aug.py 產生）:
    attn 輸出 56x56x64 ─┬→ cb3（官方）        -> 5 張 keypoint heatmap（VRB 的輸出，不變）
                        └→ seg_head（新增）   -> 1 張「這個像素是不是針」56x56（只在訓練時加 loss）
    seg_head 是 Conv1x1(64→1)：刻意只用線性讀出，讓「分辨針 / 背景 / 夾爪」的能力必須長在共用的
    decoder / cb / attn 特徵裡（keypoint 也讀同一份特徵）。loss = BCE(pos_weight) + Dice，mask 以面積平均縮到 56x56。
    評估時印出 seg IoU / precision / recall；heatmaps/*.png 第 6 格是預測的針 mask（綠框 = GT）。
    參考：Laina et al., MICCAI 2017（手術器械 segmentation + landmark heatmap 共同訓練）

特徵圖檢查（--feat_probe 1，預設開啟；在 epoch 0 = pretrained 與每 --vis_every 個 epoch 執行）:
    對 layer1~4、decoder 3 層、cb2、attn 共 9 層特徵圖：
    - energy：每個位置特徵向量的長度（L2）-> 這層「在看哪裡」
    - 原型相似度：用 train 原圖在 GT 5 點的特徵取平均當「原型」，val 每張圖每個位置與原型的 cosine 相似度
      -> 第 k 張圖的尖峰若落在第 k 個點上，代表這層特徵已經能找到那個部位（這個檢查不經過 cb3）
    - 數字（存在 feat_log.jsonl，曲線在 features.png）：
        argmax_err_px  相似度最高處與 GT 的距離（原圖 pixel）
        proto_acc      在 GT 5 個位置，特徵最像哪個原型（5 選 1，隨機 = 20%）
        head_tail_acc  在針頭 / 針尾位置，分不分得出頭和尾（2 選 1，隨機 = 50%）
    - 圖：vis/epochXXXX_val/features/*.png（列 = 9 層；欄 = 原圖、energy、5 個點的相似度）
- 裁切：以針的框為中心的正方形，邊長 = 框長邊 + 2*pad（pad = max(box_pad, box_pad_ratio*長邊)），
        再縮成 224x224。訓練時加隨機縮放 / 位移 / 翻轉 / 旋轉（--rot_deg）
- 每次訓練存到 <out_dir>_<開始時間>/，可用 --resume 接續
- 學習曲線：每 --plot_every 個 epoch 更新 <run>/curves.png（train / val loss 與 val 各點 pixel 誤差）
  也可以隨時另外畫：python train_vrb_needle5.py --plot_only runs/needle5_fc1_20261012_181530

兩種資料格式（自動辨識）:
    A. mask_to_needle5.py 的 annotations.json      -> 線上增強（每次讀圖時隨機裁切 / 翻轉 / 旋轉 / 色彩）
       預設 train = train+val、val = test
    B. build_needle5_aug.py 的 annotations.json     -> 離線增強（直接讀存好的 224x224 圖，訓練時不再做任何增強）
       ("format": "needle5_crops")                     預設 train = train、val = val，訓練完用 best 權重評估 test

用法:
    # B. 離線增強資料集（--images_root 不用給，預設就是 annotations.json 所在資料夾）
    python train_vrb_needle5.py --vrb_root . --ann ./needle5_aug/annotations.json \
        --pretrained ./models/model_checkpoint_1249.pth.tar --out_dir ./runs/needle5_aug \
        --attn_kp_fc 0 --kp_beta 10 --epochs 40

    # A. 線上增強
    python train_vrb_needle5.py --vrb_root . \
        --ann needle5_dataset/annotations.json \
        --images_root /path/to/DATA/Images \
        --pretrained ./models/model_checkpoint_1249.pth.tar \
        --out_dir runs/needle5

    python train_vrb_needle5.py --resume latest --out_dir runs/needle5

    # 用 --exp 分類：結果存到 runs/<exp>/needle5_<時間>/（每個 run 內也會存一份當時的程式碼）
    #   接續時同樣加 --exp：python train_vrb_needle5.py --resume latest --exp mine_plus_senior

    # 多個資料集一起訓練（各自沿用自己的 train/val/test 切分；終端機與 curves.png 會分開顯示各資料集誤差）
    python train_vrb_needle5.py --vrb_root . \
        --ann ./needle5_mine_aug/annotations.json ./needle5_aug/annotations.json \
        --pretrained ./models/model_checkpoint_1249.pth.tar \
        --attn_kp_fc 0 --kp_beta 10 --loss ordered --balance 1 --best_on needle5_mine_aug
"""

import argparse
import datetime
import glob
import json
import math
import os
import random
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

POINT_NAMES = ["head", "tail", "middle", "head_1_3", "tail_1_3"]
SWAP_HEAD_TAIL = [1, 0, 2, 4, 3]                 # 頭尾互換後的對應順序
POINT_COLORS = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 140, 255), (255, 255, 0)]  # BGR
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CKPT_NAME = "checkpoint_last.pth"
SEG_SIZE = 56                                    # heatmap / segmentation 解析度


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

    @staticmethod
    def _rotated_crop(img, pts, x0, y0, side, deg):
        """以裁切中心旋轉 deg 度（逆時針）：先裁一個 1.5 倍大的區域旋轉，再取中間，四角不會出現黑邊。
        旋轉後有點跑出裁切範圍就回傳 (None, None)，改用不旋轉。"""
        cx, cy = x0 + side / 2, y0 + side / 2
        big = int(math.ceil(side * 1.5))
        bx0, by0 = int(round(cx - big / 2)), int(round(cy - big / 2))
        region = img.crop((bx0, by0, bx0 + big, by0 + big)).rotate(deg, resample=Image.BILINEAR)
        o = int(round((big - side) / 2))
        crop = region.crop((o, o, o + side, o + side))
        # PIL 以 region 中心 (big/2, big/2) 旋轉；在影像座標（y 向下）中是逆時針：
        # (dx, dy) -> (dx cos + dy sin, -dx sin + dy cos)
        th = math.radians(deg)
        rc = np.array([bx0 + big / 2, by0 + big / 2]) - 0.5          # PIL 的旋轉中心換成像素索引座標
        d = pts - rc
        rd = np.stack([d[:, 0] * math.cos(th) + d[:, 1] * math.sin(th),
                       -d[:, 0] * math.sin(th) + d[:, 1] * math.cos(th)], axis=1)
        pn = (rd + rc - np.array([bx0 + o, by0 + o])) / side
        if (pn < 0).any() or (pn > 1).any():
            return None, None
        return crop.resize((224, 224), Image.BILINEAR), pn

    def __getitem__(self, i):
        s = self.samples[i]
        img = self._load(s["image"])
        W, H = img.size
        pts = np.asarray(s["points_px"], dtype=np.float64)                   # (5, 2) x, y

        # train：全域 random（DataLoader 每個 worker / epoch 會重新 seed）；val：不加擾動
        x0, y0, side = needle_crop(s["needle_bbox"], W, H, self.args, random if self.train else None)
        crop, pn = None, None
        if self.train and self.args.rot_deg > 0:
            crop, pn = self._rotated_crop(img, pts, x0, y0, side, random.uniform(-self.args.rot_deg, self.args.rot_deg))
        if crop is None:
            crop = img.crop((x0, y0, x0 + side, y0 + side)).resize((224, 224), Image.BILINEAR)
            pn = (pts - np.array([x0, y0])) / side                           # crop 內 normalized (x, y)

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
            "src": s.get("_src", "data"),
            "mask": torch.zeros(1, SEG_SIZE, SEG_SIZE),                       # 線上增強資料不支援 mask
            "has_mask": torch.tensor(0.0),
        }


class Needle5CropDataset(Dataset):
    """build_needle5_aug.py 產生的離線增強資料：224x224 圖 + 圖內 normalized 5 點，不再做任何增強。"""

    def __init__(self, ann, root, splits, orig_only=False):
        self.samples = [s for s in ann["samples"] if s["split"] in splits
                        and not (orig_only and s.get("augmented", False))]
        self.root = root
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(os.path.join(s.get("_root", self.root), s["image"])).convert("RGB")
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.BILINEAR)
        name = os.path.basename(s["image"])
        if s.get("_multi"):                       # 多個資料集合併時，檔名前加上資料集名稱避免撞名
            name = f"{s['_src']}__{name}"
        mask, has = torch.zeros(1, SEG_SIZE, SEG_SIZE), 0.0
        if s.get("mask"):
            m = cv2.imread(os.path.join(s.get("_root", self.root), s["mask"]), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                # 面積平均縮到 56x56：細針不會消失，值 = 該格有多少比例是針
                m = cv2.resize(m, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                mask, has = torch.from_numpy(m)[None], 1.0
        return {
            "img": self.tf(img),
            "img_raw": torch.from_numpy(np.asarray(img).copy()),
            "pts": torch.tensor(s["points"], dtype=torch.float32),
            "crop": torch.tensor([0, 0, s["side_px"]], dtype=torch.float32),   # side_px：誤差換算回原圖 pixel
            "name": name,
            "src": s.get("_src", "data"),
            "mask": mask,
            "has_mask": torch.tensor(has),
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


HM_CACHE = {}


def fix_heatmap_head(net, remove_relu=False, reinit=True):
    """attn_kp_fc=0 的熱度圖輸出層 cb3 = Conv1x1 -> BatchNorm -> ReLU（官方原版）。
    - 官方 pretrained 是用 fc 路徑訓練的，cb3 雖然存在 checkpoint 裡，但從來沒有被訓練過（等於隨機值）
    - reinit：重新初始化 cb3（卷積權重 std=0.01 + BatchNorm 統計量歸零），從頭訓練這一層
    - remove_relu：把最後的 ReLU 換成 Identity（--hm_relu 0；預設不拿掉，維持官方架構）"""
    head = net.cb3
    if remove_relu and hasattr(head, "relu"):
        head.relu = nn.Identity()
        print("[NOTE] 已移除熱度圖輸出層 cb3 最後的 ReLU（--hm_relu 0，與官方架構不同）")
    if reinit:
        nn.init.normal_(head.conv.weight, std=0.01)
        if head.conv.bias is not None:
            nn.init.zeros_(head.conv.bias)
        if hasattr(head, "norm") and isinstance(head.norm, nn.modules.batchnorm._BatchNorm):
            head.norm.reset_running_stats()
            nn.init.ones_(head.norm.weight)
            nn.init.zeros_(head.norm.bias)
        print("[NOTE] 已重新初始化熱度圖輸出層 cb3（官方 pretrained 沒訓練過這層）")


def patch_attention(net, mode):
    """官方：nn.MultiheadAttention(batch_first=False) 收到 (B, H*W, C)，實際是在「同一位置、跨 batch 影像」之間做 attention。
    spatial：前後轉置，改成每張影像自己 H*W 個位置之間的 attention（權重 / state_dict 完全相同）。
    只替換這個模組實例的 forward，不改官方原始碼。"""
    if mode != "spatial":
        print("[NOTE] attention 照官方原版（--attn_mode official）：沿 batch 維度做 attention，"
              "預測會受同一 batch 其他影像影響；batch=1 時退化成逐位置的線性層")
        return
    mha = net.self_attention
    orig = mha.forward

    def forward(query, key, value, *a, **kw):
        kw["need_weights"] = False                       # 用 scaled_dot_product_attention，不建出 3136x3136 權重矩陣
        out, _ = orig(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), *a, **kw)
        return out.transpose(0, 1), None

    mha.forward = forward
    print("[NOTE] attention 改為空間 attention（--attn_mode spatial）：每張影像 56x56 個位置之間互相注意")


def add_seg_head(net):
    """在 attention 輸出上加一個 Conv1x1(64→1) 的 segmentation 輔助頭，並掛 hook 取 attention 輸出。
    seg_head 是 net 的子模組（會存進 state_dict / checkpoint）；官方 encode_img 不會呼叫它，不影響 5 點預測。"""
    net.seg_head = nn.Conv2d(64, 1, kernel_size=1)
    nn.init.normal_(net.seg_head.weight, std=0.01)
    nn.init.constant_(net.seg_head.bias, -3.0)            # 起始預測「幾乎都不是針」（針只佔很少像素）

    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out     # (B, H*W, C)
        B, N, C = o.shape
        s_ = int(round(math.sqrt(N)))
        HM_CACHE["attn"] = o.transpose(1, 2).reshape(B, C, s_, s_)
    net.self_attention.register_forward_hook(hook)
    print("[NOTE] 加上 segmentation 輔助頭 seg_head：attention 輸出 -> Conv1x1(64→1) -> 56x56 針 mask")


def seg_logits(net):
    f = HM_CACHE.get("attn")
    return None if f is None or not hasattr(net, "seg_head") else net.seg_head(f)


def seg_loss(logits, target, has, pos_weight, dice_w=1.0):
    """BCE(pos_weight) + soft Dice；只對有 mask 的樣本計算。target：0~1（面積比例）。"""
    if logits is None or has.sum() == 0:
        return None
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                             pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = bce.mean(dim=(1, 2, 3))
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    per = bce + dice_w * dice
    return (per * has).sum() / has.sum()


def patch_get_coord(beta):
    """把官方 networks.model.get_coord 換成 2D softmax(beta * x) 的邊際分布版本（簽名與回傳格式相同）。"""
    import torch.nn.functional as F
    import networks.model as M

    def get_coord(x, other_axis, axis_size):
        B, C, Hh, Ww = x.shape
        HM_CACHE["x"] = x                                                   # 給 heatmap loss 用
        p2 = F.softmax(beta * x.reshape(B, C, -1), dim=-1).reshape(B, C, Hh, Ww)
        g_c_prob = p2.sum(dim=other_axis)                                   # 邊際分布 (B, C, axis)
        coord_pt = torch.linspace(0, 1.0, axis_size, device=x.device).view(1, 1, axis_size)
        return torch.sum(g_c_prob * coord_pt, dim=2), g_c_prob

    M.get_coord = get_coord
    print(f"[NOTE] 已將 networks.model.get_coord 改為 2D softmax（beta={beta}）")


def load_pretrained(net, path):
    """回傳 checkpoint 裡的 key 集合與 missing keys（給 init_report 用）。"""
    dt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(dt, dict) and "state_dict" in dt and isinstance(dt["state_dict"], dict):
        dt = dt["state_dict"]
    dt = {k[7:] if k.startswith("module.") else k: v for k, v in dt.items()}
    missing, unexpected = net.load_state_dict(dt, strict=False)
    print(f"載入 pretrained: {path}")
    print(f"  missing keys   : {len(missing)} {missing[:10]}")
    print(f"  unexpected keys: {len(unexpected)} {unexpected[:10]}")
    return {"keys": set(dt.keys()), "missing": set(missing), "path": path}


# -----------------------------------------------------------------------------
# 架構分段 / 凍結 / 初始化報告
# -----------------------------------------------------------------------------
STAGES = [
    ("backbone", ["r3m"], "ResNet18（R3M）conv1 ~ layer4"),
    ("decoder", ["deconv1", "bn1", "deconv2", "bn2", "deconv3", "bn3"], "3 層 deconv + BN + ReLU（含 skip）"),
    ("cb", ["cb1", "cb2"], "cb1、cb2：Conv3x3-BN-ReLU x2"),
    ("attn", ["self_attention"], "位置編碼 + MultiheadAttention"),
    ("head", ["cb3"], "cb3：Conv1x1-BN-ReLU -> 5 張 heatmap"),
]
STAGE_OF = {m: name for name, mods, _ in STAGES for m in mods}


def stage_modules(upto):
    """--freeze_upto：回傳該 stage（含）之前所有模組名稱。"""
    if upto in (None, "none"):
        return []
    out = []
    for name, mods, _ in STAGES:
        out += mods
        if name == upto:
            break
    return out


@torch.no_grad()
def find_used_modules(net, device):
    """跑一次 encode_img，記錄哪些子模組（含它底下任何一層）真的有被呼叫。
    注意：官方 encode_img 不是呼叫 net.r3m(...)，而是直接呼叫 net.r3m.convnet.conv1 / layer1 ...，
    所以要掛在「所有子層」上，再對應回最上層的名稱，否則 r3m 會被誤判成沒用到。"""
    used = set()
    hs = []
    for top, child in net.named_children():
        for _, m in child.named_modules():
            hs.append(m.register_forward_hook(lambda mod, i, o, n=top: used.add(n)))
    was_training = net.training
    net.eval()
    net.encode_img(torch.zeros(2, 3, 224, 224, device=device))
    net.train(was_training)
    for h in hs:
        h.remove()
    HM_CACHE.pop("x", None)
    return used


def write_init_report(net, run_dir, args, ckpt_info, reinit, frozen, used):
    """每一層：stage / 參數量 / 初始化來源 / 是否用到 / 是否訓練 / 學習率 -> init_report.md"""
    rows = []
    for name, mod in net.named_children():
        n_p = sum(p.numel() for p in mod.parameters())
        keys = [f"{name}.{k}" for k in mod.state_dict().keys() if not k.endswith("num_batches_tracked")]
        in_ckpt = bool(keys) and all(k in ckpt_info["keys"] for k in keys)
        if name in reinit:
            src = "重新初始化（本程式）"
        elif in_ckpt:
            src = "官方 pretrained"
        elif not keys:
            src = "（無參數）"
        else:
            src = "隨機初始化（checkpoint 沒有）"
        if name == "cb3" and in_ckpt:
            src += "；checkpoint 有此層，但官方訓練走 fc 路徑，從未被訓練"
        is_used = name in used or name == "seg_head"
        trained = is_used and name not in frozen and n_p > 0 and any(p.requires_grad for p in mod.parameters())
        if n_p == 0:
            tr = "（無參數）"
        elif trained:
            lr = args.lr * (args.backbone_lr_mult if name == "r3m" else 1.0)
            tr = f"訓練（lr {lr:g}）"
        elif not is_used:
            tr = "未使用（attn_kp_fc=0 的 forward 不會經過）"
        else:
            tr = "凍結"
        if name == "seg_head":
            src = "新增（本程式），隨機初始化"
            tr += "；segmentation 輔助任務，只在訓練時計算 loss，不參與 5 點預測"
        rows.append((STAGE_OF.get(name, "seg（新增）" if name == "seg_head" else "-"), name, n_p, src,
                     "是（只有 seg loss）" if name == "seg_head" else ("是" if is_used else "否"), tr))
    order = {s[0]: i for i, s in enumerate(STAGES)}
    order["seg（新增）"] = len(STAGES)
    rows.sort(key=lambda r: (order.get(r[0], 99), r[1]))

    L = ["# 模型初始化 / 訓練範圍報告", "",
         f"- pretrained：`{ckpt_info['path']}`",
         f"- attn_kp={args.attn_kp}, attn_kp_fc={args.attn_kp_fc}, n_maps=5（建構參數與官方 demo.py 相同）",
         f"- freeze_upto={args.freeze_upto}, freeze_bn={args.freeze_bn}（1 = 所有 BatchNorm 用 pretrained 統計量，不更新）",
         "", "| stage | 模組 | 參數量 | 初始化 | forward 有用到 | 訓練 |", "|---|---|---:|---|---|---|"]
    for r in rows:
        L.append(f"| {r[0]} | `{r[1]}` | {r[2]:,} | {r[3]} | {r[4]} | {r[5]} |")
    L += ["", "## 與官方程式不同的地方", ""]
    if not args.attn_kp_fc:
        L.append(f"- 座標解碼 get_coord："
                 + (f"改為 2D softmax(β·heatmap) 的邊際分布，β={args.kp_beta}（官方：先沿一軸平均再 softmax）"
                    if args.kp_beta > 0 else "官方原版"))
        L.append(f"- cb3 最後的 ReLU：{'保留（官方原版）' if args.hm_relu else '拿掉（Identity）'}")
        if args.kp_beta > 0 and args.hm_weight > 0:
            L.append(f"- 額外 loss：高斯 heatmap KL，權重 {args.hm_weight}，σ={args.hm_sigma} 格（56x56）")
        L.append(f"- 評估解碼：{'local（argmax 附近 ±%d 格加權平均）' % args.decode_radius if args.decode == 'local' else '全圖 soft-argmax'}")
    L.append(f"- attention：{'官方原版（沿 batch 維度）' if args.attn_mode == 'official' else '空間 attention（每張影像 H*W 個位置之間）'}")
    if hasattr(net, "seg_head"):
        L.append(f"- 新增 segmentation 輔助頭 seg_head（attention 輸出 → Conv1x1 → 56x56 針 mask），"
                 f"loss = {args.seg_weight}×(BCE(pos_weight={args.seg_pos_weight}) + {args.seg_dice}×Dice)；"
                 f"推論時不使用")
    L.append(f"- loss：官方 obj_loss 公式 5·mean((μ−gt)²)，{'頭尾順序固定' if args.loss == 'ordered' else '頭尾互換取較小（sym）'}；不訓練軌跡分支")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(run_dir, "init_report.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


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


def local_decode(hm_logits, beta, radius):
    """熱度圖解碼：先找機率最大的格子，只用它周圍 (2r+1)x(2r+1) 範圍的機率做加權平均。
    全圖 soft-argmax 會被分散在整張圖的小機率拉向中心；只看尖峰附近可以去掉這個偏差。
    hm_logits: (B, C, H, W) tensor -> (B, C, 2) normalized (x, y)"""
    B, C, Hh, Ww = hm_logits.shape
    p = torch.softmax(beta * hm_logits.float().reshape(B, C, -1), dim=-1).reshape(B, C, Hh, Ww)
    idx = p.reshape(B, C, -1).argmax(-1)
    iy, ix = (idx // Ww).view(B, C, 1, 1), (idx % Ww).view(B, C, 1, 1)
    yy = torch.arange(Hh, device=p.device).view(1, 1, Hh, 1)
    xx = torch.arange(Ww, device=p.device).view(1, 1, 1, Ww)
    win = ((yy - iy).abs() <= radius) & ((xx - ix).abs() <= radius)
    pl = p * win
    pl = pl / pl.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    gx = torch.linspace(0, 1, Ww, device=p.device).view(1, 1, 1, Ww)
    gy = torch.linspace(0, 1, Hh, device=p.device).view(1, 1, Hh, 1)
    return torch.stack([(pl * gx).sum(dim=(2, 3)), (pl * gy).sum(dim=(2, 3))], dim=-1)


def heatmap_loss(gt, args):
    """DSNT / 高斯 heatmap 監督：每個點的目標是以 GT 為中心的 2D 高斯分布，
    與 softmax(beta * heatmap) 算 KL divergence。讓每個 channel 在「正確位置」形成單一尖峰，
    避免座標只靠加權平均學、被拉向中心。sym 模式同樣取頭尾互換較小者。"""
    import torch.nn.functional as F
    x = HM_CACHE.get("x")
    if x is None:
        return torch.zeros((), device=gt.device)
    B, C, Hh, Ww = x.shape
    ys = torch.linspace(0, 1, Hh, device=x.device).view(1, 1, Hh, 1)
    xs = torch.linspace(0, 1, Ww, device=x.device).view(1, 1, 1, Ww)
    sig = args.hm_sigma / (Ww - 1)
    logp = F.log_softmax(args.kp_beta * x.float().reshape(B, C, -1), dim=-1)

    def kl(g):
        t = torch.exp(-((xs - g[..., 0, None, None]) ** 2 + (ys - g[..., 1, None, None]) ** 2) / (2 * sig ** 2))
        t = t.reshape(B, C, -1)
        t = t / t.sum(-1, keepdim=True).clamp_min(1e-12)
        return (t * (torch.log(t.clamp_min(1e-12)) - logp)).sum(-1).mean(-1)   # (B,)

    per = kl(gt)
    if args.loss == "sym":
        per = torch.minimum(per, kl(gt[:, SWAP_HEAD_TAIL]))
    return per.mean()


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
    sum_err_ord = torch.zeros(5)
    sum_err_global = torch.zeros(5)
    n, loss_sum, flips = 0, 0.0, 0
    zero_sum = 0.0
    seg_tp = seg_fp = seg_fn = 0.0
    seg_n = 0
    dead = np.zeros(5)                           # 每個點：整張 heatmap <= 0 的樣本數（ReLU 全死 -> 分布均勻 -> 點落在中心）
    by_src = {}                                  # 每個資料集各自的誤差（多資料集合併訓練時用）
    for batch in loader:
        img = batch["img"].to(device)
        gt = batch["pts"].to(device)
        HM_CACHE.pop("x", None)
        HM_CACHE.pop("attn", None)
        pred = predict_xy(net, img, args)
        hm_t = HM_CACHE.get("x")
        sl = seg_logits(net)
        seg_p = torch.sigmoid(sl).float() if sl is not None else None
        has = batch["has_mask"].to(device) > 0
        if seg_p is not None and has.any():
            pb = seg_p[has] > 0.5
            tb = batch["mask"].to(device)[has] > 0.25
            seg_tp += float((pb & tb).sum())
            seg_fp += float((pb & ~tb).sum())
            seg_fn += float((~pb & tb).sum())
            seg_n += int(has.sum())
        hm = hm_t.float().cpu().numpy() if hm_t is not None else None
        if hm is not None:
            zero_sum += float((hm == 0).mean()) * img.shape[0]
            if args.hm_relu:                     # 只有保留 ReLU 時才有「整張為 0」的問題
                dead += (hm.reshape(hm.shape[0], hm.shape[1], -1).max(-1) <= 0).sum(0)
        loss_sum += contact_loss(pred, gt, args.loss).item() * img.shape[0]
        pred_global = pred
        if hm_t is not None and args.decode == "local":
            pred = local_decode(hm_t, args.kp_beta if args.kp_beta > 0 else 1.0, args.decode_radius)
        # 全圖 soft-argmax 的誤差另外記錄，方便比較兩種解碼
        e_g = (pred_global - gt).norm(dim=-1)
        e_gs = (pred_global - gt[:, SWAP_HEAD_TAIL]).norm(dim=-1)
        if args.loss == "sym":
            e_g = torch.where((e_gs.mean(1) < e_g.mean(1))[:, None], e_gs, e_g)
        sum_err_global += (e_g * batch["crop"][:, 2].to(device)[:, None]).sum(0).cpu()
        side = batch["crop"][:, 2].to(device)[:, None]
        err = (pred - gt).norm(dim=-1) * side                                # (B, 5) 原圖 pixel
        err_sw = (pred - gt[:, SWAP_HEAD_TAIL]).norm(dim=-1) * side
        flipped = err_sw.mean(1) < err.mean(1)                               # (B,)
        flips += int(flipped.sum())
        sum_err_ord += err.sum(0).cpu()
        if args.loss == "sym":
            # sym：頭尾順序不計，把 GT 對齊到較接近的那個方向再算誤差（畫圖也用對齊後的 GT）
            gt = torch.where(flipped[:, None, None], gt[:, SWAP_HEAD_TAIL], gt)
            err = torch.where(flipped[:, None], err_sw, err)
        sum_err += err.sum(0).cpu()
        n += img.shape[0]
        e_mean = err.mean(1).cpu().tolist()
        for b, src in enumerate(batch["src"]):
            d = by_src.setdefault(src, {"sum": 0.0, "n": 0, "flips": 0})
            d["sum"] += e_mean[b]
            d["n"] += 1
            d["flips"] += int(flipped[b])
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            for b in range(img.shape[0]):
                draw(batch["img_raw"][b].numpy(), gt[b].cpu().numpy(), pred[b].cpu().numpy(),
                     os.path.join(vis_dir, batch["name"][b]))
                if hm is not None:
                    hdir = os.path.join(vis_dir, "heatmaps")
                    os.makedirs(hdir, exist_ok=True)
                    draw_heatmaps(batch["img_raw"][b].numpy(), hm[b], gt[b].cpu().numpy(),
                                  pred[b].cpu().numpy(), args.kp_beta if args.kp_beta > 0 else 1.0,
                                  os.path.join(hdir, batch["name"][b]),
                                  seg=seg_p[b, 0].cpu().numpy() if seg_p is not None else None,
                                  seg_gt=batch["mask"][b, 0].numpy() if bool(has[b]) else None)
    per = (sum_err / max(n, 1)).tolist()
    per_ord = (sum_err_ord / max(n, 1)).tolist()
    m = {"loss": loss_sum / max(n, 1), "mean_px": float(np.mean(per)), "flip_rate": flips / max(n, 1),
         "mean_px_ordered": float(np.mean(per_ord)), "hm_zero_frac": zero_sum / max(n, 1),
         "mean_px_softargmax": float((sum_err_global / max(n, 1)).mean()),
         "hm_dead_frac": (dead / max(n, 1)).tolist()}
    if seg_n:
        m["seg_iou"] = seg_tp / max(seg_tp + seg_fp + seg_fn, 1)
        m["seg_precision"] = seg_tp / max(seg_tp + seg_fp, 1)      # 低 = 把背景 / 夾爪當成針
        m["seg_recall"] = seg_tp / max(seg_tp + seg_fn, 1)         # 低 = 針沒找到
    m.update({f"{k}_px": v for k, v in zip(POINT_NAMES, per)})
    m["by_src"] = {s: {"mean_px": d["sum"] / d["n"], "flip_rate": d["flips"] / d["n"], "n": d["n"]}
                   for s, d in sorted(by_src.items())}
    return m


def src_msg(m):
    """多資料集時在終端機印出各資料集的 val mean。"""
    bs = m.get("by_src", {})
    if len(bs) < 2:
        return ""
    return " | " + "  ".join(f"[{s}] {d['mean_px']:.1f}px flip {d['flip_rate']:.0%}" for s, d in bs.items())


def select_metric(m, best_on):
    """選 best 權重用的數字：預設是全部 val 的 mean；--best_on 指定資料集時只看該資料集。"""
    if best_on and best_on in m.get("by_src", {}):
        return m["by_src"][best_on]["mean_px"]
    return m["mean_px"]


def draw_heatmaps(img_rgb, hm_logits, gt, pred, beta, path, tile=224, seg=None, seg_gt=None):
    """每個點一格：裁切圖 + 該點的機率分布 softmax(beta * heatmap)（JET 色階疊在影像上）。
    實心點 = GT、空心圈 = 預測。格子上方標出分布最大機率 pmax 與熵比例（越接近 0 越集中、越接近 1 越平）。
    - 尖峰落在針的正確部位 -> 有學到
    - 整片均勻 / 尖峰在中心 -> 還沒學到（或 beta 太小）
    - 5 張圖的尖峰都在同一處 -> 五個部位分不開"""
    base = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    base = cv2.resize(base, (tile, tile), interpolation=cv2.INTER_LINEAR)
    C, Hh, Ww = hm_logits.shape
    tiles = []
    for c in range(C):
        z = beta * hm_logits[c].astype(np.float64)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))
        rng = p.max() - p.min()
        vis = ((p - p.min()) / rng * 255).astype(np.uint8) if rng > 1e-9 else np.zeros_like(p, np.uint8)
        vis = cv2.resize(vis, (tile, tile), interpolation=cv2.INTER_LINEAR)
        over = cv2.addWeighted(base, 0.55, cv2.applyColorMap(vis, cv2.COLORMAP_JET), 0.45, 0)
        col = POINT_COLORS[c]
        P = lambda q: (int(round(q[0] * tile)), int(round(q[1] * tile)))
        cv2.circle(over, P(gt[c]), 3, col, -1, cv2.LINE_AA)
        cv2.circle(over, P(gt[c]), 4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(over, P(pred[c]), 6, col, 2, cv2.LINE_AA)
        head = np.zeros((34, tile, 3), np.uint8)
        cv2.putText(head, POINT_NAMES[c], (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(head, f"pmax {p.max():.3f}  H {ent:.2f}", (6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
        tiles.append(np.vstack([head, over]))
        tiles.append(np.full((tile + 34, 4, 3), 40, np.uint8))
    if seg is not None:
        # 第 6 格：預測的針 mask（機率，JET）；綠色輪廓 = GT mask
        v = cv2.resize((np.clip(seg, 0, 1) * 255).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_LINEAR)
        over = cv2.addWeighted(base, 0.5, cv2.applyColorMap(v, cv2.COLORMAP_JET), 0.5, 0)
        txt = f"max {seg.max():.2f}"
        if seg_gt is not None:
            gb = cv2.resize((seg_gt > 0.25).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(gb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(over, cs, -1, (0, 255, 0), 1, cv2.LINE_AA)
            pb, tb = seg > 0.5, seg_gt > 0.25
            txt += f"  IoU {(pb & tb).sum() / max((pb | tb).sum(), 1):.2f}"
        head = np.zeros((34, tile, 3), np.uint8)
        cv2.putText(head, "needle mask (seg_head)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(head, txt, (6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        tiles.append(np.vstack([head, over]))
        tiles.append(np.full((tile + 34, 4, 3), 40, np.uint8))
    cv2.imwrite(path, np.hstack(tiles[:-1]))


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
# 特徵圖檢查（backbone / decoder / attention 有沒有學到 5 個點的特徵）
# -----------------------------------------------------------------------------
PROBE_LAYERS = ["layer1", "layer2", "layer3", "layer4", "dec1", "dec2", "dec3", "cb2", "attn"]


class FeatureProbe:
    """在官方模組上掛 forward hook 取中間特徵圖（不改模型、平常不啟動）。
    layer1~4：ResNet18 各 stage 輸出（56/28/14/7）；dec1~3：deconv 後 BN+ReLU（14/28/56）；
    cb2：cb1、cb2 之後（56）；attn：attention 輸出（56，就是 cb3 的輸入）"""

    def __init__(self, net):
        self.on, self.feats = False, {}
        cn = net.r3m.convnet
        for i in range(1, 5):
            self._reg(getattr(cn, f"layer{i}"), f"layer{i}")
        for i in range(1, 4):
            self._reg(getattr(net, f"bn{i}"), f"dec{i}", F.relu)     # 官方在 bn 之後接 ReLU
        self._reg(net.cb2, "cb2")
        self._reg(net.self_attention, "attn", self._attn)

    @staticmethod
    def _attn(o):
        o = o[0] if isinstance(o, tuple) else o                      # (B, H*W, C)
        B, N, C = o.shape
        s = int(round(math.sqrt(N)))
        return o.transpose(1, 2).reshape(B, C, s, s)

    def _reg(self, m, name, fn=None):
        def hook(mod, inp, out):
            if self.on:
                t = fn(out) if fn else out
                self.feats[name] = t.detach().float().clone()        # clone：官方後面有 inplace ReLU
        m.register_forward_hook(hook)

    @torch.no_grad()
    def run(self, net, img):
        self.feats, self.on = {}, True
        try:
            net.encode_img(img)
        finally:
            self.on = False
        HM_CACHE.pop("x", None)
        return {k: self.feats[k] for k in PROBE_LAYERS if k in self.feats}


def sample_feat(fn, pts):
    """fn: (B, C, h, w) 已 L2 normalize；pts: (B, 5, 2) normalized (x, y) -> (B, 5, C) 雙線性取樣後再 normalize"""
    grid = (pts * 2 - 1).unsqueeze(2)                                    # (B, 5, 1, 2)
    v = F.grid_sample(fn, grid, mode="bilinear", align_corners=False)[..., 0]   # (B, C, 5)
    return F.normalize(v.transpose(1, 2), dim=-1)


@torch.no_grad()
def compute_prototypes(net, probe, loader, device, max_n=0):
    """每層、每個點：train 原圖在 GT 位置的（normalize 後）特徵平均 -> 原型 (5, C)"""
    net.eval()
    sums, n = {}, 0
    for batch in loader:
        img, gt = batch["img"].to(device), batch["pts"].to(device)
        for L, f in probe.run(net, img).items():
            v = sample_feat(F.normalize(f, dim=1), gt).sum(0)
            sums[L] = sums[L] + v if L in sums else v
        n += img.shape[0]
        if max_n and n >= max_n:
            break
    return {L: F.normalize(v, dim=-1) for L, v in sums.items()}, n


def _overlay(base, m, tile, nearest=True):
    lo, hi = float(m.min()), float(m.max())
    v = ((m - lo) / (hi - lo) * 255).astype(np.uint8) if hi - lo > 1e-9 else np.zeros(m.shape, np.uint8)
    v = cv2.resize(v, (tile, tile), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    return cv2.addWeighted(base, 0.4, cv2.applyColorMap(v, cv2.COLORMAP_JET), 0.6, 0), lo, hi


def draw_feature_grid(img_rgb, gt, side, per_layer, path, tile=128):
    """列 = 各層；欄 = 原圖 | energy | head | tail | middle | head_1_3 | tail_1_3。
    熱力圖以「最近鄰」放大，看得到該層真正的解析度（layer4 是 7x7 格）。
    實心點 = GT；白色 x = 該原型相似度最高的位置；格子上方：cos 範圍 與 argmax 誤差（原圖 px）"""
    base = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (tile, tile), interpolation=cv2.INTER_AREA)
    P = lambda q: (int(round(q[0] * tile)), int(round(q[1] * tile)))
    lab_w, head_h, gap = 92, 18, 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    def cell(im, text, col=(230, 230, 230)):
        h = np.zeros((head_h, tile, 3), np.uint8)
        cv2.putText(h, text, (3, 13), font, 0.36, col, 1, cv2.LINE_AA)
        return np.vstack([h, im])

    titles = ["image", "energy |f|"] + POINT_NAMES
    hdr = [np.zeros((20, lab_w, 3), np.uint8)]
    for j, t in enumerate(titles):
        h = np.zeros((20, tile, 3), np.uint8)
        col = POINT_COLORS[j - 2] if j >= 2 else (255, 255, 255)
        cv2.putText(h, t, (4, 14), font, 0.45, col, 1, cv2.LINE_AA)
        hdr += [np.full((20, gap, 3), 40, np.uint8), h]
    rows = [np.hstack(hdr)]
    for L, (energy, sim, am) in per_layer.items():
        lab = np.zeros((tile + head_h, lab_w, 3), np.uint8)
        cv2.putText(lab, L, (6, 60), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(lab, f"{energy.shape[0]}x{energy.shape[1]}", (6, 82), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        im0 = base.copy()
        for g, c in zip(gt, POINT_COLORS):
            cv2.circle(im0, P(g), 3, c, -1, cv2.LINE_AA)
        en, lo, hi = _overlay(base, energy, tile)
        for g, c in zip(gt, POINT_COLORS):
            cv2.circle(en, P(g), 3, c, -1, cv2.LINE_AA)
        cells = [cell(im0, ""), cell(en, f"|f| {lo:.1f}~{hi:.1f}")]
        for k in range(5):
            ov, lo, hi = _overlay(base, sim[k], tile)
            cv2.circle(ov, P(gt[k]), 4, POINT_COLORS[k], -1, cv2.LINE_AA)
            cv2.circle(ov, P(gt[k]), 5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(ov, P(am[k]), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
            err = float(np.linalg.norm(am[k] - gt[k]) * side)
            cells.append(cell(ov, f"cos {lo:.2f}~{hi:.2f} err {err:.0f}px"))
        row = [lab]
        for c in cells:
            row += [np.full((tile + head_h, gap, 3), 40, np.uint8), c]
        rows.append(np.hstack(row))
        rows.append(np.full((gap, rows[-1].shape[1], 3), 40, np.uint8))
    cv2.imwrite(path, np.vstack(rows[:-1]))


@torch.no_grad()
def feature_probe(net, probe, protos, loader, device, out_dir=None, n_draw=0):
    """在 val 上：每層的原型相似度 argmax 誤差、5 點分類、頭尾分類、針上能量比；並畫前 n_draw 張。"""
    net.eval()
    acc = {L: {"err": torch.zeros(5), "cls": 0.0, "ht": 0.0, "eratio": 0.0} for L in protos}
    n, drawn = 0, 0
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for batch in loader:
        img, gt = batch["img"].to(device), batch["pts"].to(device)
        side = batch["crop"][:, 2].to(device)
        B = img.shape[0]
        maps = {}
        for L, f in probe.run(net, img).items():
            fn = F.normalize(f, dim=1)
            sim = torch.einsum("bchw,kc->bkhw", fn, protos[L])                       # (B, 5, h, w)
            _, K, h, w = sim.shape
            idx = sim.reshape(B, K, -1).argmax(-1)
            am = torch.stack([((idx % w).float() + 0.5) / w, ((idx // w).float() + 0.5) / h], -1)
            acc[L]["err"] += ((am - gt).norm(dim=-1) * side[:, None]).sum(0).cpu()
            s = torch.einsum("bkc,jc->bkj", sample_feat(fn, gt), protos[L])          # GT 位置 k 與原型 j
            acc[L]["cls"] += float((s.argmax(-1) == torch.arange(5, device=device)).float().sum())
            acc[L]["ht"] += float((s[:, 0, 0] > s[:, 0, 1]).float().sum() + (s[:, 1, 1] > s[:, 1, 0]).float().sum())
            energy = f.norm(dim=1)                                                    # (B, h, w)
            e_pts = F.grid_sample(energy[:, None], (gt * 2 - 1).unsqueeze(2), align_corners=False)[:, 0, :, 0]
            acc[L]["eratio"] += float((e_pts.mean(1) / energy.mean((1, 2)).clamp_min(1e-8)).sum())
            if out_dir and drawn < n_draw:
                maps[L] = (energy.cpu().numpy(), sim.cpu().numpy(), am.cpu().numpy())
        n += B
        if out_dir and drawn < n_draw:
            for b in range(B):
                if drawn >= n_draw:
                    break
                draw_feature_grid(batch["img_raw"][b].numpy(), batch["pts"][b].numpy(), float(side[b]),
                                  {L: (e[b], sm[b], a[b]) for L, (e, sm, a) in maps.items()},
                                  os.path.join(out_dir, batch["name"][b]))
                drawn += 1
    out = {}
    for L, d in acc.items():
        per = (d["err"] / max(n, 1)).tolist()
        out[L] = {"argmax_err_mean": float(np.mean(per)), **{f"{k}_err": v for k, v in zip(POINT_NAMES, per)},
                  "proto_acc": d["cls"] / max(5 * n, 1), "head_tail_acc": d["ht"] / max(2 * n, 1),
                  "energy_ratio": d["eratio"] / max(n, 1)}
    return out, n


def print_feature_table(epoch, res, n_tr, n_val):
    print(f"[特徵檢查 ep {epoch}] 原型：train 原圖 {n_tr} 根針；val {n_val} 根針"
          f"（5 點分類隨機 = 20%，頭尾分類隨機 = 50%；能量比 > 1 = 針上的特徵比背景強）")
    print(f"    {'layer':7s} {'argmax誤差':>10s} {'5點分類':>8s} {'頭尾分類':>8s} {'能量比':>7s}   各點誤差 h/t/m/h13/t13 (px)")
    for L, r in res.items():
        print(f"    {L:7s} {r['argmax_err_mean']:8.1f}px {r['proto_acc']:8.0%} {r['head_tail_acc']:8.0%} "
              f"{r['energy_ratio']:7.2f}   " + " / ".join(f"{r[k + '_err']:.0f}" for k in POINT_NAMES))


def plot_feature_curves(run_dir):
    """feat_log.jsonl -> features.png：x = 層（淺到深），每個檢查過的 epoch 一條線（ep 0 = pretrained，虛線）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = os.path.join(run_dir, "feat_log.jsonl")
    if not os.path.isfile(path):
        return None
    recs = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["epoch"]] = r
    if not recs:
        return None
    eps = sorted(recs)
    layers = [L for L in PROBE_LAYERS if L in recs[eps[0]]["layers"]]
    x = np.arange(len(layers))
    ink, muted, grid = "#1f2328", "#59636e", "#e6e8eb"
    cmap = plt.get_cmap("Blues")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), dpi=120)
    specs = [("proto_acc", "5-point classification by prototype", 0.2, True),
             ("head_tail_acc", "Head vs tail at GT locations", 0.5, True),
             ("argmax_err_mean", "Prototype-similarity argmax error (px)", None, False)]
    for ax, (key, title, chance, pct) in zip(axes, specs):
        ax.set_facecolor("white")
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted, labelsize=9)
        for i, e in enumerate(eps):
            ys = [recs[e]["layers"][L][key] for L in layers]
            if e == 0:
                ax.plot(x, ys, color=muted, linewidth=1.8, linestyle="--", marker="o", markersize=3, label="ep 0 (pretrained)")
            else:
                c = cmap(0.35 + 0.65 * (i / max(len(eps) - 1, 1)))
                ax.plot(x, ys, color=c, linewidth=2 if e == eps[-1] else 1.2, marker="o", markersize=3, label=f"ep {e}")
        if chance is not None:
            ax.axhline(chance, color="#d9480f", linewidth=1, linestyle=":", label=f"chance {chance:.0%}")
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=30)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        if pct:
            ax.set_ylim(0, 1.02)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8, labelcolor=ink)
    fig.suptitle(os.path.basename(os.path.normpath(run_dir)) + "  feature probe (val)", color=ink, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    out = os.path.join(run_dir, "features.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


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

    srcs = sorted({s for e in ve for s in recs[e]["val"].get("by_src", {})})
    multi = len(srcs) > 1
    fig, axes = plt.subplots(1, 3 if multi else 2, figsize=(19 if multi else 13, 4.6), dpi=120)
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
        bpath = os.path.join(run_dir, "baseline.json")
        if os.path.isfile(bpath):
            bm = json.load(open(bpath))["mean_px"]
            ax.axhline(bm, color=muted, linewidth=1.2, linestyle=":", label=f"baseline: mean shape ({bm:.1f}px)")
        te = [e for e in ep if "train_eval" in recs[e]]
        if te:
            ax.plot(te, [recs[e]["train_eval"]["mean_px"] for e in te], color=ink, linewidth=1.5,
                    linestyle="--", marker="o", markersize=4, label="train mean (no aug)")
        b = int(np.argmin(vm))
        right = ve[b] > ep[0] + 0.7 * (ep[-1] - ep[0])
        ax.scatter([ve[b]], [vm[b]], s=40, color=ink, edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(f"best {vm[b]:.1f}px @ ep {ve[b]}", (ve[b], vm[b]), xytext=(-8 if right else 8, -16),
                    textcoords="offset points", fontsize=9, color=ink, ha="right" if right else "left")
    sym_note = ""
    apath = os.path.join(run_dir, "args.json")
    if os.path.isfile(apath) and json.load(open(apath)).get("loss") == "sym":
        sym_note = " (head/tail order ignored)"
    ax.set_title("Point error in original-image pixels" + sym_note, color=ink, fontsize=11, loc="left")
    ax.set_ylabel("px", color=muted)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, ncol=2, labelcolor=ink)

    # (3) 多資料集：各資料集的 val mean
    if multi:
        ax = axes[2]
        src_colors = ["#2a78d6", "#d9480f", "#2b9348", "#7048e8", "#c2255c"]
        for s, c in zip(srcs, src_colors * 3):
            es = [e for e in ve if s in recs[e]["val"].get("by_src", {})]
            ys = [recs[e]["val"]["by_src"][s]["mean_px"] for e in es]
            ax.plot(es, ys, color=c, linewidth=2, label=f"{s} (n={recs[es[-1]]['val']['by_src'][s]['n']})")
            b = int(np.argmin(ys))
            ax.scatter([es[b]], [ys[b]], s=36, color=c, edgecolors="white", linewidths=1.5, zorder=3)
            ax.annotate(f"{ys[b]:.1f}px @ {es[b]}", (es[b], ys[b]), xytext=(6, -14),
                        textcoords="offset points", fontsize=8, color=c)
        ax.set_title("Val mean error per dataset" + sym_note, color=ink, fontsize=11, loc="left")
        ax.set_ylabel("px", color=muted)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8, labelcolor=ink)

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
def load_annotations(paths):
    """讀一個或多個 annotations.json。多個時合併（必須都是 build_needle5_aug.py 的離線格式），
    每筆記錄自己的影像根目錄（_root）與資料集名稱（_src = annotations.json 所在資料夾名稱）。
    各資料集沿用自己原本的 train / val / test 切分。"""
    anns, srcs, dirs = [], [], []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            anns.append(json.load(f))
        d = os.path.dirname(os.path.abspath(p))
        name, k = os.path.basename(d), 2
        while name in srcs:                                     # 資料夾同名時加編號
            name, k = f"{os.path.basename(d)}_{k}", k + 1
        srcs.append(name)
        dirs.append(d)
    if len(anns) == 1:
        a = anns[0]
        for s in a.get("samples", []):
            s["_src"] = srcs[0]
        a["_srcs"], a["_dirs"] = srcs, dirs
        return a
    if any(a.get("format") != "needle5_crops" for a in anns):
        sys.exit("[ERROR] 多個 --ann 只支援 build_needle5_aug.py 產生的離線增強資料（format: needle5_crops）")
    merged = {"format": "needle5_crops", "samples": [], "_srcs": srcs, "_dirs": dirs}
    for a, s_name, d in zip(anns, srcs, dirs):
        for s in a["samples"]:
            merged["samples"].append({**s, "_src": s_name, "_root": d, "_multi": True})
    return merged


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
        # 保存這次用的程式碼，之後可以知道每個結果是哪一版程式訓練出來的
        shutil.copy2(os.path.abspath(__file__), os.path.join(run_dir, "train_vrb_needle5.py"))
        for k in ("vrb_root", "pretrained"):
            setattr(args, k, os.path.abspath(getattr(args, k)))
        args.ann = [os.path.abspath(p) for p in args.ann]
        print(f"新的訓練，結果存到：{run_dir}")

    ann = load_annotations(args.ann if isinstance(args.ann, list) else [args.ann])
    offline = ann.get("format") == "needle5_crops"
    if args.train_splits is None:
        args.train_splits = ["train"] if offline else ["train", "val"]
    if args.val_splits is None:
        args.val_splits = ["val"] if offline else ["test"]
    images_root = args.images_root or (ann["_dirs"][0] if offline else ann.get("images_root"))
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

    if offline:
        train_set = Needle5CropDataset(ann, args.images_root, args.train_splits)
        val_set = Needle5CropDataset(ann, args.images_root, args.val_splits)
        train_eval_set = Needle5CropDataset(ann, args.images_root, args.train_splits, orig_only=True)
        n_orig = sum(not x.get("augmented", False) for x in train_set.samples)
        print(f"[離線增強資料] train {len(train_set)} 張（其中原圖 {n_orig}）({args.train_splits}) / "
              f"val {len(val_set)} 張 ({args.val_splits})；訓練時不再做線上增強")
        if len(ann["_srcs"]) > 1:
            for s in ann["_srcs"]:
                print(f"    [{s}] train {sum(x['_src'] == s for x in train_set.samples)} 張 / "
                      f"val {sum(x['_src'] == s for x in val_set.samples)} 張")
    else:
        train_set = Needle5Dataset(ann, args.images_root, args.train_splits, args, train=True)
        val_set = Needle5Dataset(ann, args.images_root, args.val_splits, args, train=False)
        train_eval_set = Needle5Dataset(ann, args.images_root, args.train_splits, args, train=False)
        print(f"[線上增強] train {len(train_set)} 根針 ({args.train_splits}) / val {len(val_set)} 根針 ({args.val_splits})")
    if len(train_set) == 0:
        sys.exit("[ERROR] 沒有訓練資料")
    bs = min(args.batch_size, len(train_set))
    if args.balance and offline and len(ann["_srcs"]) > 1:
        # 每個資料集被抽到的機率相同（不然張數多的那份會主導訓練）；每個 epoch 的張數不變
        cnt = {}
        for x in train_set.samples:
            cnt[x["_src"]] = cnt.get(x["_src"], 0) + 1
        w = torch.tensor([1.0 / cnt[x["_src"]] for x in train_set.samples], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(train_set), replacement=True)
        train_loader = DataLoader(train_set, batch_size=bs, sampler=sampler, num_workers=args.num_workers)
        print(f"[balance] 各資料集抽樣機率相同：{cnt}")
    else:
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=0) if len(val_set) else None
    train_eval_loader = DataLoader(train_eval_set, batch_size=bs, shuffle=False, num_workers=0)

    net = build_vrb(args)
    if not args.attn_kp_fc and args.kp_beta > 0:
        patch_get_coord(args.kp_beta)
    elif not args.attn_kp_fc:
        print("[WARN] attn_kp_fc=0 但 kp_beta=0：使用官方 get_coord，5 點很可能都縮到裁切中心")
        import networks.model as _M
        _orig = _M.get_coord

        def _capture(x, other_axis, axis_size):                      # 只記錄 heatmap，計算照官方
            HM_CACHE["x"] = x
            return _orig(x, other_axis, axis_size)
        _M.get_coord = _capture
    ckpt_info = load_pretrained(net, args.pretrained) if resume_state is None else None
    reinit = set()
    if not args.attn_kp_fc:
        do_reinit = bool(args.reinit_hm_head) and resume_state is None
        fix_heatmap_head(net, remove_relu=args.hm_relu == 0, reinit=do_reinit)
        if do_reinit:
            reinit.add("cb3")
        if args.hm_relu and args.kp_beta <= 0:
            print("[WARN] cb3 有 ReLU 且使用官方 get_coord：分布接近均勻，5 點會縮到中心；建議 --kp_beta 10")
    patch_attention(net, args.attn_mode)
    n_mask = sum(1 for x in train_set.samples if x.get("mask"))
    use_seg = args.seg_weight > 0 and n_mask > 0
    if use_seg or (resume_state is not None and any(k.startswith("seg_head.") for k in resume_state["model"])):
        add_seg_head(net)
        print(f"    train 有 mask 的圖：{n_mask}/{len(train_set)}（沒有 mask 的圖不計 seg loss）")
    elif args.seg_weight > 0:
        print("[NOTE] 資料沒有 mask（請用新版 build_needle5_aug.py 重做），segmentation 輔助頭不啟用")
    net = net.to(device)

    # 凍結：--freeze_upto 指定的 stage（含）之前全部不訓練
    if args.freeze_backbone and args.freeze_upto == "none":
        args.freeze_upto = "backbone"
    frozen = stage_modules(args.freeze_upto)
    for name in frozen:
        for p in getattr(net, name).parameters():
            p.requires_grad = False
    if frozen:
        print(f"[NOTE] 凍結 {args.freeze_upto} 之前（含）：{frozen}")
    bids = {id(p) for p in net.r3m.parameters()}
    backbone = [p for p in net.r3m.parameters() if p.requires_grad]
    others = [p for p in net.parameters() if p.requires_grad and id(p) not in bids]
    groups = [{"params": others, "lr": args.lr}]
    if backbone:
        groups.append({"params": backbone, "lr": args.lr * args.backbone_lr_mult})
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    if ckpt_info is not None:
        write_init_report(net, run_dir, args, ckpt_info, reinit, set(frozen), find_used_modules(net, device))
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
    resumed = resume_state is not None
    if resumed:
        net.load_state_dict(resume_state["model"])
        optim.load_state_dict(resume_state["optim"])
        start_epoch, step, best = resume_state["epoch"] + 1, resume_state["step"], resume_state["best"]
        r = resume_state["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"])
        torch.cuda.set_rng_state_all(r["cuda"])
        del resume_state
    use_hm = (not args.attn_kp_fc) and args.kp_beta > 0 and args.hm_weight > 0
    if use_hm:
        print(f"[NOTE] 加上 heatmap 監督：weight={args.hm_weight}, sigma={args.hm_sigma} 格（56x56 heatmap）")

    # 平均位置 baseline：每個點都預測「訓練集該點的平均位置」，模型至少要比這個好
    if val_loader and not resumed:
        tr_pts = torch.stack([train_eval_set[i]["pts"] for i in range(len(train_eval_set))]).mean(0)
        errs = []
        for b in val_loader:
            e_o = (tr_pts[None] - b["pts"]).norm(dim=-1) * b["crop"][:, 2:3]
            if args.loss == "sym":
                e_s = (tr_pts[None] - b["pts"][:, SWAP_HEAD_TAIL]).norm(dim=-1) * b["crop"][:, 2:3]
                e_o = torch.where((e_s.mean(1) < e_o.mean(1))[:, None], e_s, e_o)
            errs.append(e_o)
        e = torch.cat(errs).mean(0)
        base = {"mean_px": float(e.mean()), **{f"{k}_px": float(v) for k, v in zip(POINT_NAMES, e)}}
        with open(os.path.join(run_dir, "baseline.json"), "w") as f:
            json.dump(base, f, indent=2)
        print(f"[baseline：一律預測訓練集平均位置] val mean {base['mean_px']:.1f}px  "
              + "  ".join(f"{k} {base[k + '_px']:.1f}" for k in POINT_NAMES))
        print("  -> 模型的 val mean 若和這個差不多，代表它只學到「平均形狀」，沒有真的看圖找針")

    probe = FeatureProbe(net) if args.feat_probe and val_loader else None

    def run_probe(ep):
        protos, n_tr = compute_prototypes(net, probe, train_eval_loader, device, args.feat_proto_n)
        res, n_val = feature_probe(net, probe, protos, val_loader, device,
                                   os.path.join(run_dir, "vis", f"epoch{ep:04d}_val", "features"), args.feat_vis_n)
        with open(os.path.join(run_dir, "feat_log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": ep, "n_proto": n_tr, "n_val": n_val, "layers": res}) + "\n")
        print_feature_table(ep, res, n_tr, n_val)
        plot_feature_curves(run_dir)

    if val_loader and not resumed:
        m = evaluate(net, val_loader, device, args, os.path.join(run_dir, "vis", "epoch0000_val"))
        print("[pretrained] val: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()
                                               if isinstance(v, (int, float))) + src_msg(m))
        if probe:
            run_probe(0)

    ckpt_path = os.path.join(run_dir, CKPT_NAME)
    log_f = open(os.path.join(run_dir, "log.jsonl"), "a")
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            net.train()
            if args.freeze_bn:
                set_bn_eval(net)
            for name in frozen:                       # 凍結的模組：BN 統計量、dropout 也固定
                getattr(net, name).eval()
            t0 = time.time()
            tot, n, seg_tot = 0.0, 0, 0.0
            for batch in train_loader:
                set_lr(step)
                img = batch["img"].to(device)
                gt = batch["pts"].to(device)
                HM_CACHE.pop("attn", None)
                loss = contact_loss(predict_xy(net, img, args), gt, args.loss)
                if use_hm:
                    loss = loss + args.hm_weight * heatmap_loss(gt, args)
                if use_seg:
                    ls = seg_loss(seg_logits(net), batch["mask"].to(device), batch["has_mask"].to(device),
                                  args.seg_pos_weight, args.seg_dice)
                    if ls is not None:
                        loss = loss + args.seg_weight * ls
                        seg_tot += ls.item() * img.shape[0]
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
            if use_seg:
                rec["train_seg_loss"] = seg_tot / n
                msg += f" (seg {seg_tot / n:.3f})"
            do_vis = epoch % args.vis_every == 0 or epoch == args.epochs
            if val_loader and (epoch % args.eval_every == 0 or do_vis):
                vm = evaluate(net, val_loader, device, args,
                              os.path.join(run_dir, "vis", f"epoch{epoch:04d}_val") if do_vis else None)
                rec["val"] = vm
                msg += (f" | val mean{' (head/tail aligned)' if args.loss == 'sym' else ''} {vm['mean_px']:.1f}px (head {vm['head_px']:.1f} tail {vm['tail_px']:.1f} "
                        f"mid {vm['middle_px']:.1f} h1/3 {vm['head_1_3_px']:.1f} t1/3 {vm['tail_1_3_px']:.1f}) "
                        f"flip {vm['flip_rate']:.0%}"
                        + (f" hm0 {vm['hm_zero_frac']:.0%} | soft-argmax(全圖) {vm['mean_px_softargmax']:.1f}px"
                           if not args.attn_kp_fc else "")
                        + (" | dead " + "/".join(f"{d:.0%}" for d in vm["hm_dead_frac"])
                           if not args.attn_kp_fc and max(vm["hm_dead_frac"]) > 0 else "")
                        + (f" | seg IoU {vm['seg_iou']:.2f} P {vm['seg_precision']:.2f} R {vm['seg_recall']:.2f}"
                           if "seg_iou" in vm else "")
                        + src_msg(vm))
                score = select_metric(vm, args.best_on)
                if score < best:
                    best = score
                    torch.save(net.state_dict(), os.path.join(run_dir, "model_best.pth.tar"))
                    msg += "  *best*"
            if do_vis:
                tm = evaluate(net, train_eval_loader, device, args,
                              os.path.join(run_dir, "vis", f"epoch{epoch:04d}_train"))
                rec["train_eval"] = tm
                msg += f" | train mean {tm['mean_px']:.1f}px"
            print(msg + f" | {time.time() - t0:.1f}s")
            if do_vis and probe:
                run_probe(epoch)
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
    print(f"完成，best val mean{f' [{args.best_on}]' if args.best_on else ''} = {best:.1f}px，"
          f"結果在 {run_dir}（學習曲線：curves.png）")

    # 用 best 權重評估 test（只在 test 沒被拿來訓練 / 選模型時）
    test_splits = [sp for sp in ("test",) if sp not in args.train_splits and sp not in args.val_splits]
    best_path = os.path.join(run_dir, "model_best.pth.tar")
    if test_splits and os.path.isfile(best_path):
        test_set = (Needle5CropDataset(ann, args.images_root, test_splits) if offline
                    else Needle5Dataset(ann, args.images_root, test_splits, args, train=False))
        if len(test_set):
            net.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
            tm = evaluate(net, DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=0), device, args,
                          os.path.join(run_dir, "vis", "test_best"))
            with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
                json.dump(tm, f, indent=2)
            print(f"[test, best 權重] {len(test_set)} 張  mean {tm['mean_px']:.1f}px  "
                  + "  ".join(f"{k} {tm[k + '_px']:.1f}" for k in POINT_NAMES) + f"  flip {tm['flip_rate']:.0%}"
                  + src_msg(tm))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vrb_root")
    p.add_argument("--ann", nargs="+",
                   help="annotations.json，可以給多個一起訓練（例如自己的 + 學長的，需為 build_needle5_aug.py 格式）")
    p.add_argument("--balance", type=int, default=0,
                   help="多個資料集時：1 = 每個資料集抽樣機率相同（張數少的那份不會被淹沒）；0 = 照張數比例")
    p.add_argument("--best_on", default=None,
                   help="多個資料集時，只用這個資料集（資料夾名稱，例如 needle5_mine_aug）的 val 選 best 權重；預設用全部 val")
    p.add_argument("--images_root", default=None, help="影像根目錄（預設用 annotations.json 裡的；container 路徑不同時要指定）")
    p.add_argument("--pretrained")
    p.add_argument("--out_dir", default="runs/needle5",
                   help="實際存到 <out_dir>_<開始時間>/；不指定就是 runs/needle5_<開始時間>/")
    p.add_argument("--exp", default=None,
                   help="實驗分類資料夾：結果存到 runs/<exp>/needle5_<開始時間>/，例如 --exp mine_plus_senior")
    p.add_argument("--resume", default=None, help="run 資料夾 / checkpoint_last.pth / latest")
    p.add_argument("--train_splits", nargs="+", default=None,
                   help="預設：離線資料 train；線上資料 train val")
    p.add_argument("--val_splits", nargs="+", default=None,
                   help="預設：離線資料 val；線上資料 test")

    # 架構（與官方 demo.py 相同；attn_kp_fc 可選 0 / 1）
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--enc_depth", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--hand_latent_dim", type=int, default=4)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--coord_dim", type=int, default=64)
    p.add_argument("--resnet_type", type=str, default="resnet18")
    p.add_argument("--attn_kp", type=int, default=1)
    p.add_argument("--attn_kp_fc", type=int, default=0, help="1: 全連接回歸（官方預設）；0: heatmap + soft-argmax")
    p.add_argument("--decode", choices=["local", "global"], default="local",
                   help="attn_kp_fc=0 時評估 / 畫圖用的解碼：local = 只看尖峰附近（去掉往中心拉的偏差）；global = 全圖 soft-argmax")
    p.add_argument("--decode_radius", type=int, default=3, help="local 解碼的範圍（熱度圖格數，56x56）")
    p.add_argument("--hm_relu", type=int, default=1,
                   help="attn_kp_fc=0 時熱度圖輸出層 cb3 是否保留官方的 ReLU（1 = 官方原版，預設；0 = 拿掉）")
    p.add_argument("--attn_mode", choices=["official", "spatial"], default="official",
                   help="official：照官方（MultiheadAttention 沒設 batch_first，實際沿 batch 維度做 attention）；"
                        "spatial：每張影像 H*W 個位置之間做 attention（同一組權重）")
    p.add_argument("--freeze_upto", choices=["none"] + [s[0] for s in STAGES[:-1]], default="none",
                   help="凍結到哪一段（含）：backbone / decoder / cb / attn；例如 attn = 只訓練 cb3")
    p.add_argument("--reinit_hm_head", type=int, default=1,
                   help="attn_kp_fc=0 時重新初始化熱度圖輸出層 cb3（官方 pretrained 沒訓練過這層）")
    p.add_argument("--hm_weight", type=float, default=0.1,
                   help="attn_kp_fc=0 且 kp_beta>0 時，高斯 heatmap 監督（KL）的權重；0 = 關閉")
    p.add_argument("--seg_weight", type=float, default=0.5,
                   help="segmentation 輔助頭的 loss 權重（資料要有 mask：新版 build_needle5_aug.py）；0 = 不使用")
    p.add_argument("--seg_pos_weight", type=float, default=5.0, help="BCE 中「針」像素的權重（針只佔很少像素）")
    p.add_argument("--seg_dice", type=float, default=1.0, help="Dice loss 的權重（相對於 BCE）")
    p.add_argument("--hm_sigma", type=float, default=1.5, help="目標高斯的標準差（heatmap 格數，56x56）")
    p.add_argument("--kp_beta", type=float, default=0,
                   help="attn_kp_fc=0 時的 softmax 溫度倒數；>0 會改用 2D softmax（建議 10~30），0 = 官方原版")
    p.add_argument("--traj_len", type=int, default=5)
    p.add_argument("--encoder_time_embed_type", default="sin", choices=["sin", "param"])

    # 裁切 / augmentation
    p.add_argument("--box_pad", type=float, default=20, help="針的框四周最少留白 (px)")
    p.add_argument("--box_pad_ratio", type=float, default=0.15, help="留白至少為框長邊的此比例")
    p.add_argument("--min_side", type=float, default=96, help="裁切邊長下限 (px)")
    p.add_argument("--scale_jitter", type=float, default=0.15)
    p.add_argument("--center_jitter", type=float, default=0.08)
    p.add_argument("--hflip", type=float, default=0.5)
    p.add_argument("--vflip", type=float, default=0.5)
    p.add_argument("--rot_deg", type=float, default=180, help="訓練時隨機旋轉 ±rot_deg 度（0 = 關閉）")
    p.add_argument("--color_aug", type=int, default=1)
    p.add_argument("--hue", type=float, default=0.05)

    # 訓練
    p.add_argument("--loss", choices=["ordered", "sym"], default="ordered")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1)
    p.add_argument("--freeze_backbone", type=int, default=0, help="舊選項，等同 --freeze_upto backbone")
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
    p.add_argument("--feat_probe", type=int, default=1,
                   help="1 = 在 epoch 0 與每 --vis_every 個 epoch 做特徵圖檢查（features.png、vis/*/features/）")
    p.add_argument("--feat_vis_n", type=int, default=16, help="特徵檢查每次畫幾張 val 圖")
    p.add_argument("--feat_proto_n", type=int, default=0, help="算原型最多用幾根 train 原圖的針（0 = 全部）")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.exp:                         # runs/needle5 -> runs/<exp>/needle5
        a.out_dir = os.path.join(os.path.dirname(a.out_dir.rstrip("/")) or ".", a.exp,
                                 os.path.basename(a.out_dir.rstrip("/")))
    if a.plot_only:
        out = plot_curves(a.plot_only)
        print(f"→ {out}" if out else f"[ERROR] {a.plot_only} 裡沒有 log.jsonl")
        fo = plot_feature_curves(a.plot_only)
        if fo:
            print(f"→ {fo}")
        sys.exit(0)
    if not a.resume and not (a.vrb_root and a.ann and a.pretrained):
        p.error("新的訓練需要 --vrb_root、--ann、--pretrained（接續只需要 --resume）")
    main(a)