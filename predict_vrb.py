#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 fine-tune 後的 VRB 預測一張影像的 hand_off contact point 與軌跡。
前處理和 train_vrb.py 完全一致（不使用官方 inference.py 的流程，原因見下）：

    官方 inference.py                          本腳本（= 訓練時的設定）
    -----------------------------------------  -----------------------------------------
    先把整張圖硬縮成 1008x756（16:9 -> 4:3 變形）  保持原始解析度，不變形
    裁切 = 物體 bbox + 20px 再隨機縮               裁切 = 以物體中心為中心、邊長 crop_size 的正方形
    推論時仍套用 ColorJitter / RandomGrayscale      推論時不做任何隨機色彩擾動
    軌跡只畫成一個箭頭方向                         把 5 個軌跡點都畫出來

裁切範圍的決定方式（擇一，優先順序由上到下）:
    --center X Y        以指定點為中心、邊長 crop_size 的正方形（原圖 pixel）
    --roi X1 Y1 X2 Y2   固定 ROI（訓練時有用 --roi 就要給一樣的）
    LangSAM（預設）      用文字（預設 "needle"）框出物體，取信心最高的 bbox：
        --crop_mode bbox  (預設) 正方形完整包住 bbox，四周再留 --box_pad px（官方 VRB 也是 +20px）
                          邊長 = max(bbox 長邊 + 2*box_pad, crop_size)
        --crop_mode fixed 以 bbox 中心為中心、邊長固定 crop_size（舊版行為，可能切掉部分物體）

用法:
    python predict_vrb.py --vrb_root . \
        --model_path runs/handoff_ft_20260926_134512/model_best.pth.tar \
        --image test.png
    # crop_size / traj_y_up / roi 會自動從 model_path 同資料夾的 args.json 讀取

輸出（每次執行一個新資料夾）:
    pred_handoff_output/<執行時間>/
        <影像名>_vrb.png     原圖上畫：裁切框(灰)、LangSAM bbox(紫)、5 個 contact 預測(藍)、
                             contact 平均(紅，空心圓，才不會蓋住軌跡第 1 點)、軌跡 5 點(黃 -> 綠)
                             超出圖片的軌跡點會畫在最靠近的圖片邊緣（空心圓）並在終端機提示
        <影像名>_input.png   實際送進 VRB 的 224x224 影像（灰框裁下來再縮放）
        <影像名>_vrb.json    所有預測的原圖 pixel 座標、執行參數
"""

import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
                    attn_kp=args.attn_kp, attn_kp_fc=args.attn_kp_fc, n_maps=args.n_maps)


def load_weights(net, path):
    dt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(dt, dict) and "model" in dt and isinstance(dt["model"], dict):
        dt = dt["model"]                       # checkpoint_last.pth（接續訓練用的格式）
    elif isinstance(dt, dict) and "state_dict" in dt:
        dt = dt["state_dict"]
    net.load_state_dict(dt)


def detect_box(image_pil, text):
    """用 LangSAM 找物體，回傳信心最高的 bbox (x1, y1, x2, y2)；找不到回傳 None。
    同時相容舊版 (predict(image, text)) 與新版 (predict([images], [texts])) 的 lang_sam。"""
    from lang_sam import LangSAM
    model = LangSAM()
    with torch.no_grad():
        try:
            out = model.predict(image_pil, text)
            masks, boxes, phrases, logits = out
        except (TypeError, ValueError):
            res = model.predict([image_pil], [text])[0]
            boxes, logits = res.get("boxes"), res.get("scores")
    if boxes is None or len(boxes) == 0:
        return None
    boxes = np.asarray(boxes.cpu() if torch.is_tensor(boxes) else boxes, dtype=np.float64)
    if logits is not None and len(logits) == len(boxes):
        lg = np.asarray(logits.cpu() if torch.is_tensor(logits) else logits, dtype=np.float64)
        return boxes[int(np.argmax(lg))]
    return boxes[0]


def square_crop(cx, cy, side, W, H):
    side = int(round(min(side, W, H)))
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return x0, y0, side


def color(i, n):
    t = i / max(n - 1, 1)
    return (0, int(round(255 * (1 - t) + 200 * t)), int(round(255 * (1 - t))))  # 黃 -> 綠 (BGR)


RUN_STAMP = None


def make_stamp(root):
    base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp, k = base, 1
    while os.path.exists(os.path.join(root, stamp)):
        stamp, k = f"{base}_{k}", k + 1
    return stamp


# -----------------------------------------------------------------------------
def main(args):
    global RUN_STAMP
    RUN_STAMP = make_stamp(args.out_root)
    # 從訓練時的 args.json 讀取前處理設定，確保推論與訓練一致
    train_args_path = args.train_args or os.path.join(os.path.dirname(os.path.abspath(args.model_path)), "args.json")
    if os.path.isfile(train_args_path):
        with open(train_args_path) as f:
            ta = json.load(f)
        for k in ("crop_size", "traj_y_up", "n_maps", "traj_len", "scale_jitter"):
            if getattr(args, k) is None and k in ta:
                setattr(args, k, ta[k])
        if args.roi is None and args.center is None and ta.get("roi"):
            args.roi = ta["roi"]
        print(f"讀取訓練設定：{train_args_path}")
    else:
        print("[WARN] 找不到訓練時的 args.json，使用預設 crop_size=480, traj_y_up=1")
    args.crop_size = 480 if args.crop_size is None else args.crop_size
    args.traj_y_up = 1 if args.traj_y_up is None else args.traj_y_up
    args.n_maps = 5 if args.n_maps is None else args.n_maps
    args.traj_len = 5 if args.traj_len is None else args.traj_len
    args.scale_jitter = 0.15 if args.scale_jitter is None else args.scale_jitter
    print(f"  crop_size={args.crop_size}  traj_y_up={args.traj_y_up}  roi={args.roi}")

    if not torch.cuda.is_available():
        sys.exit("[ERROR] 官方 VRBModel 在 __init__ 內對 BatchNorm 呼叫了 .cuda()，需要 GPU")
    device = torch.device("cuda")

    image_pil = Image.open(args.image).convert("RGB")      # 不 resize，保持原始解析度
    W, H = image_pil.size

    # ---- 決定裁切 ----
    box = None
    if args.center is not None:
        cx, cy = args.center
        side = args.crop_size
        src = "center"
    elif args.roi is not None:
        x1, y1, x2, y2 = args.roi
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
        src = "roi"
    else:
        box = detect_box(image_pil, args.text)
        if box is None:
            sys.exit(f"[ERROR] LangSAM 找不到 '{args.text}'，請改用 --center X Y 手動指定裁切中心")
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if args.crop_mode == "bbox":
            long_side = max(box[2] - box[0], box[3] - box[1])
            side = max(long_side + 2 * args.box_pad, args.crop_size)
        else:
            side = args.crop_size
        src = f"LangSAM '{args.text}' bbox={np.round(box).astype(int).tolist()} mode={args.crop_mode}"
    x0, y0, side = square_crop(cx, cy, side, W, H)
    print(f"裁切 ({src})：x0={x0} y0={y0} side={side}")
    if box is not None and not (x0 <= box[0] and y0 <= box[1] and box[2] <= x0 + side and box[3] <= y0 + side):
        print("  [NOTE] 裁切範圍沒有完整包住 LangSAM bbox（bbox 比整張圖的高還大，或 crop_mode=fixed）")

    # 裁切尺度和訓練時差太多的話，物體在 224 圖上的大小會跟訓練時不同，預測可能變差
    ratio = side / args.crop_size
    if ratio > 1 + args.scale_jitter or ratio < 1 - args.scale_jitter:
        print(f"  [WARN] 這次裁切邊長 {side}px 是訓練 crop_size {args.crop_size:.0f}px 的 {ratio:.2f} 倍，"
              f"超出訓練時的縮放範圍（±{args.scale_jitter:.0%}），物體在模型眼中的大小會和訓練資料不同")

    crop = image_pil.crop((x0, y0, x0 + side, y0 + side)).resize((224, 224), Image.BILINEAR)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    inp = tf(crop).unsqueeze(0).to(device)

    # ---- 模型 ----
    net = build_vrb(args)
    load_weights(net, args.model_path)
    net = net.to(device).eval()
    with torch.no_grad():
        pred_traj, pred_contact = net.inference(inp, None, None)
    mu = pred_contact[0, 0].cpu().numpy()          # (n_maps, 2) 裁切內 normalized
    off = pred_traj[0].cpu().numpy()               # (traj_len, 2) 相對 contact 的位移 / side
    if args.traj_y_up:
        off[:, 1] = -off[:, 1]

    # ---- 換回原圖 pixel ----
    contacts_px = np.stack([x0 + mu[:, 0] * side, y0 + mu[:, 1] * side], axis=1)
    contact_px = contacts_px.mean(0)
    traj_px = contact_px[None] + off * side

    # ---- 畫圖 ----
    img = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)
    r = args.radius
    cv2.rectangle(img, (x0, y0), (x0 + side, y0 + side), (160, 160, 160), 1)
    if box is not None:
        cv2.rectangle(img, tuple(np.round(box[:2]).astype(int)), tuple(np.round(box[2:]).astype(int)),
                      (200, 0, 200), 1)
    for p in contacts_px:
        cv2.circle(img, tuple(np.round(p).astype(int)), r, (255, 0, 0), -1, cv2.LINE_AA)
    # contact 平均：空心圓（半徑 r+2），這樣與它重疊的軌跡第 1 點仍看得到
    cv2.circle(img, tuple(np.round(contact_px).astype(int)), r + 2, (0, 0, 255), 1, cv2.LINE_AA)
    traj_in_image = []
    for i, p in enumerate(traj_px):
        inside = bool(0 <= p[0] < W and 0 <= p[1] < H)
        traj_in_image.append(inside)
        q = np.array([min(max(p[0], 0), W - 1), min(max(p[1], 0), H - 1)])
        cv2.circle(img, tuple(np.round(q).astype(int)), r if inside else r + 1,
                   color(i, len(traj_px)), -1 if inside else 1, cv2.LINE_AA)
        if not inside:
            print(f"  [NOTE] 軌跡第 {i + 1} 點 ({p[0]:.0f}, {p[1]:.0f}) 超出圖片，畫在邊緣 (空心圓)")

    out_dir = os.path.join(args.out_root, RUN_STAMP)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(args.image))[0])
    cv2.imwrite(stem + "_vrb.png", img)
    crop.save(stem + "_input.png")
    with open(stem + "_vrb.json", "w") as f:
        json.dump({
            "image": os.path.abspath(args.image),
            "model_path": os.path.abspath(args.model_path),
            "crop": {"x0": x0, "y0": y0, "side": side, "source": src},
            "langsam_box": None if box is None else np.round(box, 2).tolist(),
            "contact_px": contact_px.round(2).tolist(),
            "contact_maps_px": contacts_px.round(2).tolist(),
            "trajectory_px": traj_px.round(2).tolist(),
            "trajectory_in_image": traj_in_image,
            "contact_norm": [contact_px[0] / W, contact_px[1] / H],
            "args": vars(args),
        }, f, indent=2)
    print(f"contact = ({contact_px[0]:.1f}, {contact_px[1]:.1f}) px")
    print(f"輸出資料夾：{out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vrb_root", required=True)
    p.add_argument("--model_path", required=True, help="model_best.pth.tar / model_last.pth.tar / checkpoint_last.pth")
    p.add_argument("--image", required=True)
    p.add_argument("--out_root", default="pred_handoff_output",
                   help="輸出根目錄，每次執行會在底下建立 <執行時間> 資料夾")
    p.add_argument("--train_args", default=None, help="訓練時的 args.json，預設找 model_path 同資料夾")
    p.add_argument("--text", default="needle", help="LangSAM 的文字提示")
    p.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"))
    p.add_argument("--roi", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--crop_mode", choices=["bbox", "fixed"], default="bbox",
                   help="bbox: 裁切完整包住 LangSAM bbox；fixed: 以 bbox 中心、固定 crop_size")
    p.add_argument("--box_pad", type=float, default=20, help="bbox 四周額外留白 (px)，官方 VRB 用 20")
    p.add_argument("--crop_size", type=float, default=None)
    p.add_argument("--scale_jitter", type=float, default=None)
    p.add_argument("--traj_y_up", type=int, default=None)
    p.add_argument("--radius", type=int, default=2)
    # 架構參數（與官方 demo.py / train_vrb.py 相同）
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--enc_depth", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--hand_latent_dim", type=int, default=4)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--coord_dim", type=int, default=64)
    p.add_argument("--resnet_type", type=str, default="resnet18")
    p.add_argument("--attn_kp", type=int, default=1)
    p.add_argument("--attn_kp_fc", type=int, default=1)
    p.add_argument("--traj_len", type=int, default=None)
    p.add_argument("--n_maps", type=int, default=None)
    p.add_argument("--encoder_time_embed_type", default="sin", choices=["sin", "param"])
    main(p.parse_args())