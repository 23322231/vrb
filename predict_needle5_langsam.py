#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
照官方 demo.py / inference.py 的流程，用訓練好的 needle5 模型預測針的 5 個點：

    1. LangSAM 用文字（預設 "needle"）找針的 bbox（同 inference.py：model.predict(image, text)，取第一個框）
    2. 整張圖「不」縮放（官方 demo.py 會先 resize 成 1008x756，這裡照你的要求拿掉），
       沿著 bbox 用 inference.py 一模一樣的切法裁切：
         - bbox 四邊各外擴 20 px（bbox_offset = 20）
         - 長邊兩側各往內縮一段隨機長度，讓它接近正方形：
             左（上）縮 diff / U(1.5, 2.5)，右（下）縮 diff / (U(1.5, 2.5) + diff % 2)
         - transforms.Resize((224, 224)) 縮成 224x224（不一定是正方形，可能有些微變形，同官方）
       與官方唯一的不同：官方推論時還套用 RandomGrayscale / ColorJitter（訓練用的隨機擾動），這裡不套用；
       另外框超出圖片時夾回圖片範圍內（官方沒處理，負索引會切出錯的區域）。
       隨機縮減用 --seed 固定（官方 demo 用 manual_seed 0），每張圖重設一次，結果可重現。
    3. 用訓練好的 run（預設 v2 · p5 · both · seg · paper · pretrained）依它的 args.json 重建模型，
       解碼方式同訓練時的評估（local：heatmap 尖峰 ±3 格加權平均），把 5 點換回原圖座標。
       每個點附上「信心」= heatmap 尖峰值（訓練目標的尖峰 = 1；test 集上 5 點都 >= 0.5 的影像幾乎沒有大錯）。

放在 vrb 根目錄（和 train_vrb_needle5.py、cam_from_runs.py、inference.py 同一層），在 container 裡執行：

    python predict_needle5_langsam.py
    # 等同：
    python predict_needle5_langsam.py \
        --run runs/ablation_v2/p5_both_seg1_paper_pre_20261024_180740 \
        --image /workspaces/vrb/my_needle_thread/Images/cam_260422273204_002901_1792135221.635050781 \
        --text needle

    # 只跑前 20 張 / 每 5 張取 1 張
    python predict_needle5_langsam.py --limit 20
    python predict_needle5_langsam.py --every 5

    # （可選）有 GT 的圖順便算誤差：給 needle5 的 annotations.json
    python predict_needle5_langsam.py --gt_ann needle5_thread_aug/annotations.json

輸出：pred_needle5_output/<時間>/
    <影像名>_pred.png   左：原圖（以裁切框為中心放大）+ LangSAM 框（紫）+ 實際裁切框（灰）+ 5 點與信心
                        中：送進模型的 224x224 + 5 點；右：5 張 heatmap + seg 頭的針 mask
    <影像名>_input.png  實際送進模型的 224x224
    <影像名>.json       LangSAM 框與分數、裁切框、5 點原圖座標、信心
    summary.csv         每張一列：找到框沒、LangSAM 分數、5 點座標、信心（有 GT 時加誤差）
"""

import argparse
import csv
import datetime
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_vrb_needle5 as T  # noqa: E402
from cam_from_runs import build_net, load_args, load_state  # noqa: E402

DEFAULT_RUN = "runs/ablation_v2/p5_both_seg1_paper_pre_20261024_180740"
DEFAULT_IMAGES = "/workspaces/vrb/my_needle_thread/Images/cam_260422273204_002901_1792135221.635050781"
PT_BGR = [(0, 0, 255), (255, 80, 0), (0, 200, 0), (0, 140, 255), (255, 255, 0)]   # head 紅、tail 藍、middle 綠、head_1_3 橘、tail_1_3 青
BBOX_OFFSET = 20                                                                    # inference.py 的 bbox_offset

# inference.py 的 transform 去掉 RandomGrayscale / ColorJitter（只保留決定性的部分）
TF = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# -----------------------------------------------------------------------------
# 1. LangSAM 找 bbox（同 inference.py）
# -----------------------------------------------------------------------------
class BoxFinder:
    def __init__(self, box_threshold, text_threshold):
        try:
            from lang_sam import LangSAM
        except ImportError:
            sys.exit("[ERROR] 沒有 lang_sam（官方 inference.py 用的套件）：pip install lang-sam")
        self.model = LangSAM()                                     # 同 inference.py：只載入一次
        self.bt, self.tt = box_threshold, text_threshold
        import inspect
        first = list(inspect.signature(self.model.predict).parameters)[0]
        self.new_api = first.startswith("images")                     # 新版第一個參數叫 images_pil

    def __call__(self, image_pil, text):
        """回傳 (box [x1, y1, x2, y2], score, 全部候選數)；找不到回傳 (None, None, 0)。
        官方 inference.py 取 boxes[0]；新版 lang_sam 沒有排序保證，所以取分數最高的（舊版 boxes[0] 就是最高分）。"""
        with torch.no_grad():
            if self.new_api:                                         # 新版 lang_sam：predict([images], [texts]) -> [dict]
                r = self.model.predict([image_pil], [text], box_threshold=self.bt, text_threshold=self.tt)[0]
                boxes, scores = r.get("boxes"), r.get("scores")
            else:                                                    # 舊版（官方 demo 用的）：predict(image, text)
                masks, boxes, phrases, scores = self.model.predict(image_pil, text, self.bt, self.tt)
        if boxes is None or len(boxes) == 0:
            return None, None, 0
        boxes = np.asarray(boxes.cpu() if torch.is_tensor(boxes) else boxes, np.float64).reshape(-1, 4)
        k = 0
        if scores is not None and len(scores) == len(boxes):
            scores = np.asarray(scores.cpu() if torch.is_tensor(scores) else scores, np.float64).reshape(-1)
            k = int(np.argmax(scores))
            return boxes[k].tolist(), float(scores[k]), len(boxes)
        return boxes[k].tolist(), None, len(boxes)


# -----------------------------------------------------------------------------
# 2. 官方 inference.py 的裁切
# -----------------------------------------------------------------------------
def official_crop(box, W, H, rng):
    """照 inference.py 的寫法（它把 box 拆成 y1, x1, y2, x2，但實際上 box 是 (x1, y1, x2, y2)，
    img[x1:x2, y1:y2] 等於 img[列=上下, 欄=左右]，所以最後切的還是 bbox 本身）。
    回傳 (left, top, right, bottom)，原圖 pixel，右 / 下不含。"""
    y1, x1, y2, x2 = box                                             # 官方變數名稱：y* = 欄（左右）、x* = 列（上下）
    o = BBOX_OFFSET
    y1, x1, y2, x2 = int(y1) - o, int(x1) - o, int(y2) + o, int(x2) + o
    width = y2 - y1
    height = x2 - x1
    diff = width - height
    if width > height:
        y1 += int(diff / rng.uniform(1.5, 2.5))
        y2 -= int((diff / (rng.uniform(1.5, 2.5) + diff % 2)))
    else:
        diff = height - width
        x1 += int(diff / rng.uniform(1.5, 2.5))
        x2 -= int((diff / (rng.uniform(1.5, 2.5) + diff % 2)))
    # 官方沒有處理超出圖片的情況（負索引會切到錯的地方）；這裡夾回圖片內
    left, right = max(y1, 0), min(y2, W)
    top, bottom = max(x1, 0), min(x2, H)
    return left, top, right, bottom


# -----------------------------------------------------------------------------
# 3. needle5 模型預測
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict5(net, args, crop_pil, device):
    img = TF(crop_pil).unsqueeze(0).to(device)
    T.HM_CACHE.pop("x", None)
    T.HM_CACHE.pop("attn", None)
    pred = T.predict_xy(net, img, args)                              # (1, K, 2)，0~1，全圖 soft-argmax
    hm = T.HM_CACHE.get("x")
    if hm is not None and args.decode == "local":                    # 這個 run：decode=local, hm_loss=mse
        beta = args.kp_beta if args.kp_beta > 0 else 1.0
        pred = T.local_decode(hm, beta, args.decode_radius, "mse" if args.hm_loss == "mse" else "softmax")
    K = pred.shape[1]
    h = hm.float()[0].reshape(K, -1)
    if args.hm_loss == "mse" and args.hm_weight > 0:
        conf = h.clamp_min(0).max(-1)[0]                             # heatmap 尖峰值（目標尖峰 = 1）
    else:
        conf = torch.softmax((args.kp_beta or 1.0) * h, -1).max(-1)[0]
    sl = T.seg_logits(net)
    seg = torch.sigmoid(sl.float())[0, 0].cpu().numpy() if sl is not None else None
    return pred[0].cpu().numpy(), conf.cpu().numpy(), hm[0].float().cpu().numpy(), seg


def to_orig(pn, crop):
    """224 圖內 normalized (x, y) -> 原圖 pixel。裁切可能不是正方形，x、y 分開換算；
    -0.5 對應訓練資料的像素中心約定（build_needle5_aug.py：pn = (p + 0.5) / S）。"""
    left, top, right, bottom = crop
    return np.stack([left + pn[:, 0] * (right - left) - 0.5, top + pn[:, 1] * (bottom - top) - 0.5], -1)


# -----------------------------------------------------------------------------
# 畫圖
# -----------------------------------------------------------------------------
def draw(img_rgb, box, crop, pts, conf, names, crop224, pn, hm, seg, conf_th, gt=None):
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    if box is not None:
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 0, 200), 2)
    l, t, r, b = crop
    cv2.rectangle(vis, (l, t), (r, b), (160, 160, 160), 2)
    for k, (x, y) in enumerate(pts):
        c = PT_BGR[k % len(PT_BGR)]
        if gt is not None:
            cv2.circle(vis, (int(round(gt[k, 0])), int(round(gt[k, 1]))), 3, c, -1)
        cv2.circle(vis, (int(round(x)), int(round(y))), 6, c, 2 if conf[k] >= conf_th else 1)
        cv2.putText(vis, f"{conf[k]:.2f}", (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    s = max(r - l, b - t)
    pad = int(0.6 * s)
    X0, Y0 = max(l - pad, 0), max(t - pad, 0)
    X1, Y1 = min(r + pad, vis.shape[1]), min(b + pad, vis.shape[0])
    left = cv2.resize(vis[Y0:Y1, X0:X1], (448, max(1, int(448 * (Y1 - Y0) / max(X1 - X0, 1)))))
    mid = cv2.cvtColor(crop224, cv2.COLOR_RGB2BGR).copy()
    for k, (u, v) in enumerate(pn):
        cv2.circle(mid, (int(u * 224), int(v * 224)), 4, PT_BGR[k % len(PT_BGR)], 2)
    mid = cv2.resize(mid, (448, 448))
    tiles = []
    for k in range(hm.shape[0]):
        m = np.clip(hm[k], 0, None)
        m = (255 * m / max(m.max(), 1e-6)).astype(np.uint8)
        tl = cv2.applyColorMap(cv2.resize(m, (224, 224), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_JET)
        tl = cv2.addWeighted(cv2.cvtColor(crop224, cv2.COLOR_RGB2BGR), 0.5, tl, 0.5, 0)
        cv2.putText(tl, f"{names[k]} {conf[k]:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tl)
    if seg is not None:
        tl = cv2.cvtColor(crop224, cv2.COLOR_RGB2BGR).copy()
        sm = cv2.resize(seg, (224, 224)) > 0.5
        tl[sm] = (0.4 * tl[sm] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.putText(tl, "seg head", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tl)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    right = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    Hh = max(left.shape[0], mid.shape[0], right.shape[0])
    padh = lambda a: np.vstack([a, np.zeros((Hh - a.shape[0], a.shape[1], 3), np.uint8)])
    return np.hstack([padh(left), padh(mid), padh(right)])


def load_gt(ann_path, n_points):
    """needle5 annotations.json -> {原圖檔名: (5,2) 原圖 pixel}（只取 n0 的無增強樣本）"""
    gt = {}
    for s in json.load(open(ann_path))["samples"]:
        if s.get("augmented") or s.get("needle_id", 0) != 0:
            continue
        x0, y0, side = s["aug"]["crop"]
        S = s["side_px"]
        gx, gy = x0 + side / 2 - S / 2, y0 + side / 2 - S / 2
        pn = np.asarray(s["points"], np.float64)[:n_points]
        gt[os.path.basename(s["source_image"])] = np.stack([gx + pn[:, 0] * S - 0.5, gy + pn[:, 1] * S - 0.5], -1)
    return gt


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN, help="訓練 run 的資料夾（讀 args.json 與權重）")
    ap.add_argument("--weights", default="model_best.pth.tar")
    ap.add_argument("--image", default=DEFAULT_IMAGES, help="一張圖或一個資料夾")
    ap.add_argument("--text", default="needle", help="給 LangSAM 的文字")
    ap.add_argument("--box_threshold", type=float, default=0.3, help="LangSAM（GroundingDINO）框的門檻，lang_sam 預設 0.3")
    ap.add_argument("--text_threshold", type=float, default=0.25, help="lang_sam 預設 0.25")
    ap.add_argument("--seed", type=int, default=0, help="官方裁切的隨機縮減用的 seed（每張圖重設）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 張（0 = 全部）")
    ap.add_argument("--every", type=int, default=1, help="每 N 張取 1 張")
    ap.add_argument("--conf_th", type=float, default=0.5, help="信心門檻（只影響畫圖粗細與 summary 的 all_confident）")
    ap.add_argument("--gt_ann", default=None, help="（可選）needle5 annotations.json，有 GT 的圖會算誤差")
    ap.add_argument("--out", default="pred_needle5_output")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[ERROR] 需要 GPU（官方 VRBModel 在 __init__ 對 BatchNorm 呼叫 .cuda()）")
    device = torch.device("cuda")

    if os.path.isdir(a.image):
        paths = sorted(p for p in glob.glob(os.path.join(a.image, "*"))
                       if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    else:
        paths = [a.image]
    paths = paths[::max(a.every, 1)]
    if a.limit > 0:
        paths = paths[:a.limit]
    if not paths:
        sys.exit(f"[ERROR] {a.image} 裡沒有圖片")

    args = load_args(a.run)
    net = build_net(args, load_state(os.path.join(a.run, a.weights)), device)
    names = T.POINT_NAMES
    print(f"模型：{a.run}（{args.points} 點，decode={args.decode}，hm_loss={args.hm_loss}，"
          f"seg 頭={'有' if hasattr(net, 'seg_head') else '無'}）")
    finder = BoxFinder(a.box_threshold, a.text_threshold)
    gt_all = load_gt(a.gt_ann, args.points) if a.gt_ann else {}

    out_dir = os.path.join(a.out, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for j, path in enumerate(paths, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        image_pil = Image.open(path).convert("RGB")                  # 不縮放
        img = np.asarray(image_pil)
        H, W = img.shape[:2]
        box, score, n_box = finder(image_pil, a.text)
        row = {"image": os.path.basename(path), "found": box is not None, "langsam_score": score, "n_boxes": n_box}
        if box is None:
            print(f"[{j}/{len(paths)}] {name}：LangSAM 找不到 '{a.text}'")
            rows.append(row)
            continue
        rng = np.random.RandomState(a.seed)                          # 官方用 np.random；每張圖重設，結果可重現
        crop = official_crop(box, W, H, rng)
        l, t, r, b = crop
        if r - l < 2 or b - t < 2:
            print(f"[{j}/{len(paths)}] {name}：裁切框太小 {crop}")
            rows.append(row)
            continue
        crop_pil = image_pil.crop((l, t, r, b))
        pn, conf, hm, seg = predict5(net, args, crop_pil, device)
        pts = to_orig(pn, crop)
        crop224 = np.asarray(crop_pil.resize((224, 224), Image.BILINEAR))  # 只給畫圖用
        gt = gt_all.get(os.path.basename(path))
        info = {"image": path, "text": a.text, "langsam_box": [round(v, 1) for v in box], "langsam_score": score,
                "n_boxes": n_box, "crop_ltrb": list(crop), "crop_wh": [r - l, b - t],
                "points": {n: [round(float(x), 1), round(float(y), 1)] for n, (x, y) in zip(names, pts)},
                "confidence": {n: round(float(c), 3) for n, c in zip(names, conf)},
                "all_confident": bool((conf >= a.conf_th).all())}
        msg = (f"[{j}/{len(paths)}] {name}: 框分數 {score if score is None else round(score, 2)}，"
               f"裁切 {r - l}x{b - t}，信心 " + " ".join(f"{c:.2f}" for c in conf))
        if gt is not None:
            e = np.linalg.norm(pts - gt, axis=-1)
            info["err_px"] = e.round(2).tolist()
            msg += f" | 誤差 {e.mean():.1f}px"
        print(msg)
        json.dump(info, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1, ensure_ascii=False)
        cv2.imwrite(os.path.join(out_dir, f"{name}_input.png"), cv2.cvtColor(crop224, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, f"{name}_pred.png"),
                    draw(img, box, crop, pts, conf, names, crop224, pn, hm, seg, a.conf_th, gt))
        row.update({"crop_w": r - l, "crop_h": b - t, "all_confident": info["all_confident"],
                    **{f"{n}_x": round(float(x), 1) for n, (x, _) in zip(names, pts)},
                    **{f"{n}_y": round(float(y), 1) for n, (_, y) in zip(names, pts)},
                    **{f"{n}_conf": round(float(c), 3) for n, c in zip(names, conf)}})
        if gt is not None:
            row["err_mean_px"] = round(float(e.mean()), 2)
        rows.append(row)

    keys = []
    for r_ in rows:
        keys += [k for k in r_ if k not in keys]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    n_found = sum(r_["found"] for r_ in rows)
    n_conf = sum(bool(r_.get("all_confident")) for r_ in rows)
    print(f"\nLangSAM 找到針：{n_found}/{len(rows)} 張；5 點信心都 >= {a.conf_th}：{n_conf} 張")
    errs = [r_["err_mean_px"] for r_ in rows if "err_mean_px" in r_]
    if errs:
        print(f"有 GT 的 {len(errs)} 張：平均誤差 {np.mean(errs):.1f}px，中位數 {np.median(errs):.1f}px")
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()