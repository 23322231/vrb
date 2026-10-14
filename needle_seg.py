#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
針的 segmentation（Ultralytics YOLO26-seg）：訓練 / 評估 / 預測，一支搞定。

    train    用 masks_to_yolo_seg.py 產生的 data.yaml 訓練，結果存到 runs/needle_seg/<name>_<開始時間>/
    resume   中斷後接續（指定 run 資料夾或 'latest'）
    val      用 best.pt 評估 test（或 val）split
    predict  對一個資料夾的影像預測，輸出「和學長資料一樣格式」的彩色 mask：
                 <out>/Images/...   影像（複製，主機上也讀得到）
                 <out>/Labels/...   mask（針 = BGR (0,0,255)，其餘黑色）
                 <out>/Color_code.txt
                 <out>/overlay/...  疊圖檢查
             -> 可以直接丟給 mask_to_needle5.py 標頭尾

針很細（原圖約 3~6 px 寬），預設：
    imgsz 1280      不把影像縮太小，否則針只剩 1~2 px
    mask_ratio 2    mask loss 用 1/2 解析度（官方預設 1/4，對細針太粗）
    predict 時 retina_masks=True，mask 直接輸出原圖解析度

需要：pip install -U ultralytics   （在 VRB container 的 venv 裡）

用法:
    python needle_seg.py train   --data needle_seg_yolo/data.yaml --model yolo26s-seg.pt --epochs 150
    python needle_seg.py resume  --run latest
    python needle_seg.py val     --run runs/needle_seg/yolo26s-seg_20261013_230000 --split test
    python needle_seg.py predict --run runs/needle_seg/yolo26s-seg_20261013_230000 \
                                 --source my_frames/ --out pred_masks
"""

import argparse
import datetime
import glob
import os
import shutil
import sys

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
NEEDLE_BGR = (0, 0, 255)
COLOR_CODE = """Background: 0, [0,0,0]

Tool shaft: 1, [255,0,0]

Needle: 2, [0,0,255]

Suture pad: 3, [0,255,0]

Tool end-effector: 4, [55,250,250]

Insertion points: 5, [209,125,42]
"""


def yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] 沒有 ultralytics，請在 container 內執行：pip install -U ultralytics")
    return YOLO


def find_run(run, project):
    if run == "latest":
        cands = [d for d in glob.glob(os.path.join(project, "*")) if os.path.isfile(os.path.join(d, "weights", "last.pt"))]
        if not cands:
            sys.exit(f"[ERROR] {project} 底下找不到任何 run")
        return max(cands, key=lambda d: os.path.getmtime(os.path.join(d, "weights", "last.pt")))
    if not os.path.isdir(run):
        sys.exit(f"[ERROR] 找不到 run 資料夾：{run}")
    return run


# -----------------------------------------------------------------------------
def cmd_train(a):
    YOLO = yolo()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{os.path.splitext(os.path.basename(a.model))[0]}_{stamp}"
    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers,
        project=os.path.abspath(a.project), name=name, exist_ok=False, seed=a.seed,
        patience=a.patience, cos_lr=True, plots=True,
        mask_ratio=a.mask_ratio, overlap_mask=True,
        # 增強：針的方向任意，所以開旋轉與上下翻轉
        degrees=a.degrees, flipud=0.5, fliplr=0.5, scale=0.3, translate=0.1,
        mosaic=1.0, close_mosaic=10, hsv_h=0.01, hsv_s=0.5, hsv_v=0.4,
    )
    run_dir = os.path.join(a.project, name)
    print(f"\n訓練完成：{run_dir}")
    print("  weights/best.pt、results.png（loss 與 mAP 曲線）、val_batch*_pred.jpg（預測範例）")
    if a.eval_test:
        cmd_val(argparse.Namespace(run=run_dir, project=a.project, data=a.data, split="test",
                                   imgsz=a.imgsz, batch=a.batch))


def cmd_resume(a):
    YOLO = yolo()
    run_dir = find_run(a.run, a.project)
    print(f"接續：{run_dir}")
    YOLO(os.path.join(run_dir, "weights", "last.pt")).train(resume=True)


def cmd_val(a):
    YOLO = yolo()
    run_dir = find_run(a.run, a.project)
    model = YOLO(os.path.join(run_dir, "weights", "best.pt"))
    data = a.data or (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("data")
    if not data:
        sys.exit("[ERROR] 讀不到訓練時的 data.yaml，請用 --data 指定")
    m = model.val(data=data, split=a.split, imgsz=a.imgsz, batch=a.batch,
                  project=run_dir, name=f"eval_{a.split}", exist_ok=True, plots=True)
    print(f"\n[{a.split}] mask mAP50 = {m.seg.map50:.3f}   mask mAP50-95 = {m.seg.map:.3f}   "
          f"box mAP50 = {m.box.map50:.3f}")


def cmd_predict(a):
    YOLO = yolo()
    if a.weights:
        weights = a.weights
    else:
        weights = os.path.join(find_run(a.run, a.project), "weights", "best.pt")
    model = YOLO(weights)
    names = model.names
    needle_ids = [k for k, v in names.items() if "needle" in str(v).lower()] or [0]

    files = []
    for root, _, fs in os.walk(a.source):
        for f in sorted(fs):
            if f.lower().endswith(IMG_EXTS):
                files.append(os.path.relpath(os.path.join(root, f), a.source))
    files.sort()
    if a.every > 1:
        files = files[::a.every]
    print(f"預測 {len(files)} 張（權重 {weights}，needle class {needle_ids}）")

    for sub in ("Images", "Labels", "overlay"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    with open(os.path.join(a.out, "Color_code.txt"), "w") as f:
        f.write(COLOR_CODE)

    n_with = 0
    for i, rel in enumerate(files):
        src = os.path.abspath(os.path.join(a.source, rel))
        r = model.predict(src, imgsz=a.imgsz, conf=a.conf, retina_masks=True, verbose=False)[0]
        img = r.orig_img
        H, W = img.shape[:2]
        out_mask = np.zeros((H, W, 3), np.uint8)
        if r.masks is not None and len(r.masks):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            md = r.masks.data.cpu().numpy() > 0.5
            if md.shape[1:] != (H, W):          # 保險：不同版本可能不是原圖大小
                md = np.stack([cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                               for m in md])
            for m, c in zip(md, cls):
                if c in needle_ids:
                    out_mask[m] = NEEDLE_BGR
        has = bool(out_mask.any())
        n_with += int(has)

        stem = os.path.splitext(rel)[0]
        img_dst = os.path.join(a.out, "Images", rel)
        lab_dst = os.path.join(a.out, "Labels", stem + ".png")
        ov_dst = os.path.join(a.out, "overlay", stem + ".png")
        for p in (img_dst, lab_dst, ov_dst):
            os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(img_dst):
            shutil.copy2(src, img_dst)
        cv2.imwrite(lab_dst, out_mask)
        ov = img.copy()
        ov[out_mask[:, :, 2] == 255] = (0.4 * ov[out_mask[:, :, 2] == 255] + 0.6 * np.array(NEEDLE_BGR)).astype(np.uint8)
        cv2.imwrite(ov_dst, ov)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)}")

    print(f"\n完成：{n_with}/{len(files)} 張有偵測到針")
    print(f"  mask    → {os.path.join(a.out, 'Labels')}")
    print(f"  疊圖    → {os.path.join(a.out, 'overlay')}")
    print("  下一步：python mask_to_needle5.py --images_dir {0}/Images --labels_dir {0}/Labels "
          "--color_txt {0}/Color_code.txt --out_dir needle5_from_pred --mode interactive".format(a.out))


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    proj = "runs/needle_seg"

    t = sp.add_parser("train")
    t.add_argument("--data", required=True, help="masks_to_yolo_seg.py 產生的 data.yaml")
    t.add_argument("--model", default="yolo26s-seg.pt", help="yolo26n/s/m/l/x-seg.pt（第一次會自動下載）")
    t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--batch", type=int, default=4, help="RTX 5070 12GB + imgsz 1280：s 約 4~8、m 約 2~4")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--mask_ratio", type=int, default=2)
    t.add_argument("--degrees", type=float, default=180)
    t.add_argument("--patience", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--project", default=proj)
    t.add_argument("--eval_test", type=int, default=1, help="訓練完自動用 best.pt 評估 test")

    r = sp.add_parser("resume")
    r.add_argument("--run", default="latest")
    r.add_argument("--project", default=proj)

    v = sp.add_parser("val")
    v.add_argument("--run", default="latest")
    v.add_argument("--project", default=proj)
    v.add_argument("--data", default=None, help="預設用訓練時的 data.yaml")
    v.add_argument("--split", default="test", choices=["train", "val", "test"])
    v.add_argument("--imgsz", type=int, default=1280)
    v.add_argument("--batch", type=int, default=4)

    p = sp.add_parser("predict")
    p.add_argument("--run", default="latest")
    p.add_argument("--project", default=proj)
    p.add_argument("--weights", default=None, help="直接指定 .pt（給了就不看 --run）")
    p.add_argument("--source", required=True, help="影像資料夾（可含子資料夾）")
    p.add_argument("--out", default="pred_masks")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--every", type=int, default=1, help="每隔幾張取一張（影片影格抽樣用）")

    a = ap.parse_args()
    {"train": cmd_train, "resume": cmd_resume, "val": cmd_val, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
