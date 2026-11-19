#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 B：感知核心（不依賴 ROS）。YOLO26-seg 找針 → 切 224×224 → VRB needle5 預測 5 點。

ROS node（handover_perception_node.py）與離線評估（eval_handover_crops.py）都用這裡的 NeedlePerception，
所以「線上」與「離線」走的是同一段程式。

    from handover_perception import NeedlePerception
    perc = NeedlePerception()                         # 模型只載入一次（YOLO + VRB），會先暖機
    res, extra = perc.predict(img_bgr)                # img_bgr: HxWx3 uint8（OpenCV 的 BGR）
    res   -> 介面 §3.1 的 dict（json.dumps 後就是 Trigger 的 message）
    extra -> 畫圖 / 存檔用的中間結果（224 圖、heatmap、seg、YOLO mask…）

裁切方式（crop_mode）：
    "train"     （預設）和訓練資料完全相同的規則：build_needle5_aug.needle_crop（以框為中心的正方形，
                四邊外擴 max(20 px, 0.15×長邊)、最小 96 px、夾在圖內），再用 cv2.INTER_AREA 縮成 224。
                VRB 的 val / test 就是這樣切的（test 平均 7.3 px 就是在這個分佈下量的）。
    "official"  predict_needle5_langsam.py 用的官方 inference.py 切法（外擴 20 px、長邊隨機內縮、
                PIL bilinear 縮成 224，可能不是正方形）。只用來比較。

框的來源（box_src）：
    "box"   YOLO 的 box
    "mask"  YOLO mask（retina_masks，原圖解析度）的外接框；訓練時的 needle_bbox 就是 GT mask 的外接框

選哪一個偵測（多根針 / 誤偵測時）：先依 roi_xyxy、框的大小過濾，再取 YOLO 信心最高的。

5 點怎麼算（decode）：
    "vrb"   VRB 直接預測 5 點（原本的做法）
    "geom"  （預設）幾何解碼，見 needle_geom.py：所有通過過濾的 YOLO mask 取聯集 → 中心線（沿圓弧補遮擋，
            和產生 GT 的 mask_to_needle5.py 同一套）→ VRB 只用來決定哪端是針尖 → 5 點依弧長比例取。
            VRB 的切圖改用整根針（含補起來的部分）的外接框。mask 抽不出中心線時自動退回 "vrb"（decode_used 會寫）。
            這個模式下每個點的 conf = 方向信心 orient_conf（0～1），VRB 原本的 5 點與信心放在 res["geom"]["vrb_points"]。
            方向：YOLO 有 Thread 類別時先用線（離線近的那端 = 針尾），判斷不了再用 VRB（orient_mode）。
    "vrb_anchor"（handover 預設，計畫 v1.7 §3.1.1a）夾取點由 VRB 決定：中心線同 "geom"；VRB 的 head、tail、middle
            投影到中心線上（針頭方向由 VRB 決定），head_1_3 / tail_1_3 / head_1_5 依投影後 head → tail 弧長的
            1/3、2/3、1/5 取。points 共 6 點（原本 5 點 + head_1_5）；conf：head、tail、middle 是各自的 heatmap 尖峰，
            算出來的點是 min(head, tail) 尖峰，conf_src 註明。同時算「純幾何 + 針頭朝左」放在 res["geom_only"]，
            並記下兩者夾取點的像素差；VRB 方向和「針頭朝左」不一致時 res["vrb_agree"] = false。

離線用法（對資料夾跑，存 §3.1 JSON + 疊圖，順便量時間與 GPU 記憶體）：
    python handover_perception.py --images handover_samples/ --out pred_handoff_output/offline
    python handover_perception.py --images img.png --crop_mode official --box_src box
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import threading
import time
import types

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# 2026-11-17 起的 handover 模型（舊的：runs/segment/train/weights/best.pt、runs/ablation_v2/p5_both_seg1_paper_pre_20261024_180740）
DEFAULT_YOLO = os.path.join(_HERE, "runs/needle_seg/best_20261116_225446/weights/best.pt")
DEFAULT_RUN = os.path.join(_HERE, "runs/handover_v3/needle5_20261116_205018")
DEFAULT_VRB_WEIGHTS = "model_best.pth.tar"
# build_needle5_aug.py 建 needle5_thread_aug 時用的裁切參數（annotations.json 的 build_args）
TRAIN_CROP = types.SimpleNamespace(box_pad=20.0, box_pad_ratio=0.15, min_side=96.0)
DEPTH_POINTS = ("head", "tail_1_3", "head_1_5")       # 介面 §3.1：這兩點要有 xyz_cam
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# =============================================================================
# 裁切（純 numpy / cv2，可單獨測）
# =============================================================================
def train_crop(img_rgb, box, crop_args=TRAIN_CROP):
    """訓練（val / test）用的決定性裁切。回傳 (crop224 RGB uint8, (l, t, r, b))，r - l = b - t = side。"""
    from build_needle5_aug import needle_crop
    H, W = img_rgb.shape[:2]
    x0, y0, side = needle_crop([float(v) for v in box], W, H, crop_args, None)
    c = img_rgb[y0:y0 + side, x0:x0 + side]
    # 和 build_needle5_aug.warp_crop（deg=0、grow=1）完全相同：整數平移的 warpAffine = 直接切
    c224 = cv2.resize(np.ascontiguousarray(c), (224, 224), interpolation=cv2.INTER_AREA)
    return c224, (int(x0), int(y0), int(x0 + side), int(y0 + side))


def official_crop224(img_rgb, box, seed=0):
    """predict_needle5_langsam.py 的官方切法（只用來比較）。回傳 (crop224 RGB uint8, (l, t, r, b))。"""
    from PIL import Image
    from predict_needle5_langsam import official_crop
    H, W = img_rgb.shape[:2]
    l, t, r, b = official_crop(box, W, H, np.random.RandomState(seed))
    if r - l < 2 or b - t < 2:
        return None, (l, t, r, b)
    pil = Image.fromarray(img_rgb).crop((l, t, r, b)).resize((224, 224), Image.BILINEAR)
    return np.asarray(pil), (int(l), int(t), int(r), int(b))


def to_orig(pn, crop):
    """224 圖內 normalized (x, y) -> 原圖 pixel（訓練的像素中心約定：pn = (p + 0.5) / S）。"""
    l, t, r, b = crop
    pn = np.asarray(pn, np.float64)
    return np.stack([l + pn[:, 0] * (r - l) - 0.5, t + pn[:, 1] * (b - t) - 0.5], -1)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


# =============================================================================
# GPU 記憶體
# =============================================================================
def gpu_mem_report():
    """torch 看到的（allocated / reserved / peak）與 nvidia-smi 看到的這個 process 用量（MB）。"""
    out = {}
    try:
        import torch
        if torch.cuda.is_available():
            out["torch_allocated_mb"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
            out["torch_reserved_mb"] = round(torch.cuda.memory_reserved() / 2 ** 20, 1)
            out["torch_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import subprocess
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout
        for line in q.strip().splitlines():
            pid, mem = [s.strip() for s in line.split(",")[:2]]
            if int(pid) == os.getpid():
                out["nvidia_smi_process_mb"] = float(mem)
        if "nvidia_smi_process_mb" not in out:
            out["nvidia_smi_note"] = "nvidia-smi 列不出這個 process（container 的 PID namespace），看 torch_reserved_mb + 約 300~500 MB CUDA context"
    except Exception as e:  # noqa: BLE001
        out["nvidia_smi_note"] = f"nvidia-smi 失敗：{e}"
    return out


# =============================================================================
# 感知
# =============================================================================
class Sam3Detector:
    """SAM 3（Meta，2025-11）的文字提示分割，包成和 YOLO 一樣的介面：呼叫後回傳 ultralytics 的 Results
    （boxes.xyxy / conf / cls、masks.data），所以後面的流程（選偵測、mask 聯集、幾何解碼）完全不用改。
    透過 ultralytics（>= 8.3.237）的 SAM3SemanticPredictor，不需要另外裝 Meta 的 sam3 套件（那個要 Python 3.12）。
    權重 sam3.pt 要先在 Hugging Face（facebook/sam3）申請下載。
    prompts：文字提示，可以給好幾個（例如 "needle" "surgical needle"），每個提示的結果都當成針。"""

    def __init__(self, weights="sam3.pt", prompts=("needle",), conf=0.25, half=True):
        from ultralytics.models.sam import SAM3SemanticPredictor
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"找不到 SAM 3 權重 {weights}（要先到 Hugging Face facebook/sam3 申請並下載 sam3.pt）")
        base = dict(conf=float(conf), task="segment", mode="predict", model=weights, save=False, verbose=False)
        last = None
        for extra in (({"quantize": 16} if half else {}), ({"half": True} if half else {}), {}):   # 不同版本的 FP16 參數名稱
            try:
                self.pred = SAM3SemanticPredictor(overrides={**base, **extra})
                self.fp16 = extra
                break
            except Exception as e:  # noqa: BLE001
                last = e
        else:
            raise last
        self.prompts, self.conf = list(prompts), float(conf)
        self.names = {i: str(p) for i, p in enumerate(self.prompts)}

    def __call__(self, img_bgr, conf=None):
        self.pred.args.conf = self.conf if conf is None else float(conf)
        try:
            self.pred.set_image(img_bgr)                      # numpy（BGR，和 cv2 一樣）
        except Exception:  # noqa: BLE001               # 舊版只吃路徑
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, img_bgr)
                self.pred.set_image(f.name)
        return self.pred(text=self.prompts)[0]


class NeedlePerception:
    def __init__(self, yolo_weights=DEFAULT_YOLO, vrb_run=DEFAULT_RUN, vrb_weights=DEFAULT_VRB_WEIGHTS,
                 yolo_imgsz=640, yolo_conf=0.25, yolo_iou=0.7, box_src="box", crop_mode="train",
                 roi_xyxy=None, min_box_px=8.0, max_box_px=800.0, official_seed=0, device="cuda", warmup=True,
                 decode="geom", orient_mode="heat", detector="yolo", sam3_weights="sam3.pt", sam3_prompts=("needle",)):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("需要 GPU（官方 VRBModel 在 __init__ 對 BatchNorm 呼叫 .cuda()）")
        from ultralytics import YOLO
        import train_vrb_needle5 as T
        from cam_from_runs import build_net, load_args, load_state

        self.torch, self.T = torch, T
        self.device = torch.device(device)
        self.yolo_weights, self.vrb_run = os.path.abspath(yolo_weights), os.path.abspath(vrb_run)
        assert detector in ("yolo", "sam3")
        self.detector = detector
        if detector == "sam3":
            self.sam3 = Sam3Detector(sam3_weights, sam3_prompts, yolo_conf)
            self.yolo = types.SimpleNamespace(names=self.sam3.names)          # 只拿來當類別名稱表
        else:
            self.yolo = YOLO(self.yolo_weights)
        self.yolo_imgsz, self.yolo_conf, self.yolo_iou = int(yolo_imgsz), float(yolo_conf), float(yolo_iou)
        assert box_src in ("box", "mask") and crop_mode in ("train", "official")
        self.box_src, self.crop_mode, self.official_seed = box_src, crop_mode, int(official_seed)
        self.roi = [float(v) for v in roi_xyxy] if roi_xyxy else None
        self.min_box_px, self.max_box_px = float(min_box_px), float(max_box_px)
        assert decode in ("vrb", "geom", "vrb_anchor") and orient_mode in ("heat", "ends", "all", "left")
        self.decode, self.orient_mode = decode, orient_mode

        self.args = load_args(self.vrb_run)
        self.net = build_net(self.args, load_state(os.path.join(self.vrb_run, vrb_weights)), self.device)
        # VRB 實際輸出的點：3 點 run = head/tail/middle，5 點 run 再加 head_1_3/tail_1_3。
        # 幾何解碼不管 VRB 幾點，都沿中心線給完整 5 點（tail_1_3 = 夾取點），VRB 只用 head/tail heatmap 定方向。
        self.vrb_names = list(T.POINT_NAMES)
        self.geom_names = list(T.ALL_POINT_NAMES)
        import needle_geom as G
        self.anchor_names = list(G.ANCHOR_NAMES)                 # vrb_anchor：5 點 + head_1_5
        self.names = {"geom": self.geom_names, "vrb_anchor": self.anchor_names}.get(decode, self.vrb_names)
        # YOLO 的類別：名稱含 needle 的是針、含 thread 的是線（舊的 best.pt 只有 needle；新訓練若加 Thread 類別就會用線定方向）
        ynames = {int(k): str(v).lower() for k, v in dict(self.yolo.names).items()}
        self.needle_cls = {k for k, v in ynames.items() if "needle" in v} or set(ynames)
        if detector == "sam3":
            self.needle_cls = set(ynames)                    # SAM 3：每個文字提示都當成針
        self.thread_cls = {k for k, v in ynames.items() if "thread" in v}
        self.class_names = ynames
        self.lock = threading.Lock()                 # predict 一次只跑一個（ROS 多執行緒 executor 也安全）
        if warmup:
            dummy = np.zeros((720, 1280, 3), np.uint8)
            self.detect(dummy)
            self.keypoints(np.zeros((224, 224, 3), np.uint8))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------------------- YOLO
    def detect(self, img_bgr, conf=None):
        """回傳 [{conf, box_xyxy, mask_box_xyxy, mask(HxW bool 或 None)}]，依信心由高到低。ultralytics 吃 BGR。"""
        if self.detector == "sam3":
            r = self.sam3(img_bgr, conf)
        else:
            r = self.yolo.predict(img_bgr, imgsz=self.yolo_imgsz, conf=self.yolo_conf if conf is None else conf,
                                  iou=self.yolo_iou, retina_masks=True, verbose=False)[0]
        dets = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        confs = r.boxes.conf.cpu().numpy().astype(np.float64)
        clss = r.boxes.cls.cpu().numpy().astype(int) if getattr(r.boxes, "cls", None) is not None else np.zeros(len(confs), int)
        masks = r.masks.data.cpu().numpy() > 0.5 if getattr(r, "masks", None) is not None else None
        for i in np.argsort(-confs):
            m = masks[i] if masks is not None else None
            if m is not None and m.shape != img_bgr.shape[:2]:          # 保險：retina_masks 應該已是原圖大小
                m = cv2.resize(m.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            dets.append({"conf": float(confs[i]), "box_xyxy": boxes[i].tolist(), "cls": int(clss[i]),
                         "mask_box_xyxy": mask_bbox(m) if m is not None else None, "mask": m})
        return dets

    def select(self, dets):
        """ROI 與框大小過濾後取信心最高的；回傳 (det 或 None, 被濾掉的原因列表)。"""
        rejected = []
        for d in dets:
            if d.get("cls", 0) not in self.needle_cls:
                continue
            x1, y1, x2, y2 = d["box_xyxy"]
            cx, cy, long_side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
            if self.roi and not (self.roi[0] <= cx <= self.roi[2] and self.roi[1] <= cy <= self.roi[3]):
                rejected.append({"conf": round(d["conf"], 3), "reason": "outside_roi"})
                continue
            if not (self.min_box_px <= long_side <= self.max_box_px):
                rejected.append({"conf": round(d["conf"], 3), "reason": f"box_size_{long_side:.0f}px"})
                continue
            return d, rejected
        return None, rejected

    def det_status(self, dets, det):
        """每個偵測的處理結果（給 yolo_overlay.png）：selected = 被選中的（信心最高且通過過濾）；
        union = 也通過過濾、幾何／vrb_anchor 解碼時 mask 會併進中心線；passed = 通過過濾但沒用到（decode=vrb）；
        outside_roi / box_size_XXpx = 被濾掉；class_<名稱> = 不是針的類別（例如線）。"""
        out = []
        for d in dets:
            if d.get("cls", 0) not in self.needle_cls:
                out.append(f"class_{self.class_names.get(d.get('cls', 0), d.get('cls', 0))}")
                continue
            x1, y1, x2, y2 = d["box_xyxy"]
            cx, cy, long_side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
            if self.roi and not (self.roi[0] <= cx <= self.roi[2] and self.roi[1] <= cy <= self.roi[3]):
                out.append("outside_roi")
            elif not (self.min_box_px <= long_side <= self.max_box_px):
                out.append(f"box_size_{long_side:.0f}px")
            elif d is det:
                out.append("selected")
            else:
                out.append("union" if self.decode in ("geom", "vrb_anchor") else "passed")
        return out

    def valid(self, dets):
        """通過 ROI 與框大小過濾的偵測（幾何解碼把它們的 mask 取聯集：YOLO 常把一根針切成好幾段框）。"""
        out = []
        for d in dets:
            if d.get("cls", 0) not in self.needle_cls:
                continue
            x1, y1, x2, y2 = d["box_xyxy"]
            cx, cy, long_side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
            if self.roi and not (self.roi[0] <= cx <= self.roi[2] and self.roi[1] <= cy <= self.roi[3]):
                continue
            if self.min_box_px <= long_side <= self.max_box_px:
                out.append(d)
        return out

    def thread_mask(self, dets):
        """YOLO 線類別的 mask 聯集（模型沒有線類別時 None）。"""
        if not self.thread_cls:
            return None
        m = None
        for d in dets:
            if d.get("cls") in self.thread_cls and d.get("mask") is not None:
                m = d["mask"].copy() if m is None else (m | d["mask"])
        return m

    def geom_decode(self, img_rgb, mask, seed_mask=None, thread_mask=None, mode=None):
        """幾何解碼。mask：針的 mask（YOLO 聯集或 GT）；seed_mask：選哪一根針（信心最高的偵測）。
        回傳 dict（pts、conf、needle、orient、box、crop、crop224、VRB 的 pn/conf/hm/seg/pts）或 None（抽不出中心線）。"""
        import needle_geom as G
        nd = G.centerline(mask, seed_mask)
        if nd is None or nd["length_px"] < 10:
            return None
        box = nd["bbox"]                                          # 整根針（含補起來的部分）的外接框
        c224, crop = self.crop(img_rgb, box)
        if c224 is None:
            return None
        pn, vconf, hm, seg = self.keypoints(c224)
        vpts = to_orig(pn, crop)
        mode = mode or (self.decode if self.decode in ("geom", "vrb_anchor") else "geom")
        base = {"needle": nd, "box": box, "crop": crop, "crop224": c224, "pn": pn, "vconf": vconf, "hm": hm, "seg": seg,
                "vpts": vpts, "mode": mode}
        gpts, gpoly, gdx = G.left_points(nd["poly"], self.anchor_names)          # 純幾何 + 針頭朝左（6 點）
        base["geom_only"] = {"pts": gpts, "poly": gpoly, "orient_conf": gdx}
        if mode == "vrb_anchor":
            apts, aconf, asrc, ainfo = G.anchor_decode(nd["poly"], vpts, vconf, self.vrb_names, self.anchor_names)
            apoly = nd["poly"] if ainfo["head_end"] == "A" else nd["poly"][::-1]
            ia = {n: i for i, n in enumerate(self.anchor_names)}
            ainfo["grasp_diff_px"] = {n: round(float(np.linalg.norm(apts[ia[n]] - gpts[ia[n]])), 2)
                                      for n in ("tail_1_3", "head_1_5")}
            return {**base, "pts": apts, "conf": aconf, "conf_src": asrc, "names": self.anchor_names, "poly": apoly,
                    "anchor": ainfo, "orient": {"source": "vrb_anchor", "vrb_agree": ainfo["vrb_agree"]},
                    "orient_conf": ainfo["grasp_conf"], "pts6": apts}
        poly, oconf, oinfo = G.decide_orientation(nd["poly"], nd["length_px"], self.vrb_names, self.orient_mode,
                                                  vrb_pts=vpts, vrb_conf=vconf, hm=hm, crop=crop, thread_mask=thread_mask)
        pts, L = G.five_points(poly, self.geom_names)
        pts6, _ = G.points_along(poly, [G.POINT_FRACS[n] for n in self.anchor_names])
        return {**base, "pts": pts, "conf": np.full(len(self.geom_names), oconf), "names": self.geom_names, "poly": poly,
                "orient": oinfo, "orient_conf": oconf, "pts6": pts6}

    def det_box(self, det):
        if self.box_src == "mask" and det.get("mask_box_xyxy") is not None:
            return det["mask_box_xyxy"]
        return det["box_xyxy"]

    # ---------------------------------------------------------------- 裁切 + VRB
    def crop(self, img_rgb, box, mode=None):
        mode = mode or self.crop_mode
        if mode == "train":
            return train_crop(img_rgb, box)
        return official_crop224(img_rgb, box, self.official_seed)

    def keypoints(self, crop224_rgb):
        """224 RGB -> (pn (K,2) 0~1, conf (K,), heatmap (K,56,56), seg (56,56) 或 None)。和訓練評估相同的解碼。"""
        from PIL import Image
        from predict_needle5_langsam import predict5
        return predict5(self.net, self.args, Image.fromarray(crop224_rgb), self.device)

    # ---------------------------------------------------------------- 全流程
    def predict(self, img_bgr, stamp=None, frame_id="", depth_fn=None, box_override=None):
        """img_bgr: HxWx3 uint8。depth_fn(uv_list, box_xyxy, mask) -> [xyz 或 None]（C 的模組接這裡，None = 不算）。
        box_override：直接給框（評估「GT 框 / LangSAM 框」用），不跑 YOLO。
        回傳 (res §3.1 dict, extra dict)。"""
        torch = self.torch
        with self.lock:
            t0 = time.perf_counter()
            H, W = img_bgr.shape[:2]
            res = {"ok": False, "error": "", "stamp": stamp if stamp is not None else time.time(), "frame_id": frame_id,
                   "image_size": [int(W), int(H)], "yolo": None, "crop_xyxy": None, "points": [], "debug_dir": ""}
            extra = {"dets": [], "det": None}
            timing = {}
            if box_override is None:
                dets = self.detect(img_bgr)
                torch.cuda.synchronize()
                timing["yolo"] = (time.perf_counter() - t0) * 1e3
                det, rejected = self.select(dets)
                extra.update(dets=dets, det=det, det_status=self.det_status(dets, det), roi=self.roi)
                res["n_detections"] = len(dets)
                if rejected:
                    res["rejected"] = rejected
                if det is None:
                    res["error"] = "no_detection" if not dets else "no_valid_detection"
                    res["timing_ms"] = {k: round(v, 1) for k, v in timing.items()}
                    return res, extra
                box = self.det_box(det)
                res["yolo"] = {"conf": round(det["conf"], 4), "box_xyxy": [round(v, 1) for v in det["box_xyxy"]],
                               "mask_box_xyxy": None if det["mask_box_xyxy"] is None else [round(v, 1) for v in det["mask_box_xyxy"]],
                               "box_src": self.box_src}
            else:
                box = [float(v) for v in box_override]
                timing["yolo"] = 0.0
            t1 = time.perf_counter()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            g = None
            if self.decode in ("geom", "vrb_anchor") and box_override is None and det.get("mask") is not None:
                union = np.zeros(img_bgr.shape[:2], bool)
                for d in self.valid(extra["dets"]):
                    if d.get("mask") is not None:
                        union |= d["mask"]
                g = self.geom_decode(img_rgb, union, det["mask"], self.thread_mask(extra["dets"]))
                res["decode_used"] = self.decode if g is not None else "vrb_fallback"
            else:
                res["decode_used"] = "vrb"
            if g is not None:
                box, crop, c224 = g["box"], g["crop"], g["crop224"]
                pn, conf, hm, seg, pts = g["pn"], g["conf"], g["hm"], g["seg"], g["pts"]
                nd = g["needle"]
                res["geom"] = {"orient_conf": round(g["orient_conf"], 4), "orient": g["orient"],
                               "length_px": round(nd["length_px"], 1), "n_fragments": nd["n_fragments"],
                               "bridged_gap_px": nd["bridged_gap_px"], "fit_model": nd["fit_model"],
                               "needle_bbox": [round(v, 1) for v in nd["bbox"]],
                               "vrb_points": [{"name": n, "uv": [round(float(u), 2), round(float(v), 2)], "conf": round(float(c), 4)}
                                              for n, (u, v), c in zip(self.vrb_names, g["vpts"], g["vconf"])]}
                extra["poly"] = g["poly"]
                if g.get("anchor") is not None:
                    res["vrb_anchor"] = g["anchor"]
                    res["vrb_agree"] = g["anchor"]["vrb_agree"]
                    res["geom_only"] = {"orient": "left", "orient_conf": round(float(g["geom_only"]["orient_conf"]), 4),
                                        "points": [{"name": n, "uv": [round(float(u), 2), round(float(v), 2)]}
                                                   for n, (u, v) in zip(self.anchor_names, g["geom_only"]["pts"])],
                                        "grasp_diff_px": g["anchor"]["grasp_diff_px"]}
                    extra["vrb_raw"] = list(zip(self.vrb_names, g["vpts"], g["vconf"]))
                    extra["geom_only_pts"] = dict(zip(self.anchor_names, g["geom_only"]["pts"]))
            else:
                c224, crop = self.crop(img_rgb, box)
                if c224 is None:
                    res["error"] = f"crop_too_small {crop}"
                    return res, extra
                pn, conf, hm, seg = self.keypoints(c224)
                pts = to_orig(pn, crop)
            out_names = g["names"] if g is not None else self.vrb_names
            conf_src = g.get("conf_src") if g is not None else None
            torch.cuda.synchronize()
            timing["vrb"] = (time.perf_counter() - t1) * 1e3
            res["crop_xyxy"] = [int(v) for v in crop]
            res["crop_mode"] = self.crop_mode
            res["points"] = [{"name": n, "uv": [round(float(u), 2), round(float(v), 2)], "conf": round(float(c), 4),
                              "xyz_cam": None} for n, (u, v), c in zip(out_names, pts, conf)]
            if conf_src:
                for p_, cs in zip(res["points"], conf_src):
                    p_["conf_src"] = cs
            if "tail_1_3" not in out_names:
                res["warning"] = "VRB 是 3 點模型且這張沒有用到幾何解碼，沒有 tail_1_3（夾取點）"
            if depth_fn is not None:
                t2 = time.perf_counter()
                sel = [p for p in res["points"] if p["name"] in DEPTH_POINTS]
                try:
                    xyz = depth_fn([p["uv"] for p in sel], box, (extra["det"] or {}).get("mask"))
                    for p, x in zip(sel, xyz):
                        p["xyz_cam"] = None if x is None else [round(float(v), 5) for v in x]
                except Exception as e:  # noqa: BLE001  深度失敗不影響 2D 結果
                    res["depth_error"] = f"{type(e).__name__}: {e}"
                timing["depth"] = (time.perf_counter() - t2) * 1e3
            timing["total"] = (time.perf_counter() - t0) * 1e3
            res["timing_ms"] = {k: round(v, 1) for k, v in timing.items()}
            res["ok"] = True
            extra.update(box=box, crop=crop, crop224=c224, pn=pn, conf=conf, hm=hm, seg=seg, pts=pts, names=out_names)
            return res, extra

    def point(self, res, name="tail_1_3"):
        for p in res.get("points", []):
            if p["name"] == name:
                return p
        return None


# =============================================================================
# 存 debug（輸入影像、224 裁切圖、疊圖、JSON）
# =============================================================================
def draw_overlay(img_bgr, res, extra, names, conf_th=0.5):
    """有預測時用 predict_needle5_langsam.draw（原圖放大 + 224 + heatmap + seg）；沒有時只畫 YOLO 的框。"""
    vis = img_bgr.copy()
    for d in extra.get("dets", []):
        x1, y1, x2, y2 = [int(round(v)) for v in d["box_xyxy"]]
        c = (0, 200, 255) if d is extra.get("det") else (0, 0, 200)
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 1)
        cv2.putText(vis, f"{d['conf']:.2f}", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    if not res.get("ok"):
        cv2.putText(vis, res.get("error", ""), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return vis
    if extra.get("poly") is not None:                                     # 幾何解碼的中心線（含補起來的部分）
        cv2.polylines(vis, [np.round(extra["poly"]).astype(np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA)
    names = extra.get("names", names)
    from predict_needle5_langsam import draw
    if extra.get("vrb_raw"):
        # vrb_anchor：VRB 3 點 = 十字 + heatmap 值；夾取點（VRB 決定）= 洋紅菱形；同條中心線的純幾何夾取點 = 灰三角
        P = lambda q: (int(round(q[0])), int(round(q[1])))
        for (n, q, c), col in zip(extra["vrb_raw"], [(0, 0, 255), (255, 80, 0), (0, 200, 0)]):
            cv2.drawMarker(vis, P(q), col, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            cv2.putText(vis, f"{n} {c:.2f}", (P(q)[0] + 5, P(q)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        for n, q in extra.get("geom_only_pts", {}).items():
            if n in ("tail_1_3", "head_1_5"):
                cv2.drawMarker(vis, P(q), (200, 200, 200), cv2.MARKER_TRIANGLE_UP, 10, 2, cv2.LINE_AA)
        for n, q in zip(names, extra["pts"]):
            if n in ("tail_1_3", "head_1_5"):
                cv2.drawMarker(vis, P(q), (255, 0, 255), cv2.MARKER_DIAMOND, 11, 2, cv2.LINE_AA)
                cv2.putText(vis, n, (P(q)[0] + 5, P(q)[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
        vn = [n for n, _, _ in extra["vrb_raw"]]
        vc = np.array([c for _, _, c in extra["vrb_raw"]])
        out = draw(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), extra["box"], extra["crop"], np.zeros((0, 2)), vc, vn,
                   extra["crop224"], extra["pn"], extra["hm"], extra["seg"], conf_th)
        a = res.get("vrb_anchor", {})
        bar = np.zeros((44, out.shape[1], 3), np.uint8)
        cv2.putText(bar, f"vrb_anchor  vrb_agree={res.get('vrb_agree')}  grasp_conf={a.get('grasp_conf')}  "
                    f"span_ratio={a.get('span_ratio')}  middle_diff={a.get('middle_consistency_px')}px  "
                    f"diff_vs_geom_only={a.get('grasp_diff_px')}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255) if res.get("vrb_agree") else (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, "+ = VRB point (heatmap peak)   magenta diamond = grasp point (VRB)   gray triangle = geom_only grasp",
                    (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        return np.vstack([bar, out])
    img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return draw(img_rgb, extra["box"], extra["crop"], extra["pts"], extra["conf"], names, extra["crop224"],
                extra["pn"], extra["hm"], extra["seg"], conf_th)


DET_STYLE = {   # status -> (BGR 顏色, 說明)
    "selected": ((0, 255, 0), "selected"),
    "union": ((255, 200, 0), "merged into centerline"),
    "passed": ((0, 220, 220), "passed (not used)"),
    "outside_roi": ((0, 0, 255), "rejected: outside ROI"),
    "box_size": ((0, 0, 255), "rejected: box size"),
    "class": ((200, 0, 200), "other class"),
}


def _det_style(st):
    if st.startswith("box_size"):
        return DET_STYLE["box_size"][0]
    if st.startswith("class_"):
        return DET_STYLE["class"][0]
    return DET_STYLE.get(st, ((255, 255, 255), st))[0]


def draw_yolo_overlay(img_bgr, res, extra):
    """YOLO 輸出疊在原圖上：每個偵測的 mask（半透明）、框、編號、conf、處理結果；被選中的用綠色粗框；
    被 ROI／框大小濾掉的用紅色並標原因；ROI 畫白框。最上方一行是摘要。"""
    vis = img_bgr.copy()
    dets, status = extra.get("dets", []), extra.get("det_status") or []
    layer = vis.copy()
    order = sorted(range(len(dets)), key=lambda i: (status[i] == "selected", dets[i]["conf"]))
    for i in order:                                            # 低信心先畫，選中的最後畫在最上面
        if dets[i].get("mask") is not None:
            layer[dets[i]["mask"]] = _det_style(status[i])
    vis = cv2.addWeighted(layer, 0.45, vis, 0.55, 0)
    roi = extra.get("roi")
    if roi:
        x1, y1, x2, y2 = [int(round(v)) for v in roi]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "ROI", (x1 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for i in order:
        d, st = dets[i], status[i]
        c = _det_style(st)
        x1, y1, x2, y2 = [int(round(v)) for v in d["box_xyxy"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 3 if st == "selected" else 1, cv2.LINE_AA)
        label = f"#{i} {d['conf']:.2f} {st}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 - 5 if y1 - 5 - th > 44 else y2 + th + 5        # 上方放不下就放到框下方
        cv2.rectangle(vis, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), (0, 0, 0), -1)
        cv2.putText(vis, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    bar = np.zeros((44, vis.shape[1], 3), np.uint8)
    sel = [i for i, st in enumerate(status) if st == "selected"]
    head = (f"YOLO: {len(dets)} detections"
            + (f", selected #{sel[0]} conf {dets[sel[0]]['conf']:.2f}" if sel else "")
            + (f"   error: {res.get('error')}" if res.get("error") else "")
            + f"   stamp {res.get('stamp')}")
    cv2.putText(bar, head, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if res.get("ok") else (0, 0, 255), 1,
                cv2.LINE_AA)
    x = 8
    for key in ("selected", "union", "passed", "outside_roi", "box_size", "class"):
        c, txt = DET_STYLE[key]
        cv2.rectangle(bar, (x, 27), (x + 12, 39), c, -1)
        cv2.putText(bar, txt, (x + 16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        x += 22 + cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    return np.vstack([bar, vis])


def save_debug(out_dir, img_bgr, res, extra, names, async_write=True):
    """存到 out_dir/：input.png（給 YOLO 的原圖）、yolo_overlay.png（YOLO 輸出）、crop224.png（給 VRB 的 224 圖）、
    overlay.png（VRB 結果）、result.json。YOLO 失敗（no_detection／no_valid_detection）時只有 input、yolo_overlay、
    result.json。async_write：背景執行緒寫檔，不拖慢回應（res["debug_dir"] 在這裡先填好）。"""
    os.makedirs(out_dir, exist_ok=True)
    res["debug_dir"] = os.path.abspath(out_dir)
    payload = (img_bgr.copy(), json.loads(json.dumps(res)), extra)

    def _w():
        img, r, ex = payload
        cv2.imwrite(os.path.join(out_dir, "input.png"), img)
        if "det_status" in ex:                                  # YOLO 有跑（成功或失敗都存）
            try:
                cv2.imwrite(os.path.join(out_dir, "yolo_overlay.png"), draw_yolo_overlay(img, r, ex))
            except Exception as e:  # noqa: BLE001
                r["yolo_overlay_error"] = str(e)
        if ex.get("crop224") is not None:
            cv2.imwrite(os.path.join(out_dir, "crop224.png"), cv2.cvtColor(ex["crop224"], cv2.COLOR_RGB2BGR))
        if r.get("ok"):
            try:
                cv2.imwrite(os.path.join(out_dir, "overlay.png"), draw_overlay(img, r, ex, names))
            except Exception as e:  # noqa: BLE001
                r["overlay_error"] = str(e)
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, ensure_ascii=False)

    if async_write:
        threading.Thread(target=_w, daemon=True).start()
    else:
        _w()


def add_common_args(ap):
    ap.add_argument("--yolo", default=DEFAULT_YOLO)
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--weights", default=DEFAULT_VRB_WEIGHTS)
    ap.add_argument("--yolo_imgsz", type=int, default=1280, help="新 YOLO（best_20261116_225446）是 1280 訓練的；舊 best.pt 用 640")
    ap.add_argument("--yolo_conf", type=float, default=0.1, help="0.1 才抓得到被指頭切開的短段（2026-11-17 實測）")
    ap.add_argument("--box_src", default="box", choices=["box", "mask"])
    ap.add_argument("--crop_mode", default="train", choices=["train", "official"])
    ap.add_argument("--detector", default="yolo", choices=["yolo", "sam3"], help="找針用 YOLO 或 SAM 3（文字提示）")
    ap.add_argument("--sam3", default=os.path.join(_HERE, "sam3.pt"), help="SAM 3 權重（Hugging Face facebook/sam3）")
    ap.add_argument("--sam3_prompt", nargs="+", default=["needle"], help="SAM 3 的文字提示，可以給好幾個")
    ap.add_argument("--decode", default="vrb_anchor", choices=["vrb_anchor", "geom", "vrb"],
                    help="vrb_anchor（預設）：VRB 點投影到中心線決定夾取點；geom：mask 幾何 + 定方向；vrb：VRB 直接預測")
    ap.add_argument("--orient", default="left", choices=["heat", "ends", "all", "left"],
                    help="geom 時 VRB 怎麼定方向（有線的類別時先用線）：heat = 在中心線兩端讀 head/tail heatmap")
    ap.add_argument("--roi", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                    help="只接受框中心落在這個範圍的偵測（原圖 pixel）")


def build_from_args(a, warmup=True):
    return NeedlePerception(a.yolo, a.run, a.weights, a.yolo_imgsz, a.yolo_conf, box_src=a.box_src,
                            crop_mode=a.crop_mode, roi_xyxy=a.roi, warmup=warmup, decode=a.decode, orient_mode=a.orient,
                            detector=a.detector, sam3_weights=a.sam3, sam3_prompts=a.sam3_prompt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--images", required=True, help="一張圖或資料夾（會往下找子資料夾）")
    ap.add_argument("--glob", default="*", help="資料夾裡要用的檔名樣式，例如 '*_rgb.png'")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "pred_handoff_output"))
    a = ap.parse_args()

    if os.path.isdir(a.images):
        paths = sorted(p for p in glob.glob(os.path.join(a.images, "**", a.glob), recursive=True) if p.lower().endswith(IMG_EXTS))
    else:
        paths = [a.images]
    paths = paths[::max(1, a.every)]
    if a.limit > 0:
        paths = paths[:a.limit]
    if not paths:
        sys.exit(f"[ERROR] {a.images} 沒有影像")

    t = time.perf_counter()
    perc = build_from_args(a)
    print(f"載入 + 暖機 {time.perf_counter() - t:.1f}s；GPU {gpu_mem_report()}")
    root = os.path.join(a.out, "offline_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    rows = []
    for j, p in enumerate(paths, 1):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 讀不到 {p}")
            continue
        res, extra = perc.predict(img, stamp=os.path.getmtime(p), frame_id="offline")
        name = os.path.splitext(os.path.relpath(p, a.images if os.path.isdir(a.images) else os.path.dirname(p)))[0].replace(os.sep, "__")
        save_debug(os.path.join(root, name), img, res, extra, perc.names, async_write=False)
        q = perc.point(res)
        rows.append({"image": p, "ok": res["ok"], "error": res["error"], "n_det": res.get("n_detections"),
                     "yolo_conf": (res["yolo"] or {}).get("conf"),
                     "tail_1_3_u": q["uv"][0] if q else None, "tail_1_3_v": q["uv"][1] if q else None,
                     "tail_1_3_conf": q["conf"] if q else None,
                     **({"head_1_5_u": perc.point(res, "head_1_5")["uv"][0], "head_1_5_v": perc.point(res, "head_1_5")["uv"][1]}
                        if perc.point(res, "head_1_5") else {}),
                     "vrb_agree": res.get("vrb_agree"),
                     **{f"anchor_{k}": (res.get("vrb_anchor") or {}).get(k) for k in ("grasp_conf", "span_ratio",
                                                                                      "middle_consistency_px", "head_off_px", "tail_off_px")},
                     **{f"diff_geom_{k}": v for k, v in ((res.get("vrb_anchor") or {}).get("grasp_diff_px") or {}).items()},
                     "min_conf": min(pp["conf"] for pp in res["points"]) if res["points"] else None,
                     **{f"ms_{k}": v for k, v in res.get("timing_ms", {}).items()}})
        print(f"[{j}/{len(paths)}] {name}: ok={res['ok']} {res['error']} det={res.get('n_detections')} "
              f"yolo={rows[-1]['yolo_conf']} tail_1_3={None if not q else q['uv']} conf={None if not q else q['conf']} "
              f"{res.get('timing_ms')}")
    import csv
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tot = [r["ms_total"] for r in rows if r.get("ms_total")]
    summ = {"n": len(rows), "ok": sum(r["ok"] for r in rows),
            "ms_total_median": float(np.median(tot)) if tot else None,
            "ms_total_p90": float(np.percentile(tot, 90)) if tot else None,
            "gpu": gpu_mem_report(), "settings": {k: getattr(a, k) for k in ("detector", "yolo", "sam3", "sam3_prompt", "run", "weights", "yolo_imgsz", "yolo_conf", "box_src", "crop_mode", "decode", "orient", "roi")}}
    json.dump(summ, open(os.path.join(root, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    print(f"→ {root}")


if __name__ == "__main__":
    main()
