#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 C：D405 左右 IR → 立體匹配（FoundationStereo）→ 針上某一點的 3D 座標。

給 B 的感知 node 用的最短寫法（模型只載入一次、常駐）：

    from handover_depth import D405Calib, StereoDepth, make_backend

    calib = D405Calib.from_json("captures/xxx/calib.json")          # capture_d405_stereo.py 存的；或 D405Calib.from_camera_info(...)
    depth = StereoDepth(make_backend("fs"), calib)                   # "fs" / "ffs" / "sgbm"
    out = depth.locate(ir_left, ir_right,                            # 同一個時間戳的 infra1 / infra2（HxW uint8）
                       uv_color=[(u_head, v_head), (u_t13, v_t13)],  # VRB 在 color 影像上的像素
                       box_color=yolo_box_xyxy,                      # YOLO 的框（color 像素），只算這一塊
                       mask_color=yolo_mask)                         # YOLO 的針 mask（color 大小，可為 None）
    out[i]["xyz_cam"]   # color optical frame，公尺；查不到是 None → JSON 填 null
    out[i]["ok"], out[i]["z_sigma"], out[i]["sep_px"] ...            # 品質指標，見 lookup_disparity()

座標：
  - 立體匹配在「左 IR（infra1，已校正 rectified）」上做，視差 d = u_left - u_right（像素，>0），Z = fx * B / d。
  - D405 沒有獨立的 RGB 感測器，color 是左 imager 經 ISP 出來的，視點和 infra1 相同（平移 ≈ 0），
    但 color 有自己的內參（可能帶畸變），所以 color 的 (u, v) 不能直接拿去查 infra1 的視差圖：
    color 像素 → 去畸變的視線 → 轉到 infra1 → 用 infra1 內參投影。平移是 0 時這個對應和深度無關。
  - 回傳的 xyz_cam 在 color optical frame（介面 §3.1），也就是 infra1 的點再乘上 infra1→color 外參。

細針的處理（任務書 §5-C-2）：
  - 只算針周圍的 ROI（省 GPU 記憶體與時間），而且先把 ROI 放大 upscale 倍（預設 2）再丟進網路：
    針在 35 cm 只有 2~3 px 寬，立體網路的特徵是 1/4 解析度，不放大很容易被背景（縫合墊）蓋掉。
  - 查深度不取單一像素：取該點附近、落在針 mask 內的像素；一定做「前景／背景」兩群分離，
    針比縫合墊近（視差大），取視差大的那一群的中位數。
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = ["D405Calib", "StereoDepth", "DispResult", "lookup_disparity", "make_backend",
           "FoundationStereoBackend", "FastFoundationStereoBackend", "SGBMBackend"]

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FS_REPO = os.path.join(_HERE, "third_party", "FoundationStereo")
DEFAULT_FS_CKPT = os.path.join(DEFAULT_FS_REPO, "pretrained_models", "23-51-11", "model_best_bp2.pth")
DEFAULT_FFS_REPO = os.path.join(_HERE, "third_party", "Fast-FoundationStereo")
DEFAULT_FFS_CKPT = os.path.join(DEFAULT_FFS_REPO, "weights", "23-36-37", "model_best_bp2_serialize.pth")


# =============================================================================
# 1. 相機參數
# =============================================================================
@dataclass
class D405Calib:
    """D405 的立體參數與 color↔infra1 的關係。長度公尺、像素以 (u, v) 表示。

    K_left        infra1（已校正）的 3x3 內參
    baseline      infra1 與 infra2 的距離（正值，公尺；D405 約 0.018）
    K_color       color 的 3x3 內參
    dist_color    color 的畸變係數 [k1, k2, p1, p2, k3]（RealSense 的順序，和 OpenCV 相同）
    dist_model    "none" / "brown_conrady" / "inverse_brown_conrady"
    R_c_l, t_c_l  infra1 → color：X_color = R_c_l @ X_left + t_c_l（D405 上應該是單位矩陣與 ≈0）
    """
    width: int
    height: int
    K_left: np.ndarray
    baseline: float
    K_color: np.ndarray | None = None
    dist_color: np.ndarray = field(default_factory=lambda: np.zeros(5))
    dist_model: str = "none"
    R_c_l: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_c_l: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.K_left = np.asarray(self.K_left, np.float64).reshape(3, 3)
        self.K_color = self.K_left.copy() if self.K_color is None else np.asarray(self.K_color, np.float64).reshape(3, 3)
        self.dist_color = np.asarray(self.dist_color, np.float64).ravel()[:5]
        self.R_c_l = np.asarray(self.R_c_l, np.float64).reshape(3, 3)
        self.t_c_l = np.asarray(self.t_c_l, np.float64).ravel()
        self.baseline = abs(float(self.baseline))
        if not (0.005 < self.baseline < 0.2):
            raise ValueError(f"baseline = {self.baseline} m 不合理（D405 約 0.018）。單位是公尺嗎？")
        if not np.any(np.abs(self.dist_color) > 1e-12):
            self.dist_model = "none"

    # ---- 讀寫 ---------------------------------------------------------------
    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        c = d.get("color") or {}
        e = d.get("left_to_color") or {}
        return cls(width=d["width"], height=d["height"], K_left=d["left"]["K"], baseline=d["baseline_m"],
                   K_color=c.get("K"), dist_color=c.get("dist", [0] * 5), dist_model=c.get("model", "none"),
                   R_c_l=e.get("R", np.eye(3)), t_c_l=e.get("t", [0, 0, 0]))

    def to_dict(self):
        return {"width": self.width, "height": self.height, "baseline_m": self.baseline,
                "left": {"K": self.K_left.tolist()},
                "color": {"K": self.K_color.tolist(), "dist": self.dist_color.tolist(), "model": self.dist_model},
                "left_to_color": {"R": self.R_c_l.tolist(), "t": self.t_c_l.tolist()}}

    @classmethod
    def from_camera_info(cls, info_left, info_right, info_color=None, R_c_l=None, t_c_l=None, baseline=None):
        """從 ROS 的 sensor_msgs/CameraInfo 建立（B 的 node 用）。info_* 也可以是同欄位的 dict（錄影時存下來的）。
        baseline 取自右相機 P 矩陣：P[0,3] = -fx * B（沒填時用參數 baseline）。color 的畸變模型照 distortion_model 欄位；
        infra1→color 的外參要另外給（realsense2_camera 的 extrinsics topic 或 tf），沒給就當成同一個視點。"""
        if isinstance(info_left, dict):
            info_left, info_right = types.SimpleNamespace(**info_left), types.SimpleNamespace(**info_right)
            info_color = None if info_color is None else types.SimpleNamespace(**info_color)
        Kl = np.asarray(info_left.k, np.float64).reshape(3, 3)
        Pr = np.asarray(info_right.p, np.float64).reshape(3, 4)
        b = -Pr[0, 3] / Pr[0, 0]
        if abs(b) < 1e-4:
            if baseline is None:
                raise ValueError("infra2 camera_info 的 P[0,3] 是 0，算不出 baseline；請用 baseline= 指定（D405 約 0.018 m）")
            b = baseline
        kw = {}
        if info_color is not None:
            m = (info_color.distortion_model or "").lower()
            kw = {"K_color": np.asarray(info_color.k).reshape(3, 3), "dist_color": list(info_color.d)[:5] or [0] * 5,
                  "dist_model": "inverse_brown_conrady" if "inverse" in m else "brown_conrady"}
        if R_c_l is not None:
            kw.update(R_c_l=R_c_l, t_c_l=t_c_l if t_c_l is not None else np.zeros(3))
        return cls(width=info_left.width, height=info_left.height, K_left=Kl, baseline=b, **kw)

    # ---- 幾何 ---------------------------------------------------------------
    @property
    def fx(self):
        return float(self.K_left[0, 0])

    @property
    def same_viewpoint(self):
        return float(np.linalg.norm(self.t_c_l)) < 1e-4            # < 0.1 mm 就當成同一個視點

    def disp_to_z(self, disp):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.fx * self.baseline / np.asarray(disp, np.float64)

    def z_to_disp(self, z):
        return self.fx * self.baseline / float(z)

    def color_rays(self, uv):
        """color 像素 (N,2) → color frame 下 z=1 的視線 (N,3)（已去畸變）。"""
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        K = self.K_color
        x = (uv[:, 0] - K[0, 2]) / K[0, 0]
        y = (uv[:, 1] - K[1, 2]) / K[1, 1]
        if self.dist_model == "inverse_brown_conrady":            # librealsense：反投影方向直接套公式
            k1, k2, p1, p2, k3 = self.dist_color
            r2 = x * x + y * y
            f = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            x, y = (x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                    y * f + 2 * p2 * x * y + p1 * (r2 + 2 * y * y))
        elif self.dist_model != "none":                           # (modified) brown_conrady：反投影要迭代，交給 OpenCV
            n = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, self.dist_color).reshape(-1, 2)
            x, y = n[:, 0], n[:, 1]
        return np.stack([x, y, np.ones_like(x)], axis=1)

    def color_uv_to_left(self, uv, z=None):
        """color 像素 → infra1 像素。視點相同時和深度無關；不同時需要 z（color frame 的深度，公尺）。"""
        Xc = self.color_rays(uv)
        if not self.same_viewpoint:
            Xc = Xc * (0.3 if z is None else np.asarray(z, np.float64).reshape(-1, 1))
            Xc = Xc - self.t_c_l
        Xl = Xc @ self.R_c_l                                       # = (R^T @ Xc^T)^T
        return np.stack([self.K_left[0, 0] * Xl[:, 0] / Xl[:, 2] + self.K_left[0, 2],
                         self.K_left[1, 1] * Xl[:, 1] / Xl[:, 2] + self.K_left[1, 2]], axis=1)

    def left_uvz_to_color_xyz(self, uv, z):
        """infra1 像素 + 深度 → color optical frame 的 3D 點。"""
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        z = np.asarray(z, np.float64).reshape(-1)
        Xl = np.stack([(uv[:, 0] - self.K_left[0, 2]) / self.K_left[0, 0] * z,
                       (uv[:, 1] - self.K_left[1, 2]) / self.K_left[1, 1] * z, z], axis=1)
        return Xl @ self.R_c_l.T + self.t_c_l

    def color_mask_to_left(self, mask, z=None):
        """color 上的 mask → infra1 上的 mask（把 mask 的像素逐點搬過去，再補 3x3 的洞）。"""
        ys, xs = np.nonzero(mask)
        out = np.zeros((self.height, self.width), np.uint8)
        if len(xs) == 0:
            return out
        p = np.rint(self.color_uv_to_left(np.stack([xs, ys], 1), z)).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < self.width) & (p[:, 1] >= 0) & (p[:, 1] < self.height)
        out[p[ok, 1], p[ok, 0]] = 1
        return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


# =============================================================================
# 2. 立體匹配後端：__call__(left_rgb, right_rgb) -> 視差 (H, W) float32，單位 = 輸入影像的像素
# =============================================================================
def _to_rgb(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.repeat(img[..., None], 3, axis=2) if img.ndim == 2 else np.ascontiguousarray(img[..., :3])


def _stub_missing(names):
    """FoundationStereo 的 Utils.py 在最上面 import 了只有畫點雲才用到的套件；沒裝就塞空殼，推論用不到。

    注意（2026-11-18 實機）：torch._dynamo 第一次被 import 時會對一串套件名（含 pandas）呼叫
    importlib.util.find_spec，空殼模組沒有 __spec__ 會讓它丟 ValueError。所以：
      1. 先把 torch._dynamo、torchvision、timm 都 import 完，再塞空殼；
      2. 空殼也補上 __spec__，萬一之後還有人查也不會出錯。
    """
    import importlib.machinery
    import torch._dynamo  # noqa: F401
    import torchvision  # noqa: F401
    import timm  # noqa: F401
    for n in names:
        if n in sys.modules:
            continue
        try:
            __import__(n)
        except Exception:
            m = types.ModuleType(n)
            m.__spec__ = importlib.machinery.ModuleSpec(n, None)
            sys.modules[n] = m


class _TorchBackend:
    name = "torch"

    def __init__(self):
        self.last_time = None
        self.peak_mem_mb = None

    def _run(self, fn, left, right):
        import torch
        from core.utils.utils import InputPadder
        l = torch.as_tensor(_to_rgb(left)).cuda().float()[None].permute(0, 3, 1, 2)
        r = torch.as_tensor(_to_rgb(right)).cuda().float()[None].permute(0, 3, 1, 2)
        H, W = l.shape[-2:]
        padder = InputPadder(l.shape, divis_by=32, force_square=False)
        l, r = padder.pad(l, r)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            disp = fn(l, r)
        torch.cuda.synchronize()
        self.last_time = time.time() - t0
        self.peak_mem_mb = torch.cuda.max_memory_allocated() / 2 ** 20
        return padder.unpad(disp.float()).cpu().numpy().reshape(H, W).astype(np.float32)


class FoundationStereoBackend(_TorchBackend):
    """NVlabs/FoundationStereo（預設 23-51-11，ViT-L）。第一次建立模型會從網路抓 DINOv2 的程式與 edgenext 的權重。"""
    name = "fs"

    def __init__(self, repo_dir=DEFAULT_FS_REPO, ckpt=DEFAULT_FS_CKPT, valid_iters=32, hiera=False, low_memory=False):
        super().__init__()
        import torch
        from omegaconf import OmegaConf
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"找不到 FoundationStereo 權重：{ckpt}（照 README 從 Google Drive 下載整個 23-51-11 資料夾，含 cfg.yaml）")
        _stub_missing(["open3d", "trimesh", "joblib", "pandas", "imageio"])
        sys.path.insert(0, repo_dir)
        from core.foundation_stereo import FoundationStereo
        cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt), "cfg.yaml"))
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"], cfg["hiera"], cfg["low_memory"] = valid_iters, int(hiera), bool(low_memory)
        self.iters, self.hiera, self.low_memory = valid_iters, hiera, low_memory
        # 建模型時官方程式會上網抓兩樣東西，但權重之後都會被 checkpoint 蓋掉，所以都改成不下載：
        #   - torch.hub.load('facebookresearch/dinov2', ...)：改用 repo 內附的 dinov2/（hubconf.py），不需要連 GitHub
        #   - timm.create_model('edgenext_small', pretrained=True)：改 pretrained=False，不需要連 Hugging Face
        #     （也避開 venv 釘住的 huggingface-hub==0.13.4 和新版 timm 的相容問題）
        import timm
        hub_load, create_model = torch.hub.load, timm.create_model
        local_dino = os.path.join(repo_dir, "dinov2")

        def _hub_load(repo, model, *a, **kw):
            if repo == "facebookresearch/dinov2" and os.path.isfile(os.path.join(local_dino, "hubconf.py")):
                kw.update(source="local", pretrained=False)
                return hub_load(local_dino, model, *a, **kw)
            return hub_load(repo, model, *a, **kw)

        def _create_model(name, *a, **kw):
            kw["pretrained"] = False
            return create_model(name, *a, **kw)
        torch.hub.load, timm.create_model = _hub_load, _create_model
        try:
            self.model = FoundationStereo(cfg)
        finally:
            torch.hub.load, timm.create_model = hub_load, create_model
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        self.model.cuda().eval()

    def __call__(self, left, right):
        import torch

        def fn(l, r):
            with torch.amp.autocast("cuda", enabled=True):
                if self.hiera:
                    return self.model.run_hierachical(l, r, iters=self.iters, test_mode=True, low_memory=self.low_memory, small_ratio=0.5)
                return self.model.forward(l, r, iters=self.iters, test_mode=True, low_memory=self.low_memory)
        return self._run(fn, left, right)


class FastFoundationStereoBackend(_TorchBackend):
    """NVlabs/Fast-FoundationStereo（CVPR 2026；README 寫 640x480 時尖峰記憶體約 650 MB）。權重是整個模型的 pickle。"""
    name = "ffs"

    def __init__(self, repo_dir=DEFAULT_FFS_REPO, ckpt=DEFAULT_FFS_CKPT, valid_iters=8, max_disp=192):
        super().__init__()
        import torch
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"找不到 Fast-FoundationStereo 權重：{ckpt}（照 README 從 Google Drive 下載到 weights/）")
        _stub_missing(["open3d", "imageio"])
        sys.path.insert(0, repo_dir)
        self.model = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.args.valid_iters, self.model.args.max_disp = valid_iters, max_disp
        self.model.cuda().eval()
        self.iters, self.max_disp = valid_iters, max_disp

    def __call__(self, left, right):
        import torch

        def fn(l, r):
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                return self.model.forward(l, r, iters=self.iters, test_mode=True, optimize_build_volume="pytorch1")
        return self._run(fn, left, right)


class SGBMBackend:
    """OpenCV SGBM：不用 GPU、不用權重。只用來確認流程與當對照組，細針上不要期待它準。"""
    name = "sgbm"

    def __init__(self, max_disp=256, block=5):
        nd = int(np.ceil(max_disp / 16) * 16)
        self.m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=block, P1=8 * block * block,
                                       P2=32 * block * block, uniquenessRatio=5, speckleWindowSize=0, disp12MaxDiff=2,
                                       mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.last_time, self.peak_mem_mb = None, 0.0

    def __call__(self, left, right):
        t0 = time.time()
        g = [cv2.cvtColor(_to_rgb(x), cv2.COLOR_RGB2GRAY) for x in (left, right)]
        d = self.m.compute(g[0], g[1]).astype(np.float32) / 16.0
        d[d <= 0] = np.nan
        self.last_time = time.time() - t0
        return d


def make_backend(name="fs", **kw):
    return {"fs": FoundationStereoBackend, "ffs": FastFoundationStereoBackend, "sgbm": SGBMBackend}[name](**kw)


# =============================================================================
# 3. 查某一點的視差（細針用）
# =============================================================================
def _split_near(vals, min_gap):
    """一維兩群分離（Otsu）。兩群中心差 >= min_gap 時回傳 (近的那群, 遠的那群中位數)，否則 (全部, None)。"""
    v = np.sort(vals)
    if len(v) < 6:
        return v, None
    c1 = np.cumsum(v)
    n = np.arange(1, len(v))
    m_lo = c1[:-1] / n
    m_hi = (c1[-1] - c1[:-1]) / (len(v) - n)
    k = int(np.argmax(n * (len(v) - n) * (m_hi - m_lo) ** 2)) + 1
    lo, hi = v[:k], v[k:]
    if np.median(hi) - np.median(lo) < min_gap:
        return v, None
    return hi, float(np.median(lo))


def lookup_disparity(disp, uv, mask=None, radius=6.0, min_px=5, min_gap=0.75, ring=(2.0, 4.0)):
    """在視差圖 disp 上查 uv（disp 的像素座標）附近「屬於針」的視差。

    作法：取 uv 半徑 radius 內的有效像素（有 mask 就只取 mask 內的；不夠 min_px 就把半徑放大到 2 倍、3 倍），
    做兩群分離，取視差大（近）的那一群的中位數。min_gap 是兩群要差多少像素才算分得開。

    回傳 dict：
      disp     針的視差（像素）；查不到是 None
      n        用了幾個像素
      mad_px   這些像素的 MAD×1.4826（視差的離散程度）
      bg_disp  周圍一圈（半徑 ring[0]~ring[1] 倍、mask 外）的視差中位數，也就是背景
      sep_px   disp - bg_disp。接近 0 表示網路沒有把針和背景分開 → 這個深度其實是背景的，不能用
      method   "mask" / "window"，後面加 "+near" 表示有做前景分離
    """
    H, W = disp.shape
    u, v = float(uv[0]), float(uv[1])
    yy, xx = np.ogrid[:H, :W]
    d2 = (xx - u) ** 2 + (yy - v) ** 2
    valid = np.isfinite(disp) & (disp > 0)
    out = {"disp": None, "n": 0, "mad_px": None, "bg_disp": None, "sep_px": None, "method": None}

    rmax = radius * ring[1]
    bg_sel = valid & (d2 > (radius * ring[0]) ** 2) & (d2 <= rmax ** 2)
    if mask is not None:
        bg_sel &= cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    bg = float(np.median(disp[bg_sel])) if bg_sel.sum() >= min_px else None

    vals, method = np.empty(0), None
    for k in (1, 2, 3):
        win = valid & (d2 <= (radius * k) ** 2)
        if mask is not None and (win & (mask > 0)).sum() >= min_px:
            vals, method = disp[win & (mask > 0)], "mask"
            break
        if mask is None and win.sum() >= min_px:
            vals, method = disp[win], "window"
            break
    if method is None:                                            # mask 在這附近沒有像素 → 退回用視窗
        win = valid & (d2 <= radius ** 2)
        if win.sum() < min_px:
            return out
        vals, method = disp[win], "window"

    near, far = _split_near(vals.astype(np.float64), min_gap)
    if far is not None:
        method += "+near"
        if bg is None:
            bg = far
    d = float(np.median(near))
    out.update(disp=d, n=int(len(near)), mad_px=float(1.4826 * np.median(np.abs(near - d))), bg_disp=bg,
               sep_px=None if bg is None else d - bg, method=method)
    return out


# =============================================================================
# 4. 整合：ROI + 放大 + 立體匹配 + 查點
# =============================================================================
@dataclass
class DispResult:
    """ROI 的視差。disp 的解析度是原圖的 scale 倍，數值已經換回「原圖像素」的視差。"""
    disp: np.ndarray
    x0: int
    y0: int
    scale: float
    full_size: tuple                                               # (H, W)
    time_s: float | None = None
    peak_mem_mb: float | None = None

    def to_local(self, uv):
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        return (uv - [self.x0, self.y0] + 0.5) * self.scale - 0.5

    def full(self):
        """貼回原圖大小的視差圖（ROI 以外是 NaN）。只用來畫圖／存檔，查點請用 StereoDepth.locate()。"""
        out = np.full(self.full_size, np.nan, np.float32)
        h, w = int(round(self.disp.shape[0] / self.scale)), int(round(self.disp.shape[1] / self.scale))
        out[self.y0:self.y0 + h, self.x0:self.x0 + w] = cv2.resize(self.disp, (w, h), interpolation=cv2.INTER_NEAREST)
        return out


class StereoDepth:
    """
    backend   make_backend(...) 的結果
    calib     D405Calib
    upscale   ROI 丟進網路前放大的倍數（細針建議 2；整張圖時自動用 1）
    z_min     會出現的最近距離（公尺），用來決定 ROI 左邊要多留多少（右影像的對應點在左邊 d 個像素）
    margin    ROI 四周多留的像素（給網路一點上下文）
    radius    查點的半徑（原圖像素）
    """

    def __init__(self, backend, calib: D405Calib, upscale=2.0, z_min=0.10, margin=48, radius=6.0, max_roi_px=1024):
        self.backend, self.calib = backend, calib
        self.upscale, self.z_min, self.margin, self.radius, self.max_roi_px = float(upscale), z_min, margin, radius, max_roi_px

    def roi_from_box(self, box_left):
        x1, y1, x2, y2 = [float(v) for v in box_left]
        dmax = self.calib.z_to_disp(self.z_min)
        H, W = self.calib.height, self.calib.width
        return (int(max(0, np.floor(x1 - self.margin - dmax))), int(max(0, np.floor(y1 - self.margin))),
                int(min(W, np.ceil(x2 + self.margin))), int(min(H, np.ceil(y2 + self.margin))))

    def disparity(self, left, right, roi=None, upscale=None) -> DispResult:
        """roi = (x0, y0, x1, y1)（infra1 像素）；None = 整張圖、不放大。左右影像用同一個矩形裁切，視差不用補償。"""
        H, W = left.shape[:2]
        if roi is None:
            x0, y0, x1, y1, s = 0, 0, W, H, 1.0 if upscale is None else float(upscale)
        else:
            x0, y0, x1, y1 = roi
            s = self.upscale if upscale is None else float(upscale)
        s = min(s, self.max_roi_px / max(x1 - x0, y1 - y0)) if s > 1 else s      # 放大後最長邊不超過 max_roi_px
        l, r = left[y0:y1, x0:x1], right[y0:y1, x0:x1]
        if abs(s - 1) > 1e-6:
            size = (int(round((x1 - x0) * s)), int(round((y1 - y0) * s)))
            interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
            l, r = cv2.resize(l, size, interpolation=interp), cv2.resize(r, size, interpolation=interp)
            s = size[0] / (x1 - x0)
        d = self.backend(l, r) / s
        return DispResult(d, x0, y0, s, (H, W), self.backend.last_time, self.backend.peak_mem_mb)

    def locate(self, left, right, uv_color, box_color=None, mask_color=None, return_disp=False):
        """VRB 的點（color 像素）→ color optical frame 的 3D 點。uv_color: [(u, v), ...]。"""
        c = self.calib
        uv_color = np.asarray(uv_color, np.float64).reshape(-1, 2)
        uv_l = c.color_uv_to_left(uv_color)
        if box_color is not None:
            b = c.color_uv_to_left(np.asarray(box_color, np.float64).reshape(2, 2)).ravel()
        else:                                                      # 沒給框：包住所有點，四周各加 80 px
            b = np.r_[uv_l.min(0) - 80, uv_l.max(0) + 80]
        res = self.disparity(left, right, self.roi_from_box(b))
        mask_s = None
        if mask_color is not None:
            m = c.color_mask_to_left(mask_color)[res.y0:, res.x0:]
            m = m[:int(round(res.disp.shape[0] / res.scale)), :int(round(res.disp.shape[1] / res.scale))]
            mask_s = cv2.resize(m, res.disp.shape[::-1], interpolation=cv2.INTER_NEAREST)
        out = []
        for i, p in enumerate(res.to_local(uv_l)):
            q = lookup_disparity(res.disp, p, mask_s, radius=self.radius * res.scale, min_gap=0.75)
            q.update(uv_color=uv_color[i].tolist(), uv_left=uv_l[i].tolist(), z=None, z_sigma=None, xyz_cam=None, ok=False)
            if q["disp"] is not None:
                z = float(c.disp_to_z(q["disp"]))
                q["z"] = z
                q["z_sigma"] = float(z * z / (c.fx * c.baseline) * max(q["mad_px"], 0.05))
                q["xyz_cam"] = c.left_uvz_to_color_xyz(uv_l[i], z)[0].tolist()
                # ok：有查到，而且和背景分得開（分不開代表拿到的多半是縫合墊的深度）
                q["ok"] = q["sep_px"] is None or q["sep_px"] >= 0.75
            out.append(q)
        return (out, res) if return_disp else out
