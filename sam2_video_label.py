#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 SAM 2 影片追蹤，快速標註連續影格中的「針」與「縫線」，輸出和學長資料相同格式的彩色 mask
（針 = BGR (0,0,255)，線 = BGR (255,0,255)，見 Color_code.txt）。另有畫筆 / 橡皮擦與復原（undo）。

畫面:
    左側按鈕列                         右側影格（點擊 = 對「目前的類別 / 物件」下提示或修改）
    ├ Class   Needle / Thread          選要標哪一類；每一類可以有多個物件（例如兩根針）
    ├ Object  < / > / + new            切換 / 新增目前類別的物件
    ├ Tool    SAM2 點 / SAM2 框 / 畫筆 / 橡皮擦、筆刷大小
    ├ Track   追蹤目前類別 / 全部追蹤
    ├ Edit    修好的 mask 給 SAM2、清除提示、清除手動修改、復原
    └ Frame   上一張 / 下一張 / ±10 張、排除這張、顯示 mask、存檔

流程（一次處理一段 clip）:
    1. 左側選 Needle，在影格上點針（左鍵 = 是，右鍵 = 不是；或用 SAM2 框）
    2. 按 Track（p）：以「目前這張影格畫面上的 mask」（SAM 結果 + 畫筆 − 橡皮擦）為起點，
       只往後追蹤到 clip 結尾。目前這張以及之前的影格完全不動；只追蹤目前類別（其他類別不動）
    3. 左側選 Thread，點線（從針尾接出來的那一段即可），再按 Track
    4. 用 d 往後翻；看到第一張追錯的影格，就在那張用點 / 畫筆 / 橡皮擦修好，再按 Track
       -> 從這張往後重新追蹤，前面已經修好的都保留。重複到 clip 結尾即可
       （被重新追蹤的物件：這張之後「SAM 追蹤出來」的舊結果會被取代；u 可以復原）
       「Keep labeled frames (t)」預設開啟：這張之後已經標好的影格（YOLO / Labels 讀回的 mask、手動畫過 / 擦過、
       下過 SAM 提示的）不會被覆蓋，還會當成 SAM 2 的參考影格；關掉 = 這張之後全部覆蓋
    5. Save（s）；q 存檔並離開

手動修改（畫筆 / 橡皮擦）存在獨立的一層：最終 mask = (SAM 2 結果 ∪ 畫筆) − 橡皮擦。
Track 時目前影格與之前的影格（含手動修改）都不會被改；之後的影格才會換成新的追蹤結果。
針與線重疊的像素算「針」（針優先）。

輸出（可直接給 mask_to_needle5.py 使用；mask_to_needle5.py 會用線自動判斷針尾）:
    <out_dir>/Images/<clip>/<原檔名>              原始影格（複製）
    <out_dir>/Labels/<clip>/<原檔名去副檔名>.png   mask（針 (0,0,255)、線 (255,0,255)，其餘黑色）
    <out_dir>/overlay/<clip>/...                   疊圖檢查
    <out_dir>/Color_code.txt
    <out_dir>/sessions/<clip>.json + <clip>_edits.npz   提示點、物件類別與手動修改（重開同一段 clip 自動讀回；
                                                         舊版 session 的物件一律視為針）

快捷鍵（按鈕都有對應的鍵）:
    1 / 2          類別：Needle / Thread          n  新增物件（目前類別）   [ / ]  切換物件
    f              SAM2 點（左鍵 + / 右鍵 -）     b  SAM2 框              w  畫筆（右鍵拖曳 = 擦）   e  橡皮擦
    + / -          筆刷大小                       z  放大 1x→2x→4x→8x（以滑鼠為中心）
    p              追蹤目前類別                   P (Shift+p)  全部類別一起追蹤
    g              這張修好的 mask 交給 SAM2      c  清除這張的 SAM 提示    k  清除這張的手動修改
    u / Ctrl+Z     復原（最多 50 步）             x  排除這張影格            m  顯示 / 隱藏 mask
    a / d  ← / →   上一張 / 下一張                j / l  ±10 張
    s              存檔                           q  存檔並離開
    F11            全螢幕切換                     0  影像回到原尺寸     Esc  同 q

視窗（預設 --gui tk，需要 tkinter：sudo apt-get install -y python3-tk）:
    - 一個視窗：左邊選項欄（真正的按鈕，字 --ui_font 16px）| 中間分隔線（拖曳調整選項欄寬度）| 右邊影像
    - 影像只能縮放、不能拖曳移動：滾輪 = 以滑鼠位置為中心放大 / 縮小，z = 1x→2x→4x→8x，0 = 回到原尺寸
      縮放時視窗大小不變；畫筆 / 橡皮擦 / 框的拖曳只會畫，不會移動畫面
    - 開啟時最大化；F11 / 「Full screen」按鈕切換真正的全螢幕（--window normal = 一般大小）
    - 沒有 tkinter 時自動改用舊的 OpenCV 介面（--gui cv2；Qt 內建的滾輪縮放 / 拖曳平移無法關閉）

重開已標過的 clip：直接顯示上次存檔的 mask（新版存在 sessions/<clip>_edits.npz；舊版從 Labels/<clip>/*.png 讀回），
                   不需要重新追蹤。只想補標線時，用「Track Thread only」，針的結果完全不會動。

需要（在 VRB container 內）:
    SAM2_BUILD_CUDA=0 pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/sam2.git
    pip install "hydra-core>=1.3.2" "iopath>=0.1.10"

用法:
    python sam2_video_label.py --frames_dir /data/.../rgb --out_dir my_needle_masks \\
        --start 3000 --num 150 --stride 3 --model small
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
# 類別：label png 的顏色（BGR）、color.txt 的 id、畫面上同類別不同物件的顏色
CLASSES = {
    "needle": {"name": "Needle", "bgr": (0, 0, 255), "id": 2,
               "shades": [(0, 0, 255), (0, 140, 255), (60, 60, 190), (0, 200, 255)]},
    "thread": {"name": "Thread", "bgr": (255, 0, 255), "id": 6,
               "shades": [(255, 0, 255), (255, 120, 180), (200, 80, 255), (160, 0, 160)]},
}
CLASS_ORDER = ["needle", "thread"]              # 存檔時後面的類別蓋過前面的 -> 針優先
CKPTS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
COLOR_CODE = """Background: 0, [0,0,0]

Tool shaft: 1, [255,0,0]

Needle: 2, [0,0,255]

Suture pad: 3, [0,255,0]

Tool end-effector: 4, [55,250,250]

Insertion points: 5, [209,125,42]

Thread: 6, [255,0,255]
"""
UNDO_MAX = 50
PANEL_W = 280                                   # 選項欄預設寬度（--panel_width；執行中可拖曳右邊界調整）
PANEL_MIN, PANEL_MAX = 200, 700
FONT_PX = 16                                    # 選項欄文字大小（--ui_font）
GRIP_W = 10                                     # 選項欄右邊可拖曳的把手寬度


# -----------------------------------------------------------------------------
def load_predictor(model, ckpt_dir):
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        sys.exit("[ERROR] 沒有 sam2，請在 container 內安裝（見檔案開頭說明）")
    name, cfg = CKPTS[model]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, name)
    if not os.path.isfile(ckpt):
        print(f"下載 {name} ...")
        urllib.request.urlretrieve(CKPT_URL + name, ckpt + ".part")
        os.replace(ckpt + ".part", ckpt)
    if not torch.cuda.is_available():
        sys.exit("[ERROR] 需要 GPU")
    return build_sam2_video_predictor(cfg, ckpt, device="cuda")


def pack(m):
    return None if m is None else (np.packbits(m, axis=None), m.shape)


def unpack(p):
    return None if p is None else np.unpackbits(p[0], count=p[1][0] * p[1][1]).reshape(p[1]).astype(bool)


class Session:
    """一段 clip 的所有狀態。
    obj_ids = [oid...]   obj_cls[oid] = "needle" / "thread"
    points[oid][t] = [(x, y, label)]   boxes[oid][t] = (x1,y1,x2,y2)   mask_prompts[oid][t] = bool mask
    sam[t][oid]    = SAM 2 輸出的 bool mask
    add[t][oid] / erase[t][oid] = 手動畫筆 / 橡皮擦（uint8 0/1）
    最終 mask = (sam | add) & ~erase
    """

    def __init__(self, predictor, jpg_dir, n_frames, H, W):
        self.pred = predictor
        self.H, self.W, self.n = H, W, n_frames
        if predictor is not None:
            import torch
            self.torch = torch
            with self._ctx():
                self.state = predictor.init_state(jpg_dir, offload_video_to_cpu=True)
        self.points, self.boxes, self.mask_prompts = {}, {}, {}
        self.sam, self.add, self.erase = {}, {}, {}
        self.obj_ids = []
        self.obj_cls = {}
        self.rebuilt = False
        self.need_reset = False
        self.trusted = {}           # f -> {oid}：已確認的 mask（YOLO / Labels 讀回、或你標過的），Track 不覆蓋
        self.protect = True         # Track 時保護已標好的影格

    # ---- 物件 ----
    def new_obj(self, cls):
        oid = max(self.obj_ids, default=0) + 1
        self.obj_ids.append(oid)
        self.obj_cls[oid] = cls
        return oid

    def objs_of(self, cls):
        return [o for o in self.obj_ids if self.obj_cls.get(o, "needle") == cls]

    # ---- SAM 2 呼叫 ----
    def _ctx(self):
        torch = self.torch

        class C:
            def __enter__(s):
                s.i = torch.inference_mode()
                s.a = torch.autocast("cuda", dtype=torch.bfloat16)
                s.i.__enter__()
                s.a.__enter__()

            def __exit__(s, *e):
                s.a.__exit__(*e)
                s.i.__exit__(*e)
        return C()

    def _store(self, f, obj_ids, logits, only=None):
        d = self.sam.setdefault(f, {})
        for oid, lg in zip(obj_ids, logits):
            if only is None or int(oid) in only:
                d[int(oid)] = (lg[0] > 0.0).cpu().numpy()

    def sync(self, f, oid):
        """依目前的點 / 框 / mask 提示，更新 SAM 2 在這張影格、這個物件的結果。
        點擊只看「這張影格」的提示（每張影格獨立預測），不會改到其他影格。
        追蹤過後 SAM 2 的內部狀態含有追蹤記憶，先重設再預測。"""
        if self.need_reset:
            self.reset()
        try:
            self._sync(f, oid)
        except RuntimeError as e:
            if "reset_state" not in str(e) and "after tracking starts" not in str(e):
                raise
            self.reset()
            self._sync(f, oid)

    def reset(self):
        """清空 SAM 2 的內部狀態（提示與追蹤記憶）。self.sam / 手動修改 都不受影響。"""
        with self._ctx():
            self.pred.reset_state(self.state)
        self.need_reset = False

    def _sync(self, f, oid):
        pts = self.points.get(oid, {}).get(f, [])
        box = self.boxes.get(oid, {}).get(f)
        mp = self.mask_prompts.get(oid, {}).get(f)
        with self._ctx():
            if pts or box is not None:
                kw = {}
                if pts:
                    kw["points"] = np.array([[p[0], p[1]] for p in pts], np.float32)
                    kw["labels"] = np.array([p[2] for p in pts], np.int32)
                if box is not None:
                    kw["box"] = np.array(box, np.float32)
                _, ids, logits = self.pred.add_new_points_or_box(self.state, frame_idx=f, obj_id=oid,
                                                                  clear_old_points=True, **kw)
                self._store(f, ids, logits, only={oid})     # 只更新這個物件：SAM 2 會回傳這張影格「所有物件」的
            elif mp is not None:                            # mask，其他物件（例如已標好的針）不能被覆蓋
                _, ids, logits = self.pred.add_new_mask(self.state, frame_idx=f, obj_id=oid, mask=mp)
                self._store(f, ids, logits, only={oid})
            else:
                try:
                    self.pred.clear_all_prompts_in_frame(self.state, f, oid, need_output=False)
                except Exception:
                    pass
                self.sam.get(f, {}).pop(oid, None)

    def prompted_frames(self, oids=None):
        fs = set()
        for src in (self.points, self.boxes, self.mask_prompts):
            for o, d in src.items():
                if oids is not None and o not in oids:
                    continue
                fs.update(k for k, v in d.items() if v is not None and (not isinstance(v, list) or v))
        return sorted(fs)

    def propagate_from(self, t, only=None, on_progress=None):
        """以影格 t「畫面上看到的最終 mask」（SAM 結果 + 畫筆 − 橡皮擦）當起點，只往後追蹤。
        - 影格 t 以及 t 之前的所有內容完全不動
        - only = 只追蹤這些物件；只有在影格 t 上 mask 不是空的物件才會被追蹤
        - 被追蹤的物件：t 之後的 SAM 結果換成新的，t 之後的手動修改與提示點清掉（舊的會和新結果對不上）
        回傳 (被追蹤的物件, undo 快照)；沒有可追蹤的物件時回傳 ([], None)。"""
        oids = [o for o in self.obj_ids if (only is None or o in only) and self.final(t, o).any()]
        if not oids or t >= self.n - 1:
            return [], None
        seeds = {o: self.final(t, o) for o in oids}
        sel = set(oids)
        # 已標好的影格（保護）：不覆蓋，而且當成 SAM 2 的額外條件影格（追蹤時會參考這些正確的 mask）
        prot = {(f, o) for f in range(t + 1, self.n) for o in oids if self.is_labeled(f, o)} if self.protect else set()

        # undo 快照：t 之後、這些物件的所有內容
        snap = {"kind": "track", "t": t, "oids": oids,
                "sam": {f: {o: pack(m) for o, m in d.items() if o in sel} for f, d in self.sam.items() if f > t},
                "add": {f: {o: m for o, m in d.items() if o in sel} for f, d in self.add.items() if f > t},
                "erase": {f: {o: m for o, m in d.items() if o in sel} for f, d in self.erase.items() if f > t},
                "prompts": {name: {o: {f: v for f, v in src.get(o, {}).items() if f > t} for o in oids}
                            for name, src in (("points", self.points), ("boxes", self.boxes),
                                              ("mask_prompts", self.mask_prompts))},
                "trusted": {f: set(v) for f, v in self.trusted.items()}}

        self.reset()                                            # 只用影格 t 的 mask 當條件，其他影格的提示不參與
        new, done = {}, 0
        with self._ctx():
            for o in oids:
                self.pred.add_new_mask(self.state, frame_idx=t, obj_id=o, mask=seeds[o])
            for f, o in sorted(prot):
                m = self.final(f, o)
                if m.any():
                    self.pred.add_new_mask(self.state, frame_idx=f, obj_id=o, mask=m)
            for f, ids, logits in self.pred.propagate_in_video(self.state, start_frame_idx=t, reverse=False):
                if f > t:
                    new[f] = {int(o): (lg[0] > 0.0).cpu().numpy() for o, lg in zip(ids, logits)}
                done += 1
                if on_progress and done % 10 == 0:
                    on_progress(done)
        self.need_reset = True                                  # 之後點擊前先清掉追蹤記憶

        for f in range(t + 1, self.n):                          # 寫回：只動 t 之後、被追蹤、沒被保護的物件
            d = self.sam.setdefault(f, {})
            for o in oids:
                if (f, o) in prot:
                    continue
                m = new.get(f, {}).get(o)
                if m is None:
                    d.pop(o, None)
                else:
                    d[o] = m
                for layer in (self.add, self.erase):
                    layer.get(f, {}).pop(o, None)
                for src in (self.points, self.boxes, self.mask_prompts):
                    src.get(o, {}).pop(f, None)
                self.trusted.get(f, set()).discard(o)
        snap["n_protected"] = len(prot)
        return oids, snap

    def is_labeled(self, f, o):
        """這張影格的這個物件是否「已標好」：YOLO / Labels 讀回的 mask、手動畫過 / 擦過、或下過 SAM 提示。"""
        if o in self.trusted.get(f, ()) and self.final(f, o).any():
            return True
        a, e = self.add.get(f, {}).get(o), self.erase.get(f, {}).get(o)
        if (a is not None and a.any()) or (e is not None and e.any()):
            return True
        return bool(self.points.get(o, {}).get(f) or self.boxes.get(o, {}).get(f) is not None
                    or self.mask_prompts.get(o, {}).get(f) is not None)

    def restore_track(self, s):
        t, oids = s["t"], set(s["oids"])
        self.trusted = {f: set(v) for f, v in s.get("trusted", {}).items()}
        for f in [f for f in self.sam if f > t]:
            for o in oids:
                self.sam[f].pop(o, None)
        for layer in (self.add, self.erase):
            for f in [f for f in layer if f > t]:
                for o in oids:
                    layer[f].pop(o, None)
        for f, d in s["sam"].items():
            self.sam.setdefault(f, {}).update({o: unpack(p) for o, p in d.items()})
        for name in ("add", "erase"):
            layer = getattr(self, name)
            for f, d in s[name].items():
                layer.setdefault(f, {}).update(d)
        for name, per_obj in s["prompts"].items():
            src = getattr(self, name)
            for o, d in per_obj.items():
                src.setdefault(o, {}).update(d)

    # ---- 手動修改 ----
    def layer(self, which, f, oid):
        d = (self.add if which == "add" else self.erase).setdefault(f, {})
        if oid not in d:
            d[oid] = np.zeros((self.H, self.W), np.uint8)
        return d[oid]

    def stroke(self, f, oid, p0, p1, radius, erase):
        a, e = self.layer("add", f, oid), self.layer("erase", f, oid)
        tmp = np.zeros((self.H, self.W), np.uint8)
        p0 = tuple(int(round(v)) for v in p0)
        p1 = tuple(int(round(v)) for v in p1)
        if radius == 0:
            cv2.line(tmp, p0, p1, 1, 1, lineType=cv2.LINE_8)
        else:
            cv2.line(tmp, p0, p1, 1, 2 * radius + 1, lineType=cv2.LINE_8)
            cv2.circle(tmp, p1, radius, 1, -1)
        sel = tmp > 0
        if erase:
            e[sel], a[sel] = 1, 0
        else:
            a[sel], e[sel] = 1, 0

    def final(self, f, oid):
        m = self.sam.get(f, {}).get(oid)
        m = np.zeros((self.H, self.W), bool) if m is None else m.copy()
        a = self.add.get(f, {}).get(oid)
        e = self.erase.get(f, {}).get(oid)
        if a is not None:
            m |= a > 0
        if e is not None:
            m &= ~(e > 0)
        return m

    def objs_on(self, f):
        return sorted(set(self.sam.get(f, {})) | {o for o, a in self.add.get(f, {}).items() if a.any()})

    def has_any(self, f, cls=None):
        return any(self.final(f, o).any() for o in self.objs_on(f)
                   if cls is None or self.obj_cls.get(o, "needle") == cls)

    def edited(self, f, oid):
        a, e = self.add.get(f, {}).get(oid), self.erase.get(f, {}).get(oid)
        return (a is not None and a.any()) or (e is not None and e.any())

    # ---- undo 快照 ----
    def snap_frame(self, f, oid):
        cp = lambda d: None if d is None else d.copy()
        return {"kind": "frame", "f": f, "oid": oid,
                "points": list(self.points.get(oid, {}).get(f, [])),
                "box": self.boxes.get(oid, {}).get(f),
                "mp": pack(self.mask_prompts.get(oid, {}).get(f)),
                "sam": pack(self.sam.get(f, {}).get(oid)),
                "add": cp(self.add.get(f, {}).get(oid)), "erase": cp(self.erase.get(f, {}).get(oid))}

    def restore_frame(self, s):
        f, oid = s["f"], s["oid"]

        def put(d, v):
            if v is None or (isinstance(v, list) and not v):
                d.setdefault(oid, {}).pop(f, None)
            else:
                d.setdefault(oid, {})[f] = v
        put(self.points, s["points"])
        put(self.boxes, s["box"])
        put(self.mask_prompts, unpack(s["mp"]))
        self.sync(f, oid)
        sam = unpack(s["sam"])
        if sam is None:
            self.sam.get(f, {}).pop(oid, None)
        else:
            self.sam.setdefault(f, {})[oid] = sam
        for d, v in ((self.add, s["add"]), (self.erase, s["erase"])):
            if v is None:
                d.get(f, {}).pop(oid, None)
            else:
                d.setdefault(f, {})[oid] = v

    def snap_all_sam(self):
        return {"kind": "all_sam", "sam": {f: {o: pack(m) for o, m in d.items()} for f, d in self.sam.items()}}

    def restore_all_sam(self, s):
        self.sam = {f: {o: unpack(p) for o, p in d.items()} for f, d in s["sam"].items()}


# -----------------------------------------------------------------------------
class View:
    """顯示用：縮放 + 放大鏡，負責畫面座標 <-> 原圖座標轉換（畫面座標不含左側按鈕列）。"""

    def __init__(self, W, H, disp_w):
        self.W, self.H = W, H
        self.ds = disp_w / W
        self.zf, self.cx, self.cy = 1, W / 2, H / 2

    @property
    def size(self):
        return int(self.W * self.ds), int(self.H * self.ds)

    def region(self):
        w, h = self.W / self.zf, self.H / self.zf
        x0 = min(max(self.cx - w / 2, 0), self.W - w)
        y0 = min(max(self.cy - h / 2, 0), self.H - h)
        return x0, y0, w, h

    def zoom_at(self, factor, x, y):
        """以原圖座標 (x, y) 為定點縮放（滑鼠下的那一點不動）。"""
        x0, y0, _, _ = self.region()
        s = self.ds * self.zf
        dx, dy = (x - x0) * s, (y - y0) * s                     # 該點在畫面上的位置
        self.zf = float(min(max(self.zf * factor, 1.0), 16.0))
        if self.zf < 1.01:
            self.zf = 1.0
        s2 = self.ds * self.zf
        w, h = self.W / self.zf, self.H / self.zf
        self.cx, self.cy = x - dx / s2 + w / 2, y - dy / s2 + h / 2

    def pan(self, ddx, ddy):
        """畫面上移動 (ddx, ddy) pixel。"""
        s = self.ds * self.zf
        x0, y0, w, h = self.region()
        self.cx, self.cy = x0 + w / 2 - ddx / s, y0 + h / 2 - ddy / s

    def to_img(self, mx, my):
        x0, y0, _, _ = self.region()
        s = self.ds * self.zf
        return x0 + mx / s, y0 + my / s

    def to_disp(self, x, y):
        x0, y0, _, _ = self.region()
        s = self.ds * self.zf
        return int(round((x - x0) * s)), int(round((y - y0) * s))

    def render(self, img):
        if self.zf <= 1:
            return cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        # 放大：用與 to_disp / to_img 完全相同的仿射轉換，畫面與點 / 筆刷位置不會差半格
        x0, y0, _, _ = self.region()
        s = self.ds * self.zf
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)   # 原圖像素 i 的中心 -> 畫面 (i - x0)·s
        return cv2.warpAffine(img, M, self.size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


# -----------------------------------------------------------------------------
class App:
    """標註介面：狀態、按鈕、滑鼠、鍵盤、畫面。與 SAM 2 無關的部分都在這裡（方便不開 GPU 測試畫面）。"""

    def __init__(self, sess, imgs, clip, view, save_fn, excluded=None):
        self.sess, self.imgs, self.clip, self.view = sess, imgs, clip, view
        self.save_fn = save_fn
        self.excluded = excluded if excluded is not None else set()
        self.H, self.W = imgs[0].shape[:2]
        cls0 = "needle"
        oid0 = (sess.objs_of(cls0) or [None])[0]
        self.st = {"t": 0, "cls": cls0, "oid": oid0, "last_oid": {}, "mode": "point", "drag": None,
                   "mouse": (self.W / 2, self.H / 2), "show": True, "radius": 2, "last": None, "hover": None,
                   "msg": "choose Needle / Thread on the left, click on it, then Track (p)", "quit": False}
        self.undo = []
        self.buttons = []
        self.canvas = None                                      # 影像視窗內部大小 (w, h)
        self.pw, self.ph = PANEL_W, 800                         # 選項欄視窗大小
        self.font_px = FONT_PX
        self.fill = False                                       # 兩個視窗填滿螢幕
        self.set_fill = lambda on: None                         # main() 會換成真正排視窗的函式
        self.on_panel_resize = lambda w: None                   # 拖曳選項欄邊界時呼叫（main() 會換掉）
        self.own_wheel = True                                   # 滾輪由本程式處理（Tk 介面）；OpenCV 備案介面由 Qt 處理

    @property
    def fs(self):                                               # Hershey 字型的 scale：大寫字高 ≈ 0.8 × font_px
        return self.font_px / 26.5

    def fit(self, win_w, win_h):
        """依影像視窗大小調整影像顯示比例：影像放到最大（保持長寬比）。"""
        if win_w < 100 or win_h < 100 or self.canvas == (win_w, win_h):
            return
        self.canvas = (win_w, win_h)
        disp_w = min(win_w, win_h * self.W / self.H)
        self.view.ds = max(disp_w, 50) / self.W

    def set_panel_width(self, w, notify=True):
        w = int(min(max(w, PANEL_MIN), PANEL_MAX))
        if w != self.pw:
            self.pw = w
            if notify:
                self.on_panel_resize(w)

    def toggle_fill(self):
        self.fill = not self.fill
        self.set_fill(self.fill)
        self.st["msg"] = "full screen (F11 to toggle)" if self.fill else "window mode (F11 = full screen)"

    # ---------------- 共用 ----------------
    def push(self, entry):
        self.undo.append(entry)
        if len(self.undo) > UNDO_MAX:
            self.undo.pop(0)

    def cur_oid(self, create=True):
        """目前操作的物件；目前類別還沒有物件時自動新增一個。"""
        st = self.st
        if st["oid"] is None or self.sess.obj_cls.get(st["oid"]) != st["cls"]:
            objs = self.sess.objs_of(st["cls"])
            st["oid"] = st["last_oid"].get(st["cls"]) if st["last_oid"].get(st["cls"]) in objs else \
                (objs[0] if objs else None)
        if st["oid"] is None and create:
            st["oid"] = self.sess.new_obj(st["cls"])
            self.push({"kind": "new_obj", "oid": st["oid"]})
        return st["oid"]

    def obj_color(self, o):
        cls = self.sess.obj_cls.get(o, "needle")
        k = self.sess.objs_of(cls).index(o) if o in self.sess.objs_of(cls) else 0
        sh = CLASSES[cls]["shades"]
        return sh[k % len(sh)]

    # ---------------- 動作（按鈕與快捷鍵共用） ----------------
    def set_class(self, cls):
        st = self.st
        if st["oid"] is not None:
            st["last_oid"][st["cls"]] = st["oid"]
        st["cls"], st["oid"] = cls, None
        self.cur_oid(create=False)
        n = len(self.sess.objs_of(cls))
        st["msg"] = f"class {CLASSES[cls]['name']}" + (f" ({n} object(s))" if n else ": click on it to start")

    def switch_obj(self, step):
        objs = self.sess.objs_of(self.st["cls"])
        if not objs:
            self.st["msg"] = "no object of this class yet"
            return
        o = self.cur_oid(create=False)
        i = objs.index(o) if o in objs else 0
        self.st["oid"] = objs[(i + step) % len(objs)]

    def new_obj(self):
        o = self.sess.new_obj(self.st["cls"])
        self.push({"kind": "new_obj", "oid": o})
        self.st["oid"] = o
        self.st["msg"] = f"new {CLASSES[self.st['cls']]['name']} #{len(self.sess.objs_of(self.st['cls']))}: click on it"

    def set_mode(self, mode):
        self.st["mode"] = mode
        self.st["drag"] = None
        self.st["msg"] = {"point": "SAM2: left click = +, right click = -",
                          "box": "SAM2 box: drag a box",
                          "brush": "BRUSH: left-drag paint, right-drag erase | wheel = zoom",
                          "eraser": "ERASER: left-drag erase | wheel = zoom"}[mode]

    def toggle_mode(self, mode):
        self.set_mode("point" if self.st["mode"] == mode else mode)

    def set_radius(self, d):
        self.st["radius"] = int(min(max(self.st["radius"] + d, 0), 30))

    def goto(self, d):
        self.st["t"] = int(min(max(self.st["t"] + d, 0), len(self.clip) - 1))

    def track_cls(self, cls):
        self.track(False, cls)

    def track(self, all_classes=False, cls=None):
        """從目前這張影格往後追蹤：以這張畫面上的 mask（包含手畫的）為準，
        這張以及之前的影格完全不動。only 模式只追蹤該類別的物件，其他類別完全不動。"""
        sess, st = self.sess, self.st
        cls = cls or st["cls"]
        t = st["t"]
        only = None if all_classes else set(sess.objs_of(cls))
        what = "all classes" if all_classes else CLASSES[cls]["name"]
        if t >= len(self.clip) - 1:
            st["msg"] = "this is the last frame: nothing to track after it"
            return
        st["msg"] = f"tracking {what} from frame {t + 1} to the end ..."
        self.redraw_now()
        t0 = time.time()
        try:
            oids, snap = sess.propagate_from(t, only, lambda n: print(f"  propagated {n} frames", end="\r"))
            if snap is not None:
                self.push(snap)
                npf = snap.get("n_protected", 0)
                st["msg"] = (f"tracked {len(oids)} {what} object(s) from frame {t + 1} in {time.time() - t0:.0f}s"
                             f" - frames 1..{t + 1} unchanged"
                             + (f"; {npf} labeled frame(s) after it kept" if sess.protect else
                                "; protect OFF: later frames overwritten") + "; u = undo")
            else:
                st["msg"] = f"no {what} mask on this frame: label it here first, then Track"
        except RuntimeError as e:
            sess.need_reset = True
            st["msg"] = f"tracking failed: {str(e)[:80]}"
        print()

    def toggle_protect(self):
        self.sess.protect = not self.sess.protect
        self.st["msg"] = ("Track keeps frames already labeled (YOLO / your edits / clicks) and uses them as anchors"
                          if self.sess.protect else
                          "Track overwrites ALL frames after the current one (including YOLO masks and your edits)")

    def to_sam(self):
        t, o = self.st["t"], self.cur_oid(create=False)
        m = self.sess.final(t, o) if o is not None else None
        if m is None or not m.any():
            self.st["msg"] = "mask is empty: nothing to send to SAM 2"
            return
        self.push(self.sess.snap_frame(t, o))
        self.sess.points.get(o, {}).pop(t, None)
        self.sess.boxes.get(o, {}).pop(t, None)
        self.sess.mask_prompts.setdefault(o, {})[t] = m
        self.sess.sync(t, o)
        self.st["msg"] = "(Track already uses the mask you see on this frame, Mask->SAM2 is optional)"

    def clear_prompts(self):
        t, o = self.st["t"], self.cur_oid(create=False)
        if o is None:
            return
        self.push(self.sess.snap_frame(t, o))
        self.sess.points.get(o, {}).pop(t, None)
        self.sess.boxes.get(o, {}).pop(t, None)
        self.sess.mask_prompts.get(o, {}).pop(t, None)
        self.sess.sync(t, o)
        self.st["msg"] = "cleared SAM prompts of this object on this frame (manual edits kept)"

    def clear_edits(self):
        t, o = self.st["t"], self.cur_oid(create=False)
        if o is None:
            return
        self.push(self.sess.snap_frame(t, o))
        self.sess.add.get(t, {}).pop(o, None)
        self.sess.erase.get(t, {}).pop(o, None)
        self.st["msg"] = "cleared manual edits of this object on this frame"

    def do_undo(self):
        st, sess = self.st, self.sess
        if not self.undo:
            st["msg"] = "nothing to undo"
            return
        e = self.undo.pop()
        if e["kind"] == "frame":
            sess.restore_frame(e)
            st["t"], st["oid"] = e["f"], e["oid"]
            st["cls"] = sess.obj_cls.get(e["oid"], st["cls"])
        elif e["kind"] == "all_sam":
            sess.restore_all_sam(e)
        elif e["kind"] == "track":
            sess.restore_track(e)
        elif e["kind"] == "exclude":
            self.excluded.symmetric_difference_update({e["f"]})
            st["t"] = e["f"]
        elif e["kind"] == "new_obj":
            o = e["oid"]
            if o in sess.obj_ids and not sess.prompted_frames({o}) and \
                    not any(o in d for d in sess.add.values()) and not any(o in d for d in sess.sam.values()):
                sess.obj_ids.remove(o)
                sess.obj_cls.pop(o, None)
                if st["oid"] == o:
                    st["oid"] = None
                    self.cur_oid(create=False)
        st["msg"] = f"undo ({len(self.undo)} left)"

    def exclude(self):
        self.excluded.symmetric_difference_update({self.st["t"]})
        self.push({"kind": "exclude", "f": self.st["t"]})

    def zoom(self):
        zf = self.view.zf
        nxt = 2.0 if zf < 1.5 else 4.0 if zf < 3 else 8.0 if zf < 6 else 1.0
        self.view.zf = nxt
        self.view.cx, self.view.cy = self.st["mouse"]

    def zoom_reset(self):
        self.view.zf, self.view.cx, self.view.cy = 1.0, self.W / 2, self.H / 2

    def save(self):
        self.st["msg"] = self.save_fn()

    def quit(self):
        self.save()
        self.st["quit"] = True

    # ---------------- 選項欄（獨立視窗，不受影像縮放影響） ----------------
    def panel_spec(self):
        """選項欄內容：[("head", 標題) / ("row", [(文字, 動作 or None, 是否選取, 色塊 or None), ...])]。
        Tk 介面與 OpenCV 備案介面共用；結構（列數 / 每列格數）固定，只有文字與選取狀態會變。"""
        st, sess = self.st, self.sess
        S = []
        S.append(("head", "CLASS"))
        for cls in CLASS_ORDER:
            n = len(sess.objs_of(cls))
            S.append(("row", [(f"{CLASSES[cls]['name']}  ({n})", lambda c=cls: self.set_class(c), st["cls"] == cls,
                               CLASSES[cls]["bgr"])]))
        objs = sess.objs_of(st["cls"])
        o = st["oid"]
        k = objs.index(o) + 1 if o in objs else 0
        S.append(("head", f"OBJECT  {k}/{len(objs)}"))
        S.append(("row", [("<", lambda: self.switch_obj(-1)), (">", lambda: self.switch_obj(1)),
                          ("+ new", self.new_obj)]))
        S.append(("head", "TOOL"))
        S.append(("row", [("SAM2 point (f)", lambda: self.set_mode("point"), st["mode"] == "point")]))
        S.append(("row", [("SAM2 box (b)", lambda: self.set_mode("box"), st["mode"] == "box")]))
        S.append(("row", [("Brush (w)", lambda: self.set_mode("brush"), st["mode"] == "brush"),
                          ("Eraser (e)", lambda: self.set_mode("eraser"), st["mode"] == "eraser")]))
        S.append(("row", [("-", lambda: self.set_radius(-1)), (f"r = {st['radius']}", None),
                          ("+", lambda: self.set_radius(1))]))
        S.append(("head", "TRACK"))
        S.append(("row", [("Track Needle only", lambda: self.track_cls("needle"))]))
        S.append(("row", [("Track Thread only", lambda: self.track_cls("thread"))]))
        S.append(("row", [("Track all (P)", lambda: self.track(True))]))
        S.append(("row", [("Keep labeled frames (t)", self.toggle_protect, sess.protect)]))
        S.append(("head", "EDIT"))
        S.append(("row", [("Mask -> SAM2 (g)", self.to_sam)]))
        S.append(("row", [("Clr prompt", self.clear_prompts), ("Clr edits", self.clear_edits)]))
        S.append(("row", [("Undo (u)", self.do_undo)]))
        S.append(("head", "FRAME"))
        S.append(("row", [("<<", lambda: self.goto(-10)), ("<", lambda: self.goto(-1)), (">", lambda: self.goto(1)),
                          (">>", lambda: self.goto(10))]))
        S.append(("row", [("Exclude (x)", self.exclude, st["t"] in self.excluded),
                          ("Mask (m)", lambda: self.st.update(show=not self.st["show"]), st["show"])]))
        S.append(("row", [("Zoom (z)", self.zoom, self.view.zf > 1), ("Fit (0)", self.zoom_reset)]))
        S.append(("row", [("Save (s)", self.save), ("Full screen (F11)", self.toggle_fill, self.fill)]))
        out = []
        for kind, v in S:
            if kind == "row":
                v = [tuple(list(c) + [False, None][len(c) - 2:]) for c in v]
            out.append((kind, v))
        return out

    def layout(self):
        """OpenCV 備案介面用：把 panel_spec 排成按鈕方塊。回傳 (按鈕, 標題, 內容總高度)。"""
        B, heads = [], []
        f = self.font_px
        x0, gap = 10, max(4, f // 4)
        w = self.pw - x0 - GRIP_W - 6
        h = int(round(f * 1.9))
        y = 10
        for kind, v in self.panel_spec():
            if kind == "head":
                heads.append((y + f, v))
                y += int(f * 1.6)
                continue
            n = len(v)
            cw = (w - gap * (n - 1)) / n
            for i, (label, act, active, chip) in enumerate(v):
                B.append(((int(x0 + i * (cw + gap)), y, int(cw), h), label, act, active, chip))
            y += h + gap
        return B, heads, y

    def wrap(self, text, width):
        """依像素寬度換行（選項欄下方的狀態文字）。"""
        out, line = [], ""
        for word in text.split(" "):
            cand = (line + " " + word).strip()
            if cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, self.fs * 0.9, 1)[0][0] <= width or not line:
                line = cand
            else:
                out.append(line)
                line = word
        if line:
            out.append(line)
        return out

    def status_lines(self):
        st, sess = self.st, self.sess
        t, cur = st["t"], self.cur_oid(create=False)
        objs_c = sess.objs_of(st["cls"])
        k = objs_c.index(cur) + 1 if cur in objs_c else 0
        n_n = sum(1 for i in range(len(self.clip)) if i not in self.excluded and sess.has_any(i, "needle"))
        n_t = sum(1 for i in range(len(self.clip)) if i not in self.excluded and sess.has_any(i, "thread"))
        mode_txt = {"point": "SAM2 point", "box": "SAM2 box", "brush": f"Brush r={st['radius']}",
                    "eraser": f"Eraser r={st['radius']}"}[st["mode"]]
        warn = (0, 128, 255)
        return [
            (f"frame {t + 1} / {len(self.clip)}" + ("   EXCLUDED" if t in self.excluded else ""),
             warn if t in self.excluded else (0, 255, 255)),
            (self.clip[t], (200, 200, 200)),
            (f"{CLASSES[st['cls']]['name']} {k}/{len(objs_c)}   {mode_txt}"
             + ("   EDITED" if cur is not None and sess.edited(t, cur) else ""), (255, 255, 255)),
            (f"frames: needle {n_n} / thread {n_t}", (200, 200, 200)),
            (f"zoom {self.view.zf:.1f}x   undo {len(self.undo)}", (200, 200, 200)),
        ]

    def render_panel(self):
        B, heads, y_end = self.layout()
        self.buttons = B
        pw, ph = self.pw, max(self.ph, 200)
        fs = self.fs
        th = max(1, int(round(self.font_px / 14)))              # 字的粗細
        p = np.full((ph, pw, 3), 32, np.uint8)
        for yy, txt in heads:
            cv2.putText(p, txt, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (150, 150, 150), 1, cv2.LINE_AA)
        for (x, y, w, h), label, act, active, chip in B:
            hov = self.st["hover"] is not None and x <= self.st["hover"][0] < x + w and y <= self.st["hover"][1] < y + h
            bg = (95, 95, 95) if active else ((72, 72, 72) if hov and act else (54, 54, 54))
            if act is None:
                bg = (32, 32, 32)
            cv2.rectangle(p, (x, y), (x + w, y + h), bg, -1)
            if active:
                cv2.rectangle(p, (x, y), (x + w, y + h), (0, 220, 255), 2)
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx = x + (w - tw) // 2
            if chip is not None:
                cs = int(h * 0.45)
                cv2.rectangle(p, (x + 8, y + (h - cs) // 2), (x + 8 + cs, y + (h + cs) // 2), chip, -1)
                tx = x + 16 + cs
            ty = y + (h + tht) // 2
            cv2.putText(p, label, (max(tx, x + 4), ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255) if act or active else (200, 200, 200), th, cv2.LINE_AA)
        # 狀態文字（按鈕下方）
        y = y_end + int(self.font_px * 0.8)
        cv2.line(p, (10, y - int(self.font_px * 0.4)), (pw - GRIP_W - 6, y - int(self.font_px * 0.4)), (70, 70, 70), 1)
        lh = int(self.font_px * 1.45)
        lines = []
        for txt, col in self.status_lines():
            lines += [(ln, col) for ln in self.wrap(txt, pw - GRIP_W - 22)]
        lines += [(ln, (160, 255, 160)) for ln in self.wrap(self.st["msg"], pw - GRIP_W - 22)]
        for txt, col in lines:
            y += lh
            if y > ph - 4:
                break
            cv2.putText(p, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9, col, 1, cv2.LINE_AA)
        # 右邊的拖曳把手
        gx = pw - GRIP_W
        p[:, gx:] = (48, 48, 48) if not self.st.get("grip") else (80, 80, 80)
        for yy in range(ph // 2 - 30, ph // 2 + 31, 10):
            cv2.circle(p, (gx + GRIP_W // 2, yy), 2, (150, 150, 150), -1, cv2.LINE_AA)
        return p

    # ---------------- 影像視窗 ----------------
    def render(self):
        st, sess, view = self.st, self.sess, self.view
        t = st["t"]
        cur = self.cur_oid(create=False)
        img = self.imgs[t].copy()
        objs = sess.objs_on(t)
        finals = {o: sess.final(t, o) for o in objs} if st["show"] else {}
        # 非目前類別淡一點；先畫線再畫針
        order = sorted(finals, key=lambda o: CLASS_ORDER.index(sess.obj_cls.get(o, "needle")), reverse=True)
        for o in order:
            bm = finals[o]
            a = 0.55 if sess.obj_cls.get(o, "needle") == st["cls"] else 0.3
            img[bm] = ((1 - a) * img[bm] + a * np.array(self.obj_color(o))).astype(np.uint8)
        disp = view.render(img)
        for o in order:                                         # 細物件：再描一圈輪廓（目前物件加粗）
            cs, _ = cv2.findContours(finals[o].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cc in cs:
                pts = np.array([view.to_disp(q[0][0], q[0][1]) for q in cc], np.int32)
                cv2.polylines(disp, [pts], True, self.obj_color(o), 2 if o == cur else 1, cv2.LINE_AA)
        for o in sess.obj_ids:                                  # 提示點 / 框（只有目前物件的點是實心大點）
            big = o == cur
            for (x, y, lab) in sess.points.get(o, {}).get(t, []):
                q = view.to_disp(x, y)
                if lab:
                    cv2.circle(disp, q, 5 if big else 3, (0, 255, 0), -1 if big else 1, cv2.LINE_AA)
                else:
                    cv2.drawMarker(disp, q, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10 if big else 7, 2)
            bx = sess.boxes.get(o, {}).get(t)
            if bx:
                cv2.rectangle(disp, view.to_disp(bx[0], bx[1]), view.to_disp(bx[2], bx[3]),
                              (0, 255, 0), 2 if big else 1)
        if st["drag"]:
            d = st["drag"]
            cv2.rectangle(disp, view.to_disp(d[0], d[1]), view.to_disp(d[2], d[3]), (0, 255, 255), 1)
        if st["mode"] in ("brush", "eraser"):                   # 筆刷游標
            s_px = view.ds * view.zf
            r_disp = max(1, int(round((st["radius"] + 0.5) * s_px)))
            cv2.circle(disp, view.to_disp(*st["mouse"]), r_disp,
                       (255, 255, 255) if st["mode"] == "brush" else (0, 0, 0), 1, cv2.LINE_AA)
        if self.canvas is not None:                             # 補滿影像視窗（右 / 下補深灰），OpenCV 不再縮放 -> 不會模糊
            Wc, Hc = max(self.canvas[0], disp.shape[1]), max(self.canvas[1], disp.shape[0])
            out = np.full((Hc, Wc, 3), 24, np.uint8)
            out[:disp.shape[0], :disp.shape[1]] = disp
            disp = out
        return disp

    def redraw_now(self):
        pass                                                    # main() 會換成真正的 imshow

    # ---------------- 滑鼠 ----------------
    def _after(self):
        pass

    def on_panel_mouse(self, ev, mx, my, flags, _=None):
        """選項欄視窗：按鈕、右邊把手拖曳調整寬度。"""
        st = self.st
        st["hover"] = (mx, my)
        if st.get("grip"):                                      # 拖曳中（滑鼠可能已經跑到視窗外）
            if ev == cv2.EVENT_MOUSEMOVE:
                self.set_panel_width(mx + GRIP_W // 2)
            elif ev == cv2.EVENT_LBUTTONUP:
                st["grip"] = False
                self.set_panel_width(mx + GRIP_W // 2)
            return
        if ev == cv2.EVENT_LBUTTONDOWN:
            if mx >= self.pw - GRIP_W:
                st["grip"] = True
                return
            for (x, y, w, h), label, act, active, chip in self.buttons:
                if act is not None and x <= mx < x + w and y <= my < y + h:
                    act()
                    break
            self._after()

    def on_mouse(self, ev, mx, my, flags, _=None):
        """影像視窗。"""
        self.st["hover"] = None
        self._on_mouse(ev, mx, my, flags)
        self._after()

    def _on_mouse(self, ev, mx, my, flags):
        st, sess, view = self.st, self.sess, self.view
        # 滾輪 = 以滑鼠位置為中心放大 / 縮小影像（視窗大小不變；沒有任何拖曳移動畫面的功能）
        if ev == cv2.EVENT_MOUSEWHEEL:
            if self.own_wheel:
                x, y = view.to_img(mx, my)
                delta = (flags >> 16) & 0xFFFF                 # = cv2.getMouseWheelDelta（新舊版本都可用）
                delta = delta - 0x10000 if delta >= 0x8000 else delta
                view.zoom_at(1.25 if delta > 0 else 0.8, x, y)
            return
        dw, dh = view.size
        if my >= dh or mx >= dw or mx < 0 or my < 0:
            if ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                st["last"], st["drag"] = None, None
            return
        x, y = view.to_img(mx, my)
        x, y = min(max(x, 0), self.W - 1), min(max(y, 0), self.H - 1)
        st["mouse"] = (x, y)
        t = st["t"]

        if st["mode"] in ("brush", "eraser"):
            if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                oid = self.cur_oid()
                self.push(sess.snap_frame(t, oid))
                erase = st["mode"] == "eraser" or ev == cv2.EVENT_RBUTTONDOWN
                st["last"] = ((x, y), erase)
                sess.stroke(t, oid, (x, y), (x, y), st["radius"], erase)
            elif ev == cv2.EVENT_MOUSEMOVE and st["last"] and (flags & (cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_RBUTTON)):
                (px, py), erase = st["last"]
                sess.stroke(t, self.cur_oid(), (px, py), (x, y), st["radius"], erase)
                st["last"] = ((x, y), erase)
            elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                st["last"] = None
            return

        if st["mode"] == "box":
            if ev == cv2.EVENT_LBUTTONDOWN:
                st["drag"] = [x, y, x, y]
            elif ev == cv2.EVENT_MOUSEMOVE and st["drag"]:
                st["drag"][2:] = [x, y]
            elif ev == cv2.EVENT_LBUTTONUP and st["drag"]:
                x1, y1, x2, y2 = st["drag"]
                st["drag"] = None
                if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                    oid = self.cur_oid()
                    self.push(sess.snap_frame(t, oid))
                    sess.boxes.setdefault(oid, {})[t] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    sess.mask_prompts.get(oid, {}).pop(t, None)
                    sess.sync(t, oid)
                    st["mode"] = "point"
                    st["msg"] = "box added; add +/- points if needed, then Track (p)"
            return

        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            oid = self.cur_oid()
            self.push(sess.snap_frame(t, oid))
            lab = 1 if ev == cv2.EVENT_LBUTTONDOWN else 0
            sess.points.setdefault(oid, {}).setdefault(t, []).append((float(x), float(y), lab))
            sess.mask_prompts.get(oid, {}).pop(t, None)
            sess.sync(t, oid)
            st["msg"] = f"{'+' if lab else '-'} point on {CLASSES[st['cls']]['name']}"

    # ---------------- 鍵盤 ----------------
    def on_key(self, k):
        if k == -1:
            return
        c = k & 0xFF
        actions = {
            ord("1"): lambda: self.set_class("needle"), ord("2"): lambda: self.set_class("thread"),
            ord("d"): lambda: self.goto(1), ord("a"): lambda: self.goto(-1),
            ord("l"): lambda: self.goto(10), ord("j"): lambda: self.goto(-10),
            ord("f"): lambda: self.set_mode("point"), ord("b"): lambda: self.toggle_mode("box"),
            ord("w"): lambda: self.toggle_mode("brush"), ord("e"): lambda: self.toggle_mode("eraser"),
            ord("+"): lambda: self.set_radius(1), ord("="): lambda: self.set_radius(1),
            ord("-"): lambda: self.set_radius(-1), ord("_"): lambda: self.set_radius(-1),
            ord("k"): self.clear_edits, ord("g"): self.to_sam, ord("c"): self.clear_prompts,
            ord("u"): self.do_undo, 26: self.do_undo,
            ord("n"): self.new_obj, ord("]"): lambda: self.switch_obj(1), ord("["): lambda: self.switch_obj(-1),
            ord("z"): self.zoom, ord("0"): self.zoom_reset, ord("x"): self.exclude,
            ord("m"): lambda: self.st.update(show=not self.st["show"]),
            ord("p"): lambda: self.track(False), ord("P"): lambda: self.track(True), ord("t"): self.toggle_protect,
            ord("s"): self.save, ord("q"): self.quit, 27: self.quit,
        }
        if k in (65480, 7995392, 0x7A0000):                     # F11（Linux GTK / Windows）
            self.toggle_fill()
            return
        if k in (65363, 2555904):                               # → / ←
            self.goto(1)
        elif k in (65361, 2424832):
            self.goto(-1)
        elif k == 26 or c in actions:
            actions[26 if k == 26 else c]()


# -----------------------------------------------------------------------------
def save_outputs(sess, imgs, clip, excluded, a, clip_name, sess_path, edits_path):
    H, W = imgs[0].shape[:2]
    for sub in ("Images", "Labels", "overlay"):
        os.makedirs(os.path.join(a.out_dir, sub, clip_name), exist_ok=True)
    os.makedirs(os.path.dirname(sess_path), exist_ok=True)
    with open(os.path.join(a.out_dir, "Color_code.txt"), "w") as fcc:
        fcc.write(COLOR_CODE)
    n_saved = {"needle": 0, "thread": 0}
    for i, f in enumerate(clip):
        stem = os.path.splitext(f)[0]
        lab = os.path.join(a.out_dir, "Labels", clip_name, stem + ".png")
        img_dst = os.path.join(a.out_dir, "Images", clip_name, f)
        ov_dst = os.path.join(a.out_dir, "overlay", clip_name, stem + ".png")
        if i in excluded or not sess.has_any(i):
            for pth in (lab, img_dst, ov_dst):                  # 被排除 / 沒有 mask 的影格不輸出
                if os.path.exists(pth):
                    os.remove(pth)
            continue
        m = np.zeros((H, W, 3), np.uint8)
        ov = imgs[i].copy()
        for cls in reversed(CLASS_ORDER):                       # 線先畫、針後畫 -> 重疊處是針
            sel = np.zeros((H, W), bool)
            for o in sess.objs_on(i):
                if sess.obj_cls.get(o, "needle") == cls:
                    sel |= sess.final(i, o)
            if sel.any():
                n_saved[cls] += 1
                m[sel] = CLASSES[cls]["bgr"]
                ov[sel] = (0.4 * ov[sel] + 0.6 * np.array(CLASSES[cls]["bgr"])).astype(np.uint8)
        cv2.imwrite(lab, m)
        if not os.path.exists(img_dst):
            shutil.copy2(os.path.join(a.frames_dir, f), img_dst)
        cv2.imwrite(ov_dst, ov)
    with open(sess_path, "w") as fs:
        json.dump({"frames_dir": os.path.abspath(a.frames_dir), "frames": clip, "stride": a.stride,
                   "obj_ids": sess.obj_ids, "obj_cls": {str(o): c for o, c in sess.obj_cls.items()},
                   "excluded": sorted(excluded),
                   "trusted": {str(f): sorted(v) for f, v in sess.trusted.items() if v},
                   "points": {str(o): {str(k): v for k, v in d.items()} for o, d in sess.points.items()},
                   "boxes": {str(o): {str(k): list(v) for k, v in d.items()} for o, d in sess.boxes.items()}},
                  fs, indent=1)
    arrs = {}
    for kind, src in (("add", sess.add), ("erase", sess.erase)):
        for f, d in src.items():
            for o, arr in d.items():
                if arr.any():
                    arrs[f"{kind}_{f}_{o}"] = arr
    for o, d in sess.mask_prompts.items():
        for f, arr in d.items():
            arrs[f"mp_{f}_{o}"] = arr.astype(np.uint8)
    for f, d in sess.sam.items():                               # 追蹤結果（重開時直接讀回，不用重新追蹤）
        for o, m in d.items():
            if m is not None and m.any():
                arrs[f"sam_{f}_{o}"] = np.packbits(m, axis=None)
    np.savez_compressed(edits_path, **arrs)
    msg = f"saved: {n_saved['needle']} frames with needle, {n_saved['thread']} with thread -> {a.out_dir}"
    print(msg)
    return msg


def load_session(sess, sess_path, edits_path, clip, labels_dir=None):
    """讀回之前的提示、物件類別、手動修改與追蹤結果；回傳 excluded。舊版 session（沒有 obj_cls）的物件一律視為針。
    追蹤結果：session 有存（新版）就直接用；沒有（舊版）就從 Labels/<clip>/*.png 依顏色讀回
    （同一類別有多個物件時，整個類別的 mask 先放在該類別第一個物件上）。"""
    excluded = set()
    if not os.path.isfile(sess_path):
        return excluded
    old = json.load(open(sess_path))
    if old.get("frames") != clip:
        print("[NOTE] 已存在的 session 影格清單不同（--start / --num / --stride 改過），不讀回")
        return excluded
    excluded = set(old.get("excluded", []))
    sess.obj_ids = [int(o) for o in old.get("obj_ids", [1])]
    oc = old.get("obj_cls", {})
    sess.obj_cls = {o: oc.get(str(o), "needle") for o in sess.obj_ids}
    for oid, d in old.get("points", {}).items():
        sess.points[int(oid)] = {int(k): [tuple(p) for p in v] for k, v in d.items()}
    for oid, d in old.get("boxes", {}).items():
        sess.boxes[int(oid)] = {int(k): tuple(v) for k, v in d.items()}
    saved_sam = {}
    if os.path.isfile(edits_path):
        z = np.load(edits_path)
        for key in z.files:                                     # add_ / erase_ / mp_ / sam_<f>_<oid>
            kind, f, oid = key.split("_")
            arr = z[key]
            if kind == "mp":
                sess.mask_prompts.setdefault(int(oid), {})[int(f)] = arr.astype(bool)
            elif kind == "sam":
                saved_sam.setdefault(int(f), {})[int(oid)] = \
                    np.unpackbits(arr, count=sess.H * sess.W).reshape(sess.H, sess.W).astype(bool)
            else:
                (sess.add if kind == "add" else sess.erase).setdefault(int(f), {})[int(oid)] = arr
    for oid in sess.obj_ids:
        for f in (set(sess.points.get(oid, {})) | set(sess.boxes.get(oid, {})) | set(sess.mask_prompts.get(oid, {}))):
            sess.sync(f, oid)
    src = ""
    sess.trusted = {int(f): set(v) for f, v in old.get("trusted", {}).items()}
    if saved_sam:
        sess.sam = saved_sam
        src = "session 裡的追蹤結果"
    elif labels_dir and os.path.isdir(labels_dir):
        n_lab = 0
        for i, fn in enumerate(clip):
            lab = cv2.imread(os.path.join(labels_dir, os.path.splitext(fn)[0] + ".png"))
            if lab is None or lab.shape[:2] != (sess.H, sess.W):
                continue
            n_lab += 1
            d = sess.sam.setdefault(i, {})
            for cls in CLASS_ORDER:
                objs = sess.objs_of(cls)
                if not objs:
                    continue
                m = np.all(np.abs(lab.astype(np.int16) - np.array(CLASSES[cls]["bgr"], np.int16)) <= 10, axis=2)
                # 存檔時 final = (sam ∪ 畫筆) − 橡皮擦，所以 png 已經包含手動修改；直接當作 sam 結果即可
                d[objs[0]] = m
                for o in objs[1:]:
                    d.pop(o, None)
                if m.any():
                    sess.trusted.setdefault(i, set()).add(objs[0])   # 從 Labels（例如 YOLO）讀回的 = 已標好
        if n_lab:
            src = f"Labels 資料夾的 {n_lab} 張 mask"
            if any(len(sess.objs_of(c)) > 1 for c in CLASS_ORDER):
                print("  [NOTE] 同一類別有多個物件：舊版 Labels 分不出是哪一個，先全部放在該類別第一個物件上")
    n = {c: len(sess.objs_of(c)) for c in CLASS_ORDER}
    print(f"已讀回之前的提示與手動修改（{len(sess.prompted_frames())} 張有提示；needle {n['needle']} / "
          f"thread {n['thread']} 個物件）"
          + (f"，mask 來自{src}（不需要重新追蹤）" if src else "，按 Track 重新追蹤"))
    return excluded


def run_cv2(app, a):
    """備案介面：OpenCV（Qt）兩個視窗。注意 Qt 內建的滾輪縮放 / 拖曳平移無法關閉。"""
    win, pwin = "SAM2 labeling - image", "SAM2 labeling - tools"
    cv2.namedWindow(pwin, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(pwin, app.on_panel_mouse)
    cv2.setMouseCallback(win, app.on_mouse)

    def rect(w):
        try:
            x, y, ww, hh = cv2.getWindowImageRect(w)
            return (x, y, ww, hh) if ww > 0 and hh > 0 else None
        except Exception:                                       # 舊版 OpenCV 沒有這個函式
            return None

    def screen_size():
        if a.screen:
            sw, sh = a.screen.lower().split("x")
            return int(sw), int(sh)
        try:
            import tkinter
            r = tkinter.Tk()
            r.withdraw()
            wh = r.winfo_screenwidth(), r.winfo_screenheight()
            r.destroy()
            if wh[0] > 0:
                return wh
        except Exception:
            pass
        cv2.imshow(win, app.render())                           # 備案：暫時全螢幕量一次
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        for _ in range(10):
            cv2.waitKey(30)
        r = rect(win)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.waitKey(30)
        return (r[2], r[3]) if r else (1920, 1080)

    SW, SH = screen_size()
    print(f"螢幕 {SW}x{SH}；選項欄寬 {app.pw}px（拖曳右邊把手可調整）")
    usable_h = SH - a.screen_margin
    gap = 12                                                    # 兩個視窗之間預留給視窗邊框的空間

    img_pos = [app.pw + gap, 0]

    def place():
        app.ph = usable_h
        cv2.resizeWindow(pwin, app.pw, usable_h)
        cv2.moveWindow(pwin, 0, 0)
        cv2.resizeWindow(win, max(SW - app.pw - gap, 200), usable_h)
        img_pos[:] = [app.pw + gap, 0]
        cv2.moveWindow(win, *img_pos)
        app.canvas = None

    def set_fill(on):
        if on:
            place()
    app.set_fill = set_fill

    def on_panel_resize(w):
        cv2.resizeWindow(pwin, w, app.ph)
        if app.fill:
            cv2.resizeWindow(win, max(SW - w - gap, 200), usable_h)
            img_pos[:] = [w + gap, 0]
            cv2.moveWindow(win, *img_pos)
            app.canvas = None
    app.on_panel_resize = on_panel_resize

    cv2.imshow(pwin, app.render_panel())
    cv2.imshow(win, app.render())
    vw, vh = app.view.size
    cv2.resizeWindow(pwin, app.pw, min(usable_h, max(vh, 700)))
    cv2.resizeWindow(win, vw, vh)
    cv2.moveWindow(win, app.pw + gap, 0)
    if a.window == "fullscreen":
        app.toggle_fill()
    cv2.waitKey(50)

    def redraw_now():
        cv2.imshow(pwin, app.render_panel())
        cv2.imshow(win, app.render())
        cv2.waitKey(1)
    app.redraw_now = redraw_now

    if True:
        while not app.st["quit"]:
            r = rect(win)
            if r:
                app.fit(r[2], r[3])
            r = rect(pwin)
            if r and not app.st.get("grip"):
                app.ph = r[3]
                if abs(r[2] - app.pw) > 2:                      # 使用者拖曳了選項欄視窗的原生邊框
                    app.set_panel_width(r[2])
            cv2.imshow(pwin, app.render_panel())
            cv2.imshow(win, app.render())
            app.on_key(cv2.waitKeyEx(15))
    cv2.destroyAllWindows()


class TkGUI:
    """主介面：一個 Tk 視窗 = 左邊選項欄（真正的按鈕）| 可拖曳的分隔線 | 右邊影像。
    影像只會用滾輪 / z 鍵縮放（以滑鼠為中心），沒有任何拖曳移動畫面的功能；縮放時視窗大小不變。"""

    def __init__(self, app, a):
        import tkinter as tk
        self.tk, self.app, self.a = tk, app, a
        try:
            from PIL import Image, ImageTk
            self.Image, self.ImageTk = Image, ImageTk
        except Exception:
            self.Image = self.ImageTk = None
        root = tk.Tk()
        root.title("SAM2 needle / thread labeling")
        root.configure(bg="#202020")
        self.root = root
        fp = a.ui_font
        self.font = ("DejaVu Sans", -fp)
        self.font_head = ("DejaVu Sans", -max(10, fp - 3), "bold")
        self.font_stat = ("DejaVu Sans", -max(10, fp - 2))

        pw = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=8, sashrelief=tk.RAISED,
                            bg="#3c3c3c", bd=0, opaqueresize=True)
        pw.pack(fill=tk.BOTH, expand=True)
        self.panel = tk.Frame(pw, bg="#202020")
        self.canvas = tk.Canvas(pw, bg="#181818", highlightthickness=0, cursor="crosshair")
        pw.add(self.panel, minsize=PANEL_MIN, width=app.pw)
        pw.add(self.canvas, minsize=200)
        self.chips = {}
        self.cells = []                                     # 每個按鈕 / 標籤（與 panel_spec 攤平後的順序相同）
        self.heads = []
        self.build_panel()
        self.img_item = self.canvas.create_image(0, 0, anchor="nw")
        self.photo = None
        self.pending = False

        c = self.canvas
        E = cv2
        c.bind("<ButtonPress-1>", lambda e: self.mouse(E.EVENT_LBUTTONDOWN, e, E.EVENT_FLAG_LBUTTON))
        c.bind("<ButtonPress-3>", lambda e: self.mouse(E.EVENT_RBUTTONDOWN, e, E.EVENT_FLAG_RBUTTON))
        c.bind("<B1-Motion>", lambda e: self.mouse(E.EVENT_MOUSEMOVE, e, E.EVENT_FLAG_LBUTTON))
        c.bind("<B3-Motion>", lambda e: self.mouse(E.EVENT_MOUSEMOVE, e, E.EVENT_FLAG_RBUTTON))
        c.bind("<Motion>", lambda e: self.mouse(E.EVENT_MOUSEMOVE, e, 0))
        c.bind("<ButtonRelease-1>", lambda e: self.mouse(E.EVENT_LBUTTONUP, e, 0))
        c.bind("<ButtonRelease-3>", lambda e: self.mouse(E.EVENT_RBUTTONUP, e, 0))
        c.bind("<Button-4>", lambda e: self.wheel(e, 1))                      # Linux X11 滾輪
        c.bind("<Button-5>", lambda e: self.wheel(e, -1))
        c.bind("<MouseWheel>", lambda e: self.wheel(e, 1 if e.delta > 0 else -1))   # Windows / macOS
        c.bind("<Configure>", lambda e: self.schedule())
        self.panel.bind("<Configure>", lambda e: self.on_panel_size(e.width))
        root.bind_all("<KeyPress>", self.key)
        root.protocol("WM_DELETE_WINDOW", self.close)

        app.set_fill = lambda on: root.attributes("-fullscreen", bool(on))
        app.redraw_now = self.redraw_now
        if a.window == "fullscreen":
            try:
                root.attributes("-zoomed", True)            # Linux：最大化（保留標題列；F11 = 真正全螢幕）
            except Exception:
                try:
                    root.state("zoomed")                    # Windows
                except Exception:
                    pass
        else:
            vw, vh = app.view.size
            root.geometry(f"{app.pw + vw + 12}x{max(vh, 760)}")
        self.schedule()

    # ---------------- 選項欄 ----------------
    def chip(self, bgr):
        if bgr not in self.chips:
            im = self.tk.PhotoImage(width=14, height=14)
            im.put("#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0]), to=(0, 0, 14, 14))
            self.chips[bgr] = im
        return self.chips[bgr]

    def build_panel(self):
        tk = self.tk
        pad = max(3, self.a.ui_font // 5)
        for kind, v in self.app.panel_spec():
            if kind == "head":
                lb = tk.Label(self.panel, text=v, font=self.font_head, bg="#202020", fg="#9a9a9a", anchor="w")
                lb.pack(fill=tk.X, padx=10, pady=(pad * 2, 1))
                self.heads.append(lb)
                continue
            fr = tk.Frame(self.panel, bg="#202020")
            fr.pack(fill=tk.X, padx=10, pady=pad // 2 + 1)
            for i, (label, act, active, chip) in enumerate(v):
                fr.columnconfigure(i, weight=1, uniform="r")
                idx = len(self.cells)
                if act is None:
                    w = tk.Label(fr, text=label, font=self.font, bg="#202020", fg="#d0d0d0")
                else:
                    w = tk.Button(fr, text=label, font=self.font, command=lambda k=idx: self.click(k),
                                  bg="#383838", fg="#ffffff", activebackground="#505050", activeforeground="#ffffff",
                                  relief=tk.FLAT, bd=0, highlightthickness=2, highlightbackground="#202020",
                                  takefocus=0, pady=pad)
                    if chip is not None:
                        w.configure(image=self.chip(chip), compound="left", padx=8)
                w.grid(row=0, column=i, sticky="nsew", padx=2)
                self.cells.append(w)
        sep = tk.Frame(self.panel, bg="#404040", height=1)
        sep.pack(fill=tk.X, padx=10, pady=(pad * 3, pad))
        self.stats = []
        for _ in range(len(self.app.status_lines()) + 1):
            lb = tk.Label(self.panel, text="", font=self.font_stat, bg="#202020", fg="#d0d0d0",
                          anchor="w", justify="left", wraplength=self.app.pw - 24)
            lb.pack(fill=tk.X, padx=10, pady=1)
            self.stats.append(lb)

    def flat_cells(self):
        return [c for kind, v in self.app.panel_spec() if kind == "row" for c in v]

    def click(self, k):
        cells = self.flat_cells()
        if k < len(cells) and cells[k][1] is not None:
            cells[k][1]()
        self.after_action()

    def refresh_panel(self):
        spec = self.app.panel_spec()
        heads = [v for kind, v in spec if kind == "head"]
        for lb, txt in zip(self.heads, heads):
            if lb.cget("text") != txt:
                lb.configure(text=txt)
        for w, (label, act, active, chip) in zip(self.cells, [c for kind, v in spec if kind == "row" for c in v]):
            cfg = {"text": label}
            if act is not None:
                cfg.update(bg="#5a5a5a" if active else "#383838",
                           highlightbackground="#00c8ff" if active else "#202020")
            w.configure(**cfg)
        lines = self.app.status_lines() + [(self.app.st["msg"], (160, 255, 160))]
        for lb, (txt, col) in zip(self.stats, lines):
            lb.configure(text=txt, fg="#%02x%02x%02x" % (col[2], col[1], col[0]))

    def on_panel_size(self, w):
        self.app.pw = w
        for lb in self.stats:
            lb.configure(wraplength=max(w - 24, 50))

    # ---------------- 影像 ----------------
    def schedule(self):
        if not self.pending:
            self.pending = True
            self.root.after_idle(self.redraw)

    def redraw(self):
        self.pending = False
        app = self.app
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w > 1 and h > 1 and app.canvas != (w, h):
            app.canvas = None
            app.fit(w, h)
        bgr = app.render()
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        if self.ImageTk is not None:
            self.photo = self.ImageTk.PhotoImage(self.Image.fromarray(rgb))
        else:                                               # 沒有 PIL.ImageTk：用 PPM 給 Tk
            hdr = f"P6 {rgb.shape[1]} {rgb.shape[0]} 255 ".encode()
            self.photo = self.tk.PhotoImage(data=hdr + rgb.tobytes(), format="PPM")
        self.canvas.itemconfigure(self.img_item, image=self.photo)
        self.refresh_panel()

    def redraw_now(self):
        self.redraw()
        self.root.update_idletasks()

    # ---------------- 事件 ----------------
    def after_action(self):
        if self.app.st["quit"]:
            self.root.after(10, self.root.destroy)
            return
        self.schedule()

    def mouse(self, ev, e, flags):
        self.app.on_mouse(ev, e.x, e.y, flags)
        self.after_action()

    def wheel(self, e, direction):
        flags = ((120 if direction > 0 else -120) & 0xFFFF) << 16
        self.app.on_mouse(cv2.EVENT_MOUSEWHEEL, e.x, e.y, flags)
        self.after_action()

    KEYSYMS = {"Left": 65361, "Right": 65363, "F11": 65480, "Escape": 27}

    def key(self, e):
        code = self.KEYSYMS.get(e.keysym)
        if code is None and e.char:
            code = ord(e.char)                              # Ctrl+Z -> 26
        if code is None:
            return
        self.app.on_key(code)
        self.after_action()
        return "break"

    def close(self):
        self.app.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def run_tk(app, a):
    TkGUI(app, a).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="影片拆出的影格資料夾（依檔名排序 = 時間順序）")
    ap.add_argument("--out_dir", default="my_needle_masks")
    ap.add_argument("--clip_name", default=None, help="這段 clip 的名稱（預設：<資料夾名>_<第一張檔名>）")
    ap.add_argument("--start", default="0", help="起始影格：index（整數）或檔名的一部分")
    ap.add_argument("--num", type=int, default=150, help="這段 clip 取幾張")
    ap.add_argument("--stride", type=int, default=3, help="每幾張取一張（15fps、stride 3 = 每秒 5 張）")
    ap.add_argument("--goto", default=None, help="開啟後直接跳到這張影格：檔名的一部分或 clip 內的 index（從 0 開始）")
    ap.add_argument("--force_new", action="store_true",
                    help="這段 clip 已有 session 但影格清單不同時，仍然用新的清單開啟（存檔會蓋掉之前的標註）")
    ap.add_argument("--model", choices=list(CKPTS), default="small")
    ap.add_argument("--ckpt_dir", default="checkpoints_sam2")
    ap.add_argument("--display_width", type=int, default=1280,
                    help="一般視窗模式的起始影格顯示寬度（不含左側按鈕列）；之後會跟著視窗大小調整")
    ap.add_argument("--window", choices=["fullscreen", "normal"], default="fullscreen",
                    help="fullscreen：選項欄 + 影像兩個視窗並排填滿螢幕（F11 切換）；normal：不自動排列")
    ap.add_argument("--panel_width", type=int, default=PANEL_W, help="選項欄起始寬度（之後可拖曳分隔線）")
    ap.add_argument("--gui", choices=["tk", "cv2"], default="tk",
                    help="tk（預設）：一個視窗，選項欄 + 影像，沒有拖曳平移；cv2：舊的 OpenCV 兩個視窗（Qt 會有滾輪縮放 / 拖曳）")
    ap.add_argument("--ui_font", type=int, default=FONT_PX, help="選項欄文字大小（px）")
    ap.add_argument("--screen", default=None, help="螢幕大小，例如 1920x1080（自動偵測失敗時用）")
    ap.add_argument("--screen_margin", type=int, default=80, help="螢幕高度預留給標題列 / 工作列的空間（px）")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.frames_dir) if f.lower().endswith(IMG_EXTS))
    if not files:
        sys.exit(f"{a.frames_dir} 沒有影像")
    if a.start.isdigit():
        s0 = int(a.start)
    else:
        s0 = next((i for i, f in enumerate(files) if a.start in f), None)
        if s0 is None:
            sys.exit(f"找不到檔名包含 '{a.start}' 的影格")
    clip = files[s0: s0 + a.num * a.stride: a.stride]
    parent = os.path.basename(os.path.normpath(os.path.dirname(os.path.abspath(a.frames_dir))))
    clip_name = a.clip_name or f"{parent}_{os.path.splitext(clip[0])[0]}"
    print(f"clip {clip_name}：{len(clip)} 張（{clip[0]} ~ {clip[-1]}，stride {a.stride}）")

    # 防呆：這段 clip 已經標過、但這次的 --start / --num / --stride 不同 -> 影格清單對不上，讀不回之前的標註，
    # 這時存檔（s / q / Ctrl+C）會用空的結果蓋掉之前的 session 與 Labels，所以直接停止並告訴你正確的參數
    sess_path0 = os.path.join(a.out_dir, "sessions", f"{clip_name}.json")
    if os.path.isfile(sess_path0) and not a.force_new:
        old = json.load(open(sess_path0))
        of = old.get("frames", [])
        if of and of != clip:
            print(f"\n[停止] {clip_name} 已經標過，但這次挑出的影格和之前不同（之前 {len(of)} 張、stride {old.get('stride')}；"
                  f"這次 {len(clip)} 張、stride {a.stride}）。\n"
                  f"  用這次的參數開啟會讀不回之前的標註，存檔時還會把它們蓋掉。請用之前的參數：\n"
                  f"  python sam2_video_label.py --frames_dir {a.frames_dir} --out_dir {a.out_dir} "
                  f"--start {of[0]} --num {len(of)} --stride {old.get('stride', a.stride)}"
                  + (f" --goto {a.goto}" if a.goto else "") +
                  "\n  （確定要用新的影格清單重新標這段 clip，才加 --force_new）")
            sys.exit(1)
    goto_idx = None
    if a.goto:
        goto_idx = int(a.goto) if a.goto.isdigit() else next((i for i, f in enumerate(clip) if a.goto in f), None)
        if goto_idx is None or not 0 <= goto_idx < len(clip):
            sys.exit(f"[停止] --goto {a.goto}：這段 clip（stride {a.stride}）裡沒有這張影格。"
                     f"stride 不是 1 時只有每 {a.stride} 張的那幾張在 clip 裡，請改用 clip 裡的檔名或 index（0 ~ {len(clip) - 1}）")

    tmp = tempfile.mkdtemp(prefix="sam2clip_")                 # SAM 2 只吃 <index>.jpg 的資料夾
    imgs = []
    for i, f in enumerate(clip):
        im = cv2.imread(os.path.join(a.frames_dir, f))
        imgs.append(im)
        cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    H, W = imgs[0].shape[:2]

    print("載入 SAM 2 ...")
    sess = Session(load_predictor(a.model, a.ckpt_dir), tmp, len(clip), H, W)
    sess_path = os.path.join(a.out_dir, "sessions", f"{clip_name}.json")
    edits_path = os.path.join(a.out_dir, "sessions", f"{clip_name}_edits.npz")
    excluded = load_session(sess, sess_path, edits_path, clip, os.path.join(a.out_dir, "Labels", clip_name))

    app = App(sess, imgs, clip, View(W, H, a.display_width),
              lambda: save_outputs(sess, imgs, clip, app.excluded, a, clip_name, sess_path, edits_path), excluded)
    app.pw, app.font_px = int(min(max(a.panel_width, PANEL_MIN), PANEL_MAX)), a.ui_font
    if goto_idx is not None:
        app.st["t"] = goto_idx
        print(f"開在第 {goto_idx + 1}/{len(clip)} 張：{clip[goto_idx]}")
    gui = a.gui
    if gui == "tk":
        try:
            import tkinter  # noqa: F401
        except ImportError:
            print("[WARN] 沒有 tkinter，改用 OpenCV 介面（Qt 的滾輪縮放 / 拖曳平移無法關閉）。\n"
                  "       安裝：sudo apt-get install -y python3-tk（venv 用的是系統的 python，裝完重開即可）")
            gui = "cv2"
    app.own_wheel = gui == "tk"
    try:
        if gui == "tk":
            run_tk(app, a)
        else:
            run_cv2(app, a)
    except KeyboardInterrupt:
        print("\n[中斷] 存檔中 ...")
        app.save()
    cv2.destroyAllWindows()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()