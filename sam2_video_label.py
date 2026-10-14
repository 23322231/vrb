#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 SAM 2 影片追蹤，快速標註連續影格中的針，輸出和學長資料「相同格式」的彩色 mask。
另外提供畫筆 / 橡皮擦（逐 pixel 修正）與復原（undo）。

流程（一次處理一段 clip）:
    1. 從影格資料夾取一段：從 --start 開始，每 --stride 張取 1 張，共 --num 張
    2. 在任一張上用滑鼠點出針（左鍵 = 是針，右鍵 = 不是針；或按 b 拉框）
    3. 按 p：SAM 2 把 mask 往前、往後追蹤到整段 clip
    4. 用 a / d 翻影格檢查；追錯的影格：再點幾下修正後按 p，或用畫筆 / 橡皮擦直接修 pixel
    5. 按 s 存檔（q 存檔並離開）

手動修改（畫筆 / 橡皮擦）的說明:
    - 修改存在獨立的一層，最終 mask = (SAM 2 的結果 ∪ 畫筆) − 橡皮擦，所以重新按 p 追蹤也不會被蓋掉
    - 修完某張後按 g：把這張「修好的 mask」交給 SAM 2 當作提示，再按 p 時其他影格會參考它來追蹤
      （適合 SAM 2 在某段一直追錯時：修好一張 -> g -> p）

輸出（可以直接給 masks_to_yolo_seg.py 與 mask_to_needle5.py 使用）:
    <out_dir>/Images/<clip>/<原檔名>              原始影格（複製）
    <out_dir>/Labels/<clip>/<原檔名去副檔名>.png   mask（針 = BGR (0,0,255)，其餘黑色）
    <out_dir>/overlay/<clip>/...                   疊圖檢查
    <out_dir>/Color_code.txt
    <out_dir>/sessions/<clip>.json + <clip>_edits.npz   提示點與手動修改（重開同一段 clip 會自動讀回）

按鍵:
  [提示點模式]（預設）
    滑鼠左鍵 / 右鍵      目前影格、目前這根針加「正 / 負」提示點
    b                    拉框模式：左鍵拖曳畫框
    c                    清除目前這根針在這張影格的 SAM 提示點 / 框
  [畫筆 / 橡皮擦]
    w                    畫筆模式（左鍵拖曳 = 畫上；右鍵拖曳 = 擦掉），再按一次回到提示點模式
    e                    橡皮擦模式（左鍵拖曳 = 擦掉），再按一次回到提示點模式
    + / -                筆刷大小（半徑 0 = 單一 pixel）
    k                    清除這張影格、這根針的所有手動修改
    g                    把這張修好的 mask 交給 SAM 2 當提示（之後按 p 會參考）
  [共通]
    u  或 Ctrl+Z         復原上一步（最多 50 步；點、框、畫筆、橡皮擦、清除、追蹤、排除都可以復原）
    n                    新增一根針          [ / ]   切換目前操作的針
    p                    追蹤：從有提示的影格往前、往後傳遞到整段 clip
    a / d  或 ← / →      上一張 / 下一張      j / l   跳 10 張
    z                    放大 1x → 2x → 4x → 8x（以滑鼠位置為中心）；逐 pixel 修改請放大到 4x 以上
    x                    把這張影格標為「不要」（不輸出），再按一次取消
    m                    切換 mask 顯示       s 存檔       q / Esc 存檔並離開

需要（在 VRB container 內）:
    SAM2_BUILD_CUDA=0 pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/sam2.git
    pip install "hydra-core>=1.3.2" "iopath>=0.1.10"

用法:
    python sam2_video_label.py --frames_dir /data/.../rgb --out_dir my_needle_masks \
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
NEEDLE_BGR = (0, 0, 255)
OBJ_COLORS = [(0, 0, 255), (255, 128, 0), (0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0)]
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
"""
HELP1 = ("POINT: L/R click +/- | b box | c clear prompts   BRUSH: w brush (R-drag erase) | e eraser | +/- size | "
         "k clear edits | g edited mask->SAM")
HELP2 = ("u / Ctrl+Z undo | n new needle  [ ] switch | p PROPAGATE | a/d j/l frame | z zoom | x exclude | "
         "m mask on/off | s save | q save&quit")
UNDO_MAX = 50


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
    points[oid][t] = [(x, y, label)]   boxes[oid][t] = (x1,y1,x2,y2)   mask_prompts[oid][t] = bool mask
    sam[t][oid]    = SAM 2 輸出的 bool mask
    add[t][oid] / erase[t][oid] = 手動畫筆 / 橡皮擦（uint8 0/1）
    最終 mask = (sam | add) & ~erase
    """

    def __init__(self, predictor, jpg_dir, n_frames, H, W):
        import torch
        self.torch = torch
        self.pred = predictor
        self.H, self.W, self.n = H, W, n_frames
        with self._ctx():
            self.state = predictor.init_state(jpg_dir, offload_video_to_cpu=True)
        self.points, self.boxes, self.mask_prompts = {}, {}, {}
        self.sam, self.add, self.erase = {}, {}, {}
        self.obj_ids = [1]

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

    def _store(self, f, obj_ids, logits):
        d = self.sam.setdefault(f, {})
        for oid, lg in zip(obj_ids, logits):
            d[int(oid)] = (lg[0] > 0.0).cpu().numpy()

    def sync(self, f, oid):
        """依目前的點 / 框 / mask 提示，更新 SAM 2 在這張影格、這根針的條件。"""
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
                self._store(f, ids, logits)
            elif mp is not None:
                _, ids, logits = self.pred.add_new_mask(self.state, frame_idx=f, obj_id=oid, mask=mp)
                self._store(f, ids, logits)
            else:
                try:
                    self.pred.clear_all_prompts_in_frame(self.state, f, oid, need_output=False)
                except Exception:
                    pass
                self.sam.get(f, {}).pop(oid, None)

    def prompted_frames(self):
        fs = set()
        for d in list(self.points.values()) + list(self.boxes.values()) + list(self.mask_prompts.values()):
            fs.update(k for k, v in d.items() if v is not None and (not isinstance(v, list) or v))
        return sorted(fs)

    def propagate(self, on_progress=None):
        fs = self.prompted_frames()
        if not fs:
            return False
        first, done = fs[0], 0
        with self._ctx():
            runs = [dict(start_frame_idx=first)] + ([dict(start_frame_idx=first, reverse=True)] if first > 0 else [])
            for kw in runs:
                for f, ids, logits in self.pred.propagate_in_video(self.state, **kw):
                    self._store(f, ids, logits)
                    done += 1
                    if on_progress and done % 10 == 0:
                        on_progress(done)
        return True

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

    def has_any(self, f):
        return any(self.final(f, o).any() for o in self.objs_on(f))

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
    """顯示用：縮放 + 放大鏡，負責畫面座標 <-> 原圖座標轉換。"""

    def __init__(self, W, H, disp_w):
        self.W, self.H = W, H
        self.ds = disp_w / W
        self.zf, self.cx, self.cy = 1, W / 2, H / 2

    def region(self):
        w, h = self.W / self.zf, self.H / self.zf
        x0 = min(max(self.cx - w / 2, 0), self.W - w)
        y0 = min(max(self.cy - h / 2, 0), self.H - h)
        return x0, y0, w, h

    def to_img(self, mx, my):
        x0, y0, _, _ = self.region()
        s = self.ds * self.zf
        return x0 + mx / s, y0 + my / s

    def to_disp(self, x, y):
        x0, y0, _, _ = self.region()
        s = self.ds * self.zf
        return int(round((x - x0) * s)), int(round((y - y0) * s))

    def render(self, img):
        x0, y0, w, h = self.region()
        crop = img[int(y0):int(y0 + h), int(x0):int(x0 + w)]
        return cv2.resize(crop, (int(self.W * self.ds), int(self.H * self.ds)),
                          interpolation=cv2.INTER_NEAREST if self.zf > 1 else cv2.INTER_AREA)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="影片拆出的影格資料夾（依檔名排序 = 時間順序）")
    ap.add_argument("--out_dir", default="my_needle_masks")
    ap.add_argument("--clip_name", default=None, help="這段 clip 的名稱（預設：<資料夾名>_<第一張檔名>）")
    ap.add_argument("--start", default="0", help="起始影格：index（整數）或檔名的一部分")
    ap.add_argument("--num", type=int, default=150, help="這段 clip 取幾張")
    ap.add_argument("--stride", type=int, default=3, help="每幾張取一張（15fps、stride 3 = 每秒 5 張）")
    ap.add_argument("--model", choices=list(CKPTS), default="small")
    ap.add_argument("--ckpt_dir", default="checkpoints_sam2")
    ap.add_argument("--display_width", type=int, default=1280)
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

    tmp = tempfile.mkdtemp(prefix="sam2clip_")             # SAM 2 只吃 <index>.jpg 的資料夾
    imgs = []
    for i, f in enumerate(clip):
        im = cv2.imread(os.path.join(a.frames_dir, f))
        imgs.append(im)
        cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    H, W = imgs[0].shape[:2]

    print("載入 SAM 2 ...")
    sess = Session(load_predictor(a.model, a.ckpt_dir), tmp, len(clip), H, W)

    # ---- 讀回之前的狀態 ----
    sess_path = os.path.join(a.out_dir, "sessions", f"{clip_name}.json")
    edits_path = os.path.join(a.out_dir, "sessions", f"{clip_name}_edits.npz")
    excluded = set()
    if os.path.isfile(sess_path):
        old = json.load(open(sess_path))
        if old.get("frames") == clip:
            excluded = set(old.get("excluded", []))
            sess.obj_ids = old.get("obj_ids", [1])
            for oid, d in old.get("points", {}).items():
                sess.points[int(oid)] = {int(k): [tuple(p) for p in v] for k, v in d.items()}
            for oid, d in old.get("boxes", {}).items():
                sess.boxes[int(oid)] = {int(k): tuple(v) for k, v in d.items()}
            if os.path.isfile(edits_path):
                z = np.load(edits_path)
                for key in z.files:                          # add_<f>_<oid> / erase_<f>_<oid> / mp_<f>_<oid>
                    kind, f, oid = key.split("_")
                    arr = z[key]
                    if kind == "mp":
                        sess.mask_prompts.setdefault(int(oid), {})[int(f)] = arr.astype(bool)
                    else:
                        (sess.add if kind == "add" else sess.erase).setdefault(int(f), {})[int(oid)] = arr
            for oid in sess.obj_ids:
                for f in (set(sess.points.get(oid, {})) | set(sess.boxes.get(oid, {}))
                          | set(sess.mask_prompts.get(oid, {}))):
                    sess.sync(f, oid)
            print(f"已讀回之前的提示與手動修改（{len(sess.prompted_frames())} 張有提示），按 p 重新追蹤")

    view = View(W, H, a.display_width)
    st = {"t": 0, "oid": sess.obj_ids[0], "mode": "point", "drag": None, "mouse": (W / 2, H / 2),
          "show": True, "radius": 0, "last": None, "msg": "click on the needle, then press p"}
    undo = []

    def push(entry):
        undo.append(entry)
        if len(undo) > UNDO_MAX:
            undo.pop(0)

    def do_undo():
        if not undo:
            st["msg"] = "nothing to undo"
            return
        e = undo.pop()
        if e["kind"] == "frame":
            sess.restore_frame(e)
            st["t"], st["oid"] = e["f"], e["oid"]
        elif e["kind"] == "all_sam":
            sess.restore_all_sam(e)
        elif e["kind"] == "exclude":
            excluded.symmetric_difference_update({e["f"]})
            st["t"] = e["f"]
        elif e["kind"] == "new_obj":
            if e["oid"] in sess.obj_ids and len(sess.obj_ids) > 1:
                sess.obj_ids.remove(e["oid"])
                st["oid"] = sess.obj_ids[-1]
        st["msg"] = f"undo ({len(undo)} left)"

    def save():
        for sub in ("Images", "Labels", "overlay"):
            os.makedirs(os.path.join(a.out_dir, sub, clip_name), exist_ok=True)
        os.makedirs(os.path.dirname(sess_path), exist_ok=True)
        with open(os.path.join(a.out_dir, "Color_code.txt"), "w") as fcc:
            fcc.write(COLOR_CODE)
        n_saved = 0
        for i, f in enumerate(clip):
            stem = os.path.splitext(f)[0]
            lab = os.path.join(a.out_dir, "Labels", clip_name, stem + ".png")
            img_dst = os.path.join(a.out_dir, "Images", clip_name, f)
            ov_dst = os.path.join(a.out_dir, "overlay", clip_name, stem + ".png")
            if i in excluded or not sess.has_any(i):
                for pth in (lab, img_dst, ov_dst):              # 被排除 / 沒有 mask 的影格不輸出
                    if os.path.exists(pth):
                        os.remove(pth)
                continue
            m = np.zeros((H, W, 3), np.uint8)
            for o in sess.objs_on(i):
                m[sess.final(i, o)] = NEEDLE_BGR
            cv2.imwrite(lab, m)
            if not os.path.exists(img_dst):
                shutil.copy2(os.path.join(a.frames_dir, f), img_dst)
            ov = imgs[i].copy()
            sel = m[:, :, 2] == 255
            ov[sel] = (0.4 * ov[sel] + 0.6 * np.array(NEEDLE_BGR)).astype(np.uint8)
            cv2.imwrite(ov_dst, ov)
            n_saved += 1
        with open(sess_path, "w") as fs:
            json.dump({"frames_dir": os.path.abspath(a.frames_dir), "frames": clip, "stride": a.stride,
                       "obj_ids": sess.obj_ids, "excluded": sorted(excluded),
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
        np.savez_compressed(edits_path, **arrs)
        st["msg"] = f"saved {n_saved} frames -> {a.out_dir}"
        print(st["msg"])

    def on_mouse(ev, mx, my, flags, _):
        x, y = view.to_img(mx, my)
        x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
        st["mouse"] = (x, y)
        t, oid = st["t"], st["oid"]

        if st["mode"] in ("brush", "eraser"):
            if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                push(sess.snap_frame(t, oid))
                erase = st["mode"] == "eraser" or ev == cv2.EVENT_RBUTTONDOWN
                st["last"] = ((x, y), erase)
                sess.stroke(t, oid, (x, y), (x, y), st["radius"], erase)
            elif ev == cv2.EVENT_MOUSEMOVE and st["last"] and (flags & (cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_RBUTTON)):
                (px, py), erase = st["last"]
                sess.stroke(t, oid, (px, py), (x, y), st["radius"], erase)
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
                    push(sess.snap_frame(t, oid))
                    sess.boxes.setdefault(oid, {})[t] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    sess.mask_prompts.get(oid, {}).pop(t, None)
                    sess.sync(t, oid)
                    st["mode"] = "point"
                    st["msg"] = "box added; add +/- points if needed, then p"
            return

        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            push(sess.snap_frame(t, oid))
            lab = 1 if ev == cv2.EVENT_LBUTTONDOWN else 0
            sess.points.setdefault(oid, {}).setdefault(t, []).append((float(x), float(y), lab))
            sess.mask_prompts.get(oid, {}).pop(t, None)
            sess.sync(t, oid)
            st["msg"] = f"{'+' if lab else '-'} point on needle {oid}"

    win = "SAM2 needle labeling"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        t, oid = st["t"], st["oid"]
        img = imgs[t].copy()
        finals = {o: sess.final(t, o) for o in sess.objs_on(t)} if st["show"] else {}
        for o, bm in finals.items():
            c = OBJ_COLORS[(o - 1) % len(OBJ_COLORS)]
            img[bm] = (0.45 * img[bm] + 0.55 * np.array(c)).astype(np.uint8)
        disp = view.render(img)
        for o, bm in finals.items():                          # 細針：再描一圈輪廓
            c = OBJ_COLORS[(o - 1) % len(OBJ_COLORS)]
            cs, _ = cv2.findContours(bm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cc in cs:
                pts = np.array([view.to_disp(p[0][0], p[0][1]) for p in cc], np.int32)
                cv2.polylines(disp, [pts], True, c, 1, cv2.LINE_AA)
        for o in sess.obj_ids:                                # 提示點 / 框
            for (x, y, lab) in sess.points.get(o, {}).get(t, []):
                p = view.to_disp(x, y)
                if lab:
                    cv2.circle(disp, p, 4, (0, 255, 0), -1, cv2.LINE_AA)
                else:
                    cv2.drawMarker(disp, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 9, 2)
            bx = sess.boxes.get(o, {}).get(t)
            if bx:
                cv2.rectangle(disp, view.to_disp(bx[0], bx[1]), view.to_disp(bx[2], bx[3]), (0, 255, 0), 1)
        if st["drag"]:
            d = st["drag"]
            cv2.rectangle(disp, view.to_disp(d[0], d[1]), view.to_disp(d[2], d[3]), (0, 255, 255), 1)
        if st["mode"] in ("brush", "eraser"):                 # 筆刷游標
            s_px = view.ds * view.zf
            r_disp = max(1, int(round((st["radius"] + 0.5) * s_px)))
            cv2.circle(disp, view.to_disp(*st["mouse"]), r_disp,
                       (255, 255, 255) if st["mode"] == "brush" else (0, 0, 0), 1, cv2.LINE_AA)

        bar = np.zeros((76, disp.shape[1], 3), np.uint8)
        n_tracked = sum(1 for i in range(len(clip)) if i not in excluded and sess.has_any(i))
        mode_txt = {"point": "POINT", "box": "BOX", "brush": f"BRUSH r={st['radius']}",
                    "eraser": f"ERASER r={st['radius']}"}[st["mode"]]
        edited = " EDITED" if (sess.add.get(t, {}).get(oid) is not None and sess.add[t][oid].any()) or \
                              (sess.erase.get(t, {}).get(oid) is not None and sess.erase[t][oid].any()) else ""
        info = (f"frame {t + 1}/{len(clip)} [{clip[t]}]  needle {oid}/{len(sess.obj_ids)}  mode {mode_txt}{edited}  "
                f"zoom {view.zf}x  {'EXCLUDED  ' if t in excluded else ''}tracked {n_tracked}  undo {len(undo)}")
        cv2.putText(bar, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 128, 255) if t in excluded else (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, st["msg"], (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(bar, HELP1, (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(bar, HELP2, (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow(win, np.vstack([disp, bar]))

        k = cv2.waitKeyEx(15)
        if k == -1:
            continue
        c = k & 0xFF
        if c == ord("d") or k in (65363, 2555904):
            st["t"] = min(t + 1, len(clip) - 1)
        elif c == ord("a") or k in (65361, 2424832):
            st["t"] = max(t - 1, 0)
        elif c == ord("l"):
            st["t"] = min(t + 10, len(clip) - 1)
        elif c == ord("j"):
            st["t"] = max(t - 10, 0)
        elif c == ord("b"):
            st["mode"] = "point" if st["mode"] == "box" else "box"
            st["msg"] = "drag a box around the needle" if st["mode"] == "box" else ""
        elif c == ord("w"):
            st["mode"] = "point" if st["mode"] == "brush" else "brush"
            st["msg"] = "BRUSH: L-drag paint, R-drag erase (+/- size, z zoom)" if st["mode"] == "brush" else ""
        elif c == ord("e"):
            st["mode"] = "point" if st["mode"] == "eraser" else "eraser"
            st["msg"] = "ERASER: L-drag erase (+/- size, z zoom)" if st["mode"] == "eraser" else ""
        elif c in (ord("+"), ord("=")):
            st["radius"] = min(st["radius"] + 1, 30)
        elif c in (ord("-"), ord("_")):
            st["radius"] = max(st["radius"] - 1, 0)
        elif c == ord("k"):
            push(sess.snap_frame(t, oid))
            sess.add.get(t, {}).pop(oid, None)
            sess.erase.get(t, {}).pop(oid, None)
            st["msg"] = f"cleared manual edits of needle {oid} on this frame"
        elif c == ord("g"):
            m = sess.final(t, oid)
            if not m.any():
                st["msg"] = "mask is empty: nothing to send to SAM 2"
            else:
                push(sess.snap_frame(t, oid))
                sess.points.get(oid, {}).pop(t, None)
                sess.boxes.get(oid, {}).pop(t, None)
                sess.mask_prompts.setdefault(oid, {})[t] = m
                sess.sync(t, oid)
                st["msg"] = "edited mask sent to SAM 2 as prompt; press p to re-propagate"
        elif c == ord("u") or k == 26 or c == 26:
            do_undo()
        elif c == ord("n"):
            new = max(sess.obj_ids) + 1
            sess.obj_ids.append(new)
            push({"kind": "new_obj", "oid": new})
            st["oid"] = new
            st["msg"] = f"new needle {new}: click on it"
        elif c == ord("]"):
            st["oid"] = sess.obj_ids[(sess.obj_ids.index(oid) + 1) % len(sess.obj_ids)]
        elif c == ord("["):
            st["oid"] = sess.obj_ids[(sess.obj_ids.index(oid) - 1) % len(sess.obj_ids)]
        elif c == ord("c"):
            push(sess.snap_frame(t, oid))
            sess.points.get(oid, {}).pop(t, None)
            sess.boxes.get(oid, {}).pop(t, None)
            sess.mask_prompts.get(oid, {}).pop(t, None)
            sess.sync(t, oid)
            st["msg"] = f"cleared SAM prompts of needle {oid} on this frame (manual edits kept; k to clear them)"
        elif c == ord("z"):
            view.zf = {1: 2, 2: 4, 4: 8, 8: 1}[view.zf]
            view.cx, view.cy = st["mouse"]
        elif c == ord("x"):
            excluded.symmetric_difference_update({t})
            push({"kind": "exclude", "f": t})
        elif c == ord("m"):
            st["show"] = not st["show"]
        elif c == ord("p"):
            st["msg"] = "propagating ..."
            cv2.imshow(win, np.vstack([disp, bar]))
            cv2.waitKey(1)
            snap = sess.snap_all_sam()
            t0 = time.time()
            try:
                ok = sess.propagate(lambda n: print(f"  propagated {n} frames", end="\r"))
                if ok:
                    push(snap)
                st["msg"] = (f"propagated in {time.time() - t0:.0f}s - check with a/d; manual edits are kept"
                             if ok else "no prompts yet: click on the needle first")
            except RuntimeError as e:
                sess.restore_all_sam(snap)
                st["msg"] = f"propagate failed: {str(e)[:80]} (give every needle at least one prompt)"
            print()
        elif c == ord("s"):
            save()
        elif c in (ord("q"), 27):
            save()
            break

    cv2.destroyAllWindows()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()