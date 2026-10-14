#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 SAM 2 影片追蹤，快速標註連續影格中的針，輸出和學長資料「相同格式」的彩色 mask。

流程（一次處理一段 clip）:
    1. 從影格資料夾取一段：從 --start 開始，每 --stride 張取 1 張，共 --num 張
    2. 在任一張上用滑鼠點出針（左鍵 = 是針，右鍵 = 不是針；或按 b 拉框）
    3. 按 p：SAM 2 把 mask 往前、往後追蹤到整段 clip
    4. 用 a / d 翻影格檢查；追錯的影格直接再點幾下修正，然後再按 p 重新追蹤
    5. 按 s 存檔（q 存檔並離開）

輸出（可以直接給 masks_to_yolo_seg.py 與 mask_to_needle5.py 使用）:
    <out_dir>/Images/<clip>/<原檔名>          原始影格（複製）
    <out_dir>/Labels/<clip>/<原檔名去副檔名>.png   mask（針 = BGR (0,0,255)，其餘黑色）
    <out_dir>/overlay/<clip>/...               疊圖檢查
    <out_dir>/Color_code.txt
    <out_dir>/sessions/<clip>.json             這段 clip 的提示點紀錄（之後可重開修改）

按鍵:
    滑鼠左鍵 / 右鍵      在目前影格、目前物體加「正 / 負」提示點（每點一下即時更新 mask）
    b                    切換「拉框模式」：左鍵拖曳畫框（細針建議先框再補點）
    n                    新增一根針（新物體）       [ / ]   切換目前操作的針
    c                    清除目前針在這張影格的所有提示
    p                    追蹤：從有提示的影格往前、往後傳遞到整段 clip
    a / d  或 ← / →      上一張 / 下一張          j / l    跳 10 張
    z                    放大 1x → 2x → 4x（以滑鼠位置為中心），細針請放大再點
    x                    把這張影格標為「不要」（不輸出），再按一次取消
    m                    切換 mask 顯示（看原圖用）
    s                    存檔                      q / Esc  存檔並離開

需要（在 VRB container 內）:
    SAM2_BUILD_CUDA=0 pip install git+https://github.com/facebookresearch/sam2.git
    第一次執行會自動下載 SAM 2.1 權重到 --ckpt_dir

用法:
    python sam2_video_label.py --frames_dir /path/to/rgb --out_dir my_needle_masks \
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
HELP = ("L/R click: +/- point | b: box mode | n: new needle  [ ]: switch | c: clear frame | p: PROPAGATE | "
        "a/d j/l: frame | z: zoom | x: exclude | m: mask on/off | s: save | q: save&quit")


# -----------------------------------------------------------------------------
def load_predictor(model, ckpt_dir):
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        sys.exit("[ERROR] 沒有 sam2，請在 container 內執行：\n"
                 "  SAM2_BUILD_CUDA=0 pip install git+https://github.com/facebookresearch/sam2.git")
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


class Session:
    """管理一段 clip 的提示點與 mask。points[obj_id][frame] = [(x, y, label), ...]，box[obj_id][frame] = (x1,y1,x2,y2)"""

    def __init__(self, predictor, jpg_dir, n_frames):
        import torch
        self.torch = torch
        self.pred = predictor
        with self._ctx():
            self.state = predictor.init_state(jpg_dir, offload_video_to_cpu=True)
        self.n = n_frames
        self.points, self.boxes = {}, {}
        self.masks = {}                      # frame -> {obj_id: bool mask}
        self.obj_ids = [1]

    def _ctx(self):
        torch = self.torch

        class C:
            def __enter__(s):
                s.a = torch.autocast("cuda", dtype=torch.bfloat16)
                s.i = torch.inference_mode()
                s.i.__enter__()
                s.a.__enter__()

            def __exit__(s, *e):
                s.a.__exit__(*e)
                s.i.__exit__(*e)
        return C()

    def _store(self, f, obj_ids, logits):
        d = self.masks.setdefault(f, {})
        for oid, lg in zip(obj_ids, logits):
            d[int(oid)] = (lg[0] > 0.0).cpu().numpy()

    def prompt(self, f, oid):
        """以這張影格、這根針目前所有的點 / 框，重新送給 SAM 2。"""
        pts = self.points.get(oid, {}).get(f, [])
        box = self.boxes.get(oid, {}).get(f)
        if not pts and box is None:
            return
        with self._ctx():
            kw = {}
            if pts:
                kw["points"] = np.array([[p[0], p[1]] for p in pts], np.float32)
                kw["labels"] = np.array([p[2] for p in pts], np.int32)
            if box is not None:
                kw["box"] = np.array(box, np.float32)
            _, ids, logits = self.pred.add_new_points_or_box(self.state, frame_idx=f, obj_id=oid,
                                                              clear_old_points=True, **kw)
        self._store(f, ids, logits)

    def clear(self, f, oid):
        self.points.get(oid, {}).pop(f, None)
        self.boxes.get(oid, {}).pop(f, None)
        try:
            with self._ctx():
                self.pred.clear_all_prompts_in_frame(self.state, f, oid, need_output=False)
        except Exception:
            pass
        self.masks.get(f, {}).pop(oid, None)

    def prompted_frames(self):
        fs = set()
        for d in list(self.points.values()) + list(self.boxes.values()):
            fs.update(k for k, v in d.items() if v)
        return sorted(fs)

    def propagate(self, on_progress=None):
        fs = self.prompted_frames()
        if not fs:
            return False
        first = fs[0]
        done = 0
        with self._ctx():
            for f, ids, logits in self.pred.propagate_in_video(self.state, start_frame_idx=first):
                self._store(f, ids, logits)
                done += 1
                if on_progress and done % 10 == 0:
                    on_progress(done)
            if first > 0:
                for f, ids, logits in self.pred.propagate_in_video(self.state, start_frame_idx=first,
                                                                   reverse=True):
                    self._store(f, ids, logits)
                    done += 1
                    if on_progress and done % 10 == 0:
                        on_progress(done)
        return True


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

    # SAM 2 只吃 <index>.jpg 的資料夾 -> 轉成暫存 JPEG
    tmp = tempfile.mkdtemp(prefix="sam2clip_")
    imgs = []
    for i, f in enumerate(clip):
        im = cv2.imread(os.path.join(a.frames_dir, f))
        imgs.append(im)
        cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    H, W = imgs[0].shape[:2]

    print("載入 SAM 2 ...")
    sess = Session(load_predictor(a.model, a.ckpt_dir), tmp, len(clip))

    # 讀回之前的提示點（同一個 clip 重開時）
    sess_path = os.path.join(a.out_dir, "sessions", f"{clip_name}.json")
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
            for oid in sess.obj_ids:
                for f in set(sess.points.get(oid, {})) | set(sess.boxes.get(oid, {})):
                    sess.prompt(f, oid)
            print(f"已讀回之前的提示點（{len(sess.prompted_frames())} 張影格），按 p 重新追蹤")

    view = View(W, H, a.display_width)
    st = {"t": 0, "oid": sess.obj_ids[0], "box_mode": False, "drag": None, "mouse": (W / 2, H / 2),
          "show": True, "msg": "click on the needle, then press p"}

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
            if i in excluded or i not in sess.masks:
                for pth in (lab, img_dst, ov_dst):        # 被排除 / 沒追蹤到的影格不輸出
                    if os.path.exists(pth):
                        os.remove(pth)
                continue
            m = np.zeros((H, W, 3), np.uint8)
            for bm in sess.masks[i].values():
                m[bm] = NEEDLE_BGR
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
        st["msg"] = f"saved {n_saved} frames -> {a.out_dir}"
        print(st["msg"])

    def on_mouse(ev, mx, my, flags, _):
        x, y = view.to_img(mx, my)
        st["mouse"] = (x, y)
        t, oid = st["t"], st["oid"]
        if st["box_mode"]:
            if ev == cv2.EVENT_LBUTTONDOWN:
                st["drag"] = [x, y, x, y]
            elif ev == cv2.EVENT_MOUSEMOVE and st["drag"]:
                st["drag"][2:] = [x, y]
            elif ev == cv2.EVENT_LBUTTONUP and st["drag"]:
                x1, y1, x2, y2 = st["drag"]
                st["drag"] = None
                if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                    sess.boxes.setdefault(oid, {})[t] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    sess.prompt(t, oid)
                    st["box_mode"] = False
                    st["msg"] = "box added; add +/- points if needed, then p"
            return
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            lab = 1 if ev == cv2.EVENT_LBUTTONDOWN else 0
            sess.points.setdefault(oid, {}).setdefault(t, []).append((float(x), float(y), lab))
            sess.prompt(t, oid)
            st["msg"] = f"{'+' if lab else '-'} point on needle {oid}"

    win = "SAM2 needle labeling"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        t, oid = st["t"], st["oid"]
        img = imgs[t].copy()
        if st["show"]:
            for o, bm in sess.masks.get(t, {}).items():
                c = OBJ_COLORS[(o - 1) % len(OBJ_COLORS)]
                img[bm] = (0.45 * img[bm] + 0.55 * np.array(c)).astype(np.uint8)
        disp = view.render(img)
        if st["show"]:                                     # 細針：再描一圈輪廓，放大時也看得到
            for o, bm in sess.masks.get(t, {}).items():
                c = OBJ_COLORS[(o - 1) % len(OBJ_COLORS)]
                cs, _ = cv2.findContours(bm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cc in cs:
                    pts = np.array([view.to_disp(p[0][0], p[0][1]) for p in cc], np.int32)
                    cv2.polylines(disp, [pts], True, c, 1, cv2.LINE_AA)
        for o in sess.obj_ids:                              # 提示點 / 框
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

        bar = np.zeros((54, disp.shape[1], 3), np.uint8)
        n_tracked = sum(1 for i in range(len(clip)) if i in sess.masks and i not in excluded)
        info = (f"frame {t + 1}/{len(clip)} [{clip[t]}]  needle {oid}/{len(sess.obj_ids)}  "
                f"zoom {view.zf}x  {'BOX MODE  ' if st['box_mode'] else ''}"
                f"{'EXCLUDED  ' if t in excluded else ''}tracked {n_tracked}  | {st['msg']}")
        cv2.putText(bar, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 128, 255) if t in excluded else (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, HELP, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow(win, np.vstack([disp, bar]))

        k = cv2.waitKeyEx(30)
        if k == -1:
            continue
        c = k & 0xFF
        if c in (ord("d"),) or k in (65363, 2555904):
            st["t"] = min(t + 1, len(clip) - 1)
        elif c in (ord("a"),) or k in (65361, 2424832):
            st["t"] = max(t - 1, 0)
        elif c == ord("l"):
            st["t"] = min(t + 10, len(clip) - 1)
        elif c == ord("j"):
            st["t"] = max(t - 10, 0)
        elif c == ord("b"):
            st["box_mode"] = not st["box_mode"]
            st["msg"] = "drag a box around the needle" if st["box_mode"] else ""
        elif c == ord("n"):
            new = max(sess.obj_ids) + 1
            sess.obj_ids.append(new)
            st["oid"] = new
            st["msg"] = f"new needle {new}: click on it"
        elif c == ord("]"):
            st["oid"] = sess.obj_ids[(sess.obj_ids.index(oid) + 1) % len(sess.obj_ids)]
        elif c == ord("["):
            st["oid"] = sess.obj_ids[(sess.obj_ids.index(oid) - 1) % len(sess.obj_ids)]
        elif c == ord("c"):
            sess.clear(t, oid)
            st["msg"] = f"cleared needle {oid} on this frame"
        elif c == ord("z"):
            view.zf = {1: 2, 2: 4, 4: 1}[view.zf]
            view.cx, view.cy = st["mouse"]
        elif c == ord("x"):
            excluded.symmetric_difference_update({t})
        elif c == ord("m"):
            st["show"] = not st["show"]
        elif c == ord("p"):
            st["msg"] = "propagating ..."
            cv2.imshow(win, np.vstack([disp, bar]))
            cv2.waitKey(1)
            t0 = time.time()
            try:
                ok = sess.propagate(lambda n: print(f"  propagated {n} frames", end="\r"))
                st["msg"] = (f"propagated in {time.time() - t0:.0f}s - check frames with a/d, fix, p again"
                             if ok else "no prompts yet: click on the needle first")
            except RuntimeError as e:
                # 例如某根針的提示全部被清掉：SAM 2 要求每根針至少在一張影格有提示
                st["msg"] = f"propagate failed: {str(e)[:80]} (give every needle at least one click)"
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
