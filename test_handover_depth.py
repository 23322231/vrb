#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 C：handover_depth.py 的獨立測試腳本。放在 vrb 根目錄，在 container 裡執行。

1) 不用相機、不用 GPU 的自我檢查（合成的針 + 背景，檢查 ROI／放大／color→infra1／查點的幾何）：
    python test_handover_depth.py --selftest

2) 量測（capture_d405_stereo.py 錄的資料夾）。針上的點用下列其中一種方式指定（都是 color 影像的像素）：
    python test_handover_depth.py --dir captures/z350_a --click                 # 第一張圖上用滑鼠點針上的一點
    python test_handover_depth.py --dir captures/z350_a --uv 652,371
    python test_handover_depth.py --dir captures/z350_a --yolo runs/segment/train/weights/best.pt   # 用 YOLO 的框與 mask，取 mask 的中心點
   其他：
    --backend fs|ffs|sgbm      預設 fs（FoundationStereo ViT-L）
    --known_z_mm 350           已知的 Z（相機光軸方向、從 infra1 原點算起；D405 的原點在前玻璃後方 3.7 mm）
    --upscale 2 --iters 32     ROI 放大倍數、refine 次數
    --full                     不切 ROI，整張 1280x720 丟進去（看記憶體放不放得下）
   輸出：<dir>/depth_test_<backend>/ 的 results.csv、summary.json、每張的視差疊圖。
   同一個點同時用「原生深度」查一次（一樣的查法）當對照。

3) 位移檢查（不需要知道絕對距離）：手臂夾著針沿直線移動已知距離，前後各錄一組、各跑一次 2)，再：
    python test_handover_depth.py --displacement captures/p0/depth_test_fs captures/p1/depth_test_fs --true_mm 20
"""

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handover_depth import D405Calib, StereoDepth, lookup_disparity, make_backend  # noqa: E402


# -----------------------------------------------------------------------------
# 自我檢查
# -----------------------------------------------------------------------------
def _synthetic(calib, z_bg=0.37, z_needle=0.33, width=2.5, seed=0):
    """合成一組左右影像：有紋理的平面背景 + 一根細的弧形「針」浮在前面。回傳 left, right, GT 視差, 針 mask, 針上一點。"""
    rng = np.random.default_rng(seed)
    H, W = calib.height, calib.width
    tex = cv2.GaussianBlur(rng.integers(0, 255, (H, W + 400), dtype=np.uint8), (0, 0), 2.0)
    tex = cv2.normalize(tex, None, 60, 200, cv2.NORM_MINMAX).astype(np.uint8)
    d_bg, d_n = calib.z_to_disp(z_bg), calib.z_to_disp(z_needle)
    left = tex[:, 200:200 + W].copy()
    M = np.float32([[1, 0, -(200 + d_bg)], [0, 1, 0]])                       # right(x) = tex(x + 200 + d_bg) → left 的 x 對到 right 的 x - d_bg
    right = cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    t = np.linspace(np.deg2rad(200), np.deg2rad(340), 80)
    pts = np.stack([640 + 75 * np.cos(t), 330 - 75 * np.sin(t)], 1)        # 半徑 75 px 的弧，約 150 px 寬
    mask = np.zeros((H, W), np.uint8)
    for img, dx in ((left, 0.0), (right, -d_n), (mask, 0.0)):
        p = np.round((pts + [dx, 0]) * 16).astype(np.int32)
        cv2.polylines(img, [p], False, 255 if img is mask else 235, max(1, int(round(width))), cv2.LINE_AA, shift=4)
    mask = (mask > 127).astype(np.uint8)
    gt = np.full((H, W), d_bg, np.float32)
    gt[mask > 0] = d_n
    return left, right, gt, mask, pts[len(pts) * 2 // 3]


class _GTBackend:
    """回傳事先算好的 GT 視差（照 StereoDepth 給的裁切與放大重新取樣），用來單獨檢查 ROI／放大／查點的邏輯。"""
    name, last_time, peak_mem_mb = "gt", 0.0, 0.0

    def __init__(self, gt, left_full):
        self.gt, self.left_full = gt, left_full

    def __call__(self, l, r):
        h, w = l.shape[:2]
        small = l if l.ndim == 2 else l[..., 0]
        best = None
        for s in (1.0, 2.0):                                               # 用樣板比對找回這塊 ROI 在原圖的位置與倍率
            tw, th = int(round(w / s)), int(round(h / s))
            templ = cv2.resize(small, (tw, th), interpolation=cv2.INTER_AREA) if s != 1 else small
            res = cv2.matchTemplate(self.left_full, templ, cv2.TM_SQDIFF)
            mn, _, loc, _ = cv2.minMaxLoc(res)
            if best is None or mn / (tw * th) < best[0]:
                best = (mn / (tw * th), loc, s, tw, th)
        _, (x0, y0), s, tw, th = best
        return cv2.resize(self.gt[y0:y0 + th, x0:x0 + tw], (w, h), interpolation=cv2.INTER_NEAREST) * s


def selftest():
    K = [[650.0, 0, 640.0], [0, 650.0, 360.0], [0, 0, 1]]
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"[{'OK' if cond else 'FAIL'}] {name} {detail}")

    # (a) color → infra1：視點相同、color 內參不同且帶畸變
    for model in ("none", "brown_conrady", "inverse_brown_conrady"):
        c = D405Calib(1280, 720, K, 0.018, K_color=[[640.0, 0, 648.0], [0, 641.0, 355.0], [0, 0, 1]],
                      dist_color=[-0.05, 0.06, 0.0005, -0.0004, -0.02], dist_model=model)
        uv = np.array([[100.0, 80.0], [640, 360], [1200, 650]])
        rays = c.color_rays(uv)
        if model == "brown_conrady":                                        # 把視線用 OpenCV 投影回去應該回到原像素
            back = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), c.K_color, c.dist_color)[0].reshape(-1, 2)
            check("brown_conrady 反投影→投影", np.abs(back - uv).max() < 1e-3, f"max {np.abs(back - uv).max():.2e} px")
        ul = c.color_uv_to_left(uv)
        xyz = c.left_uvz_to_color_xyz(ul, [0.3, 0.3, 0.3])
        check(f"color→infra1→3D 與視線一致（{model}）", np.abs(xyz[:, :2] / xyz[:, 2:] - rays[:, :2]).max() < 1e-9)

    # (b) ROI + 放大 + 查點：用 GT 視差
    calib = D405Calib(1280, 720, K, 0.018)
    left, right, gt, mask, p = _synthetic(calib)
    box = [565, 330, 715, 410]
    for up in (1.0, 2.0):
        sd = StereoDepth(_GTBackend(gt, left), calib, upscale=up)
        for m, tag in ((mask, "有 mask"), (None, "沒有 mask")):
            q = sd.locate(left, right, [p], box_color=box, mask_color=m)[0]
            err = abs(q["z"] - 0.33) * 1000
            check(f"GT 視差, upscale {up:.0f}, {tag}", err < 0.3 and q["ok"],
                  f"z = {q['z'] * 1000:.2f} mm（真值 330）, sep = {q['sep_px']:.2f} px, n = {q['n']}, {q['method']}")
    # 背景上的點應該查到背景
    q = StereoDepth(_GTBackend(gt, left), calib).locate(left, right, [(640, 300)], box_color=box)[0]
    check("背景上的點", abs(q["z"] - 0.37) < 3e-4, f"z = {q['z'] * 1000:.2f} mm（真值 370）")

    # (c) 真的做立體匹配（SGBM，僅供參考：它在 2~3 px 寬的針上本來就不可靠，不列入通過條件）
    sd = StereoDepth(make_backend("sgbm"), calib, upscale=2.0)
    q = sd.locate(left, right, [p], box_color=box, mask_color=mask)[0]
    print(f"[參考] SGBM：z = {None if q['z'] is None else round(q['z'] * 1000, 2)} mm（真值 330；背景 370）, "
          f"sep = {q['sep_px']}, ok = {q['ok']}, {sd.backend.last_time * 1000:.0f} ms")
    print("自我檢查", "全部通過" if ok_all else "有失敗")
    return 0 if ok_all else 1


# -----------------------------------------------------------------------------
# 量測
# -----------------------------------------------------------------------------
def _click(img):
    pt = []
    cv2.namedWindow("click the needle point, then press any key", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("click the needle point, then press any key",
                         lambda e, x, y, *_: pt.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    cv2.imshow("click the needle point, then press any key", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if not pt:
        raise SystemExit("沒有點到")
    return pt[-1]


def _yolo(model, color, conf):
    r = model.predict(color, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None, None
    i = int(r.boxes.conf.argmax())
    box = r.boxes.xyxy[i].cpu().numpy()
    mask = None
    if r.masks is not None:
        mask = cv2.resize(r.masks.data[i].cpu().numpy(), color.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0.5
    if mask is not None and mask.any():                                    # 取 mask 上離重心最近的像素
        ys, xs = np.nonzero(mask)
        k = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
        uv = (float(xs[k]), float(ys[k]))
    else:
        uv = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return uv, box, mask


def measure(args):
    calib = D405Calib.from_json(os.path.join(args.dir, "calib.json"))
    with open(os.path.join(args.dir, "calib.json")) as f:
        depth_scale = json.load(f).get("depth_scale")
    kw = {}
    if args.backend in ("fs", "ffs"):
        kw["valid_iters"] = args.iters if args.iters else (32 if args.backend == "fs" else 8)
        if args.ckpt:
            kw["ckpt"] = args.ckpt
        if args.repo:
            kw["repo_dir"] = args.repo
    backend = make_backend(args.backend, **kw)
    sd = StereoDepth(backend, calib, upscale=args.upscale, radius=args.radius)
    out_dir = os.path.join(args.dir, f"depth_test_{args.backend}")
    os.makedirs(out_dir, exist_ok=True)
    stems = sorted(s[:-len("_ir1.png")] for s in glob.glob(os.path.join(args.dir, "*_ir1.png")))
    if not stems:
        raise SystemExit(f"{args.dir} 裡沒有 *_ir1.png")
    yolo = None
    if args.yolo:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)
    fixed_uv = tuple(float(v) for v in args.uv.split(",")) if args.uv else None
    print(f"fx = {calib.fx:.2f} px, baseline = {calib.baseline * 1000:.3f} mm；1 px 視差在 "
          + "、".join(f"{z} cm ≈ {(z / 100) ** 2 / (calib.fx * calib.baseline) * 1000:.2f} mm" for z in (15, 25, 35)))

    rows = []
    for n, stem in enumerate(stems):
        l, r = cv2.imread(stem + "_ir1.png", 0), cv2.imread(stem + "_ir2.png", 0)
        color = cv2.imread(stem + "_color.png")
        uv, box, mask = fixed_uv, None, None
        if yolo is not None:
            uv, box, mask = _yolo(yolo, color, args.conf)
            if uv is None:
                print(f"{os.path.basename(stem)}: YOLO 沒找到針，跳過")
                continue
        elif uv is None:
            uv = fixed_uv = _click(color)
            print(f"點選的位置（之後可以直接用 --uv {uv[0]},{uv[1]}）")
        if args.full:
            res = sd.disparity(l, r, None)
            q = lookup_disparity(res.disp, calib.color_uv_to_left([uv])[0],
                                 None if mask is None else calib.color_mask_to_left(mask), radius=args.radius)
            q["z"] = None if q["disp"] is None else float(calib.disp_to_z(q["disp"]))
            q["xyz_cam"] = None if q["z"] is None else calib.left_uvz_to_color_xyz(calib.color_uv_to_left([uv]), q["z"])[0].tolist()
            q["uv_left"] = calib.color_uv_to_left([uv])[0].tolist()
        else:
            (q,), res = sd.locate(l, r, [uv], box_color=box, mask_color=mask, return_disp=True)
        row = {"frame": os.path.basename(stem), "u": uv[0], "v": uv[1], "z_mm": None if q["z"] is None else q["z"] * 1000,
               "disp_px": q["disp"], "sep_px": q["sep_px"], "mad_px": q["mad_px"], "n": q["n"], "method": q["method"],
               "x_mm": None, "y_mm": None, "native_z_mm": None, "native_n": 0,
               "time_ms": None if res.time_s is None else res.time_s * 1000, "peak_mem_mb": res.peak_mem_mb}
        if q["xyz_cam"]:
            row["x_mm"], row["y_mm"] = q["xyz_cam"][0] * 1000, q["xyz_cam"][1] * 1000
        if depth_scale and os.path.isfile(stem + "_depth.png"):            # 原生深度：換成視差後用同一個查法
            dn = cv2.imread(stem + "_depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32) * depth_scale
            with np.errstate(divide="ignore"):
                dn = np.where(dn > 0, calib.fx * calib.baseline / dn, np.nan).astype(np.float32)
            qn = lookup_disparity(dn, q["uv_left"], None if mask is None else calib.color_mask_to_left(mask), radius=args.radius)
            if qn["disp"] is not None:
                row["native_z_mm"], row["native_n"] = float(calib.disp_to_z(qn["disp"])) * 1000, qn["n"]
        rows.append(row)
        if n < args.save_vis:
            full = res.full()
            fin = np.isfinite(full)
            lo, hi = np.percentile(full[fin], [2, 98]) if fin.any() else (0, 1)
            vis = cv2.applyColorMap(np.uint8(np.clip((np.nan_to_num(full, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1) * 255), cv2.COLORMAP_TURBO)
            vis[~fin] = cv2.cvtColor(l, cv2.COLOR_GRAY2BGR)[~fin] // 2
            ul = np.round(q["uv_left"]).astype(int)
            for im in (vis, color):
                cv2.drawMarker(im, tuple(ul if im is vis else np.round(uv).astype(int)), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(stem) + "_disp.png"), np.hstack([color, vis]))
        print(f"{row['frame']}: z = {row['z_mm'] and round(row['z_mm'], 2)} mm, 原生 = {row['native_z_mm'] and round(row['native_z_mm'], 2)} mm, "
              f"sep = {row['sep_px'] and round(row['sep_px'], 2)} px, n = {row['n']}, {row['method']}, "
              f"{row['time_ms'] and round(row['time_ms'])} ms, GPU 尖峰 {row['peak_mem_mb'] and round(row['peak_mem_mb'])} MB")

    if not rows:
        raise SystemExit("沒有任何一張有結果")
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def stat(key):
        v = np.array([r[key] for r in rows if r[key] is not None], float)
        return None if len(v) == 0 else {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
    summ = {"dir": args.dir, "backend": args.backend, "upscale": args.upscale, "full": bool(args.full), "frames": len(rows),
            "found": sum(r["z_mm"] is not None for r in rows), "known_z_mm": args.known_z_mm,
            "z_mm": stat("z_mm"), "x_mm": stat("x_mm"), "y_mm": stat("y_mm"), "native_z_mm": stat("native_z_mm"),
            "native_found": sum(r["native_z_mm"] is not None for r in rows), "sep_px": stat("sep_px"),
            "time_ms_first": rows[0]["time_ms"], "time_ms_rest": stat("time_ms") if len(rows) < 2 else
            {"mean": float(np.mean([r["time_ms"] for r in rows[1:] if r["time_ms"] is not None] or [np.nan]))},
            "peak_mem_mb": stat("peak_mem_mb")}
    if args.known_z_mm is not None:
        for k in ("z_mm", "native_z_mm"):
            if summ[k]:
                summ[k]["error_vs_known"] = summ[k]["mean"] - args.known_z_mm
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    print(f"→ {out_dir}")


def displacement(a, b, true_mm):
    P = []
    for d in (a, b):
        with open(os.path.join(d, "summary.json")) as f:
            s = json.load(f)
        P.append(np.array([s["x_mm"]["mean"], s["y_mm"]["mean"], s["z_mm"]["mean"]]))
    dist = float(np.linalg.norm(P[1] - P[0]))
    print(f"P0 = {P[0].round(2)} mm, P1 = {P[1].round(2)} mm, |ΔP| = {dist:.2f} mm"
          + ("" if true_mm is None else f"，真值 {true_mm} mm，誤差 {dist - true_mm:+.2f} mm"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir")
    ap.add_argument("--backend", default="fs", choices=["fs", "ffs", "sgbm"])
    ap.add_argument("--repo")
    ap.add_argument("--ckpt")
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--uv")
    ap.add_argument("--click", action="store_true")
    ap.add_argument("--yolo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--known_z_mm", type=float)
    ap.add_argument("--save_vis", type=int, default=5)
    ap.add_argument("--displacement", nargs=2, metavar=("DIR_A", "DIR_B"))
    ap.add_argument("--true_mm", type=float)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.displacement:
        return displacement(*args.displacement, args.true_mm)
    if not args.dir:
        ap.error("要給 --dir（或 --selftest / --displacement）")
    if not (args.uv or args.click or args.yolo):
        ap.error("要用 --uv、--click 或 --yolo 指定針上的點")
    measure(args)


if __name__ == "__main__":
    main()
