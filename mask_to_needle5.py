#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從針的 segmentation mask 自動算出 5 個點（依序）:
    0 head      針頭 (針尖)
    1 tail      針尾
    2 middle    針的中間        (弧長 1/2)
    3 head_1_3  靠針頭的 1/3    (從針頭量起弧長 1/3)
    4 tail_1_3  靠針尾的 1/3    (從針頭量起弧長 2/3)

流程（每根針）:
    1. 依 color.txt 取出 Needle 類別的像素
    2. 被夾爪擋住而斷開的碎片：距離 <= --max_gap 的碎片視為同一根針
    3. 每塊碎片做細線化 (Zhang-Suen) 取中心線，再把碎片依「斷口總長度最短」的順序接起來
    4. 沿整條中心線算弧長，取出 5 個點
    5. 針頭是哪一端：mask 分不出來 -> 互動模式開視窗讓你按 1 / 2 選
       （決定會存在 head_decisions.json，中斷後重跑會從沒標過的繼續）

資料夾結構（Images 與 Labels 底下的相對路徑與檔名要對得上，副檔名可以不同）:
    DATA/Images/train/xxx.png   DATA/Labels/train/xxx.png
    DATA/Images/val/...         DATA/Labels/val/...
    DATA/Images/test/...        DATA/Labels/test/...

用法:
    # 1) 先不開視窗，只算幾何並輸出檢查圖（頭尾尚未確認）
    python mask_to_needle5.py --images_dir DATA/Images --labels_dir DATA/Labels \
        --color_txt DATA/color.txt --out_dir needle5_dataset --mode auto

    # 2) 互動標頭尾（1 = 綠色那端是針頭，2 = 紫色那端是針頭，s 跳過，b 排除，q 存檔離開）
    python mask_to_needle5.py --images_dir DATA/Images --labels_dir DATA/Labels \
        --color_txt DATA/color.txt --out_dir needle5_dataset --mode interactive

輸出:
    needle5_dataset/annotations.json      每根針一筆
    needle5_dataset/head_decisions.json   你按過的頭尾決定
    needle5_dataset/vis/<split>/*.png     5 點檢查圖（2px 圓：頭紅、尾藍、中綠、頭1/3橘、尾1/3青）
"""

import argparse
import itertools
import json
import os
import re
from collections import deque

import cv2
import numpy as np

POINT_NAMES = ["head", "tail", "middle", "head_1_3", "tail_1_3"]
POINT_FRACS = [0.0, 1.0, 0.5, 1 / 3, 2 / 3]          # 從針頭量起的弧長比例
POINT_COLORS = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 140, 255), (255, 255, 0)]  # BGR
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# -----------------------------------------------------------------------------
# color.txt / mask
# -----------------------------------------------------------------------------
def parse_color_txt(path):
    """'Needle: 2, [0,0,255]' -> {'needle': (2, (0,0,255))}"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*(.+?)\s*:\s*(\d+)\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", line)
            if m:
                out[m.group(1).strip().lower()] = (int(m.group(2)),
                                                    (int(m.group(3)), int(m.group(4)), int(m.group(5))))
    return out


def needle_mask(mask, class_id, color, tol):
    """支援彩色 mask（依 color.txt 的顏色，BGR 順序比對）與單通道 index mask。"""
    if mask.ndim == 2:
        return mask == class_id
    m = mask[:, :, :3].astype(np.int16)
    return np.all(np.abs(m - np.array(color, dtype=np.int16)) <= tol, axis=2)


# -----------------------------------------------------------------------------
# 幾何
# -----------------------------------------------------------------------------
def zhang_suen(img):
    """Zhang-Suen 細線化（純 numpy），img: bool。"""
    im = np.pad(img.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P2 = im[:-2, 1:-1]; P3 = im[:-2, 2:]; P4 = im[1:-1, 2:]; P5 = im[2:, 2:]
            P6 = im[2:, 1:-1]; P7 = im[2:, :-2]; P8 = im[1:-1, :-2]; P9 = im[:-2, :-2]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(n.astype(np.int32) for n in nb)
            seq = nb + [P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                c = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                c = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            rm = (im[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                im[1:-1, 1:-1][rm] = 0
                changed = True
    return im[1:-1, 1:-1].astype(bool)


def longest_path(skel):
    """skeleton 上最長的路徑（兩次 BFS），回傳 (N, 2) 的 (x, y)。"""
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return None
    pts = set(zip(xs.tolist(), ys.tolist()))
    nbrs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    def bfs(src):
        prev = {src: None}
        q = deque([src])
        last = src
        while q:
            p = q.popleft()
            last = p
            for dx, dy in nbrs:
                n = (p[0] + dx, p[1] + dy)
                if n in pts and n not in prev:
                    prev[n] = p
                    q.append(n)
        return last, prev

    a, _ = bfs(next(iter(pts)))
    b, prev = bfs(a)
    path = []
    p = b
    while p is not None:
        path.append(p)
        p = prev[p]
    return np.array(path[::-1], dtype=np.float64)


def extend_ends(path, comp_pts, thickness):
    """細線化會讓兩端各縮短約半個線寬：沿端點切線方向，往外延伸到 mask 在「正前方」最遠的像素。
    只看距離端點 <= 1.5 倍線寬、且離切線很近的像素，避免跳到弧線的其他部分。"""
    if len(path) < 3:
        return path
    reach = max(3.0, 1.5 * thickness)
    lateral = max(1.5, 0.75 * thickness)
    out = [path]
    k = min(5, len(path) - 1)
    for end, ref, prepend in ((path[0], path[k], True), (path[-1], path[-1 - k], False)):
        t = end - ref
        nt = np.linalg.norm(t)
        if nt < 1e-6:
            continue
        t /= nt
        d = comp_pts - end
        along = d @ t
        perp = np.abs(d @ np.array([-t[1], t[0]]))
        ok = (along > 0.5) & (along <= reach) & (perp <= lateral)
        if ok.any():
            q = end + t * along[ok].max()
            if prepend:
                out.insert(0, q[None])
            else:
                out.append(q[None])
    return np.concatenate(out, axis=0)


def min_dist(a, b, max_pts=400):
    if len(a) > max_pts:
        a = a[np.linspace(0, len(a) - 1, max_pts).astype(int)]
    if len(b) > max_pts:
        b = b[np.linspace(0, len(b) - 1, max_pts).astype(int)]
    return float(np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min())


def chain_segments(segs, exhaustive_max=4):
    """把多段中心線接成一條，回傳 (polyline, 總斷口長度)。
    段數 <= exhaustive_max：窮舉順序與方向，取斷口總長最短；
    段數更多（通常是 mask 有很多零碎小塊）：貪婪法——從最長的一段開始，每次把最近的一段接到頭或尾。"""
    if len(segs) == 1:
        return segs[0], 0.0
    if len(segs) <= exhaustive_max:
        best = None
        for order in itertools.permutations(range(len(segs))):
            if order[0] > order[-1]:          # 反向的排列等價，省一半
                continue
            for flips in itertools.product((False, True), repeat=len(segs)):
                parts = [segs[i][::-1] if f else segs[i] for i, f in zip(order, flips)]
                gap = sum(np.linalg.norm(parts[k][-1] - parts[k + 1][0]) for k in range(len(parts) - 1))
                if best is None or gap < best[0]:
                    best = (gap, parts)
        return np.concatenate(best[1], axis=0), float(best[0])

    lengths = [np.linalg.norm(np.diff(sg, axis=0), axis=1).sum() if len(sg) > 1 else 0 for sg in segs]
    first = int(np.argmax(lengths))
    chain = [segs[first]]
    rest = [sg for i, sg in enumerate(segs) if i != first]
    total = 0.0
    while rest:
        head, tail = chain[0][0], chain[-1][-1]
        best = None                                   # (距離, index, 接在尾巴?, 是否反向)
        for i, sg in enumerate(rest):
            for at_tail, ref in ((True, tail), (False, head)):
                for rev in (False, True):
                    end = sg[-1] if rev else sg[0]    # 接在尾巴：sg 的起點要靠近 tail
                    if not at_tail:
                        end = sg[0] if rev else sg[-1]  # 接在頭：sg 的終點要靠近 head
                    d = float(np.linalg.norm(end - ref))
                    if best is None or d < best[0]:
                        best = (d, i, at_tail, rev)
        d, i, at_tail, rev = best
        sg = rest.pop(i)
        sg = sg[::-1] if rev else sg
        if at_tail:
            chain.append(sg)
        else:
            chain.insert(0, sg)
        total += d
    return np.concatenate(chain, axis=0), total


def points_along(poly, fracs):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    return np.stack([np.interp(np.array(fracs) * L, s, poly[:, 0]),
                     np.interp(np.array(fracs) * L, s, poly[:, 1])], axis=1), float(L)


def extract_needles(nmask, args):
    """回傳每根針的 dict：polyline（從端點 A 到端點 B）、bbox、旗標。"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(nmask.astype(np.uint8), 8)
    comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= args.min_area]
    if not comps:
        return []
    pts = {i: np.stack(np.nonzero(lab == i)[::-1], axis=1).astype(np.float64) for i in comps}

    # union-find：距離 <= max_gap 的碎片歸為同一根針
    parent = {i: i for i in comps}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in itertools.combinations(comps, 2):
        if min_dist(pts[a], pts[b]) <= args.max_gap:
            parent[find(a)] = find(b)
    groups = {}
    for i in comps:
        groups.setdefault(find(i), []).append(i)

    needles = []
    for members in groups.values():
        segs = []
        for i in members:
            x, y, w, h = st[i, :4]
            comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
            # 線寬 = 距離轉換最大值 * 2
            thickness = 2.0 * float(cv2.distanceTransform(np.pad(comp, 1), cv2.DIST_L2, 3).max())
            sk = zhang_suen(comp.astype(bool))
            path = longest_path(sk)
            if path is None or len(path) < 2:
                path = pts[i].mean(0, keepdims=True).repeat(2, axis=0)
            else:
                path += np.array([x, y], dtype=np.float64)
                path = extend_ends(path, pts[i], thickness)
            segs.append(path)
        poly, gap = chain_segments(segs)
        allpts = np.concatenate([pts[i] for i in members])
        x1, y1 = allpts.min(0)
        x2, y2 = allpts.max(0)
        five, L = points_along(poly, POINT_FRACS)
        chord = np.linalg.norm(poly[-1] - poly[0])
        needles.append({
            "poly": poly,
            "bbox": [float(x1), float(y1), float(x2 + 1), float(y2 + 1)],
            "length_px": L,
            "n_fragments": len(members),
            "bridged_gap_px": round(gap, 1),
            "straightness": round(float(chord / max(L, 1e-6)), 3),   # ~1 代表幾乎是直線（側面看的針）
            "area": int(len(allpts)),
        })
    needles.sort(key=lambda d: d["bbox"][0])                         # 由左到右編號，結果穩定
    return needles


# -----------------------------------------------------------------------------
# 畫圖 / 互動
# -----------------------------------------------------------------------------
def draw_five(img, five, r=2):
    for p, c in zip(five, POINT_COLORS):
        cv2.circle(img, tuple(np.round(p).astype(int)), r, c, -1, cv2.LINE_AA)
    return img


def draw_legend(img):
    y = 18
    for name, c in zip(POINT_NAMES, POINT_COLORS):
        cv2.circle(img, (12, y - 4), 4, c, -1, cv2.LINE_AA)
        cv2.putText(img, name, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, name, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return img


HELP_LINES = [
    "Choose which end is the needle HEAD (sharp tip). Tail = end where the thread attaches.",
    "1: green end is HEAD   2: magenta end is HEAD   b: bad (exclude)   SPACE / Enter: accept suggestion",
    "n / d / ->: next (no decision)   p / a / <-: previous   q / Esc: save & quit",
]


def render_task(rgb, nd, dec, title, view=720, pad=120, r=3, suggest=None):
    """左：同一個裁切範圍的原圖放大，不畫任何東西（方便看清楚針頭 / 針尾）；
    右：同樣的裁切，畫中心線、兩端點（1 / 2）與依目前選擇算出的 5 點。"""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = nd["bbox"]
    cx0, cy0 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
    cx1, cy1 = int(min(x2 + pad, W)), int(min(y2 + pad, H))
    crop = rgb[cy0:cy1, cx0:cx1].copy()
    s = view / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    clean = crop.copy()                                       # 左側用：完全不標註
    P = lambda p: tuple(np.round((np.asarray(p) - [cx0, cy0]) * s).astype(int))

    poly = nd["poly"]
    for k in range(len(poly) - 1):
        cv2.line(crop, P(poly[k]), P(poly[k + 1]), (0, 255, 255), 1, cv2.LINE_AA)
    a, b = P(poly[0]), P(poly[-1])
    cv2.circle(crop, a, 12, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(crop, "1", (a[0] + 12, a[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.circle(crop, b, 12, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(crop, "2", (b[0] + 12, b[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # 已選頭尾（或有建議時）-> 畫出 5 點預覽
    show = dec if dec in ("A", "B") else (suggest if dec is None else None)
    if show in ("A", "B"):
        pl = poly if show == "A" else poly[::-1]
        five, _ = points_along(pl, POINT_FRACS)
        for p, c in zip(five, POINT_COLORS):
            cv2.circle(crop, P(p), r, c, -1, cv2.LINE_AA)
        draw_legend(crop)

    canvas = np.hstack([clean, np.full((crop.shape[0], 6, 3), 40, np.uint8), crop])

    status = {"A": "HEAD = 1 (green)", "B": "HEAD = 2 (magenta)", "bad": "BAD (excluded)",
              None: "not decided"}.get(dec, str(dec))
    color = (0, 255, 255) if dec else (180, 180, 180)
    if dec is None and suggest in ("A", "B"):
        status = f"SUGGESTED HEAD = {'1 (green)' if suggest == 'A' else '2 (magenta)'}  -> SPACE to accept"
        color = (0, 165, 255)
    bar = np.zeros((26 * (len(HELP_LINES) + 2), canvas.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{title}   status: {status}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 1, cv2.LINE_AA)
    for j, t in enumerate(HELP_LINES):
        cv2.putText(bar, t, (8, 22 + 26 * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([canvas, bar])


def suggest_head(tasks, i, decisions, window=30, max_jump=150.0, min_margin=0.25):
    """連續影格：用最近一張已標好的影格（同一個資料夾），推測這根針哪一端是頭。
    找同資料夾、前後 window 根針內最近的已決定影格，取中心最接近的那根針，
    比較「1 端靠近上一張的頭」與「2 端靠近上一張的頭」哪個比較合理。回傳 'A' / 'B' / None。"""
    key, rel, nd = tasks[i]
    folder = os.path.dirname(rel)
    c = np.array([(nd["bbox"][0] + nd["bbox"][2]) / 2, (nd["bbox"][1] + nd["bbox"][3]) / 2])
    best = None
    for off in range(1, window + 1):
        for j in (i - off, i + off):
            if not (0 <= j < len(tasks)):
                continue
            k2, r2, n2 = tasks[j]
            if r2 == rel or os.path.dirname(r2) != folder or decisions.get(k2) not in ("A", "B"):
                continue
            c2 = np.array([(n2["bbox"][0] + n2["bbox"][2]) / 2, (n2["bbox"][1] + n2["bbox"][3]) / 2])
            d = float(np.linalg.norm(c - c2))
            if d <= max_jump and (best is None or (off, d) < best[0]):
                best = ((off, d), n2, decisions[k2])
        if best is not None:
            break
    if best is None:
        return None
    _, n2, d2 = best
    p2 = n2["poly"] if d2 == "A" else n2["poly"][::-1]
    head, tail = p2[0], p2[-1]
    poly = nd["poly"]
    cost_a = np.linalg.norm(poly[0] - head) + np.linalg.norm(poly[-1] - tail)
    cost_b = np.linalg.norm(poly[-1] - head) + np.linalg.norm(poly[0] - tail)
    if abs(cost_a - cost_b) < min_margin * max(cost_a, cost_b, 1e-6):
        return None                                     # 兩種都差不多（例如針轉了很多）：不給建議
    return "A" if cost_a < cost_b else "B"


def run_ui(tasks, rgb_loader, decisions, dec_path, goto=None, review=False, use_suggest=True):
    """tasks: [(key, rel, nd)]。可前後翻頁，每次決定立即存檔（已標過的也可以重按 1 / 2 / b 覆蓋）。
    起點：--goto 指定的圖 > --review 從第 1 根 > 第一個未決定的。"""
    win = "needle head labeling"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if goto:
        i = next((k for k, t in enumerate(tasks) if goto in t[1]), None)
        if i is None:
            print(f"[WARN] 找不到檔名包含 '{goto}' 的圖，從頭開始")
            i = 0
    elif review:
        i = 0
    else:
        i = next((k for k, t in enumerate(tasks) if decisions.get(t[0]) not in ("A", "B", "bad")), 0)
    cache = {}
    while 0 <= i < len(tasks):
        key, rel, nd = tasks[i]
        if rel not in cache:
            cache.clear()
            cache[rel] = rgb_loader(rel)
        n_done = sum(decisions.get(t[0]) in ("A", "B", "bad") for t in tasks)
        title = f"[{i + 1}/{len(tasks)}] {rel} needle {nd['id']}   done {n_done}/{len(tasks)}"
        sug = suggest_head(tasks, i, decisions) if use_suggest and decisions.get(key) is None else None
        cv2.imshow(win, render_task(cache[rel], nd, decisions.get(key), title, suggest=sug))
        k = cv2.waitKeyEx(0)
        c = k & 0xFF
        if c in (ord("1"), ord("2"), ord("b")) or (c in (32, 13, 10) and sug):
            decisions[key] = sug if c in (32, 13, 10) else {"1": "A", "2": "B", "b": "bad"}[chr(c)]
            with open(dec_path, "w") as f:
                json.dump(decisions, f, indent=1)
            i += 1
        elif c in (ord("n"), ord("d")) or k in (65363, 2555904):          # 右方向鍵
            i += 1
        elif c in (ord("p"), ord("a")) or k in (65361, 2424832):          # 左方向鍵
            i = max(i - 1, 0)
        elif c in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
def find_pairs(images_dir, labels_dir):
    labels = {}
    for root, _, files in os.walk(labels_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                rel = os.path.relpath(os.path.join(root, f), labels_dir)
                labels[os.path.splitext(rel)[0]] = os.path.join(root, f)
    pairs = []
    for root, _, files in os.walk(images_dir):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTS):
                rel = os.path.relpath(os.path.join(root, f), images_dir)
                key = os.path.splitext(rel)[0]
                if key in labels:
                    pairs.append((rel, labels[key]))
    pairs.sort()
    return pairs


def split_of(rel):
    first = rel.replace("\\", "/").split("/")[0].lower()
    return {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}.get(first, "train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--color_txt", required=True)
    ap.add_argument("--out_dir", default="needle5_dataset")
    ap.add_argument("--mode", choices=["auto", "interactive"], default="interactive",
                    help="auto: 只算幾何並輸出（沿用 head_decisions.json 裡已標的頭尾）；interactive: 開視窗標頭尾")
    ap.add_argument("--needle_name", default="needle", help="color.txt 裡針的類別名稱")
    ap.add_argument("--tol", type=int, default=10, help="顏色比對容許誤差")
    ap.add_argument("--min_area", type=int, default=15, help="小於此面積的碎片忽略 (px)")
    ap.add_argument("--max_gap", type=float, default=60, help="碎片距離小於此值視為同一根針 (px)")
    ap.add_argument("--min_length", type=float, default=20, help="總長小於此值的針不輸出 (px)")
    ap.add_argument("--max_fragments", type=int, default=6,
                    help="mask 碎片數超過此值的針直接排除（不標註、不輸出）；0 = 不限制")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 張（測試用）")
    ap.add_argument("--goto", default=None, help="互動模式直接跳到檔名包含此字串的圖，例如 --goto test/left_34")
    ap.add_argument("--review", action="store_true", help="互動模式從第 1 根針開始（檢查 / 修改已標過的）")
    ap.add_argument("--no_suggest", action="store_true", help="關閉「依前後影格自動建議針頭」")
    args = ap.parse_args()

    colors = parse_color_txt(args.color_txt)
    if args.needle_name.lower() not in colors:
        raise SystemExit(f"color.txt 裡找不到 '{args.needle_name}'，現有類別：{list(colors)}")
    nid, ncolor = colors[args.needle_name.lower()]
    print(f"Needle: id={nid} color(BGR)={ncolor}")

    os.makedirs(args.out_dir, exist_ok=True)
    dec_path = os.path.join(args.out_dir, "head_decisions.json")
    decisions = json.load(open(dec_path)) if os.path.isfile(dec_path) else {}

    pairs = [p for p in find_pairs(args.images_dir, args.labels_dir) if split_of(p[0]) in args.splits]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"找到 {len(pairs)} 組 影像/mask")

    # 1) 先算所有針的幾何
    stats = {"no_needle": 0, "too_short": 0, "fragmented": 0, "bad": 0, "unconfirmed": 0}
    fragmented = []
    per_image = []                                            # (rel, mpath, (H, W), [needles])
    for i, (rel, mpath) in enumerate(pairs):
        mask = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"  [WARN] 讀不到 mask {mpath}")
            continue
        needles = extract_needles(needle_mask(mask, nid, ncolor, args.tol), args)
        keep = []
        for k, nd in enumerate(needles):
            nd["id"] = k
            if nd["length_px"] < args.min_length:
                stats["too_short"] += 1
            elif args.max_fragments > 0 and nd["n_fragments"] > args.max_fragments:
                stats["fragmented"] += 1
                fragmented.append(f"{rel} needle {k}（{nd['n_fragments']} 個碎片）")
            else:
                keep.append(nd)
        if not keep:
            stats["no_needle"] += 1
            continue
        per_image.append((rel, mpath, mask.shape[:2], keep))
        if (i + 1) % 20 == 0:
            print(f"  幾何 {i + 1}/{len(pairs)}")

    if fragmented:
        print(f"\n排除 {len(fragmented)} 根碎片超過 {args.max_fragments} 個的針（mask 品質不佳，不加入訓練）：")
        for x in fragmented:
            print(f"  - {x}")
        print("  想保留的話，可以在 sam2_video_label.py 用橡皮擦把零碎的像素清掉，或調大 --max_fragments\n")

    # 舊版存的 "skip" 一律視為未決定
    decisions = {k: v for k, v in decisions.items() if v in ("A", "B", "bad")}

    # 2) 互動標頭尾
    if args.mode == "interactive":
        tasks = [(f"{rel}#{nd['id']}", rel, nd) for rel, _, _, nds in per_image for nd in nds]
        run_ui(tasks, lambda rel: cv2.imread(os.path.join(args.images_dir, rel)), decisions, dec_path,
               goto=args.goto, review=args.review, use_suggest=not args.no_suggest)

    # 3) 輸出
    samples = []
    for rel, mpath, (H, W), nds in per_image:
        rgb = cv2.imread(os.path.join(args.images_dir, rel))
        if rgb is None or rgb.shape[:2] != (H, W):
            print(f"  [WARN] 影像讀不到或尺寸與 mask 不同：{rel}")
            continue
        split = split_of(rel)
        vis = draw_legend(rgb.copy())
        for nd in nds:
            dec = decisions.get(f"{rel}#{nd['id']}")
            if dec == "bad":
                stats["bad"] += 1
                continue
            poly = nd["poly"] if dec != "B" else nd["poly"][::-1]
            five, _ = points_along(poly, POINT_FRACS)
            confirmed = dec in ("A", "B")
            if not confirmed:
                stats["unconfirmed"] += 1
            draw_five(vis, five, args.radius)
            samples.append({
                "image": rel.replace("\\", "/"),
                "mask": os.path.relpath(mpath, args.labels_dir).replace("\\", "/"),
                "split": split,
                "needle_id": nd["id"],
                "width": W, "height": H,
                "point_names": POINT_NAMES,
                "points_px": np.round(five, 2).tolist(),
                "points": np.round(five / [W, H], 6).tolist(),
                "needle_bbox": [round(v, 1) for v in nd["bbox"]],
                "head_confirmed": confirmed,
                "length_px": round(nd["length_px"], 1),
                "n_fragments": nd["n_fragments"],
                "bridged_gap_px": nd["bridged_gap_px"],
                "straightness": nd["straightness"],
            })
        vdir = os.path.join(args.out_dir, "vis", split)
        os.makedirs(vdir, exist_ok=True)
        cv2.imwrite(os.path.join(vdir, os.path.splitext(os.path.basename(rel))[0] + ".png"), vis)

    out = {
        "images_root": os.path.abspath(args.images_dir),
        "labels_root": os.path.abspath(args.labels_dir),
        "point_names": POINT_NAMES,
        "point_fracs_from_head": POINT_FRACS,
        "coordinate_convention": "points: x/W, y/H；points_px: 原圖 pixel",
        "samples": samples,
    }
    with open(os.path.join(args.out_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    by_split = {s: sum(x["split"] == s for x in samples) for s in ("train", "val", "test")}
    print(f"\n完成：{len(samples)} 根針 {by_split}")
    print(f"  略過：{stats}")
    if stats["unconfirmed"]:
        print("  [NOTE] 有頭尾尚未確認的針（head_confirmed=false），請用 --mode interactive 標完再拿去訓練")
    print(f"  → {os.path.join(args.out_dir, 'annotations.json')}")


if __name__ == "__main__":
    main()