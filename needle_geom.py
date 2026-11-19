#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover 對話 B：幾何解碼（decode=geom）。只用 numpy / cv2，不需要 GPU。

為什麼要有這個：VRB 在「被遞針手夾著、中段被指頭擋住」的針上，兩個 1/3 點會左右對調（現場 30 張有 8 張），
但「哪一端是針尖」大多判斷得對。所以把工作拆開：

    1. 針的形狀（中心線）    ← YOLO 的 mask（幾何，不靠學習去猜位置）
    2. 哪一端是針尖（head）   ← VRB 的預測（只拿來投票決定方向）
    3. 5 個點                ← 沿中心線依弧長比例取：head 0、head_1_3 1/3、middle 1/2、tail_1_3 2/3、tail 1

第 1 步完全沿用 mask_to_needle5.py 的 extract_needles（產生 GT 用的同一套）：
    mask → 每個碎片細線化（Zhang-Suen）取中心線 → 碎片若落在同一段圓弧上就合併 →
    被指頭擋住的空隙沿擬合的圓弧補起來 → 一條從端點 A 到端點 B 的折線 poly
所以「GT 5 點」與「幾何解碼的 5 點」定義完全相同；框與方向都對時，誤差只剩 mask 本身的誤差。

第 2 步（決定方向，decide_orientation）：poly 有兩種方向（A 端是針尖，或 B 端是針尖）。依序：
    (1) 線（有線的 mask 時，例如 YOLO 有 Thread 類別）：離線最近的那端 = 針尾。和 mask_to_needle5.py 產生 GT 時
        自動判斷頭尾的規則完全相同（thread_head：近端要在 15 px 內，兩端距離差要 > max(15 px, 0.15×針長)）。
        判斷不了（線不在任何一端附近、或兩端都靠近線）才交給 (2)。
    (2) VRB，三種模式：
        mode="heat"  （預設）直接在中心線兩端讀 VRB 的 head / tail heatmap：
                     s_A = head_hm(A) + tail_hm(B)，s_B = head_hm(B) + tail_hm(A)，大的勝出。
                     不管 VRB 的 argmax 落在哪（現場常見「head、tail 的 argmax 落在同一端」），只比兩端誰比較像 head。
        mode="ends"  VRB 預測的 head、tail 位置，和兩種方向下 head、tail 應在的位置比距離（信心加權），小的勝出
        mode="all"   同上，再加兩個 1/3 點（middle 兩種方向都一樣，沒有資訊）
    orient_conf = |s_A − s_B| / (s_A + s_B)，0 = 兩種方向一樣好（無法判斷）、1 = 非常確定；用線決定時 = 1
"""

from __future__ import annotations

import types

import cv2
import numpy as np

POINT_FRACS = {"head": 0.0, "tail": 1.0, "middle": 0.5, "head_1_3": 1 / 3, "tail_1_3": 2 / 3, "head_1_5": 0.2}
ANCHOR_NAMES = ["head", "tail", "middle", "head_1_3", "tail_1_3", "head_1_5"]     # decode=vrb_anchor 輸出的 6 點
# mask_to_needle5.py 的預設（碎片距離 <= 60 px 直接合併；<= 250 px 且在同一段圓弧上也合併）
GEOM_ARGS = types.SimpleNamespace(min_area=15, max_gap=60.0, merge_gap=250.0, one_needle=False)


def points_along(poly, fracs):
    """沿折線依弧長比例取點（和 mask_to_needle5.points_along 相同）。"""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    f = np.asarray(fracs, np.float64) * L
    return np.stack([np.interp(f, s, poly[:, 0]), np.interp(f, s, poly[:, 1])], axis=1), float(L)


def centerline(mask, seed_mask=None, args=GEOM_ARGS):
    """mask（HxW bool，可以是好幾個 YOLO 偵測的聯集）→ 一根針的中心線 dict（poly、bbox、length_px…），沒有就 None。
    同一張圖可能抽出好幾根「針」（真的有兩根針、或誤偵測）：取離 seed_mask（信心最高那個偵測的 mask）最近的那根。"""
    from mask_to_needle5 import extract_needles
    needles = extract_needles(mask.astype(bool), args)
    if not needles:
        return None
    if seed_mask is None or len(needles) == 1:
        return max(needles, key=lambda d: d["length_px"])
    dt = cv2.distanceTransform((~seed_mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)   # 到 seed 的距離
    H, W = dt.shape

    def score(nd):
        p = np.round(nd["poly"]).astype(int)
        p[:, 0] = p[:, 0].clip(0, W - 1)
        p[:, 1] = p[:, 1].clip(0, H - 1)
        return float(np.mean(dt[p[:, 1], p[:, 0]]))
    return min(needles, key=score)


def orient(poly, vrb_pts, vrb_conf, names, mode="ends"):
    """決定 poly 哪一端是針尖。vrb_pts (K,2) 原圖 pixel、vrb_conf (K,)、names 依序的點名。
    回傳 (head 在前的 poly, orient_conf, 說明 dict)。"""
    use = ["head", "tail"] if mode == "ends" else ["head", "tail", "head_1_3", "tail_1_3"]
    idx = [names.index(n) for n in use if n in names]
    fr = [POINT_FRACS[names[i]] for i in idx]
    w = np.clip(np.asarray(vrb_conf, np.float64)[idx], 1e-3, None)
    P = np.asarray(vrb_pts, np.float64)[idx]
    cand_A, _ = points_along(poly, fr)                 # A 端是針尖
    cand_B, _ = points_along(poly[::-1], fr)           # B 端是針尖
    sA = float(np.sum(w * np.linalg.norm(P - cand_A, axis=1)))
    sB = float(np.sum(w * np.linalg.norm(P - cand_B, axis=1)))
    conf = abs(sA - sB) / max(sA + sB, 1e-6)
    out = poly if sA <= sB else poly[::-1]
    return out, conf, {"score_A": round(sA, 1), "score_B": round(sB, 1), "mode": mode, "used": [names[i] for i in idx]}


def heat_at(hm_k, crop, p, r=2):
    """在 VRB 的一張 heatmap（56×56，蓋住整個裁切框）上讀原圖點 p 附近 (2r+1)² 格的最大值。
    座標約定和訓練一致：pn = (p − 裁切左上 + 0.5) / 邊長，格子 j 的中心在 pn = j / (56 − 1)。"""
    l, t, rr, b = crop
    Hh, Wh = hm_k.shape
    gx = (p[0] - l + 0.5) / (rr - l) * (Wh - 1)
    gy = (p[1] - t + 0.5) / (b - t) * (Hh - 1)
    x, y = int(round(gx)), int(round(gy))
    x0, x1, y0, y1 = max(0, x - r), min(Wh, x + r + 1), max(0, y - r), min(Hh, y + r + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(np.max(hm_k[y0:y1, x0:x1]))


def orient_heat(poly, hm, crop, names):
    """mode="heat"：在中心線兩端讀 head、tail heatmap。回傳 (head 在前的 poly, conf, 說明)。"""
    ih, it = names.index("head"), names.index("tail")
    A, B = poly[0], poly[-1]
    hA, hB = heat_at(hm[ih], crop, A), heat_at(hm[ih], crop, B)
    tA, tB = heat_at(hm[it], crop, A), heat_at(hm[it], crop, B)
    sA, sB = max(hA, 0) + max(tB, 0), max(hB, 0) + max(tA, 0)          # A 是針尖 / B 是針尖 的分數（大的勝）
    conf = abs(sA - sB) / max(sA + sB, 1e-6)
    out = poly if sA >= sB else poly[::-1]
    return out, conf, {"mode": "heat", "score_A": round(sA, 4), "score_B": round(sB, 4),
                       "head_hm_A": round(hA, 4), "head_hm_B": round(hB, 4), "tail_hm_A": round(tA, 4), "tail_hm_B": round(tB, 4)}


THREAD_ARGS = types.SimpleNamespace(thread_touch=15.0, thread_margin=15.0, thread_margin_ratio=0.15)


def orient_thread(poly, length_px, thread_mask, args=THREAD_ARGS):
    """線：離線最近的端點 = 針尾（mask_to_needle5.thread_head）。回傳 (head 在前的 poly 或 None, 說明)。"""
    from mask_to_needle5 import thread_head
    if thread_mask is None or not np.any(thread_mask):
        return None, {"thread": "no_thread_mask"}
    dt = cv2.distanceTransform((~thread_mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)
    dec, da, db, why = thread_head({"poly": poly, "length_px": length_px}, dt, args)
    info = {"thread": why, "thread_dist_A": None if da is None else round(da, 1), "thread_dist_B": None if db is None else round(db, 1)}
    if dec is None:
        return None, info
    return (poly if dec == "A" else poly[::-1]), info


def decide_orientation(poly, length_px, names, mode="heat", vrb_pts=None, vrb_conf=None, hm=None, crop=None,
                       thread_mask=None):
    """先用線，線判斷不了再用 VRB（mode）。回傳 (head 在前的 poly, orient_conf, 說明 dict（含 source））。"""
    out, tinfo = orient_thread(poly, length_px, thread_mask)
    if out is not None:
        return out, 1.0, {"source": "thread", **tinfo}
    if mode == "left":
        # 流程保證針頭在畫面左邊（2026-11-17 使用者確認）：中心線 x 較小的一端 = head。
        # VRB 的 heat 仍然算，只用來檢查（vrb_agree），不改變結果。
        A, B = np.asarray(poly[0], float), np.asarray(poly[-1], float)
        out = poly if A[0] <= B[0] else poly[::-1]
        dx = abs(A[0] - B[0]) / max(float(np.linalg.norm(A - B)), 1e-6)     # 兩端連線有多「水平」：1 = 完全水平
        info = {"mode": "left", "end_dx_ratio": round(dx, 3)}
        if hm is not None and crop is not None:
            vout, vconf, vinfo = orient_heat(poly, hm, crop, names)
            info.update(vrb_agree=bool(np.allclose(vout[0], out[0])), vrb_conf=round(float(vconf), 3))
        return out, dx, {"source": "rule_left", **info, **tinfo}
    if mode == "heat":
        out, conf, info = orient_heat(poly, hm, crop, names)
    else:
        out, conf, info = orient(poly, vrb_pts, vrb_conf, names, mode)
    return out, conf, {"source": "vrb", **info, **tinfo}


def five_points(poly_head_first, names):
    """head 在前的中心線 → 依 names 的順序回傳 (K,2) 點。"""
    pts, L = points_along(poly_head_first, [POINT_FRACS[n] for n in names])
    return pts, L


# =============================================================================
# decode=vrb_anchor（計畫 v1.7 §3.1.1a）：夾取點由 VRB 決定
# =============================================================================
def arclen(poly):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(seg)])


def project_to_poly(poly, p):
    """點 p 投影到折線上最近的位置。回傳 (弧長 s, 到折線的距離, 投影點)。"""
    poly = np.asarray(poly, np.float64)
    p = np.asarray(p, np.float64)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
    q = a + t[:, None] * ab
    d = np.linalg.norm(q - p, axis=1)
    i = int(np.argmin(d))
    s = arclen(poly)
    return float(s[i] + t[i] * np.sqrt(L2[i])), float(d[i]), q[i]


def point_at(poly, s_query):
    s = arclen(poly)
    sq = np.atleast_1d(np.asarray(s_query, np.float64))
    return np.stack([np.interp(sq, s, poly[:, 0]), np.interp(sq, s, poly[:, 1])], axis=1)


def anchor_decode(poly, vrb_pts, vrb_conf, vrb_names, out_names=ANCHOR_NAMES):
    """VRB 的 head、tail（、middle）投影到中心線 poly（任一方向皆可）。
    head、tail、middle 的位置 = 各自的投影點（VRB 直接預測的點，只是貼到中心線上）；
    head_1_3、tail_1_3、head_1_5 = 投影後 head → tail 弧長的 1/3、2/3、1/5。
    信心：VRB 直接預測的點 = 自己的 heatmap 尖峰值；投影算出的點 = min(head 尖峰, tail 尖峰)。
    回傳 (pts (K,2), conf (K,), conf_src [K], info dict)。"""
    P = {n: np.asarray(p, np.float64) for n, p in zip(vrb_names, vrb_pts)}
    C = {n: float(c) for n, c in zip(vrb_names, vrb_conf)}
    L = float(arclen(poly)[-1])
    sh, dh, _ = project_to_poly(poly, P["head"])
    st, dt, _ = project_to_poly(poly, P["tail"])
    sm = dm = None
    if "middle" in P:
        sm, dm, _ = project_to_poly(poly, P["middle"])
    derived = min(C["head"], C["tail"])
    pos, conf, src = [], [], []
    for n in out_names:
        if n == "head":
            pos.append(sh); conf.append(C["head"]); src.append("vrb_head")
        elif n == "tail":
            pos.append(st); conf.append(C["tail"]); src.append("vrb_tail")
        elif n == "middle" and sm is not None:
            pos.append(sm); conf.append(C["middle"]); src.append("vrb_middle")
        else:
            pos.append(sh + POINT_FRACS[n] * (st - sh)); conf.append(derived); src.append("min(head,tail)")
    pts = point_at(poly, pos)
    A, B = np.asarray(poly[0], float), np.asarray(poly[-1], float)
    head_end = "A" if sh <= st else "B"                    # VRB 認為針頭在中心線的哪一端
    left_end = "A" if A[0] <= B[0] else "B"                # 「針頭朝左」規則
    info = {"length_px": round(L, 1), "s_head": round(sh, 1), "s_tail": round(st, 1),
            "span_px": round(abs(st - sh), 1), "span_ratio": round(abs(st - sh) / max(L, 1e-6), 3),
            "head_off_px": round(dh, 1), "tail_off_px": round(dt, 1),
            "head_end": head_end, "vrb_agree": head_end == left_end,
            "grasp_conf": round(derived, 4)}
    if sm is not None:
        info.update(s_middle=round(sm, 1), middle_off_px=round(dm, 1),
                    middle_consistency_px=round(abs(sm - (sh + st) / 2), 1))
    return pts, np.asarray(conf), src, info


def left_points(poly, names=ANCHOR_NAMES):
    """純幾何 + 針頭朝左（geom_only）：中心線 x 較小的一端 = head，依弧長比例取點。"""
    A, B = np.asarray(poly[0], float), np.asarray(poly[-1], float)
    pl = poly if A[0] <= B[0] else poly[::-1]
    pts, L = points_along(pl, [POINT_FRACS[n] for n in names])
    dx = abs(A[0] - B[0]) / max(float(np.linalg.norm(A - B)), 1e-6)
    return pts, pl, dx
