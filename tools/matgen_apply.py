"""Apply the material generator (matgen_v0) to real clips through the frozen B1 front end (10-22, research track, gate R2).
Inputs per clip: the RAW cache of fix_bench2 (B1 variant): erased background crops, the glyph canvas + its per-frame quad
(geom), the source ink mask at the reference frame, and the window; plus the original clip. The model works in the
448x192 UV canvas of training: the glyph canvas is placed in it with the training framing (text height 60% of the
canvas, or width 86%, whichever binds); every frame is warped into that canvas, the model predicts the text layer once
on the reference frame (RGB + alpha, glyph locked by the canvas alpha), and the layer is warped back onto every frame's
erased background. Multi-row clips (the cache holds only the first row's canvas) are skipped and listed.
Modes: model (default) | flat (reference median colour in the glyph: the geometry pipeline alone, for checking).
Usage: python matgen_apply.py <ckpt|flat> <pkg_csv> <pkg_root> <raw_cache_dir> <out_dir> [--canvas 448x192]"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composite_local import frames, write_vid


def uv_from_canvas(wc, hc, W, H):
    """similarity placing the glyph canvas in the training framing: height 0.6H or width 0.86W, centred"""
    s = min(0.6 * H / max(hc, 1), 0.86 * W / max(wc, 1)); tx, ty = (W - s * wc) / 2, (H - s * hc) / 2
    return np.float32([[s, 0, tx], [0, s, ty], [0, 0, 1]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("csv"); ap.add_argument("root"); ap.add_argument("cache"); ap.add_argument("out")
    ap.add_argument("--canvas", default="448x192"); A = ap.parse_args(); W, H = (int(v) for v in A.canvas.split("x"))
    ROOT, CACHE, OUT = Path(A.root), Path(A.cache), Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
    model = None
    if A.ckpt != "flat":
        import torch; from matgen_v0 import build_model, compose, flat_paste, region
        dev = "cuda" if torch.cuda.is_available() else "cpu"; ck = torch.load(A.ckpt, map_location=dev)
        model = build_model(ck["args"].get("base", 48)).to(dev).eval(); model.load_state_dict(ck["model"])
    log = {}; t0 = time.time()
    for r in csv.DictReader(open(A.csv)):
        uid = Path(r["video"]).parent.name; op = OUT / uid / "fixed.mp4"
        if op.exists(): continue
        cp = CACHE / f"{uid}.npz"
        if not cp.exists(): log[uid] = "no cache"; continue
        z = np.load(cp, allow_pickle=True)
        if bool(z["failed"]) or "canvas_B1" not in z.files or "bg" not in z.files: log[uid] = "failed/slim cache"; continue
        src, fps = frames(ROOT / r["video"]); n = len(src); Hf, Wf = src[0].shape[:2]
        wx0, wy0, side = (int(v) for v in z["window"]); bg = z["bg"]; C = bg.shape[1]; canvas = z["canvas_B1"]; geom = np.float32(z["geom_B1"])
        if canvas.ndim != 2 or canvas.shape[0] < 4: log[uid] = "no single canvas"; continue
        hc, wc = canvas.shape; rect = np.float32([[0, 0], [wc, 0], [wc, hc], [0, hc]]); ref = int(z["ref"]) if "ref" in z.files else n // 2
        # erased full frames: the original with the window replaced by the erased crop (what B1 composites onto)
        base = []
        for t in range(n):
            b = src[t].copy(); b[wy0:wy0 + side, wx0:wx0 + side] = cv2.resize(bg[min(t, len(bg) - 1)], (side, side), interpolation=cv2.INTER_LINEAR); base.append(b)
        # consistency check against the rendered coverage (multi-row clips keep only the first row's canvas)
        a_can = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(rect, geom[ref]), (Wf, Hf)) > 127
        a_ref = cv2.resize(z["alpha_B1"][ref], (side, side)) > 127; a_full = np.zeros((Hf, Wf), bool); a_full[wy0:wy0 + side, wx0:wx0 + side] = a_ref[:min(side, Hf - wy0), :min(side, Wf - wx0)]
        iou = (a_can & a_full).sum() / max(1, (a_can | a_full).sum())
        if iou < 0.6: log[uid] = f"canvas/coverage IoU {iou:.2f} (multi-row?)"; continue
        S = uv_from_canvas(wc, hc, W, H); Hs = [S @ np.linalg.inv(cv2.getPerspectiveTransform(rect, geom[t])) for t in range(n)]   # UV <- frame
        atgt = cv2.warpAffine(canvas, S[:2], (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
        bg_uv = cv2.warpPerspective(base[ref], Hs[ref], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ref_uv = cv2.warpPerspective(src[ref], Hs[ref], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ink = z["inkref"] if "inkref" in z.files else None
        if ink is not None:
            ink_full = np.zeros((Hf, Wf), np.uint8); ink_full[wy0:wy0 + side, wx0:wx0 + side] = cv2.resize(ink, (side, side), interpolation=cv2.INTER_NEAREST)[:min(side, Hf - wy0), :min(side, Wf - wx0)]
            aref = cv2.warpPerspective(ink_full, Hs[ref], (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
        else: aref = np.zeros((H, W), np.float32)
        if model is None:
            m = aref > 0.5; col = np.median(ref_uv[m], axis=0) if m.sum() > 10 else np.array([230, 230, 230], np.float32)
            rgb_uv = np.ones((H, W, 3), np.float32) * col; a_uv = atgt
        else:
            import torch
            tt = lambda im: torch.from_numpy(im).to(dev).float().div_(255.0)
            bgT = tt(cv2.cvtColor(bg_uv, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None]; refT = tt(cv2.cvtColor(ref_uv, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None]
            arefT = torch.from_numpy(aref).to(dev)[None, None]; atgtT = torch.from_numpy(atgt).to(dev)[None, None]
            with torch.no_grad():
                _, rgb, a = compose(model, bgT, refT, arefT, atgtT); a = a * region(atgtT, 15)
            rgb_uv = cv2.cvtColor((rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32); a_uv = a[0, 0].cpu().numpy()
        outs = []
        for t in range(n):
            Hi = np.linalg.inv(Hs[t]); rgb_t = cv2.warpPerspective(rgb_uv, Hi, (Wf, Hf), flags=cv2.INTER_LINEAR); a_t = cv2.warpPerspective(a_uv, Hi, (Wf, Hf), flags=cv2.INTER_LINEAR)[..., None]
            outs.append((base[t].astype(np.float32) * (1 - a_t) + rgb_t * a_t).clip(0, 255).astype(np.uint8))
        write_vid(outs, op, fps); log[uid] = "ok"; print("applied", uid, f"iou {iou:.2f}", flush=True)
    json.dump({"ckpt": A.ckpt, "clips": log}, open(OUT / "apply_log.json", "w"), indent=1)
    print("APPLY_DONE", sum(v == "ok" for v in log.values()), "ok of", len(log), f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
