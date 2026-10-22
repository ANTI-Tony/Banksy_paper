"""Local compositing of a generative editor's output into the ORIGINAL clip (10-22, system track, pre-registered in
ideas/2026-10-22_混合系统_预注册.md). The editor (SteerVTE) regenerates the whole frame; here only its text region is kept:
    out_t = (1 - a_t) * orig_t + a_t * colourfix(gen_t)
a_t = 1 inside the benchmark's per-frame text box, stays 1 up to `dilate` box heights outside it (shadow, glow), then falls
linearly to 0 over `feather` box heights. colourfix matches the editor's per-channel mean and spread to the original's on
an annulus just outside the blended region (a global exposure/colour shift of the regenerated frame would otherwise show
as a faint rectangle at the seam). Frames are matched by index (same length) or by relative time; different sizes are
resized to the original. Output is written with the same writer as every other output (cv2 mp4v at the source fps).
Usage: python composite_local.py <pkg_csv> <pkg_root> <gen_dir with <uid>.mp4> <out_dir> [--dilate 0.3] [--feather 0.12] [--no-colourfix]"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import cv2, numpy as np


def frames(p):
    cap = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release(); return out, (float(fps) if fps and 1 < fps < 61 else 15.0)


def write_vid(fr, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(f".{os.getpid()}.tmp.mp4"); H, W = fr[0].shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened(): raise RuntimeError(f"VideoWriter could not open {tmp}")
    for f in fr: vw.write(f)
    vw.release(); os.replace(tmp, path)


def alpha_of(box, shape, dilate, feather):
    """box = (x0, y0, x1, y1) inclusive; 1 inside and up to `dilate`*h outside, linear falloff over `feather`*h."""
    H, W = shape; x0, y0, x1, y1 = box; h = max(1.0, y1 - y0 + 1)
    inside = np.zeros((H, W), np.uint8); inside[y0:y1 + 1, x0:x1 + 1] = 1
    dist = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)             # distance to the box, 0 inside
    a = np.clip(1.0 - (dist - dilate * h) / max(1e-6, feather * h), 0.0, 1.0)
    annulus = (dist > (dilate + feather) * h) & (dist <= (dilate + feather + 0.5) * h)
    return a.astype(np.float32), annulus


def colourfix(gen, src, annulus):
    """per-channel affine match of gen to src on the annulus; gains clipped to [0.9, 1.1] so noise is not amplified"""
    g = gen.astype(np.float32); s = src.astype(np.float32); out = g.copy(); gains = []
    if annulus.sum() < 200: return gen, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
    offs = []
    for c in range(3):
        gs, ss = g[..., c][annulus], s[..., c][annulus]
        k = float(np.clip(ss.std() / max(1e-3, gs.std()), 0.9, 1.1)); b = float(ss.mean() - k * gs.mean())
        out[..., c] = k * g[..., c] + b; gains.append(k); offs.append(b)
    return np.clip(out, 0, 255), gains, offs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("root"); ap.add_argument("gen"); ap.add_argument("out")
    ap.add_argument("--dilate", type=float, default=0.3); ap.add_argument("--feather", type=float, default=0.12); ap.add_argument("--no-colourfix", action="store_true")
    ap.add_argument("--colourfix", default="frame", choices=["frame", "clip"], help="frame: per-frame gains (V1); clip: one median gain/offset per clip (V1.1, defined after V1's temporal result)")
    A = ap.parse_args(); ROOT, GEN, OUT = Path(A.root), Path(A.gen), Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
    log = {}; t0 = time.time()
    for r in csv.DictReader(open(A.csv)):
        uid = Path(r["video"]).parent.name; op = OUT / uid / "fixed.mp4"
        if op.exists(): continue
        gp = GEN / f"{uid}.mp4"
        if not gp.exists(): gp = GEN / uid / "fixed.mp4"
        if not gp.exists(): print("no editor output", uid, flush=True); continue
        src, fps = frames(ROOT / r["video"]); mk, _ = frames(ROOT / r["vace_video_mask"]); gen, _ = frames(gp)
        if not src or not mk or not gen: print("unreadable", uid, flush=True); continue
        H, W = src[0].shape[:2]; n = len(src); idx = np.linspace(0, len(gen) - 1, n).astype(int)
        outs = []; last = None; gains = []; fixes = []
        if not A.no_colourfix and A.colourfix == "clip":            # V1.1: estimate per frame, apply the per-clip median (temporally constant)
            for t in range(n):
                m = cv2.resize(mk[min(t, len(mk) - 1)], (W, H), interpolation=cv2.INTER_NEAREST)[..., 0] > 127; ys, xs = np.where(m)
                if not len(xs): continue
                _, ann = alpha_of((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), (H, W), A.dilate, A.feather)
                g = gen[idx[t]]
                if g.shape[:2] != (H, W): g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
                _, k, b = colourfix(g, src[t], ann); fixes.append((k, b))
            KB = (np.median([f[0] for f in fixes], 0), np.median([f[1] for f in fixes], 0)) if fixes else ([1, 1, 1], [0, 0, 0])
        for t in range(n):
            m = cv2.resize(mk[min(t, len(mk) - 1)], (W, H), interpolation=cv2.INTER_NEAREST)[..., 0] > 127
            ys, xs = np.where(m)
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else last
            if box is None: outs.append(src[t]); continue
            last = box; a, ann = alpha_of(box, (H, W), A.dilate, A.feather)
            g = gen[idx[t]]
            if g.shape[:2] != (H, W): g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
            if not A.no_colourfix and A.colourfix == "clip":
                g = np.clip(np.stack([KB[0][c] * g[..., c].astype(np.float32) + KB[1][c] for c in range(3)], -1), 0, 255); gains.append(list(KB[0]))
            elif not A.no_colourfix: g, k, b = colourfix(g, src[t], ann); gains.append(k)
            a3 = a[..., None]; outs.append((src[t].astype(np.float32) * (1 - a3) + g.astype(np.float32) * a3).clip(0, 255).astype(np.uint8))
        write_vid(outs, op, fps)
        log[uid] = {"frames": n, "gen_frames": len(gen), "box_last": last, "gain_mean": [float(np.mean([k[c] for k in gains])) for c in range(3)] if gains else None}
        print("composited", uid, flush=True)
    json.dump({"params": vars(A), "clips": log}, open(OUT / "composite_log.json", "w"), indent=1)
    print(f"COMPOSITE_DONE {len(log)} clips in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
