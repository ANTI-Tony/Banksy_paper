"""Material generator v0 (10-22, research track; pre-registered: ideas/2026-10-22_材质生成器_预注册.md).
Glyph-locked: the glyph alpha of the TARGET string is an input and fixes identity, layout and stroke topology; the model
only produces the text layer's colour / shading / thickness / glow (RGB + alpha), composited onto the given background.
Trained on the counterfactual pairs of factory_v2_pair.py in the board's UV space (board corners -> fixed canvas):
    input  = [background RGB, target glyph alpha, reference RGB (the other string, real material), reference alpha]  (8 ch)
    output = text layer RGB + alpha  ->  composite = bg * (1 - a) + rgb * a
    target = the rendered composite of the target string; both directions (A->B, B->A) are training samples.
Loss: L1 + LPIPS on the composite inside the dilated text region, L1 on alpha vs the input alpha (glyph lock),
L1 outside the text region vs the background (locality).
Baseline for the 'closed' metric: flat paste = background + glyph alpha filled with the reference text's median colour.
Sub-commands:
  prep  <factory_root> <out.npz dir> [--canvas 448x192]      rectify every pair/frame into UV crops (uint8/16 npz shards)
  train <npz dir> <run dir> [--steps 20000 --bs 16 --lr 2e-4 --val <npz dir>]
  eval  <npz dir> <run dir> [--out metrics.json]              closed metric, glyph IoU, per surface/style"""
import argparse, glob, json, math, os, sys, time
from pathlib import Path
import numpy as np


# ----------------------------------------------------------------------------------------------------------------- prep
def rectify_pair(pdir, canvas):
    import cv2
    th = json.load(open(pdir / "theta.json")); meta = json.load(open(pdir / "meta.json")); W, H = canvas
    dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])          # board-local (-1,-1),(1,-1),(1,1),(-1,1): y up in Blender
    out = []
    for f in sorted(th["board"], key=int):
        src = np.float32(th["board"][f]); Hm = cv2.getPerspectiveTransform(src, dst)
        fn = f"{int(f):03d}.png"
        bg = cv2.imread(str(pdir / "background" / fn)); comp = {}; alpha = {}
        for s in ("A", "B"):
            c = cv2.imread(str(pdir / s / "composite" / fn)); t = cv2.imread(str(pdir / s / "textlayer" / fn), cv2.IMREAD_UNCHANGED)
            if c is None or t is None or bg is None: return None
            a = t[..., 3].astype(np.float32) / (65535.0 if t.dtype == np.uint16 else 255.0)
            comp[s] = cv2.warpPerspective(c, Hm, (W, H), flags=cv2.INTER_AREA); alpha[s] = cv2.warpPerspective(a, Hm, (W, H), flags=cv2.INTER_AREA)
        bgw = cv2.warpPerspective(bg, Hm, (W, H), flags=cv2.INTER_AREA)
        out.append({"bg": bgw, "A": comp["A"], "B": comp["B"], "aA": (alpha["A"] * 255).astype(np.uint8), "aB": (alpha["B"] * 255).astype(np.uint8)})
    return out, meta


def cmd_prep(a):
    root, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True); W, H = (int(v) for v in a.canvas.split("x"))
    pairs = sorted(p for p in root.iterdir() if (p / "meta.json").exists()); n_ok = 0; index = []
    for p in pairs:
        r = rectify_pair(p, (W, H))
        if r is None: print("skip (missing frames)", p.name, flush=True); continue
        frames, meta = r
        np.savez_compressed(out / f"{p.name}.npz", bg=np.stack([x["bg"] for x in frames]), A=np.stack([x["A"] for x in frames]), B=np.stack([x["B"] for x in frames]),
                            aA=np.stack([x["aA"] for x in frames]), aB=np.stack([x["aB"] for x in frames]), meta=json.dumps(meta, ensure_ascii=False))
        index.append({"id": p.name, "surface": meta["surface"], "style": meta["style"], "frames": len(frames), "cjk": any(ord(c) > 0x2E7F for c in meta["text_b"])}); n_ok += 1
    json.dump(index, open(out / "index.json", "w"), ensure_ascii=False, indent=0); print("PREP_DONE", n_ok, "pairs ->", out)


# ------------------------------------------------------------------------------------------------------------- dataset
class PairData:
    """every (pair, frame, direction) is a sample; loads npz lazily with a small cache"""
    def __init__(self, d, ids=None):
        self.d = Path(d); self.index = [x for x in json.load(open(self.d / "index.json")) if ids is None or x["id"] in ids]
        self.items = [(x["id"], f, dr) for x in self.index for f in range(x["frames"]) for dr in ("AB", "BA")]; self.cache = {}
    def __len__(self): return len(self.items)
    def load(self, pid):
        if pid not in self.cache:
            if len(self.cache) > 64: self.cache.clear()
            z = np.load(self.d / f"{pid}.npz"); self.cache[pid] = {k: z[k] for k in ("bg", "A", "B", "aA", "aB")}
        return self.cache[pid]
    def __getitem__(self, i):
        pid, f, dr = self.items[i]; z = self.load(pid); ref, tgt = ("A", "B") if dr == "AB" else ("B", "A")
        bg = z["bg"][f]; x = {"bg": bg, "ref": z[ref][f], "aref": z["a" + ref][f], "atgt": z["a" + tgt][f], "tgt": z[tgt][f], "id": pid, "frame": f, "dir": dr}
        return x


def to_tensor(batch, dev):
    import torch
    f = lambda k, c: torch.from_numpy(np.stack([b[k] for b in batch])).to(dev).float().div_(255.0)
    bg, ref, tgt = (f(k, 3).permute(0, 3, 1, 2) for k in ("bg", "ref", "tgt")); aref, atgt = (f(k, 1).unsqueeze(1) for k in ("aref", "atgt"))
    return bg, ref, aref, atgt, tgt


# --------------------------------------------------------------------------------------------------------------- model
def build_model(base=48):
    import torch, torch.nn as nn

    class Block(nn.Module):
        def __init__(s, i, o):
            super().__init__(); s.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())
        def forward(s, x): return s.net(x)

    class UNet(nn.Module):
        def __init__(s, cin=8, cout=4, b=base):
            super().__init__(); ch = [b, b * 2, b * 4, b * 8]
            s.e1, s.e2, s.e3, s.e4 = Block(cin, ch[0]), Block(ch[0], ch[1]), Block(ch[1], ch[2]), Block(ch[2], ch[3]); s.pool = nn.MaxPool2d(2)
            s.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            s.d3, s.d2, s.d1 = Block(ch[3] + ch[2], ch[2]), Block(ch[2] + ch[1], ch[1]), Block(ch[1] + ch[0], ch[0]); s.out = nn.Conv2d(ch[0], cout, 1)
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(s.pool(e1)); e3 = s.e3(s.pool(e2)); e4 = s.e4(s.pool(e3))
            d3 = s.d3(torch.cat([s.up(e4), e3], 1)); d2 = s.d2(torch.cat([s.up(d3), e2], 1)); d1 = s.d1(torch.cat([s.up(d2), e1], 1)); return s.out(d1)
    return UNet()


def compose(model, bg, ref, aref, atgt):
    import torch
    y = model(torch.cat([bg, atgt, ref, aref], 1)); rgb = torch.sigmoid(y[:, :3]); a = torch.sigmoid(y[:, 3:4])
    return bg * (1 - a) + rgb * a, rgb, a


def flat_paste(bg, ref, aref, atgt):
    """baseline: glyph filled with the reference text's median colour (what a deterministic renderer does)"""
    import torch
    B = bg.shape[0]; col = torch.zeros(B, 3, 1, 1, device=bg.device)
    for i in range(B):
        m = aref[i, 0] > 0.5
        col[i, :, 0, 0] = ref[i][:, m].median(dim=1).values if m.sum() > 10 else torch.tensor([0.9, 0.9, 0.9], device=bg.device)
    return bg * (1 - atgt) + col * atgt


def region(atgt, k=15):
    import torch, torch.nn.functional as F
    return (F.max_pool2d(atgt, k, stride=1, padding=k // 2) > 0.05).float()


# --------------------------------------------------------------------------------------------------------------- train
def cmd_train(a):
    import torch, torch.nn.functional as F, lpips
    import os
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True); ds = PairData(a.data); val = PairData(a.val) if a.val else None
    model = build_model(a.base).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); perc = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in perc.parameters(): p.requires_grad_(False)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=max(0.05, min(0.3, 3.0 / a.steps))); scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    start = 1; log = []
    cks = sorted(run.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if cks:                                               # resume (2-hour interactive slots): model, optimiser, schedule, RNG position
        ck = torch.load(cks[-1], map_location=dev); model.load_state_dict(ck["model"])
        if "opt" in ck: opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start = ck["step"] + 1; log = json.load(open(run / "log.json")) if (run / "log.json").exists() else []
        rng = np.random.default_rng([a.seed, start]); print(f"resumed from {cks[-1].name} at step {start}", flush=True)
        if start > a.steps: print("TRAIN_DONE (already complete)", run); return
    print(f"train samples {len(ds)} | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | device {dev}", flush=True); t0 = time.time(); t_budget = a.time_budget
    for step in range(start, a.steps + 1):
        batch = [ds[int(i)] for i in rng.integers(0, len(ds), a.bs)]; bg, ref, aref, atgt, tgt = to_tensor(batch, dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out, rgb, alpha = compose(model, bg, ref, aref, atgt); R = region(atgt)
            l1 = (F.l1_loss(out, tgt, reduction="none") * R).sum() / (R.sum() * 3 + 1)
            lp = perc(out * 2 - 1, tgt * 2 - 1).mean()
            lg = F.l1_loss(alpha, atgt)                                                   # glyph lock
            lb = (F.l1_loss(out, bg, reduction="none") * (1 - R)).sum() / ((1 - R).sum() * 3 + 1)   # locality
            loss = l1 + 0.5 * lp + 2.0 * lg + 1.0 * lb
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        if step % 50 == 0:
            log.append({"step": step, "loss": float(loss), "l1": float(l1), "lpips": float(lp), "glyph": float(lg), "bg": float(lb)})
            print(f"step {step} loss {loss:.4f} l1 {l1:.4f} lpips {lp:.4f} glyph {lg:.4f} bg {lb:.4f} | {time.time() - t0:.0f}s", flush=True)
        out_of_time = t_budget > 0 and (time.time() - t0) > t_budget
        if step % a.ckpt_every == 0 or step == a.steps or out_of_time:
            tmp = run / f"ckpt_{step}.pt.tmp"; torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step, "args": vars(a)}, tmp); os.replace(tmp, run / f"ckpt_{step}.pt")
            json.dump(log, open(run / "log.json", "w"))
            for old in sorted(run.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[:-2]:
                if int(old.stem.split("_")[1]) % a.keep_every != 0: old.unlink()
            if val and (step % a.ckpt_every == 0 or step == a.steps): print("VAL", json.dumps(evaluate(model, val, dev, n=min(400, len(val)))), flush=True)
            if out_of_time: print(f"TIME_BUDGET at step {step}; resubmit to continue", flush=True); return
    print("TRAIN_DONE", run)


@np.errstate(all="ignore")
def evaluate(model, ds, dev, n=None, bs=16, per_group=False):
    import torch
    model.eval(); idx = list(range(len(ds))) if n is None else list(np.linspace(0, len(ds) - 1, n).astype(int)); rows = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            batch = [ds[j] for j in idx[i:i + bs]]; bg, ref, aref, atgt, tgt = to_tensor(batch, dev)
            out, rgb, alpha = compose(model, bg, ref, aref, atgt); P = flat_paste(bg, ref, aref, atgt); R = region(atgt)
            for k in range(len(batch)):
                r = R[k]; e_m = ((out[k] - tgt[k]).abs().sum(0) * r[0]).sum().item(); e_p = ((P[k] - tgt[k]).abs().sum(0) * r[0]).sum().item()
                iou = (((alpha[k, 0] > 0.5) & (atgt[k, 0] > 0.5)).sum() / (((alpha[k, 0] > 0.5) | (atgt[k, 0] > 0.5)).sum() + 1)).item()
                meta = json.loads(str(np.load(ds.d / f"{batch[k]['id']}.npz")["meta"]))
                rows.append({"id": batch[k]["id"], "closed": 1 - e_m / max(e_p, 1e-6), "err_model": e_m, "err_paste": e_p, "iou": iou, "surface": meta["surface"], "style": meta["style"]})
    model.train()
    agg = lambda rs: {"n": len(rs), "closed_mean": float(np.mean([x["closed"] for x in rs])), "closed_median": float(np.median([x["closed"] for x in rs])),
                      "closed_sum": 1 - sum(x["err_model"] for x in rs) / max(1e-6, sum(x["err_paste"] for x in rs)), "iou": float(np.mean([x["iou"] for x in rs]))}
    res = agg(rows)
    if per_group:
        res["by_surface"] = {s: agg([x for x in rows if x["surface"] == s]) for s in sorted({x["surface"] for x in rows})}
        res["by_style"] = {s: agg([x for x in rows if x["style"] == s]) for s in sorted({x["style"] for x in rows})}
    return res


def cmd_eval(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; ck = torch.load(a.ckpt, map_location=dev); model = build_model(ck["args"].get("base", 48)).to(dev); model.load_state_dict(ck["model"])
    ds = PairData(a.data); res = evaluate(model, ds, dev, per_group=True); res["ckpt"] = a.ckpt; res["samples"] = len(ds)
    json.dump(res, open(a.out, "w"), indent=1); print("EVAL_DONE", json.dumps({k: v for k, v in res.items() if k not in ("by_surface", "by_style")}))
    for g in ("by_surface", "by_style"):
        for k, v in res[g].items(): print(f"  {g[3:]:8s} {k:10s} n={v['n']:4d} closed(sum)={v['closed_sum']:+.3f} median={v['closed_median']:+.3f} iou={v['iou']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("src"); p.add_argument("out"); p.add_argument("--canvas", default="448x192")
    p = sp.add_parser("train"); p.add_argument("data"); p.add_argument("run"); p.add_argument("--val", default=""); p.add_argument("--steps", type=int, default=20000); p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--base", type=int, default=48); p.add_argument("--seed", type=int, default=0); p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--keep-every", type=int, default=5000, help="checkpoints at multiples of this are kept; others are pruned to the last two"); p.add_argument("--time-budget", type=float, default=0, help="seconds; save and exit when exceeded (interactive slots)")
    p = sp.add_parser("eval"); p.add_argument("data"); p.add_argument("ckpt"); p.add_argument("--out", default="matgen_eval.json")
    A = ap.parse_args(); {"prep": cmd_prep, "train": cmd_train, "eval": cmd_eval}[A.cmd](A)
