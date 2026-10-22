"""Paper figures (10-22): export the photo cells of the teaser / qualitative / failure figures as separate JPGs at native
crop resolution, plus difference maps for the background-fidelity figure, and write LaTeX figure bodies that lay the
cells out with the paper's own fonts (labels and reader readings are typeset by LaTeX, not burnt into the pixels).
Inputs are local copies of the cluster files (see make_paper_figs.py). Readings: eval_vlm.json at the same frame.
Usage: python make_paper_cells.py <root> <eval_vlm.json> <picks.json> <paper_dir> [--banksy align/sel/B1] [--frame 24]"""
import argparse, json
from pathlib import Path
import cv2, numpy as np
import make_paper_figs as F

ESC = {"%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}


CJK_DIR = None                                               # set in main(): <paper>/figs/cjk


def cjk_run(buf):
    """\\zh{} for runs the GB2312 font (gbsn) can typeset; otherwise an image of the run in STHeiti (\\cjkimg)."""
    try:
        buf.encode("gb2312"); return r"\zh{" + buf + "}"
    except UnicodeEncodeError:
        from PIL import Image, ImageDraw, ImageFont
        import hashlib
        name = "cjk_" + hashlib.md5(buf.encode()).hexdigest()[:10]
        f = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 160); l, t, r, b = f.getbbox(buf)
        im = Image.new("LA", (r - l + 16, 200), (0, 0)); ImageDraw.Draw(im).text((8 - l, 20), buf, font=f, fill=(0, 255))
        im.save(CJK_DIR / f"{name}.png"); return r"\cjkimg{" + name + "}"


def tex(s):
    """LaTeX-safe text; runs of CJK characters become \\zh{} (or \\cjkimg{} when gbsn lacks a glyph)."""
    out, buf = [], ""
    for ch in s:
        if ord(ch) > 0x2E7F:
            buf += ch; continue
        if buf: out.append(cjk_run(buf)); buf = ""
        out.append(ESC.get(ch, ch))
    if buf: out.append(cjk_run(buf))
    return "".join(out)


def paths(root, banksy, u):
    pk = Path(root) / "out" / "repair_pkg2" / u
    return pk, {"orig": pk / "src.mp4", "steervte": Path(root) / "out" / "repair_out2" / "steervte" / f"{u}.mp4",
                "B1": Path(root) / banksy / u / "fixed.mp4"}


def cells(root, banksy, E, u, t, methods, out_dir, tag):
    """Save the same crop of every version; returns {method: (file, reading, correct)}."""
    pk, P = paths(root, banksy, u)
    src = F.frame(P["orig"], t); mk = F.frame(pk / "mask.mp4", t)
    if mk is None or (mk > 127).sum() == 0: mk = F.frame(pk / "mask.mp4", 0)
    x0, y0, w, h = F.crop_box(mk, src.shape); res = {}
    for m in methods:
        im = F.frame(P[m], t)
        if im.shape[:2] != src.shape[:2]: im = cv2.resize(im, (src.shape[1], src.shape[0]))
        fn = f"{tag}_{u}_{m}.jpg"; cv2.imwrite(str(out_dir / fn), im[y0:y0 + h, x0:x0 + w], [cv2.IMWRITE_JPEG_QUALITY, 95])
        e = E[m][u]; rd = (e["reads"].get(str(t)) or next(iter(e["reads"].values()))).replace("\n", " ")
        res[m] = (fn, rd, e["fixed"] >= 0.6)
    return res


def diffmap(root, banksy, u, t, out_dir, tag, clip_at=24.0):
    """Original frame + |SteerVTE - original| and |Banksy - original| (grey levels, clipped), full frame, magma."""
    pk, P = paths(root, banksy, u)
    src = F.frame(P["orig"], t); mk = F.frame(pk / "mask.mp4", t)
    if mk is None or (mk > 127).sum() == 0: mk = F.frame(pk / "mask.mp4", 0)
    H, W = src.shape[:2]; m = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)[..., 0] > 127
    ys, xs = np.where(m); box = (xs.min(), ys.min(), xs.max(), ys.max())
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = {"orig": f"{tag}_{u}_orig.jpg"}; stats = {}
    cv2.imwrite(str(out_dir / out["orig"]), src, [cv2.IMWRITE_JPEG_QUALITY, 95])
    for mth in ("steervte", "B1"):
        im = F.frame(P[mth], t)
        if im.shape[:2] != (H, W): im = cv2.resize(im, (W, H))
        d = np.abs(g(im) - g(src)); outside = ~cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        stats[mth] = {"mean_abs_outside": round(float(d[outside].mean()), 2), "share_gt8_outside": round(float((d[outside] > 8).mean()), 4)}
        col = cv2.applyColorMap((np.clip(d / clip_at, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        cv2.rectangle(col, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 2)
        out[mth] = f"{tag}_{u}_{mth}_diff.png"; cv2.imwrite(str(out_dir / out[mth]), col)
    return out, stats


def mark(ok):
    return r"\textcolor{okgreen}{\ding{51}}" if ok else r"\textcolor{badred}{\ding{55}}"


def grid_tex(rows, colnames, width, label_w, fig_env, caption, label, star=True):
    """rows: list of (row label, [(file, reading, correct), ...]). One tabular, every cell = image over its reading."""
    n = len(colnames); cw = f"{width}"
    L = [r"\begin{figure*}[t]" if star else r"\begin{figure}[t]", r"\centering", r"\setlength{\tabcolsep}{1.2pt}\renewcommand{\arraystretch}{0.6}",
         r"\begin{tabular}{@{}" + (f"m{{{label_w}}}" if label_w else "") + "c" * n + "@{}}"]
    L.append(("&" if label_w else "") + " & ".join(r"\footnotesize\textbf{" + c + "}" for c in colnames) + r" \\[2pt]")
    for lab, cs in rows:
        imgs = [r"\includegraphics[width=" + cw + r"]{figs/cells/" + f + "}" for f, _, _ in cs]
        reads = [r"\parbox[t]{" + cw + r"}{\centering\scriptsize " + mark(ok) + r"\,\texttt{" + tex(rd[:22]) + "}}" for _, rd, ok in cs]
        L.append((r"\scriptsize " + lab + " & " if label_w else "") + " & ".join(imgs) + r" \\")
        L.append(("&" if label_w else "") + " & ".join(reads) + r" \\[3pt]")
    L += [r"\end{tabular}", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{figure*}" if star else r"\end{figure}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("evalj"); ap.add_argument("picks"); ap.add_argument("paper")
    ap.add_argument("--banksy", default="align/sel/B1"); ap.add_argument("--frame", type=int, default=24)
    A = ap.parse_args(); E = json.load(open(A.evalj))["per_clip"]; P = json.load(open(A.picks)); t = A.frame
    O = Path(A.paper) / "figs" / "cells"; O.mkdir(parents=True, exist_ok=True); G = Path(A.paper) / "figs"; tgt = lambda u: tex(E["orig"][u]["target"])
    global CJK_DIR; CJK_DIR = G / "cjk"; CJK_DIR.mkdir(exist_ok=True)
    meta = {}
    # teaser: three examples side by side, each Original / SteerVTE / Banksy stacked
    T = [cells(A.root, A.banksy, E, u, t, ["orig", "steervte", "B1"], O, "teaser") for u, _ in P["teaser"]]
    meta["teaser"] = {u: c for (u, _), c in zip(P["teaser"], T)}
    # qualitative grid: per material, EN | ZH, Original / SteerVTE / Banksy
    mats = ["neon", "signboard", "wood", "metal", "glass", "paper", "screen"]; rows = []
    for mt in mats:
        cs = []
        for lang in ("en", "zh"):
            u = P["fig4"][f"{mt}_{lang}"]; c = cells(A.root, A.banksy, E, u, t, ["orig", "steervte", "B1"], O, "qual")
            meta.setdefault("qual", {})[u] = c; cs += [c["orig"], c["steervte"], c["B1"]]
        rows.append((mt, cs))
    cap = (r"\textbf{Qualitative comparison on randomly sampled clips} (one English and one Chinese clip per material, "
           r"drawn with a fixed seed among garbled originals outside the development set; not curated). Each cell is the same crop "
           r"of frame 24; below it, the reader's transcription (\textcolor{okgreen}{\ding{51}}~correct, \textcolor{badred}{\ding{55}}~wrong). "
           r"SteerVTE produces more natural glyphs but frequently keeps or invents wrong characters; Banksy writes the target "
           r"but its glyphs often look pasted (see \cref{sec:limits}).")
    (G / "fig_qual.tex").write_text(grid_tex(rows, ["Original", "SteerVTE", "Banksy"] * 2, r"0.146\linewidth", r"0.95cm", None, cap, "fig:qual"))
    # selected favourable examples (10-22, user): clips that only Banksy repairs, two per row; the caption states the rule
    if P.get("wins"):
        rows = []
        for i in range(0, len(P["wins"]), 2):
            cs = []
            for u in P["wins"][i:i + 2]:
                c = cells(A.root, A.banksy, E, u, t, ["orig", "steervte", "B1"], O, "wins"); meta.setdefault("wins", {})[u] = c; cs += [c["orig"], c["steervte"], c["B1"]]
            rows.append(("", cs))
        (G / "fig_wins.tex").write_text(grid_tex(rows, ["Original", "SteerVTE", "Banksy"] * 2, r"0.16\linewidth", None, None, P["wins_caption"], "fig:wins"))
    # failure cases: Original / Banksy / SteerVTE (reference)
    rows = []
    for u, lab in P["fig7"]:
        c = cells(A.root, A.banksy, E, u, t, ["orig", "B1", "steervte"], O, "fail"); meta.setdefault("fail", {})[u] = c
        rows.append((tex(lab.replace("\n", " ")), [c["orig"], c["B1"], c["steervte"]]))
    cap = (r"\textbf{Failure modes of Banksy} (hand-picked to cover each type; SteerVTE shown for reference). "
           r"The first three keep the target readable but degrade appearance; the last three come from clips whose original "
           r"was already correct and that Banksy broke.")
    (G / "fig_fail.tex").write_text(grid_tex(rows, ["Original", "Banksy", "SteerVTE (ref.)"], r"0.27\linewidth", r"2.6cm", None, cap, "fig:fail", star=True))
    # background fidelity: difference maps
    D = {}
    for u in P.get("diff", []):
        o, s = diffmap(A.root, A.banksy, u, t, O, "diff"); D[u] = {"files": o, "stats": s}
    meta["diff"] = D
    (G / "cells_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print("cells:", len(list(O.glob("*"))), "| diff stats:", {u: v["stats"] for u, v in D.items()})


if __name__ == "__main__":
    main()
