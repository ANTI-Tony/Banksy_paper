"""External benchmarks with the frozen B1 (10-22, pre-registered). One evaluation directory per benchmark with
eval_vlm.json (Qwen), eval_vlm2_internvl.json, eval_easyocr.json, bg.json, temporal.json, mat_stats.json and
B1_shipped_unchanged.txt (front-end failures, scored as the unchanged clip).
VTE-Bench Real: readability (both VLM readers, EasyOCR) of the original, SteerVTE and B1, exact McNemar B1 vs SteerVTE,
background PSNR, flicker / jitter, appearance distance.
VidScribe V2V: readability on the 29 replacement / translation clips (both readers); for the 15 deletions the share whose
source string is no longer read (similarity < 0.5 on >= 4 of 5 frames); the historical all-44 readability; McNemar B1 vs
each commercial model on the 29; background PSNR, flicker / jitter, appearance distance.
Usage: python analyze_external.py vte <dir> <Real.csv> <out.json> | vs <dir> <infer.csv> <out.json>"""
import csv, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from textsim import ned_sim

KIND, D, CSV, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
rows = {Path(r["video"]).parent.name: r for r in csv.DictReader(open(CSV))}
J = lambda f: json.load(open(D / f)) if (D / f).exists() else None
Q, IV, EO, BG, TM, MS = J("eval_vlm.json"), J("eval_vlm2_internvl.json"), J("eval_easyocr.json"), J("bg.json"), J("temporal.json"), J("mat_stats.json")
fails = [l.strip() for l in open(D / "B1_shipped_unchanged.txt") if l.strip()] if (D / "B1_shipped_unchanged.txt").exists() else []


def ok(E, m, u): return E["per_clip"][m][u]["fixed"] >= 0.6


def read_rate(E, m, us): us = [u for u in us if u in E["per_clip"].get(m, {})]; return (100 * float(np.mean([ok(E, m, u) for u in us])) if us else None), len(us)


def mcnemar(E, a, b, us):
    us = [u for u in us if u in E["per_clip"].get(a, {}) and u in E["per_clip"].get(b, {})]
    oa = sum(ok(E, a, u) and not ok(E, b, u) for u in us); ob = sum(ok(E, b, u) and not ok(E, a, u) for u in us); n = oa + ob
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(oa, ob) + 1)) / 2 ** n) if n else 1.0
    return {"only_" + a: oa, "only_" + b: ob, "p": p, "n": len(us)}


def removed(E, m, us):
    """deletions: source string no longer read (similarity to src_text < 0.5) on >= 4 of 5 frames"""
    out = []
    for u in us:
        if u not in E["per_clip"].get(m, {}): continue
        sims = [ned_sim(x.replace("\n", " "), rows[u]["src_text"]) for x in E["per_clip"][m][u]["reads"].values()]
        out.append(float(np.mean([s < 0.5 for s in sims]) >= 0.75))
    return (100 * float(np.mean(out)) if out else None), len(out)


def mean_of(S, m, k):
    if not S or m not in S["per_clip"]: return None, 0
    v = [x[k] for x in S["per_clip"][m].values() if isinstance(x.get(k), (int, float)) and np.isfinite(x[k])]
    return (float(np.mean(v)) if v else None), len(v)


methods = list(Q["per_clip"].keys()); U = sorted(rows); R = {"kind": KIND, "methods": methods, "front_end_failures": fails, "n_clips": len(U)}
R["per_method"] = {}
for m in methods:
    r = {}
    if KIND == "vte":
        r["qwen_read"], r["qwen_n"] = read_rate(Q, m, U)
        if IV: r["internvl_read"], r["internvl_n"] = read_rate(IV, m, U)
        if EO and m in EO["per_clip"]: L = EO["per_clip"][m]; r["easyocr_read"] = 100 * float(np.mean([x["fixed"] for x in L])); r["easyocr_sim"] = float(np.mean([x["sim"] for x in L])); r["easyocr_flicker"] = float(np.mean([x["flicker"] for x in L]))
    else:
        rt = [u for u in U if rows[u]["op"] in ("replace", "translate")]; de = [u for u in U if rows[u]["op"] == "delete"]
        r["qwen_read29"], r["n29"] = read_rate(Q, m, rt); r["qwen_read44_hist"], r["n44"] = read_rate(Q, m, U)
        r["qwen_removed15"], r["n15"] = removed(Q, m, de)
        if IV: r["internvl_read29"], _ = read_rate(IV, m, rt); r["internvl_removed15"], _ = removed(IV, m, de)
    r["bg_psnr"], r["bg_n"] = mean_of(BG, m, "psnr_bg") if m != "orig" else (None, 0)
    r["flicker"], _ = mean_of(TM, m, "mc_flicker"); r["jitter"], _ = mean_of(TM, m, "edge_jitter")
    r["mad"], r["mad_n"] = mean_of(MS, m, "mad") if m != "orig" else (None, 0)
    R["per_method"][m] = r
if KIND == "vte":
    R["mcnemar_qwen"] = mcnemar(Q, "B1", "steervte", U)
    if IV: R["mcnemar_internvl"] = mcnemar(IV, "B1", "steervte", U)
    R["failure_clips"] = [{"uid": u, "src": rows[u].get("src_text"), "tgt": rows[u]["gt_text"]} for u in fails]
else:
    rt = [u for u in U if rows[u]["op"] in ("replace", "translate")]
    R["mcnemar_qwen29"] = {m: mcnemar(Q, "B1", m, rt) for m in methods if m not in ("orig", "B1")}
    if IV: R["mcnemar_internvl29"] = {m: mcnemar(IV, "B1", m, rt) for m in methods if m not in ("orig", "B1")}
    R["failure_clips"] = [{"uid": u, "op": rows[u]["op"], "src": rows[u]["src_text"], "tgt": rows[u]["gt_text"]} for u in fails]
json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=1)
for m, r in R["per_method"].items(): print(f"{m:24s}", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
for k in ("mcnemar_qwen", "mcnemar_internvl", "mcnemar_qwen29", "mcnemar_internvl29"):
    if k in R: print(k, R[k])
print("front-end failures:", R["failure_clips"])
