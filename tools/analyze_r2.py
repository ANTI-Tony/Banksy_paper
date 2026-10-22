"""Gate R2 of the material generator (pre-registered: ideas/2026-10-22_材质生成器_预注册.md §3): on the 129 R2 clips (original
garbled, both B1 and SteerVTE repair it, outside the development sets), ONE evaluation run over original / SteerVTE / B1 /
flat geometry paste / model output. Criteria: appearance distance of the model vs B1, paired bootstrap 95% CI upper bound
< 0 and mean <= -0.30; readability (Qwen) not more than 1 point below B1; BG PSNR >= 41. Also reported: model vs SteerVTE,
flat vs B1 (what the geometry pipeline alone does), per material, glyph preservation via readability.
Usage: python analyze_r2.py <r2_eval dir> <r2_clips.txt> <out.json>"""
import json, math, sys
from pathlib import Path
import numpy as np

D, OUT = Path(sys.argv[1]), sys.argv[3]; U = [l.strip() for l in open(sys.argv[2]) if l.strip()]
E = json.load(open(D / "eval_vlm.json"))["per_clip"]; M = json.load(open(D / "mat_stats.json"))["per_clip"]
B = json.load(open(D / "bg.json"))["per_clip"]; T = json.load(open(D / "temporal.json"))["per_clip"]
rng = np.random.default_rng(0); ok = lambda m, u: u in E.get(m, {}) and E[m][u]["fixed"] >= 0.6
methods = [m for m in ("orig", "steervte", "B1", "flat", "matgen") if m in E]


def paired(S, k, a, b, us):
    d = np.array([S[a][u][k] - S[b][u][k] for u in us if u in S.get(a, {}) and u in S.get(b, {}) and np.isfinite(S[a][u][k]) and np.isfinite(S[b][u][k])], float)
    if len(d) < 3: return {"n": int(len(d))}
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    return {"mean": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "frac_better": float((d < 0).mean()), "n": int(len(d))}


def mcnemar(a, b, us):
    oa = sum(ok(a, u) and not ok(b, u) for u in us); ob = sum(ok(b, u) and not ok(a, u) for u in us); n = oa + ob
    return {"only_" + a: oa, "only_" + b: ob, "p": (min(1.0, 2 * sum(math.comb(n, k) for k in range(min(oa, ob) + 1)) / 2 ** n) if n else 1.0)}


R = {"n_clips": len(U), "methods": methods, "per_method": {}}
for m in methods:
    us = [u for u in U if u in E.get(m, {})]
    r = {"n": len(us), "read": 100 * float(np.mean([ok(m, u) for u in us])) if us else None}
    for S, k, name in ((M, "mad", "mad"), (B, "psnr_bg", "bg"), (T, "mc_flicker", "flicker"), (T, "edge_jitter", "jitter")):
        v = [S[m][u][k] for u in U if u in S.get(m, {}) and np.isfinite(S[m][u][k])]; r[name] = float(np.mean(v)) if v else None
    R["per_method"][m] = r
if "matgen" in E:
    R["matgen_vs_B1"] = {"mad": paired(M, "mad", "matgen", "B1", U), "read": mcnemar("matgen", "B1", U), "bg": paired(B, "psnr_bg", "matgen", "B1", U),
                         "flicker": paired(T, "mc_flicker", "matgen", "B1", U), "jitter": paired(T, "edge_jitter", "matgen", "B1", U)}
    R["matgen_vs_steervte"] = {"mad": paired(M, "mad", "matgen", "steervte", U), "read": mcnemar("matgen", "steervte", U)}
    mats = sorted({u.split("_")[2] for u in U})
    R["by_material"] = {k: {"n": len(us), "mad_matgen": R and float(np.mean([M["matgen"][u]["mad"] for u in us if u in M.get("matgen", {})] or [np.nan])),
                            "mad_B1": float(np.mean([M["B1"][u]["mad"] for u in us if u in M.get("B1", {})] or [np.nan])),
                            "mad_steervte": float(np.mean([M["steervte"][u]["mad"] for u in us if u in M.get("steervte", {})] or [np.nan])),
                            "read_matgen": 100 * float(np.mean([ok("matgen", u) for u in us])), "read_B1": 100 * float(np.mean([ok("B1", u) for u in us]))}
                        for k in mats for us in [[u for u in U if u.split("_")[2] == k]]}
    x = R["matgen_vs_B1"]; b1 = R["per_method"]["B1"]; mg = R["per_method"]["matgen"]
    R["criteria_R2"] = {"mad_ci_hi<0_and_mean<=-0.30": bool(x["mad"].get("hi", 1) < 0 and x["mad"].get("mean", 1) <= -0.30),
                        "read>=B1-1": bool(mg["read"] >= b1["read"] - 1.0), "bg>=41": bool((mg["bg"] or 0) >= 41.0)}
    R["criteria_R2"]["all"] = all(R["criteria_R2"].values())
if "flat" in E: R["flat_vs_B1"] = {"mad": paired(M, "mad", "flat", "B1", U), "read": mcnemar("flat", "B1", U)}
json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=1)
for m, r in R["per_method"].items(): print(f"{m:9s}", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
for k in ("matgen_vs_B1", "matgen_vs_steervte", "flat_vs_B1"):
    if k in R: print(k, json.dumps(R[k]))
if "criteria_R2" in R: print("CRITERIA R2:", R["criteria_R2"])
if "by_material" in R:
    for k, v in R["by_material"].items(): print(f"  {k:10s} n={v['n']:3d} mad matgen {v['mad_matgen']:.2f} B1 {v['mad_B1']:.2f} SteerVTE {v['mad_steervte']:.2f} | read matgen {v['read_matgen']:.0f} B1 {v['read_B1']:.0f}")
