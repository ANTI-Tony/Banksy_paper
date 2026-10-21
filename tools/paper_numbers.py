"""Every number the paper takes from ONE evaluation run (10-22, user: no mixing of evaluation runs). Reads the run's
eval_vlm / temporal / mat_stats / bg / collateral / decisions JSONs and writes
  <out>/numbers.json   all values (for the Excel sheet and the manifest)
  <out>/numbers.tex    LaTeX macros (\\rXxx) used by the abstract, introduction and experiments
  <out>/tab_main.tex   Table 1 (original, SteerVTE, Banksy; all clips and clips outside the development set)
  <out>/tab_reuse.tex  the frozen negative-result variants B1f / R1 / R1f against B1 (same run)
Definitions follow the paper: readability = share of clips read exactly on >= 4 of 5 frames; fixed / broken relative to
the original as read in the same job; appearance = per-run-normalised glyph-agnostic material distance (mat_stats);
BG PSNR outside the dilated text box; flicker = motion-compensated, jitter = edge jitter (sys_temporal).
Paired intervals: 10000 bootstrap resamples over clips (seed 0); readability: exact McNemar on discordant clips.
Usage: python paper_numbers.py <run_dir> <sel_dev_uids.txt> <r1_dev_uids.txt> <out_dir>"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

D, OUT = Path(sys.argv[1]), Path(sys.argv[4]); OUT.mkdir(parents=True, exist_ok=True)
DEV40 = {l.strip() for l in open(sys.argv[2]) if l.strip()}; DEV9 = {l.strip() for l in open(sys.argv[3]) if l.strip()}
E = json.load(open(D / "eval_vlm.json"))["per_clip"]; T = json.load(open(D / "temporal.json"))["per_clip"]
M = json.load(open(D / "mat_stats.json"))["per_clip"]; B = json.load(open(D / "bg.json"))["per_clip"]
C = json.load(open(D / "collateral.json"))["per_clip"]; DEC = json.load(open(D / "decisions.json"))
ALL = sorted(E["orig"]); NONDEV = [u for u in ALL if u not in DEV40]; NONDEV49 = [u for u in ALL if u not in DEV40 | DEV9]
assert len(ALL) == 526 and len(DEV40) == 40 and len(DEV9) == 9, (len(ALL), len(DEV40), len(DEV9))
ok = lambda m, u: E[m][u]["fixed"] >= 0.6
rng = np.random.default_rng(0); NB = 10000
R = {}


def mean_of(src, m, key, us):
    v = [src[m][u][key] for u in us if u in src.get(m, {}) and np.isfinite(src[m][u][key])]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def row(m, us):
    r = {"n": len(us), "read": 100 * float(np.mean([ok(m, u) for u in us])), "sim": float(np.mean([E[m][u]["sim"] for u in us]))}
    if m != "orig":
        r["fixed"] = sum((not ok("orig", u)) and ok(m, u) for u in us); r["broken"] = sum(ok("orig", u) and not ok(m, u) for u in us)
        r["net"] = r["fixed"] - r["broken"]
        r["mad"], r["mad_n"] = mean_of(M, m, "mad", us); r["bg"], r["bg_n"] = mean_of(B, m, "psnr_bg", us)
    r["flicker"], _ = mean_of(T, m, "mc_flicker", us); r["jitter"], _ = mean_of(T, m, "edge_jitter", us)
    if m in C: r["collat"], r["collat_n"] = mean_of(C, m, "collat", us)
    return r


def mcnemar(a, b, us):
    only_a = sum(ok(a, u) and not ok(b, u) for u in us); only_b = sum(ok(b, u) and not ok(a, u) for u in us); n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return {"only_" + a: only_a, "only_" + b: only_b, "p": p}


def paired(src, key, a, b, us):
    """mean of (a - b) over clips both have, bootstrap CI, share of clips where a > b"""
    d = np.array([src[a][u][key] - src[b][u][key] for u in us if u in src.get(a, {}) and u in src.get(b, {})
                  and np.isfinite(src[a][u][key]) and np.isfinite(src[b][u][key])], float)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)])
    return {"mean": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "frac_pos": float((d > 0).mean()), "frac_neg": float((d < 0).mean()), "n": int(len(d))}


R["main"] = {blk: {m: row(m, us) for m in ("orig", "steervte", "B1")} for blk, us in (("all", ALL), ("nondev", NONDEV))}
R["mcnemar_all"] = mcnemar("B1", "steervte", ALL); R["mcnemar_nondev"] = mcnemar("B1", "steervte", NONDEV)
R["orig_counts"] = {"correct": sum(ok("orig", u) for u in ALL), "garbled": sum(not ok("orig", u) for u in ALL)}
R["bg_diff"] = paired(B, "psnr_bg", "B1", "steervte", ALL)
R["mad_diff"] = paired(M, "mad", "B1", "steervte", ALL)
R["flicker_diff"] = paired(T, "mc_flicker", "B1", "steervte", ALL)
R["jitter_diff"] = paired(T, "edge_jitter", "B1", "steervte", ALL)
# appearance where both methods repair a garbled clip (excludes clips the preservation gate kept, whose distance is ~0)
both = [u for u in ALL if not ok("orig", u) and ok("B1", u) and ok("steervte", u) and u in M["B1"] and u in M["steervte"]]
R["both_repaired_appearance"] = {"n": len(both), "steervte_closer": sum(M["steervte"][u]["mad"] < M["B1"][u]["mad"] for u in both),
                                 "median_steervte": float(np.median([M["steervte"][u]["mad"] for u in both])), "median_B1": float(np.median([M["B1"][u]["mad"] for u in both]))}
R["both_repaired_appearance"]["frac"] = R["both_repaired_appearance"]["steervte_closer"] / len(both)
R["at_least_one_correct"] = 100 * float(np.mean([ok("B1", u) or ok("steervte", u) for u in ALL]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from textsim import norm, fold
from collections import Counter


def overlap(t, r):
    """share of the target's characters (multiset) present in one transcription"""
    t = norm(fold(t)); rc = Counter(norm(fold(r)))
    return sum(min(v, rc[k]) for k, v in Counter(t).items()) / len(t) if t else float("nan")


R["target_chars_in_garbled_reading"] = {lang: float(np.median([max(overlap(E["orig"][u]["target"], x) for x in E["orig"][u]["reads"].values())
                                                              for u in ALL if u.split("_")[1] == lang and not ok("orig", u)])) for lang in ("en", "zh")}
mats = sorted({u.split("_")[2] for u in ALL}); langs = ("en", "zh")
R["by_material"] = {k: {"n": len(us), **{m: 100 * float(np.mean([ok(m, u) for u in us])) for m in ("orig", "steervte", "B1")}}
                    for k in mats for us in [[u for u in ALL if u.split("_")[2] == k]]}
R["by_language"] = {k: {"n": len(us), **{m: 100 * float(np.mean([ok(m, u) for u in us])) for m in ("orig", "steervte", "B1")}}
                    for k in langs for us in [[u for u in ALL if u.split("_")[1] == k]]}
noedit = [u for u in ALL if (DEC.get(u, {}).get("select", {}).get("B1") or {}).get("noedit")]
R["preservation"] = {"skipped": len(noedit), "skipped_reader_correct": sum(ok("orig", u) for u in noedit),
                     "orig_correct": R["orig_counts"]["correct"], "broken_b1": R["main"]["all"]["B1"]["broken"]}
# frozen negative results: the variants of the same pass against B1
V = ("B1", "B1f", "R1", "R1f")
R["variants"] = {blk: {m: row(m, us) for m in V} for blk, us in (("all", ALL), ("nondev49", NONDEV49))}
R["variant_diffs"] = {blk: {m: {"mad": paired(M, "mad", m, "B1", us), "read": mcnemar(m, "B1", us),
                                "flicker": paired(T, "mc_flicker", m, "B1", us), "jitter": paired(T, "edge_jitter", m, "B1", us)}
                            for m in V[1:]} for blk, us in (("all", ALL), ("nondev49", NONDEV49))}
reuse = {u: (DEC[u]["select"].get("R1") or {}).get("reuse") or {} for u in ALL}
drawn = [u for u in ALL if reuse[u].get("drawn")]
why = defaultdict(int)
for u in ALL:
    if not reuse[u].get("drawn"): why[reuse[u].get("why", "not evaluated (no edit / not English)")] += 1
garbled_en = [u for u in ALL if u.split("_")[1] == "en" and not ok("orig", u)]
R["reuse"] = {"applied": len(drawn), "applied_dev9": sum(u in DEV9 for u in drawn), "applied_clips": drawn, "why_not": dict(sorted(why.items(), key=lambda x: -x[1])),
              "garbled_en": len(garbled_en), "applied_garbled_en": sum(u in garbled_en for u in drawn),
              "applied_nondev_mad": paired(M, "mad", "R1", "B1", [u for u in drawn if u not in DEV9]) if len([u for u in drawn if u not in DEV9]) >= 2 else None}
# test-retest noise of the appearance estimator (unseeded k-means++ in the style estimator): R1 / R1f leave every clip
# without reuse byte-identical to B1 / B1f (identical flicker on all of them), yet each gets its own appearance score
R["mad_retest"] = {}
for a_, b_ in (("R1", "B1"), ("R1f", "B1f")):
    us = [u for u in ALL if u not in drawn and u in M[a_] and u in M[b_] and T[a_][u]["mc_flicker"] == T[b_][u]["mc_flicker"]]
    d = np.array([M[a_][u]["mad"] - M[b_][u]["mad"] for u in us]); bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)])
    R["mad_retest"][f"{a_}-{b_}"] = {"n": len(us), "identical_score": int((d == 0).sum()), "mean": float(d.mean()), "lo": float(np.percentile(bs, 2.5)),
                                     "hi": float(np.percentile(bs, 97.5)), "sd": float(d.std()), "abs_gt_0.5": int((np.abs(d) > 0.5).sum())}
R["broken_B1"] = [u for u in ALL if ok("orig", u) and not ok("B1", u)]
R["broken_steervte"] = [u for u in ALL if ok("orig", u) and not ok("steervte", u)]
json.dump(R, open(OUT / "numbers.json", "w"), ensure_ascii=False, indent=1)

f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
a, nd = R["main"]["all"], R["main"]["nondev"]
mac = {
    "rOrigRead": f1(a["orig"]["read"]), "rSteerRead": f1(a["steervte"]["read"]), "rOursRead": f1(a["B1"]["read"]),
    "rReadGain": f1(a["B1"]["read"] - a["steervte"]["read"]), "rSteerUnread": f"{100 - a['steervte']['read']:.0f}",
    "rOursFixed": str(a["B1"]["fixed"]), "rSteerFixed": str(a["steervte"]["fixed"]), "rOursBroken": str(a["B1"]["broken"]),
    "rSteerBroken": str(a["steervte"]["broken"]), "rOursNet": str(a["B1"]["net"]), "rSteerNet": str(a["steervte"]["net"]),
    "rGarbled": str(R["orig_counts"]["garbled"]), "rOrigCorrect": str(R["orig_counts"]["correct"]),
    "rOnlyOurs": str(R["mcnemar_all"]["only_B1"]), "rOnlySteer": str(R["mcnemar_all"]["only_steervte"]),
    "rMcNemarP": (lambda p: f"{p / 10 ** math.floor(math.log10(p)):.1f}\\times10^{{{math.floor(math.log10(p))}}}")(R["mcnemar_all"]["p"]),
    "rOursReadND": f1(nd["B1"]["read"]), "rSteerReadND": f1(nd["steervte"]["read"]), "rOursNetND": str(nd["B1"]["net"]), "rSteerNetND": str(nd["steervte"]["net"]),
    "rOursBG": f1(a["B1"]["bg"]), "rSteerBG": f1(a["steervte"]["bg"]), "rBGDiff": f1(R["bg_diff"]["mean"]),
    "rBGDiffLo": f1(R["bg_diff"]["lo"]), "rBGDiffHi": f1(R["bg_diff"]["hi"]), "rBGHigherPct": f1(100 * R["bg_diff"]["frac_pos"]),
    "rOursJitter": f2(a["B1"]["jitter"]), "rSteerJitter": f2(a["steervte"]["jitter"]), "rOursFlicker": f2(a["B1"]["flicker"]), "rSteerFlicker": f2(a["steervte"]["flicker"]),
    "rOursMad": f2(a["B1"]["mad"]), "rSteerMad": f2(a["steervte"]["mad"]),
    "rBothRepairN": str(R["both_repaired_appearance"]["n"]), "rSteerCloserPct": f"{100 * R['both_repaired_appearance']['frac']:.0f}",
    "rBothRepairMedSteer": f2(R["both_repaired_appearance"]["median_steervte"]), "rBothRepairMedOurs": f2(R["both_repaired_appearance"]["median_B1"]),
    "rAtLeastOne": f1(R["at_least_one_correct"]), "rCharsEn": f"{100 * R['target_chars_in_garbled_reading']['en']:.0f}", "rCharsZh": f"{100 * R['target_chars_in_garbled_reading']['zh']:.0f}",
    "rOursCollat": f2(a["B1"]["collat"]), "rOursCollatN": str(a["B1"]["collat_n"]),
    "rEnN": str(R["by_language"]["en"]["n"]), "rZhN": str(R["by_language"]["zh"]["n"]),
    "rOursEn": f1(R["by_language"]["en"]["B1"]), "rOursZh": f1(R["by_language"]["zh"]["B1"]),
    "rSteerEn": f1(R["by_language"]["en"]["steervte"]), "rSteerZh": f1(R["by_language"]["zh"]["steervte"]),
    "rSkipped": str(R["preservation"]["skipped"]), "rSkippedOK": str(R["preservation"]["skipped_reader_correct"]),
    "rReuseApplied": str(R["reuse"]["applied"]), "rReuseAppliedDev": str(R["reuse"]["applied_dev9"]),
    "rGarbledEn": str(R["reuse"]["garbled_en"]), "rReuseGarbledEn": str(R["reuse"]["applied_garbled_en"]),
    "rReuseNDMad": f"{R['reuse']['applied_nondev_mad']['mean']:+.2f}" if R["reuse"]["applied_nondev_mad"] else "--",
    "rReuseWhyLetters": str(R["reuse"]["why_not"].get("letters != reading", 0)), "rReuseWhySource": str(R["reuse"]["why_not"].get("source reads the target", 0)),
    "rReuseWhyFont": str(R["reuse"]["why_not"].get("new letters would not match the kept letters (font IoU)", 0)),
    "rRetestN": str(R["mad_retest"]["R1-B1"]["n"]), "rRetestSame": str(R["mad_retest"]["R1-B1"]["identical_score"]),
    "rRetestSD": f2(R["mad_retest"]["R1-B1"]["sd"]), "rRetestMean": f"{R['mad_retest']['R1-B1']['mean']:+.3f}",
    "rRetestLo": f"{R['mad_retest']['R1-B1']['lo']:+.3f}", "rRetestHi": f"{R['mad_retest']['R1-B1']['hi']:+.3f}",
    "rRetestMeanB": f"{R['mad_retest']['R1f-B1f']['mean']:+.3f}",
}
for k in mats:
    mac["rGain" + k.capitalize()] = f1(R["by_material"][k]["B1"] - R["by_material"][k]["steervte"])
vd = R["variant_diffs"]
for blk, tag in (("all", ""), ("nondev49", "ND")):
    for m in V[1:]:
        x = vd[blk][m]["mad"]; mac[f"r{m}MadDiff{tag}"] = f"{x['mean']:+.3f}"; mac[f"r{m}MadLo{tag}"] = f"{x['lo']:+.3f}"; mac[f"r{m}MadHi{tag}"] = f"{x['hi']:+.3f}"
CS = OUT / "cache_stats.json"                        # statistics read from the run's caches on the cluster (same pass)
if CS.exists():
    cs = json.load(open(CS)); R["cache_stats"] = cs
    mac.update({"rTiltN": str(cs["tilt_rotated_clips"]), "rFshapeN": str(cs["fshape_font_changed_clips"]),
                "rLatinFam": str(cs["b1_fonts"]["latin_families_selected"]), "rZhFam": str(cs["b1_fonts"]["chinese_families_selected"]),
                "rNotoPct": f"{100 * cs['b1_fonts']['noto_sans_sc_share_zh']:.0f}"})
    if "box_policy_clips" in cs:
        mac.update({"rBoxPolicy": str(cs["box_policy_clips"]), "rInstancePath": str(cs["instance_path_clips"]), "rSatOn": str(cs["sat_enabled"]),
                    "rNotoSerifPct": f"{100 * cs['fonts_zh_share']['Noto Serif SC']:.0f}", "rZcoolPct": f"{100 * cs['fonts_zh_share']['ZCOOL XiaoWei']:.0f}"})
    json.dump(R, open(OUT / "numbers.json", "w"), ensure_ascii=False, indent=1)
with open(OUT / "numbers.tex", "w") as fh:
    fh.write(f"% generated by paper_numbers.py from {D.name} (one evaluation job); do not edit by hand\n")
    for k, v in mac.items(): fh.write(f"\\newcommand{{\\{k.replace('1', 'One')}}}{{{v}}}\n")


def tline(name, r, bold):
    cells = [f1(r["read"]), str(r.get("fixed", "--")), str(r.get("broken", "--")), str(r.get("net", "--")), f3(r["sim"]),
             f2(r["mad"]) if "mad" in r else "--", f2(r["bg"]) if "bg" in r else "--", f2(r["flicker"]), f2(r["jitter"])]
    cells = [f"\\textbf{{{c}}}" if i in bold else c for i, c in enumerate(cells)]
    return f"{name} & " + " & ".join(cells) + " \\\\\n"


def bold_cols(rows):
    """index -> best row name per column (higher better except broken, appearance, flicker, jitter)"""
    keys = [("read", 1), ("fixed", 1), ("broken", -1), ("net", 1), ("sim", 1), ("mad", -1), ("bg", 1), ("flicker", -1), ("jitter", -1)]
    best = defaultdict(set)
    for i, (k, s) in enumerate(keys):
        vals = {n: round(r[k], 3) for n, r in rows.items() if k in r}
        if len(vals) < 2: continue
        top = max(vals.values()) if s > 0 else min(vals.values())
        for n, v in vals.items():
            if v == top: best[n].add(i)
    return best


names = {"orig": "Original (generated)", "steervte": "\\steer~\\cite{zeng2026steervte}", "B1": "\\method"}
with open(OUT / "tab_main.tex", "w") as fh:
    fh.write(f"% generated by paper_numbers.py: Table 1 from {D.name} (one front-end pass, one evaluation job)\n")
    fh.write("\\begin{table*}[t]\n\\centering\n\\caption{\\textbf{Main results} (526 clips; one front-end pass and one evaluation job for every row). "
             "Readability: share of clips whose text the reader transcribes exactly in at least 4 of 5 frames. Fixed / broken: garbled $\\to$ correct and correct $\\to$ wrong transitions relative to the original as read in the same job; net = fixed $-$ broken. "
             "Appearance: glyph-agnostic material distance to the original text (normalised within this run; lower is closer). BG PSNR: outside the dilated text region. Flicker and jitter: motion-compensated, inside the text box (grey levels). "
             "The lower block excludes the 40 clips inspected while developing the selective-editing components.}\n\\label{tab:main}\n\\setlength{\\tabcolsep}{4.5pt}\n\\resizebox{\\linewidth}{!}{%\n"
             "\\begin{tabular}{@{}lccccccccc@{}}\n\\toprule\n & Readability $\\uparrow$ & Fixed $\\uparrow$ & Broken $\\downarrow$ & Net $\\uparrow$ & NED sim.\\ $\\uparrow$ & Appearance $\\downarrow$ & BG PSNR $\\uparrow$ & Flicker $\\downarrow$ & Jitter $\\downarrow$ \\\\\n\\midrule\n")
    for blk, title in (("all", "All 526 clips"), ("nondev", f"{len(NONDEV)} clips outside the development set")):
        rows = R["main"][blk]; bc = bold_cols({k: v for k, v in rows.items() if k != "orig"})
        fh.write(f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\\n")
        for m in ("orig", "steervte", "B1"): fh.write(tline(names[m], rows[m], bc.get(m, set()) if m != "orig" else set()))
        if blk == "all": fh.write("\\midrule\n")
    fh.write("\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
with open(OUT / "tab_supp_comp.tex", "w") as fh:
    fh.write(f"% generated by paper_numbers.py from {D.name}\n\\begin{{table}}[h]\n\\centering\n\\caption{{Composition of the benchmark and readability (\\%) per group (same run as \\cref{{tab:main}}).}}\n\\label{{tab:supp_comp}}\n\\small\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{tabular}}{{@{{}}lccccc@{{}}}}\n\\toprule\nGroup & EN & ZH & Original & \\steer & \\method \\\\\n\\midrule\n")
    for k in ("neon", "signboard", "wood", "metal", "glass", "paper", "screen"):
        en = sum(1 for u in ALL if u.split("_")[2] == k and u.split("_")[1] == "en"); zh = sum(1 for u in ALL if u.split("_")[2] == k and u.split("_")[1] == "zh")
        b_ = R["by_material"][k]; fh.write(f"{k} & {en} & {zh} & {f1(b_['orig'])} & {f1(b_['steervte'])} & {f1(b_['B1'])} \\\\\n")
    fh.write("\\midrule\n")
    for k, nm in (("en", "English"), ("zh", "Chinese")):
        b_ = R["by_language"][k]; fh.write(f"{nm} ({b_['n']}) & & & {f1(b_['orig'])} & {f1(b_['steervte'])} & {f1(b_['B1'])} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
vnames = {"B1": "\\method (B1, final)", "B1f": "+ baseline layout + glyph-shape font (B1f)", "R1": "+ source-glyph reuse (R1)", "R1f": "+ both (R1f)"}
with open(OUT / "tab_reuse.tex", "w") as fh:
    fh.write(f"% generated by paper_numbers.py: frozen variants from {D.name} (same front-end pass and evaluation job as Table 1)\n")
    fh.write("\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Two appearance modules that did not pass their pre-registered criteria} (same front-end pass and evaluation job as \\cref{tab:main}). "
             "$\\Delta$App.: paired appearance difference to B1 with 95\\% bootstrap CI, on the clips outside both development sets "
             f"({len(NONDEV49)} clips); the criterion required the upper bound to be below 0. R1 changed {R['reuse']['applied']} clips, {R['reuse']['applied_dev9']} of them development clips.}}\n"
             "\\label{tab:reuse}\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
             " & Read.\\ $\\uparrow$ & Net $\\uparrow$ & App.\\ $\\downarrow$ & $\\Delta$App.\\ vs.\\ B1 \\\\\n\\midrule\n")
    for m in V:
        r = R["variants"]["all"][m]
        d = "--" if m == "B1" else (lambda x: f"${x['mean']:+.3f}$ [${x['lo']:+.3f}$, ${x['hi']:+.3f}$]")(vd["nondev49"][m]["mad"])
        fh.write(f"{vnames[m]} & {f1(r['read'])} & {r['net']} & {f2(r['mad'])} & {d} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
print(json.dumps({"main_all": {m: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for m, r in a.items()},
                  "mcnemar": R["mcnemar_all"], "bg_diff": R["bg_diff"], "both_repaired_appearance": R["both_repaired_appearance"],
                  "at_least_one": R["at_least_one_correct"], "chars": R["target_chars_in_garbled_reading"],
                  "by_language": R["by_language"], "preservation": R["preservation"], "reuse": {k: v for k, v in R["reuse"].items() if k != "applied_clips"}},
                 ensure_ascii=False, indent=1))
print("PAPER_NUMBERS_DONE", OUT)
