"""Numbers and tables for the glyph-free reconstruction rounds v2-v4.1 and the sealed cross-generator test (T4) of v4.1.
Every macro is read from one of the source files below; nothing is typed by hand. Writes numbers_g4.tex, tables/tab_g4.tex
(registered gates of the sealed test, per generator) and tables/tab_g_rounds.tex (all rounds v1-v4.1 with their verdicts).
Sources (archived under results/ of the project folder; cluster paths in MANIFEST.md):
  --t4      out/l3/v4/verdict_T4.json                (the single sealed run, job 56702391; frozen code frozen_v41_20261025)
  --t4nat-cog / --t4nat-wan   out/l3/v4/{cog,wan}_T4/analysis_v4nat.json (same run: per-row natural-subset counts, broken clips, raw temporal)
  --d4v40   out/l3/v4/verdict_D4_devcheck_v40.json   (D4 development check of the frozen v4-LT; development evidence only)
  --d4v41   out/l3/v4/verdict_D4_devcheck_v41.json   (the same after the two locality fixes of v4.1)
  --cmp     out/l3/v4/{cog,wan}_D4/compare_v41.json  (v4.0 vs v4.1 regression on D4)
  --v2syn / --v2nat / --v3syn / --v3l2   analysis_g.json, analysis_n.json, analysis_g2.json, analysis_b3.json (registered v2 / v3 runs)
Usage: python tools/make_numbers_g4.py --t4 F --d4v40 F --d4v41 F --cmp-cog F --cmp-wan F --v2syn F --v2nat F --v3syn F --v3l2 F [--out .]"""
import argparse, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser()
for k in ("t4", "t4nat-cog", "t4nat-wan", "d4v40", "d4v41", "cmp-cog", "cmp-wan", "v2syn", "v2nat", "v3syn", "v3l2"): ap.add_argument("--" + k, required=True)
ap.add_argument("--out", default=".")
A = ap.parse_args(); OUT = Path(A.out)
src = {k: getattr(A, k.replace("-", "_")) for k in ("t4", "t4nat-cog", "t4nat-wan", "d4v40", "d4v41", "cmp-cog", "cmp-wan", "v2syn", "v2nat", "v3syn", "v3l2")}
J = {k: json.load(open(v)) for k, v in src.items()}
M = []                                                   # (name, value)


def S(x, nd):                                            # signed number, proper minus sign in text and math mode
    v = round(float(x), nd)
    if v == 0: v = 0.0
    s = f"{abs(v):.{nd}f}"
    return ("\\ensuremath{-}" if v < 0 else "+") + s


def U(x, nd): return f"{float(x):.{nd}f}"
def P(x, nd=1): return f"{100 * float(x):.{nd}f}"                     # proportion -> percent
def PS(x, nd=1): return S(100 * float(x), nd)                          # signed percentage points
def put(name, val): assert name.isalpha(), name; M.append((name, str(val)))
def ci(name, d, nd, pct=False):
    f = PS if pct else (lambda x, n=nd: S(x, n))
    put(name, f(d["mean"], nd) if pct else f(d["mean"])); put(name + "Lo", f(d["lo"], nd) if pct else f(d["lo"])); put(name + "Hi", f(d["hi"], nd) if pct else f(d["hi"]))


T = J["t4"]; G = {"Cog": T["per_generator"]["cog"], "Wan": T["per_generator"]["wan"]}
assert T["verdict"] == "FAIL"
# ---------------------------------------------------------------- sealed test T4
for g, V in G.items():
    c = V["counts"]; put(f"gT{g}Raw", c["raw_clips"]); put(f"gT{g}Lone", c["classes"]["L1"]); put(f"gT{g}Ltwo", c["classes"]["L2"]); put(f"gT{g}Lthree", c["classes"]["L3"])
    put(f"gT{g}N", c["synthetic_subset"]); put(f"gT{g}Prompts", V["read_v4LT_minus_b1fb"]["n_prompts"]); put(f"gT{g}LocFail", P(c["loc_fail_rate"])); put(f"gT{g}FallbackRate", P(c["fallback_rate"]))
    r = V["read"]; put(f"gT{g}Read", P(r["v4LT"])); put(f"gT{g}ReadFb", P(r["b1fb"])); put(f"gT{g}ReadVthree", P(r["v3"])); ci(f"gT{g}ReadGain", V["read_v4LT_minus_b1fb"], 1, pct=True)
    n = V["nat_read"]; put(f"gT{g}NatOrig", P(n["orig"])); put(f"gT{g}NatBone", P(n["B1"])); put(f"gT{g}NatVthree", P(n["v3"])); put(f"gT{g}NatRead", P(n["v4LT"]))
    ci(f"gT{g}NatGain", V["nat_read_v4LT_minus_B1"], 1, pct=True); put(f"gT{g}NatN", V["nat_read_v4LT_minus_B1"]["n"]); put(f"gT{g}NatPrompts", V["nat_read_v4LT_minus_B1"]["n_prompts"])
    t = V["temporal_v4LT_minus_b1fb"]; ci(f"gT{g}JitN", t["edge_jitter_norm"], 3); ci(f"gT{g}FlkN", t["mc_flicker_norm"], 3); ci(f"gT{g}JitRaw", t["edge_jitter_raw"], 1); ci(f"gT{g}FlkRaw", t["mc_flicker_raw"], 2)
    put(f"gT{g}TempN", t["edge_jitter_norm"]["n"])
    b = V["bg_synthetic"]; put(f"gT{g}BgSyn", U(b["means"]["v4LT"], 2)); put(f"gT{g}BgSynNoop", U(b["means"]["noop"], 2)); put(f"gT{g}BgSynFb", U(b["means"]["b1fb"], 2)); ci(f"gT{g}BgSynDiff", b["v4LT_minus_b1fb"], 2)
    b = V["bg_natural"]; put(f"gT{g}BgNat", U(b["means"]["v4LT"], 2)); put(f"gT{g}BgNatNoop", U(b["means"]["noop"], 2)); put(f"gT{g}BgNatBone", U(b["means"]["B1"], 2)); ci(f"gT{g}BgNatDiff", b["v4LT_minus_B1"], 2)
    put(f"gT{g}Overflow", P(V["overflow_rate"]["v4LT"])); put(f"gT{g}OverflowFb", P(V["overflow_rate"]["b1fb"])); put(f"gT{g}Spill", P(V["spill_mean"]))
    k = V["ctrl_T_minus_A1"]; ci(f"gT{g}CtrlJit", k["edge_jitter"], 2); ci(f"gT{g}CtrlFlk", k["mc_flicker"], 3); put(f"gT{g}CtrlN", k["edge_jitter"]["n"]); put(f"gT{g}CtrlFb", V["ctrl_fallback"])
    q = V["geometry_v4LT_minus_b1fb"]
    for key, nm in (("offset", "Offset"), ("drift", "Drift"), ("motion_err", "MotionErr"), ("iou", "IoU"), ("corner_err", "Corner")): ci(f"gT{g}{nm}", q[key], 3)
    put(f"gT{g}Height", U(V["height_ratio_median"], 3))
    o = V["containment"]; put(f"gT{g}Cont", U(o["v4LT_mean"], 3)); put(f"gT{g}ContFb", U(o["b1fb_mean"], 3)); ci(f"gT{g}ContDiff", o["v4LT_minus_b1fb"], 3); put(f"gT{g}ContRate", P(o["v4LT_rate_ge_095"]))
    s = V["severe"]; put(f"gT{g}Severe", s["any"]); put(f"gT{g}SevereCross", s["counts"]["cross"]); put(f"gT{g}SevereOther", s["counts"]["outside"] + s["counts"]["clipped"] + s["counts"]["missing_chars"])
    sn = V["severe_natural"]; put(f"gT{g}NatClipped", sn["clipped"]); put(f"gT{g}NatMissing", sn["missing_chars"]); put(f"gT{g}NatSevN", sn["n"])
for g, key in (("Cog", "t4nat-cog"), ("Wan", "t4nat-wan")):
    rw = J[key]["rows"]; put(f"gT{g}NatBroken", rw["v4LT"]["broken"]); put(f"gT{g}NatBoneBroken", rw["B1"]["broken"]); put(f"gT{g}NatFixed", rw["v4LT"]["fixed"]); put(f"gT{g}NatBoneFixed", rw["B1"]["fixed"])
    put(f"gT{g}NatFallback", P(rw["v4LT"]["fallback_rate"])); put(f"gT{g}NatAll", J[key]["n"]); put(f"gT{g}NatBoneN", rw["B1"]["n"])
    for m, nm in (("v4LT", ""), ("B1", "Bone")): put(f"gT{g}NatJit{nm}", U(rw[m]["jitter"], 2)); put(f"gT{g}NatFlk{nm}", U(rw[m]["flicker"], 2))
PL = T["pooled"]; ci("gTReadGain", PL["read_v4LT_minus_b1fb"], 1, pct=True); put("gTReadGainU", P(PL["read_v4LT_minus_b1fb"]["mean"]))
put("gTPoolRead", P(PL["read_v4LT"])); put("gTPoolReadVthree", P(PL["read_v3"]))
put("gTSynN", G["Cog"]["counts"]["synthetic_subset"] + G["Wan"]["counts"]["synthetic_subset"])
put("gTNatN", G["Cog"]["nat_read_v4LT_minus_B1"]["n"] + G["Wan"]["nat_read_v4LT_minus_B1"]["n"])
gap = max(V[k]["means"]["noop"] - V[k]["means"]["v4LT"] for V in G.values() for k in ("bg_synthetic", "bg_natural")); put("gTBgMaxGap", U(gap, 2))
gates = {k: v for k, v in T["gates"].items()}; booleans = {k: v for k, v in gates.items() if isinstance(v, bool)}
put("gTGatesTotal", len(booleans)); put("gTGatesFailed", sum(1 for v in booleans.values() if not v)); put("gTGatesPending", sum(1 for v in gates.values() if v is None))
assert sorted(T["failed"]) == sorted(k for k, v in booleans.items() if not v)
# ---------------------------------------------------------------- D4 development (v4-LT = v4.0, then v4.1)
D0, D1 = J["d4v40"], J["d4v41"]
for g, key in (("Cog", "cog"), ("Wan", "wan")):
    V = D1["per_generator"][key]; c = V["counts"]
    put(f"gD{g}Raw", c["raw_clips"]); put(f"gD{g}Lone", c["classes"]["L1"]); put(f"gD{g}Ltwo", c["classes"]["L2"]); put(f"gD{g}Lthree", c["classes"]["L3"]); put(f"gD{g}N", c["synthetic_subset"])
    ci(f"gD{g}ReadGain", V["read_v4LT_minus_b1fb"], 1, pct=True); put(f"gD{g}Height", U(V["height_ratio_median"], 3))
    put(f"gD{g}NatBgVfour", U(D0["per_generator"][key]["bg_natural"]["means"]["v4LT"], 2)); put(f"gD{g}NatBgVfourone", U(V["bg_natural"]["means"]["v4LT"], 2)); put(f"gD{g}NatBgNoop", U(V["bg_natural"]["means"]["noop"], 2))
    put(f"gD{g}Cont", U(V["containment"]["v4LT_mean"], 3))
ci("gDReadGain", D1["pooled"]["read_v4LT_minus_b1fb"], 1, pct=True)
put("gDFailedVfour", len(D0["failed"])); put("gDFailedVfourone", len(D1["failed"]))
for g, key in (("Cog", "cmp-cog"), ("Wan", "cmp-wan")):
    C = J[key]; f = C["natural"]["fallback"]; o = C["natural"]["own"]
    put(f"gC{g}FbN", f["n"]); put(f"gC{g}FbBgBefore", S(f["bg_v40_minus_noop_mean"], 2)); put(f"gC{g}FbBgAfter", S(f["bg_v41_minus_noop_mean"], 2))
    put(f"gC{g}FbReadBefore", f["read_v40"]); put(f"gC{g}FbReadAfter", f["read_v41"]); put(f"gC{g}LeakFrac", U(100 * f["leak_before_compositing_mean"]["changed_outside_frac"], 2))
    put(f"gC{g}LeakShare", U(100 * f["leak_before_compositing_mean"]["changed_mass_outside_share"], 1))
    put(f"gC{g}OwnBgBefore", S(o["bg_v40_minus_noop_mean"], 2)); put(f"gC{g}OwnBgAfter", S(o["bg_v41_minus_noop_mean"], 2))
    inv = C["invariant_and_layout"]; put(f"gC{g}Checked", sum(v["checked"] for v in inv.values())); put(f"gC{g}LayoutSame", sum(v["layout_identical_ref_quad"] for v in inv.values()))
    assert all(v["checked"] == v["clips"] == v["layout_identical_ref_quad"] == v["fallback_same"] for v in inv.values())
    assert not C["synthetic"]["own"]["read_flips_lost"] and not C["natural"]["own"]["read_flips_lost"]
# ---------------------------------------------------------------- v2 / v3 (registered runs)
g2, n2, g3, b3 = J["v2syn"], J["v2nat"], J["v3syn"], J["v3l2"]
R2 = g2["rows_locatable"]; put("gVtwoSynN", g2["n_locatable"]); put("gVtwoSynRead", U(R2["carrier"]["read"], 1)); put("gVtwoSynFb", U(R2["b1fb"]["read"], 1)); put("gVoneSynRead", U(R2["carrier1"]["read"], 1))
put("gVtwoAoneRead", U(R2["v2A1"]["read"], 1)); put("gVtwoAtwoRead", U(R2["v2A2"]["read"], 1)); put("gVtwoHeight", U(R2["carrier"]["height_ratio"], 2)); ci("gVtwoJit", g2["paired_v2_minus_b1fb"]["jitter"], 2)
put("gVtwoFailed", sum(1 for k, v in g2["criteria_G"].items() if k != "all" and v is False)); put("gVtwoCriteria", sum(1 for k in g2["criteria_G"] if k != "all"))
N2 = n2["rows"]; put("gVtwoNatN", n2["n"]); put("gVtwoNatRead", U(N2["carrier"]["read"], 1)); put("gVtwoNatBone", U(N2["B1"]["read"], 1)); put("gVoneNatRead", U(N2["carrier1"]["read"], 1))
R3 = g3["rows_locatable"]; put("gVthreeSynN", g3["n_locatable"]); put("gVthreeSynAll", g3["n_locatable"] + g3["n_not_locatable"]); put("gVthreeSynRead", U(R3["carrier"]["read"], 1)); put("gVthreeSynFb", U(R3["b1fb"]["read"], 1))
put("gVthreeHeight", U(R3["carrier"]["height_ratio"], 2)); put("gVthreeBg", U(R3["carrier"]["bg"], 2)); ci("gVthreeJit", g3["paired_v2_minus_b1fb"]["jitter"], 1); ci("gVthreeFlk", g3["paired_v2_minus_b1fb"]["flicker"], 2)
rd = g3["paired_v2_minus_b1fb"]["read"]; put("gVthreeWins", rd["only_carrier"]); put("gVthreeLosses", rd["only_b1fb"]); put("gVthreeP", U(rd["p"], 3))
put("gVthreeFailed", sum(1 for k, v in g3["criteria_G"].items() if k != "all" and v is False)); put("gVthreeCriteria", sum(1 for k in g3["criteria_G"] if k != "all"))
B3 = b3["rows"]; put("gVthreeLtwoRead", U(B3["carrier"]["read"], 1)); put("gVthreeLtwoBone", U(B3["B1"]["read"], 1)); put("gVthreeLtwoBg", U(B3["carrier"]["bg"], 2))
rd = b3["carrier_vs_B1"]["read"]; put("gVthreeLtwoWins", rd["only_carrier"]); put("gVthreeLtwoLosses", rd["only_B1"]); put("gVthreeLtwoP", U(rd["p"], 2))

# ---------------------------------------------------------------- write
names = [n for n, _ in M]; assert len(names) == len(set(names)), [n for n in names if names.count(n) > 1]
hdr = ["%% generated by tools/make_numbers_g4.py -- do not edit by hand", "%% sources (sha256):"]
hdr += [f"%%   {k}: {Path(v).name} {hashlib.sha256(Path(v).read_bytes()).hexdigest()}" for k, v in src.items()]
(OUT / "numbers_g4.tex").write_text("\n".join(hdr + [f"\\newcommand{{\\{n}}}{{{v}}}" for n, v in M]) + "\n")
mk = lambda ok: "\\ding{51}" if ok else "\\textbf{\\ding{55}}"
gk = lambda gen, key: gates[f"[{gen}] {key}"]
V = {"cog": G["Cog"], "wan": G["Wan"]}
rows = [("\\multicolumn{6}{l}{\\textit{Correctness (synthetic glyph-free; natural pseudo-text is a separate task)}}", None),
        ("Readability gain over the static fallback, pp", "CI lo $\\geq -5$", lambda g: f"{PS(V[g]['read_v4LT_minus_b1fb']['mean'])} [{PS(V[g]['read_v4LT_minus_b1fb']['lo'])}, {PS(V[g]['read_v4LT_minus_b1fb']['hi'])}]", "1 safety: read CI lo >= -5 pts (synthetic)"),
        ("Natural pseudo-text: gain over deployed \\method, pp", "CI lo $\\geq -5$", lambda g: f"{PS(V[g]['nat_read_v4LT_minus_B1']['mean'])} [{PS(V[g]['nat_read_v4LT_minus_B1']['lo'])}, {PS(V[g]['nat_read_v4LT_minus_B1']['hi'])}]", "1 natural: read CI lo >= -5 pts vs deployed B1"),
        ("\\multicolumn{6}{l}{\\textit{Temporal stability and locality}}", None),
        ("Edge-normalised jitter, difference to fallback", "CI hi $\\leq 0$", lambda g: f"{S(V[g]['temporal_v4LT_minus_b1fb']['edge_jitter_norm']['mean'], 3)} [{S(V[g]['temporal_v4LT_minus_b1fb']['edge_jitter_norm']['lo'], 3)}, {S(V[g]['temporal_v4LT_minus_b1fb']['edge_jitter_norm']['hi'], 3)}]", "2 jitter_norm CI hi <= 0"),
        ("Edge-normalised flicker, difference to fallback", "CI hi $\\leq 0$", lambda g: f"{S(V[g]['temporal_v4LT_minus_b1fb']['mc_flicker_norm']['mean'], 3)} [{S(V[g]['temporal_v4LT_minus_b1fb']['mc_flicker_norm']['lo'], 3)}, {S(V[g]['temporal_v4LT_minus_b1fb']['mc_flicker_norm']['hi'], 3)}]", "2 flicker_norm CI hi <= 0"),
        ("BG PSNR synthetic / identity re-encoding, dB", "$\\geq$ no-op $-0.3$", lambda g: f"{U(V[g]['bg_synthetic']['means']['v4LT'], 2)} / {U(V[g]['bg_synthetic']['means']['noop'], 2)}", "2 bg synthetic >= noop - 0.3"),
        ("BG PSNR natural / identity re-encoding, dB", "$\\geq$ no-op $-0.3$", lambda g: f"{U(V[g]['bg_natural']['means']['v4LT'], 2)} / {U(V[g]['bg_natural']['means']['noop'], 2)}", "2 bg natural >= noop - 0.3"),
        ("Spill outside the allowed region, \\%", "$\\leq 2$", lambda g: P(V[g]["spill_mean"]), "2 spill <= 2%"),
        ("Frame overflow, \\% (fallback)", "$\\leq$ fallback", lambda g: f"{P(V[g]['overflow_rate']['v4LT'])} ({P(V[g]['overflow_rate']['b1fb'])})", "2 overflow <= b1fb"),
        ("\\multicolumn{6}{l}{\\textit{Tracking and geometry}}", None),
        ("Same rendering, new tracker $-$ A1: raw jitter", "$\\leq +0.2$", lambda g: f"{S(V[g]['ctrl_T_minus_A1']['edge_jitter']['mean'], 2)} [{S(V[g]['ctrl_T_minus_A1']['edge_jitter']['lo'], 2)}, {S(V[g]['ctrl_T_minus_A1']['edge_jitter']['hi'], 2)}]", "3 ctrl jitter (T - A1) <= +0.2"),
        ("Same rendering, new tracker $-$ A1: raw flicker", "$\\leq +0.05$", lambda g: f"{S(V[g]['ctrl_T_minus_A1']['mc_flicker']['mean'], 3)} [{S(V[g]['ctrl_T_minus_A1']['mc_flicker']['lo'], 3)}, {S(V[g]['ctrl_T_minus_A1']['mc_flicker']['hi'], 3)}]", "3 ctrl flicker (T - A1) <= +0.05"),
        ("Centre offset, difference to fallback", "CI hi $< 0$", lambda g: f"{S(V[g]['geometry_v4LT_minus_b1fb']['offset']['mean'], 3)} [{S(V[g]['geometry_v4LT_minus_b1fb']['offset']['lo'], 3)}, {S(V[g]['geometry_v4LT_minus_b1fb']['offset']['hi'], 3)}]", "3 offset CI hi < 0"),
        ("Drift, difference to fallback", "CI hi $< 0$", lambda g: f"{S(V[g]['geometry_v4LT_minus_b1fb']['drift']['mean'], 3)} [{S(V[g]['geometry_v4LT_minus_b1fb']['drift']['lo'], 3)}, {S(V[g]['geometry_v4LT_minus_b1fb']['drift']['hi'], 3)}]", "3 drift CI hi < 0"),
        ("Motion error, difference to fallback", "CI hi $< 0$", lambda g: f"{S(V[g]['geometry_v4LT_minus_b1fb']['motion_err']['mean'], 3)} [{S(V[g]['geometry_v4LT_minus_b1fb']['motion_err']['lo'], 3)}, {S(V[g]['geometry_v4LT_minus_b1fb']['motion_err']['hi'], 3)}]", "3 motion_err CI hi < 0"),
        ("Median text-height ratio", "$[0.8, 1.2]$", lambda g: U(V[g]["height_ratio_median"], 3), "3 height ratio median in [0.8, 1.2]"),
        ("\\multicolumn{6}{l}{\\textit{Containment in the carrier}}", None),
        ("Mean containment (fallback)", "$\\geq 0.90$", lambda g: f"{U(V[g]['containment']['v4LT_mean'], 3)} ({U(V[g]['containment']['b1fb_mean'], 3)})", "4 containment mean >= 0.90"),
        ("Containment, difference to fallback", "CI lo $\\geq -0.03$", lambda g: f"{S(V[g]['containment']['v4LT_minus_b1fb']['mean'], 3)} [{S(V[g]['containment']['v4LT_minus_b1fb']['lo'], 3)}, {S(V[g]['containment']['v4LT_minus_b1fb']['hi'], 3)}]", "4 containment CI lo >= -0.03 vs b1fb"),
        ("Severe-failure audit; layout acceptability", "human", lambda g: f"{V[g]['severe']['any']} automatic candidates", None)]
L = ["% generated by tools/make_numbers_g4.py -- do not edit by hand", "\\begin{table*}[t]", "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\caption{Sealed cross-generator test of the glyph-free branch v4.1: every registered gate, per generator (one run of the frozen code; "
     "prompt-clustered 95\\% bootstrap CIs; differences are v4.1 minus the frozen code's static-hint fallback on the same erased inputs, or minus the "
     "deployed \\method on natural pseudo-text). Synthetic: " + f"{G['Cog']['counts']['synthetic_subset']} (CogVideoX-5B, {G['Cog']['read_v4LT_minus_b1fb']['n_prompts']} prompts) and "
     f"{G['Wan']['counts']['synthetic_subset']} (Wan2.2-TI2V-5B, {G['Wan']['read_v4LT_minus_b1fb']['n_prompts']} prompts) clips. The pooled primary gate (equal-weight mean of the two generators) "
     f"is {PS(PL['read_v4LT_minus_b1fb']['mean'])} points [{PS(PL['read_v4LT_minus_b1fb']['lo'])}, {PS(PL['read_v4LT_minus_b1fb']['hi'])}] (passed). "
     "A1: the trajectory selection of v2/v3. \\ding{55}: failed gate. The two human gates could no longer change the verdict and were replaced by a diagnostic audit (Sec.~\\ref{sec:supp_g4_audit}).}",
     "\\label{tab:g4}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{llrcrc}", "\\toprule",
     "Gate & Threshold & CogVideoX-5B (8\\,fps) & & Wan2.2-TI2V-5B (24\\,fps) & \\\\", "\\midrule"]
for row in rows:
    if row[1] is None: L.append(row[0] + "\\\\"); continue
    lab, thr, val, key = row
    if key is None: L.append(f"{lab} & {thr} & {val('cog')} & -- & {val('wan')} & -- \\\\"); continue
    L.append(f"{lab} & {thr} & {val('cog')} & {mk(gk('cog', key))} & {val('wan')} & {mk(gk('wan', key))} \\\\")
L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
(OUT / "tables" / "tab_g4.tex").write_text("\n".join(L) + "\n")
# rounds table
l3 = None
R = ["% generated by tools/make_numbers_g4.py -- do not edit by hand", "\\begin{table*}[t]", "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\caption{All rounds of the glyph-free branch. Each verdict was taken by criteria fixed before that round's test run and none was revised afterwards. "
     "Readability: branch vs.\\ comparator on the round's test data (first reader). D4 is development data of v4 and v4.1 and is not a test.}",
     "\\label{tab:g_rounds}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{lp{4.6cm}p{4.4cm}p{3.3cm}p{4.4cm}}", "\\toprule",
     "Round & What changed & Test data & Readability & Registered verdict\\\\", "\\midrule",
     "v1 & carrier-only layout, SIFT tracking & \\lthreeALoc{} synthetic glyph-free clips (benchmark); \\lthreeLtwo{} pseudo-text clips & \\lthreeACarRead{} vs.\\ \\lthreeABfbRead{} (fallback); \\lthreeBCarRead{} vs.\\ \\lthreeBBoneRead{} (\\method) & not passed: \\lthreeAPassed{} of \\lthreeACriteria{} criteria met (geometry and stability missed)\\\\",
     f"v2 & motion-model selection; in-carrier layout with a font search & {g2['n_locatable']} synthetic clips (benchmark); {n2['n']} CogVideoX pseudo-text clips & {U(R2['carrier']['read'], 1)} vs.\\ {U(R2['b1fb']['read'], 1)}; {U(N2['carrier']['read'], 1)} vs.\\ {U(N2['B1']['read'], 1)} & not passed: readability, IoU, height ratio {U(R2['carrier']['height_ratio'], 2)}, jitter; the font search chose display and calligraphic faces\\\\",
     f"v3 & v2 tracking; frozen default fonts & {g3['n_locatable']} locatable CogVideoX synthetic clips; the benchmark's {b3['n']} pseudo-text clips & {U(R3['carrier']['read'], 1)} vs.\\ {U(R3['b1fb']['read'], 1)}; {U(B3['carrier']['read'], 1)} vs.\\ {U(B3['B1']['read'], 1)} & not passed: synthetic {sum(1 for k, v in g3['criteria_G'].items() if k != 'all' and v is False)} of {sum(1 for k in g3['criteria_G'] if k != 'all')} criteria failed; pseudo-text only BG ({U(B3['carrier']['bg'], 2)}\\,dB $<$ 41\\,dB)\\\\",
     f"v4 (v4-LT) & dense-flow tracker with keyframe anchors; layout in carrier $\\cap$ allowed region & D4 (development, two generators) & -- & not a test; on D4 two implementation violations of the edit-support constraint were found\\\\",
     f"v4.1 & unified region compositor; inward feather; outside-region invariant & sealed T4: {G['Cog']['counts']['synthetic_subset'] + G['Wan']['counts']['synthetic_subset']} synthetic and {G['Cog']['nat_read_v4LT_minus_B1']['n'] + G['Wan']['nat_read_v4LT_minus_B1']['n']} pseudo-text clips, two generators & pooled {PS(PL['read_v4LT_minus_b1fb']['mean'])} pp [{PS(PL['read_v4LT_minus_b1fb']['lo'])}, {PS(PL['read_v4LT_minus_b1fb']['hi'])}] over the fallback & not passed: {sum(1 for v in booleans.values() if not v)} of {len(booleans)} gates failed (CogVideoX height ratio and normalised jitter; Wan2.2 normalised flicker)\\\\",
     "\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
(OUT / "tables" / "tab_g_rounds.tex").write_text("\n".join(R) + "\n")
print(f"numbers_g4.tex: {len(M)} macros | tables/tab_g4.tex | tables/tab_g_rounds.tex | T4 failed {T['failed']}")
