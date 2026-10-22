"""Assemble the hybrid system (Banksy-V, 10-22, pre-registered: ideas/2026-10-22_混合系统_预注册.md) from the per-clip metrics
of ONE evaluation run over {orig, steervte, steervte_s2, steer_local, steer2_local, B1}, and check the registered criteria.
Decision (V1, no tunable threshold): preservation gate kept the clip -> original; verifier (InternVL) reads SteerVTE seed 1
-> its local composite; else seed 2 -> its composite; else B1. Scored by the first reader (Qwen) and, on the English clips,
by Pixtral; the verifier never scores. Also reports the whole-frame variant and the seed-1-only variant, the verifier's
false accepts, and an ORACLE row (selector = scorer; circular, upper bound only).
Usage: python assemble_hybrid.py <run_dir> <decisions.json of the B1 run> <out_dir>
  run_dir: eval_vlm.json (Qwen), eval_vlm2_internvl.json, eval_vlm2_pixtral_en.json, mat_stats.json, bg.json, temporal.json"""
import json, math, sys
from pathlib import Path
import numpy as np

D, OUT = Path(sys.argv[1]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
Q = json.load(open(D / "eval_vlm.json"))["per_clip"]; V = json.load(open(D / "eval_vlm2_internvl.json"))["per_clip"]
P = json.load(open(D / "eval_vlm2_pixtral_en.json"))["per_clip"] if (D / "eval_vlm2_pixtral_en.json").exists() else None
M = json.load(open(D / "mat_stats.json"))["per_clip"]; B = json.load(open(D / "bg.json"))["per_clip"]; T = json.load(open(D / "temporal.json"))["per_clip"]
DEC = json.load(open(sys.argv[2])); U = sorted(Q["orig"]); rng = np.random.default_rng(0)
ok = lambda E, m, u: E[m][u]["fixed"] >= 0.6
noedit = lambda u: bool((DEC[u]["select"]["B1"] or {}).get("noedit"))
LOCAL = {"steervte": "steer_local", "steervte_s2": "steer2_local"}
LOCAL_C = {"steervte": "steer_local_c", "steervte_s2": "steer2_local_c"}        # V1.1: per-clip (temporally constant) colour matching
HAS_C = all(k in Q for k in LOCAL_C.values())


def policy(u, verifier, local=True, seeds=("steervte", "steervte_s2"), table=None):
    table = table or LOCAL
    if noedit(u): return "orig"
    for s in seeds:
        if u in verifier[s] and ok(verifier, s, u): return table[s] if local else s
    return "B1"


def mcnemar(ch_a, ch_b, E, us):
    a = sum(ok(E, ch_a[u], u) and not ok(E, ch_b[u], u) for u in us); b = sum(ok(E, ch_b[u], u) and not ok(E, ch_a[u], u) for u in us); n = a + b
    return {"only_a": a, "only_b": b, "p": (min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n) if n else 1.0)}


def score(ch, E=Q, us=None):
    us = us or [u for u in U if all(u in E.get(ch[v], {}) for v in [u])]
    r = {"n": len(us), "read": 100 * float(np.mean([ok(E, ch[u], u) for u in us]))}
    r["fixed"] = sum((not ok(E, "orig", u)) and ok(E, ch[u], u) for u in us); r["broken"] = sum(ok(E, "orig", u) and not ok(E, ch[u], u) for u in us); r["net"] = r["fixed"] - r["broken"]
    mv = [M[ch[u]][u]["mad"] for u in us if ch[u] != "orig" and u in M.get(ch[u], {})]; bv = [B[ch[u]][u]["psnr_bg"] for u in us if ch[u] != "orig" and u in B.get(ch[u], {})]
    r["mad"] = float(np.mean(mv)) if mv else float("nan"); r["bg"] = float(np.mean(bv)) if bv else float("nan")
    r["flicker"] = float(np.mean([T[ch[u]][u]["mc_flicker"] for u in us])); r["jitter"] = float(np.mean([T[ch[u]][u]["edge_jitter"] for u in us]))
    r["chosen"] = {k: int(sum(ch[u] == k for u in us)) for k in sorted(set(ch[u] for u in us))}
    return r


# B1 and SteerVTE as evaluated in THIS run (for a like-for-like comparison), then the systems
CH = {"B1": {u: "B1" for u in U}, "steervte": {u: "steervte" for u in U},
      "V1 (verified selection, local composite)": {u: policy(u, V) for u in U},
      "V1 whole-frame (no compositing)": {u: policy(u, V, local=False) for u in U},
      "V1 seed 1 only": {u: policy(u, V, seeds=("steervte",)) for u in U},
      "ORACLE (selector = scorer; circular)": {u: policy(u, Q) for u in U}}
if HAS_C: CH["V1.1 (per-clip colour matching; post hoc)"] = {u: policy(u, V, table=LOCAL_C) for u in U}
R = {"systems": {}, "criteria": {}}
for name, ch in CH.items():
    r = score(ch); r["mcnemar_vs_B1"] = mcnemar(ch, CH["B1"], Q, U); r["mcnemar_vs_steervte"] = mcnemar(ch, CH["steervte"], Q, U)
    if P:
        en = [u for u in U if u in P["B1"] and all(u in P.get(ch[v], {}) for v in [u])]
        r["pixtral_read_en"] = 100 * float(np.mean([ok(P, ch[u], u) for u in en])); r["pixtral_n"] = len(en)
    R["systems"][name] = r
v1 = CH["V1 (verified selection, local composite)"]
fa = [u for u in U if v1[u].startswith("steer") and not ok(Q, v1[u], u)]
R["verifier_false_accepts"] = {"n": len(fa), "B1_would_be_right": int(sum(ok(Q, "B1", u) for u in fa)), "clips": fa}
R["local_vs_whole"] = {s: {"read_same": 100 * float(np.mean([ok(Q, s, u) == ok(Q, LOCAL[s], u) for u in U])),
                           "bg_whole": float(np.mean([B[s][u]["psnr_bg"] for u in U if u in B[s]])), "bg_local": float(np.mean([B[LOCAL[s]][u]["psnr_bg"] for u in U if u in B[LOCAL[s]]]))} for s in LOCAL}
b1, s1 = R["systems"]["B1"], R["systems"]["V1 (verified selection, local composite)"]
pix_ok = (s1.get("pixtral_read_en", 0) >= R["systems"]["B1"].get("pixtral_read_en", 0) - 1.0) if P else None
R["criteria"] = {"1_read_gain>=2_and_p<0.05": bool(s1["read"] >= b1["read"] + 2.0 and s1["mcnemar_vs_B1"]["p"] < 0.05), "2_broken<=B1": bool(s1["broken"] <= b1["broken"]),
                 "3_bg>=41": bool(s1["bg"] >= 41.0), "4_mad<=B1-0.30": bool(s1["mad"] <= b1["mad"] - 0.30), "5_flicker<=1.85_jitter<=8.40": bool(s1["flicker"] <= 1.85 and s1["jitter"] <= 8.40),
                 "6_pixtral_en>=B1-1": pix_ok}
R["criteria"]["all"] = all(v for v in R["criteria"].values() if v is not None)
if HAS_C:
    sc = R["systems"]["V1.1 (per-clip colour matching; post hoc)"]; pix_c = (sc.get("pixtral_read_en", 0) >= b1.get("pixtral_read_en", 0) - 1.0) if P else None
    R["criteria_v11"] = {"1_read_gain>=2_and_p<0.05": bool(sc["read"] >= b1["read"] + 2.0 and sc["mcnemar_vs_B1"]["p"] < 0.05), "2_broken<=B1": bool(sc["broken"] <= b1["broken"]), "3_bg>=41": bool(sc["bg"] >= 41.0),
                         "4_mad<=B1-0.30": bool(sc["mad"] <= b1["mad"] - 0.30), "5_flicker<=1.85_jitter<=8.40": bool(sc["flicker"] <= 1.85 and sc["jitter"] <= 8.40), "6_pixtral_en>=B1-1": pix_c}
    R["criteria_v11"]["all"] = all(v for v in R["criteria_v11"].values() if v is not None)
json.dump(R, open(OUT / "hybrid.json", "w"), ensure_ascii=False, indent=1)
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"
pexp = lambda p: f"{p:.2f}" if p >= 0.01 else (lambda e: f"{p / 10 ** e:.1f}\\times10^{{{e}}}")(math.floor(math.log10(p)))
rows = [("\\method (B1)", "B1"), ("\\steer", "steervte"), ("Banksy-V: verified selection, local composite", "V1 (verified selection, local composite)"),
        ("\\quad whole-frame instead of local composite", "V1 whole-frame (no compositing)"), ("\\quad seed 1 only", "V1 seed 1 only")]
if HAS_C: rows.append(("\\quad per-clip colour matching (V1.1, defined after V1)", "V1.1 (per-clip colour matching; post hoc)"))
rows.append(("Oracle selection (selector = reader; upper bound)", "ORACLE (selector = scorer; circular)"))
L = ["% generated by assemble_hybrid.py (one evaluation run; verifier InternVL, scorer Qwen)", "\\begin{table*}[t]", "\\centering",
     "\\caption{\\textbf{Verified selection between \\method and \\steer} (526 clips; one evaluation run for every row, so \\method and \\steer are re-evaluated here). "
     "Banksy-V keeps the clip when the preservation gate fires, otherwise takes a \\steer output (seed 1, then seed 2) whose text an independent verifier (InternVL2.5-8B, the protocol of \\cref{sec:metrics}) reads as the target, composited back into the original frame around the text box, and falls back to \\method otherwise. "
     "Scored by the first reader; the verifier never scores. W/L vs.\\ \\method: clips correct only with the system / only with \\method (McNemar $p$). Pixtral: readability on the 295 English clips under the third reader.}",
     "\\label{tab:hybrid}", "\\setlength{\\tabcolsep}{4pt}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{@{}lcccccccccc@{}}", "\\toprule",
     " & Read.\\ $\\uparrow$ & Fixed & Broken $\\downarrow$ & Net $\\uparrow$ & W/L vs.\\ \\method ($p$) & Appear.\\ $\\downarrow$ & BG PSNR $\\uparrow$ & Flicker $\\downarrow$ & Jitter $\\downarrow$ & Pixtral $\\uparrow$ \\\\", "\\midrule"]
for lab, key in rows:
    r = R["systems"][key]; mc = r["mcnemar_vs_B1"]; wl = "--" if key == "B1" else f"{mc['only_a']}/{mc['only_b']} (${pexp(mc['p'])}$)"
    L.append(f"{lab} & {f1(r['read'])} & {r['fixed']} & {r['broken']} & {r['net']} & {wl} & {f2(r['mad'])} & {f2(r['bg'])} & {f2(r['flicker'])} & {f2(r['jitter'])} & {f1(r['pixtral_read_en']) if 'pixtral_read_en' in r else '--'} \\\\")
L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
(OUT / "tab_hybrid.tex").write_text("\n".join(L) + "\n")
mac = {}
for key, tag in [("B1", "Bone"), ("steervte", "Steer"), ("V1 (verified selection, local composite)", "V"), ("V1 whole-frame (no compositing)", "VWhole"), ("V1 seed 1 only", "VOne"), ("ORACLE (selector = scorer; circular)", "Oracle")] + ([("V1.1 (per-clip colour matching; post hoc)", "VC")] if HAS_C else []):
    r = R["systems"][key]; mc = r["mcnemar_vs_B1"]
    mac.update({f"h{tag}Read": f1(r["read"]), f"h{tag}Fixed": str(r["fixed"]), f"h{tag}Broken": str(r["broken"]), f"h{tag}Net": str(r["net"]), f"h{tag}Mad": f2(r["mad"]), f"h{tag}BG": f2(r["bg"]),
                f"h{tag}Flicker": f2(r["flicker"]), f"h{tag}Jitter": f2(r["jitter"]), f"h{tag}OnlySys": str(mc["only_a"]), f"h{tag}OnlyBone": str(mc["only_b"]), f"h{tag}P": pexp(mc["p"])})
    if "pixtral_read_en" in r: mac[f"h{tag}Pix"] = f1(r["pixtral_read_en"])
    for k, v in r["chosen"].items(): mac[f"h{tag}Chosen{k.replace('_', '').replace('1', 'One').replace('2', 'Two')}"] = str(v)
mac.update({"hFalseAccepts": str(R["verifier_false_accepts"]["n"]), "hFalseAcceptsBone": str(R["verifier_false_accepts"]["B1_would_be_right"])})
with open(OUT / "numbers_hybrid.tex", "w") as fh:
    fh.write("% generated by assemble_hybrid.py; do not edit by hand\n")
    for k, v in mac.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for name, r in R["systems"].items():
    print(f"{name:46s} read {r['read']:.1f} fixed {r['fixed']} broken {r['broken']} net {r['net']} | mad {r['mad']:.2f} bg {r['bg']:.2f} fl {r['flicker']:.2f} jt {r['jitter']:.2f} | vsB1 {r['mcnemar_vs_B1']['only_a']}/{r['mcnemar_vs_B1']['only_b']} p={r['mcnemar_vs_B1']['p']:.2e} | pix {r.get('pixtral_read_en', float('nan')):.1f} | {r['chosen']}")
print("false accepts:", R["verifier_false_accepts"]["n"], "| local vs whole:", R["local_vs_whole"]); print("CRITERIA V1:", R["criteria"])
if HAS_C: print("CRITERIA V1.1:", R["criteria_v11"])
