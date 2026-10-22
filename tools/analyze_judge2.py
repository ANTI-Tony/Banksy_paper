"""Second-reader analysis (10-22, pre-registered: ideas/2026-10-22_第二读者与外部基准复测_预注册.md). For each reader
(the first reader Qwen from the al_r1 evaluation job, InternVL on all clips, Pixtral on the English clips): readability of
the original, SteerVTE and B1; fixed / broken / net relative to the original as read by the SAME reader; exact McNemar
B1 vs SteerVTE; English / Chinese split; agreement with Qwen per clip and method (share, Cohen's kappa) and the
disagreeing clips. Prints the pre-registered verdict and writes <out>/judge2.json and <out>/numbers_judge2.tex.
Usage: python analyze_judge2.py <qwen eval_vlm.json (A40, al_r1_eval)> <qwen re-read json (second environment)> <internvl json> <pixtral json> <out_dir>
The second readers and the Qwen re-read ran in the same environment (OSC Pitzer V100, decoded crops byte-identical to the
A40 environment); environment effect = Qwen re-read vs Qwen A40, reader effect = second reader vs Qwen re-read."""
import json, math, sys
from pathlib import Path
import numpy as np

Q, QR, I, P, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5]); OUT.mkdir(parents=True, exist_ok=True)
load = lambda p: json.load(open(p))
RD = {"qwen": load(Q), "qwen_rr": load(QR) if Path(QR).exists() else None, "internvl": load(I), "pixtral": load(P) if Path(P).exists() else None}
M3 = ("orig", "steervte", "B1")


def ok(E, m, u): return E["per_clip"][m][u]["fixed"] >= 0.6


def mcnemar(E, a, b, us):
    oa = sum(ok(E, a, u) and not ok(E, b, u) for u in us); ob = sum(ok(E, b, u) and not ok(E, a, u) for u in us); n = oa + ob
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(oa, ob) + 1)) / 2 ** n) if n else 1.0
    return {"only_" + a: oa, "only_" + b: ob, "p": p}


def kappa(x, y):
    x, y = np.asarray(x, int), np.asarray(y, int); po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


R = {"readers": {}}
for name, E in RD.items():
    if E is None: continue
    us = sorted(set.intersection(*[set(E["per_clip"][m]) for m in M3]))
    r = {"n": len(us), "reader": E.get("reader"), "read_errors": E.get("read_errors")}
    for blk, sub in (("all", us), ("en", [u for u in us if u.split("_")[1] == "en"]), ("zh", [u for u in us if u.split("_")[1] == "zh"])):
        if not sub: continue
        b = {"n": len(sub)}
        for m in M3:
            b[m] = 100 * float(np.mean([ok(E, m, u) for u in sub]))
            if m != "orig":
                fx = sum((not ok(E, "orig", u)) and ok(E, m, u) for u in sub); br = sum(ok(E, "orig", u) and not ok(E, m, u) for u in sub)
                b[m + "_fixed"], b[m + "_broken"], b[m + "_net"] = fx, br, fx - br
        b["mcnemar"] = mcnemar(E, "B1", "steervte", sub); r[blk] = b
    for ref in ("qwen", "qwen_rr"):
        Eq = RD.get(ref)
        if name == ref or Eq is None or (name == "qwen" and ref == "qwen_rr"): continue
        agree = {}
        for m in M3:
            common = [u for u in us if u in Eq["per_clip"][m]]
            po, k = kappa([ok(E, m, u) for u in common], [ok(Eq, m, u) for u in common])
            dis = [u for u in common if ok(E, m, u) != ok(Eq, m, u)]
            agree[m] = {"n": len(common), "agreement": po, "kappa": k, "disagree": len(dis),
                        "examples": [(u, E["per_clip"][m][u]["target"], E["per_clip"][m][u]["reads"].get("24", ""), Eq["per_clip"][m][u]["reads"].get("24", ""), int(ok(E, m, u)), int(ok(Eq, m, u))) for u in dis[:12]]}
        r["agreement_with_" + ref] = agree
    R["readers"][name] = r
iv = R["readers"]["internvl"]["all"]
R["verdict"] = {"internvl_b1_minus_steervte": iv["B1"] - iv["steervte"], "p": iv["mcnemar"]["p"],
                "reader_robust": bool(iv["B1"] > iv["steervte"] and iv["mcnemar"]["p"] < 0.05)}
# exploratory (defined after the reads, reported as such): frames where a reader transcribes the target plus extra text, and
# readability when a frame only has to CONTAIN the target (>= 4 of 5 frames)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from textsim import norm, fold
def _contains(E, m, u):
    t = norm(fold(E["per_clip"][m][u]["target"])); return float(np.mean([t in norm(fold(r)) for r in E["per_clip"][m][u]["reads"].values()]) >= 0.75)
def _extra(E, m, u):
    t = norm(fold(E["per_clip"][m][u]["target"])); rs = [norm(fold(r)) for r in E["per_clip"][m][u]["reads"].values()]
    return float(np.mean([(t in r) and len(r) > len(t) for r in rs]))
R["exploratory"] = {}
for name in ("pixtral", "internvl", "qwen"):
    E = RD.get(name)
    if E is None: continue
    us = sorted(set.intersection(*[set(E["per_clip"][m]) for m in M3]))
    if name == "qwen": us = [u for u in us if u.split("_")[1] == "en"] if RD.get("pixtral") is None else sorted(RD["pixtral"]["per_clip"]["B1"])
    R["exploratory"][name] = {m: {"contains": 100 * float(np.mean([_contains(E, m, u) for u in us])), "extra_frames": 100 * float(np.mean([_extra(E, m, u) for u in us])), "n": len(us)} for m in M3}
json.dump(R, open(OUT / "judge2.json", "w"), ensure_ascii=False, indent=1)
f1 = lambda x: f"{x:.1f}"
pexp = lambda p: f"{p:.2f}" if p >= 0.01 else (lambda e: f"{p / 10 ** e:.1f}\\times10^{{{e}}}")(math.floor(math.log10(p)))
mac = {}
for name, tag in (("internvl", "Ivl"), ("pixtral", "Pix"), ("qwen", "Qw"), ("qwen_rr", "Qr")):
    if name not in R["readers"]: continue
    for blk in ("all", "en", "zh"):
        b = R["readers"][name].get(blk)
        if not b: continue
        t = tag + {"all": "", "en": "En", "zh": "Zh"}[blk]
        mac.update({f"j{t}Orig": f1(b["orig"]), f"j{t}Steer": f1(b["steervte"]), f"j{t}Ours": f1(b["B1"]),
                    f"j{t}OursNet": str(b["B1_net"]), f"j{t}SteerNet": str(b["steervte_net"]), f"j{t}OursBroken": str(b["B1_broken"]), f"j{t}SteerBroken": str(b["steervte_broken"]),
                    f"j{t}OnlyOurs": str(b["mcnemar"]["only_B1"]), f"j{t}OnlySteer": str(b["mcnemar"]["only_steervte"]), f"j{t}P": pexp(b["mcnemar"]["p"])})
    for ref, rt in (("qwen", ""), ("qwen_rr", "Rr")):
        if "agreement_with_" + ref in R["readers"][name]:
            for m, mm in (("orig", "Orig"), ("steervte", "Steer"), ("B1", "Ours")):
                a = R["readers"][name]["agreement_with_" + ref][m]
                mac[f"j{tag}Agree{rt}{mm}"] = f"{100 * a['agreement']:.1f}"; mac[f"j{tag}Kappa{rt}{mm}"] = f"{a['kappa']:.2f}"; mac[f"j{tag}Dis{rt}{mm}"] = str(a["disagree"])
if "pixtral" in R["exploratory"]:
    X = R["exploratory"]["pixtral"]
    mac.update({"jPixExtraOrig": f1(X["orig"]["extra_frames"]), "jPixExtraSteer": f1(X["steervte"]["extra_frames"]), "jPixExtraOurs": f1(X["B1"]["extra_frames"]),
                "jPixContSteer": f1(X["steervte"]["contains"]), "jPixContOurs": f1(X["B1"]["contains"])})
with open(OUT / "numbers_judge2.tex", "w") as fh:
    fh.write("% generated by analyze_judge2.py (second readers on the al_r1 outputs); do not edit by hand\n")
    for k, v in mac.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
RL = [("qwen", "Qwen2.5-VL-7B (A40; \\cref{tab:main})", "all"), ("qwen_rr", "Qwen2.5-VL-7B, re-read (V100)", "all"), ("internvl", "InternVL2.5-8B (V100)", "all"),
      ("pixtral", "Pixtral-12B, English only (V100)", "all"), ("qwen", "Qwen2.5-VL-7B (A40), English only", "en")]
TL = ["% generated by analyze_judge2.py", "\\begin{table}[t]", "\\centering",
      "\\caption{\\textbf{Readability under different readers} (the outputs of \\cref{tab:main}; identical crops, frames, prompt, decoding and scoring). "
      "W/L: clips correct only with \\method / only with \\steer, with the exact McNemar $p$. Agree: share of clip--method verdicts equal to the first reader's (\\cref{tab:main}). "
      "The re-read uses the first reader on the second readers' hardware.}", "\\label{tab:readers}", "\\setlength{\\tabcolsep}{2.5pt}", "\\resizebox{\\linewidth}{!}{%",
      "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule", "Reader & $n$ & Original & \\steer & \\method & W/L ($p$) & Agree \\\\", "\\midrule"]
for key, lab, blk in RL:
    r = R["readers"].get(key)
    if not r or blk not in r: continue
    b = r[blk]; mc = b["mcnemar"]; ag = r.get("agreement_with_qwen")
    agr = "--" if key == "qwen" or not ag else f"{100 * np.mean([ag[m]['agreement'] for m in M3]):.1f}"
    TL.append(f"{lab} & {b['n']} & {b['orig']:.1f} & {b['steervte']:.1f} & {b['B1']:.1f} & {mc['only_B1']}/{mc['only_steervte']} (${pexp(mc['p'])}$) & {agr} \\\\")
TL += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
(OUT / "tab_readers.tex").write_text("\n".join(TL) + "\n")
for name, r in R["readers"].items():
    a = r["all"]; print(f"{name:9s} n={r['n']} orig {a['orig']:.1f} steer {a['steervte']:.1f} B1 {a['B1']:.1f} | net steer {a['steervte_net']} B1 {a['B1_net']} | broken steer {a['steervte_broken']} B1 {a['B1_broken']} | McNemar {a['mcnemar']} | errors {r.get('read_errors')}")
    for blk in ("en", "zh"):
        if blk in r: b = r[blk]; print(f"   {blk}: n={b['n']} orig {b['orig']:.1f} steer {b['steervte']:.1f} B1 {b['B1']:.1f} McNemar {b['mcnemar']}")
    for ref in ("qwen", "qwen_rr"):
        if "agreement_with_" + ref in r: print(f"   agreement with {ref}:", {m: (round(v['agreement'], 3), round(v['kappa'], 3), v['disagree']) for m, v in r["agreement_with_" + ref].items()})
print("VERDICT (pre-registered):", R["verdict"])
