# Frozen evaluation manifest

Every number in the paper must trace back to one row of this file. Paths are relative to the project root on the
compute cluster (`$ROOT`); large data are not in this repository. Times are cluster time (US Central).

## 1. Frozen system

| Item | Value |
|---|---|
| Final system | B1 = `inst+lay+track+font+photo+shade+cerase+anchor+gate+sat+noop` (decided 2026-10-22; R1 and B1f frozen as negative results) |
| Code snapshot | `$ROOT/code/frozen_B1_1022` — its own git repository, commit `b6059c8f9e0160a378a8163e32cab5fc4e51ab92` |
| Key files (md5) | `src_a3x/glyphlayer_fix.py` f547675e, `src_a3x/reuse.py` c25eac2b, `src_a3x/realpairs_build.py` 805bd006, `src_a3x/sat.py` 7bc11055, `src_a3x/fix_bench2.py` 7e5642a6 |
| Front-end environment | `GL_FALLBACK=detect FIXB_SLIM=1 HF_HUB_OFFLINE=1`, fonts `GL_FONT_LATIN=DejaVuSans-Bold`, `GL_FONT_CJK=NotoSansSC`, OCR `GL_OCR=easy` |
| Check | every valid slice of the al_r1 pass started after the last code change (10-21 10:37); the 10:33 slice belongs to the discarded run `align/r1_stale_1022` |

## 2. Inputs

| Item | Value |
|---|---|
| Benchmark package | `out/repair_pkg2/infer.csv` (sha256 `7852a68c…`), 526 clips, IDs in `manifest/clips_526.txt` |
| Development clips | selective editing: `manifest/dev_selective_40.txt`; source-glyph reuse: `manifest/dev_reuse_9.txt` |
| SteerVTE outputs | `out/repair_out2/steervte/<uid>.mp4` (released model, hint box as mask, target as text) |
| English subset (second reader, Pixtral) | `lists/judge2_en.csv` (sha256 `5f9103fb…`), 295 clips |
| VTE-Bench Real | `out/vtebench/Real.csv` (sha256 `0f8f0dab…`), 100 clips; SteerVTE outputs `out/vtebench_out/steervte/Real` |
| VidScribe V2V | `bench_ext/vs_v2v_pkg/infer.csv` (sha256 `39738763…`), 44 clips (packing rule in `pkg_vidscribe.py`) |

## 3. Runs used by the paper

| Run | Where the numbers appear | Outputs | Evaluation job | Status |
|---|---|---|---|---|
| **al_r1** (one front-end pass: B1, B1f, R1, R1f) | Tab. 1, Tab. 7, Fig. 1 panel, Fig. 3, Secs. 5.1–5.3, 5.7, 6, suppl. Tab. composition, failure list | `align/r1/{B1,B1f,R1,R1f}/<uid>/fixed.mp4` | `al_r1_eval` (job 23102605, Delta A40) | final |
| sel2 (selective editing B0–B4; 427 clips from `align/sel` + 99 two-row clips from `align/selfix`) | Tab. 5 | `align/sel2/{B0..B4}` (symlinks) | `al_sel2_eval` → `align/sel2/ev1/` | running (one job for all rows) |
| second reader | Sec. 5 / suppl. | reads the al_r1 outputs | `al_judge2` (InternVL, 526 clips), `al_judge2p` (Pixtral, 295 English clips) → `judge2/` | running |
| VTE-Bench Real, frozen B1 | Sec. 5.8 | `bench_b1/vte_real/B1` | `vte_b1_eval` | front end done (92/100 located) |
| VidScribe V2V, frozen B1 | Sec. 5.8 | `bench_b1/vs_v2v/B1` | `vs_b1_eval` | front end done (43/44 located) |
| full8 / sat53 / sat53p | Tab. SAT, Fig. SAT, Sec. 5.4 | `align/full8`, `align/sat53*` | their own jobs | historical, self-contained |
| full6 / full7 / T / D | Tab. render (two tables) | `align/full*`, `repair_outT` | their own jobs | historical, self-contained |
| refiner (fuse) | Tab. refiner, suppl. ladder | `align/full*`, `repair_outT` | their own jobs | historical, self-contained |

Appearance distances are normalised within each evaluation job and are compared only within one table.

## 4. Evaluator

| Item | Value |
|---|---|
| First reader | Qwen/Qwen2.5-VL-7B-Instruct, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`; `src/eval_vlm.py` md5 ceb1f8b2; `src/textsim.py` md5 1c186036 |
| Crop and frames | frame-0 mask box padded by 15% of its height; frames 0, 12, 24, 36, 48 |
| Prompt | `Transcribe all the text visible in this image exactly as written, keeping the original characters (Chinese or English). Output only the text, with no explanation.` |
| Decoding | greedy, max_new_tokens 40, batch 24; processor min_pixels 28·28·16, max_pixels 448·448 |
| Scoring | normalise (remove whitespace, lower-case, OpenCC t2s); frame exact if NED similarity ≥ 0.999; clip correct if ≥ 4 of 5 frames exact |
| Second readers | OpenGVLab/InternVL2_5-8B-MPO-hf, revision `543db189852edd2dbf0c0395c6afe4159cdc842f` (all clips); mistral-experimental/pixtral-12b (HF conversion of Pixtral-12B), revision `c2756cbbb9422eba9f6c5c439a214b0392dfc998` (English only); `src/eval_vlm2.py` md5 26d364b5 (same crop, frames, prompt, decoding, scoring; crops resized with the first reader's pixel budget) |
| Other metrics (md5) | `src/mat_stats.py` 988dffb8, `src/bg_preserve.py` a817dd62, `src/sys_temporal.py` f022743a, `src/eval_bench.py` 5262b45d, `src_a3x/collateral.py` baa7dcf7, `src_a3x/sel_decisions.py` dfee63ae, `src/ship_unchanged.py` 46996cb0 |
| Software | Python venv on Delta x86: transformers 4.57.6, torch 2.8.0+cu128, Pillow 9.5.0, OpenCV 4.12.0, numpy 1.26.4; GPU NVIDIA A40 |
| Known non-determinism | (i) the reader re-reading identical videos in another job: 98/99 identical verdicts; (ii) the appearance estimator uses unseeded k-means++: per-clip test–retest SD 0.47 on 517 byte-identical outputs, mean shift +0.021 [−0.020, +0.062]; (iii) the front end's GPU OCR is not bit-reproducible across passes |

## 5. Seeds

| Use | Seed |
|---|---|
| Paired bootstrap in the paper (`tools/paper_numbers.py`, md5 e51a74eb) | numpy default_rng(0), 10,000 resamples |
| Qualitative figure picks | `random.Random(2027)`, one RNG, materials in figure order, English then Chinese (`figs/picks.json`) |
| Banksy-vs-SteerVTE study sampling and left/right | `random.Random(20261022)` |
| R1 30-clip sample (pre-registered, later contaminated) | 20261022 |

## 6. Result hashes (sha256)

| File | sha256 |
|---|---|
| `align/r1/eval_vlm.json` | `ce9715c0e477e06f4a51222b4228634edc0e4b663b2a4e7ec8ac9bd11622b489` |
| `align/r1/temporal.json` | `34a21f57d9d3908f56a670948413089a7b5c8e0b7724504329bb9acf1f9d0272` |
| `align/r1/mat_stats.json` | `10c93d29cadd2554944b24d8279f881a426b0f732a88b5ebdc3775e3ba4a2365` |
| `align/r1/bg.json` | `5493389e48498b1497f4e13a57bf9eb6087686965c382fe76b6b513131bcb92a` |
| `align/r1/collateral.json` | `1fd8ef657826b6cb1800fc44908f42a466432028b06570477fae433bc3b028b7` |
| `align/r1/decisions.json` | `990a09a548ccfcdec6eb10cdf7d9f35bdcb64b489f05d665517d12d22dc61ba6` |
| second reader, sel2/ev1, external | to be added when the jobs finish |

## 7. How the paper's numbers are produced

`python tools/paper_numbers.py <dir with the six al_r1 JSONs> manifest/dev_selective_40.txt manifest/dev_reuse_9.txt <out>`
writes `numbers.tex` (copied to `numbers_r1.tex`), `tables/tab_main.tex`, `tables/tab_reuse.tex` and
`tables/tab_supp_comp.tex`; `manifest/cache_stats_r1.json` holds the statistics read from the run's caches (font choices,
rotations, gate decisions). Figures: `figs/plots/make_plots.py <eval_vlm.json> figs/plots <bg.json>` and
`make_paper_cells.py` with `--banksy align/r1/B1` on the picks in `figs/picks.json`.
