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
| sel2 (selective editing B0–B4; 427 clips from `align/sel` + 99 two-row clips from `align/selfix`) | Tab. 5 | `align/sel2/{B0..B4}` (symlinks) | one evaluation run (§4b): reads + temporal + appearance + background on OSC, decisions + collateral on Delta → `align/sel2/ev1/` | final |
| second readers | Sec. 5.1, Tab. readers | reads the al_r1 outputs | InternVL (526 clips, 6 shards), Pixtral (295 English clips, 4 shards), Qwen re-read control (3 shards) on OSC (§4b) → `judge2/` | final |
| VTE-Bench Real, frozen B1 | Sec. 5.8, Tab. external | `bench_b1/vte_real/B1` (8 clips shipped unchanged) | one evaluation run on OSC (§4b) | final |
| VidScribe V2V, frozen B1 | Sec. 5.8, Tab. external | `bench_b1/vs_v2v/B1` (1 clip shipped unchanged) | one evaluation run on OSC (§4b) | final |
| Material generator v0 (research track; `factory_v2_pair.py`, `make_pairs.py`, `matgen_v0.py`, `matgen_apply.py`, `analyze_r2.py`) | Sec. 5.7 (negative result), Fig. matgen | synthetic pairs `factory/{train,val}`; `r2_raw` (frozen B1 RAW caches of the 129 R2 clips); `r2_flat`, `r2_matgen` | training on OSC V100 (20k steps, seed 0); validation `runs/v0/eval_val.json`; one R2 evaluation run `r2_eval/` (Qwen reads, temporal, appearance, background) | R1 not met (0.482 < 0.50), R2 not met; the one registered capacity test `v1_base128` (base width 128, 31.1M params, same data/seed/schedule; jobs 56531351+56543013, 6.1 V100 GPU-h) closes 0.498 on the full validation set, R1 still not met, never applied to real clips; photometric ablation below |
| Photometric ablation of the generator (`matgen_apply.py --photometric`, `matgen_stats.py`, `analyze_r2p.py`, `make_tab_r2p.py`, `r2p.sbatch`; registered 2026-10-23 before the run) | Sec. 5.7 (attribution sentence), suppl. Sec. prereg_matgen, Tab. r2p | `r2p/{flat,p0,p1,p2,p12}` (frozen v0 weights `runs/v0/ckpt_20000.pt`, md5 27e5a655; erase-mask compositing; `r2p/train_stats.json` = training-set mean/std for P1) | one evaluation run on OSC over 8 rows (`r2p/eval/`: Qwen reads in 2 shards merged, temporal, appearance, background; jobs 56547889/56547890/56547893/56547894, 0.5 V100 GPU-h, 0.35 CPU-node-h) | verdict (b): P2/P12 better than P0 on the 50 non-emissive clips, (a) not met, P1 (c), P2 worse than the flat paste |
| Reproducibility audit of the frozen front end (10-23; `xenv_audit.py`, `xenv_extract.py`, `xenv_compare.py`) | suppl. (reproducibility note) | the 71 pseudo-text clips: deployed outputs (Delta, cv2 4.12.0, torch 2.8.0+cu128, A40) vs a same-code re-run (OSC, cv2 4.11.0, torch 2.8.0+cu126, V100); a same-environment repeat of 5 clips | decoded frames identical; erase masks IoU 0.98, layout quads 0.07 px, OCR quads 0 px; glyph alphas IoU 0.72 with the font matcher choosing differently on 17/71 clips and OCR instance decisions differing on 8/71; final frames never bit-identical (PSNR 39.2 overall, 35.4 in the text window, 46.3 outside); the same-environment repeat is bit-exact (IoU 1.0 everywhere) | within-environment deterministic; across OpenCV/CUDA versions readability moves by ~3 points and background PSNR by 1.4 dB on these clips |
| Banksy-V (verified selection; `composite_local.py`, `assemble_hybrid.py`) | Sec. 5.9, Tab. hybrid | `hyb/steer_local`, `hyb/steer2_local` (composites of the SteerVTE seeds) | one evaluation run on OSC over 8 methods (`hyb/`); verifier = InternVL reads of that run, scorer = Qwen + Pixtral | run 2 (`hyb2/`, 8 methods incl. the post-hoc V1.1) is Tab. hybrid; run 1 (`hyb/`) reported in the supplement |
| full8 / sat53 / sat53p | Tab. SAT, Fig. SAT, Sec. 5.4 | `align/full8`, `align/sat53*` | their own jobs | historical, self-contained |
| full6 / full7 / T / D | Tab. render (two tables) | `align/full*`, `repair_outT` | their own jobs | historical, self-contained |
| refiner (fuse) | Tab. refiner, suppl. ladder | `align/full*`, `repair_outT` | their own jobs | historical, self-contained |
| Glyph-free rounds v2 / v3 (registered 2026-10-24; `carrier_v2.py`, `carrier_fix.py --v2`) | suppl. Sec. D (rounds), Tab. g_rounds | `out/l3/` G-syn, G-nat2, G-syn2 and benchmark-L2 trees | one evaluation run per set on OSC Pitzer (PCS0296) → `analysis_g.json`, `analysis_n.json`, `analysis_g2.json`, `analysis_b3.json` | final (both not passed) |
| Glyph-free v4 / v4.1 on D4 (development; `carrier_v4.py`, `edit_support.py`, `compare_v41.py`) | suppl. Sec. D (v4.1) | `out/l3/v4/{cog,wan}_D4` (v4 outputs kept in `v40/`) | `verdict_D4_devcheck_v40.json`, `verdict_D4_devcheck_v41.json`, `{cog,wan}_D4/compare_v41.json` | development evidence only |
| **Glyph-free v4.1 sealed cross-generator test (T4)** — frozen code `code/frozen_v41_20261025` (manifest sha256 `477f1e9a…`); CogVideoX-5B and Wan2.2-TI2V-5B, 480 clips each | Sec. 5.7 paragraph (main8), Sec. 5.7 (long version), suppl. Sec. D, Tab. g4, Tab. g_rounds | `out/l3/v4/{cog,wan}_T4/` | full pipeline once per generator (OSC Pitzer, PCS0296); verdict job 56702391 → `out/l3/v4/verdict_T4.json` | final: FAIL (3 of 34 automatic gates); diagnostic human audit `out/l3/v4/audit_T4_diag/` pending |

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

## 4b. Second environment (second readers, Table 5 re-read, external benchmarks)

The Delta and DeltaAI queues were blocked for days on 10-22, so these evaluations ran on OSC Pitzer (account PCS0296).

| Item | Value |
|---|---|
| Hardware | NVIDIA V100 32 GB (`gpu-exp`), one GPU per job; CPU-only metrics on `cpu-exp` |
| Software | uv venv, Python 3.11.17: torch 2.8.0+cu126, transformers 4.57.6, Pillow 9.5.0, opencv-python-headless 4.12.0.88 (installed without dependencies, running on numpy 1.26.4 as on Delta), numpy 1.26.4, opencc-python-reimplemented 0.1.7, easyocr 1.7.2, scikit-image 0.24.0 |
| Model files | same revisions as Delta; `model.safetensors.index.json` sha256 identical for all three readers |
| Crop equality | the evaluation crops (450 crops: 30 clips × 3 methods × 5 frames, raw and resized) are byte-identical to the first reader's environment (max abs diff 0) |
| Precision | bf16 as in the protocol; V100 has no fused bf16 attention, so PyTorch falls back to the math kernel; batch 1 for InternVL / Pixtral, 8 for Qwen (left padding with attention mask) |
| Sharding | reads split over clips (`SHARD=k NSHARD=n`) on identical GPUs and merged with `tools/merge_shards.py` (md5 9e2dfba8; checks for overlaps and a single reader) |
| Re-read control | the first reader (Qwen) re-read the Table 1 outputs in this environment: 1576 of 1578 clip–method verdicts equal (original and SteerVTE 100%, B1 two clips wrong → right) |
| Evaluator patch | `src/sys_temporal.py` zero-division guard for static clips (frozen repository commit `6bf2a64`, md5 721c40eb; values unchanged wherever the old code returned one) |
| Table 5 split | readability (7 methods, sharded), temporal, appearance and background on OSC, each over all 526 clips in one process; decisions and collateral change computed once on Delta (`align/sel2/ev1_cpu`, CPU-only, deterministic) from the same merged tree |
| Job script | `osc_eval.sbatch` (md5 aef51d4b); OSC usage for this round: 17.6 V100 GPU-hours, 1.3 CPU-node hours |

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
| `judge2/eval_vlm2_internvl_r1.json` | `3753c6627250901d579ca19bc6c563d8e28afdd1b256e6f0c0cccf7419f601af` |
| `judge2/eval_vlm2_pixtral_r1_en.json` | `b9f4557fa3eea287c87a3eed7e83fdf89b91f22d3c4881a72fe43ef91b1d9fca` |
| `judge2/eval_vlm_qwen_osc_r1.json` | `5fa37936e6cd5707f20d941214bb28e2c00fa0e27af0b1f28c35102e63ea6ef7` |
| `align/sel2/ev1/eval_vlm.json` | `5a4974ae960c1852830ac4340e28e3e0b20e98391fe37e4e93d76aa9c40b11f6` |
| `align/sel2/ev1/temporal.json` | `19870687f72277cb96f55bdba6b0d522cbdec404ef9e518e8783e13fb5258f53` |
| `align/sel2/ev1/mat_stats.json` | `56caa263e104ce0201042bcc7ca2d37608890d8b5ecd74ddcf1d7c716a1873c8` |
| `align/sel2/ev1/bg.json` | `de0f80d390b6428dc99e558b7e9e3e111de4bca5ad8cda9308fe92c63055e6fc` |
| `align/sel2/ev1/decisions.json` | `ff85c45aaced19db0035f8e24a68eae730df843c5fde79864ea36f09d2688ea5` |
| `align/sel2/ev1/collateral.json` | `32f4b35a0ae59fad2f000ab516879c466aac2cb18c4d2522a1c0ce103dfcd3ab` |
| `bench_b1/vte_real/eval/eval_vlm.json` | `54e7163714a37c49425f376bd335a93e7a5804ddba19430797969ee10de7b501` |
| `bench_b1/vte_real/eval/eval_vlm2_internvl.json` | `c9732166d7df8cf5fa7549fa4328081c3af44bca58193b4f0585e3f1dd0a56d3` |
| `bench_b1/vte_real/eval/eval_easyocr.json` | `351f56ade783064bd53e802717597cae6e043e78f8383d4eb24ba2b8c2bd4c29` |
| `bench_b1/vte_real/eval/bg.json` | `fb74eeacc6910c5e7cb42e7b5cc61651de4d4563e8008e22859a257f3acfbb73` |
| `bench_b1/vte_real/eval/temporal.json` | `6926644d03219e9da682b6218ead00387180109ed6096c4c1b216bbd6b9b7615` |
| `bench_b1/vte_real/eval/mat_stats.json` | `65565f6380468cd3b7ae69efc62b96162dbdbecf068c1fee3a202fe6e85e72e8` |
| `bench_b1/vs_v2v/eval/eval_vlm.json` | `aee75af1c868433fae9968688404f44bb7e97bcd0bb8b3df80a16fce10448646` |
| `bench_b1/vs_v2v/eval/eval_vlm2_internvl.json` | `28854536c582cc5dccf8e55b224a5b05149215b0db0df9c40626da9f299b2a36` |
| `bench_b1/vs_v2v/eval/eval_easyocr.json` | `93c8229449465be36ad6983da192269550f8bd9e8da42cc1953200c13ed494a5` |
| `bench_b1/vs_v2v/eval/bg.json` | `a5eb118da113d60bfc41053a6136079f491d9db2791956cb2a8012ee565e109f` |
| `bench_b1/vs_v2v/eval/temporal.json` | `e0a349993bbb999c96dc56bbe574480a5774186616686a3d7388f474117e1a57` |
| `bench_b1/vs_v2v/eval/mat_stats.json` | `79f9b9442824fe15ac5aa8952cdd870410d93763492342141fce46c2d5e902a2` |
| `hyb/eval_vlm.json` (Banksy-V run 1) | `478dc475f1618f1f8471daeb19aeb94910551a90e455295582c634cd31ab8faf` |
| `hyb/eval_vlm2_internvl.json` (Banksy-V run 1) | `b1a14396e3ce07464c6f1b74dca81365497461aedfa68851a32ba94146d35af3` |
| `hyb/eval_vlm2_pixtral_en.json` (Banksy-V run 1) | `f67f0f628bb2115d0a85664bd5d538eed71b80e54414b4a29d21fc605f5aa35f` |
| `hyb/mat_stats.json` (Banksy-V run 1) | `f37be2ef2e66fc3a27734af2c39482b0f406b25bf10171e6e0ce489bf97eb8f1` |
| `hyb/bg.json` (Banksy-V run 1) | `e10e8ed1635e2bda6ffd97df78581479f3521f1f21d579c86a5979ed526f15a9` |
| `hyb/temporal.json` (Banksy-V run 1) | `4168025f83d53505cabdf2e752e708e173a3eeca519e3334a69b0f032de3b2c2` |
| `hyb2/eval_vlm.json` (Banksy-V run 2, Tab. hybrid) | `e1a52870006530b7ba462bcc7cb9dc5d349d050843a8b5672054d340d2c01029` |
| `hyb2/eval_vlm2_internvl.json` (Banksy-V run 2, Tab. hybrid) | `7cf3caf87a92c1a0dcf4a3ecc5bec401a2a1ca3ada4204e24a17074e59ca085c` |
| `hyb2/eval_vlm2_pixtral_en.json` (Banksy-V run 2, Tab. hybrid) | `22f09825e41125ea9ea934ba271f883307885896bffcb9529e735db8794b0dfa` |
| `hyb2/mat_stats.json` (Banksy-V run 2, Tab. hybrid) | `6e8867ea6cac03ff08ddda6074de77d1acba08cff639547bfbc59e41ed62b782` |
| `hyb2/bg.json` (Banksy-V run 2, Tab. hybrid) | `6f9c1fa56493193e6d930f051d43467c551616520ce6715110d7523b9869bb75` |
| `hyb2/temporal.json` (Banksy-V run 2, Tab. hybrid) | `9313457582e5a389834751332f1e4739c40c98d5a6d040f922da6d96ba44cc76` |
| `hyb/temporal_reenc.json` (re-encoding diagnostic: SteerVTE, re-encoded SteerVTE, composites, B1; temporal metric only) | `5ae0feecbc36d97e4be76915453f39f81265bb1b57488b01e2f942659960d7b5` |
| `r2_eval/eval_vlm.json` (material generator v0; R2 run / validation / weights) | `b97f8cfe1968d456145b64da3e9bc3b40d04b9ceef229702466866803be0c231` |
| `r2_eval/mat_stats.json` (material generator v0; R2 run / validation / weights) | `6bad2c1dc83c166897d0f4afb3ba5b3f2146a28b0ca766be94a3bbcf3df81ec2` |
| `r2_eval/bg.json` (material generator v0; R2 run / validation / weights) | `8656be4e8cd067aed8c86efa458f70283b63afe0ca63064739cd3f0f286a8f93` |
| `r2_eval/temporal.json` (material generator v0; R2 run / validation / weights) | `88a448ec60895efc78d0eea090e03c5f3e586c67be94f1501691b6702b013d94` |
| `matgen_v0/eval_val.json` (material generator v0; R2 run / validation / weights) | `7d89f2f9be11f975e656c1f70362e3db4c9a193d867128d1fc950fe5fc0b2016` |
| `matgen_v0/ckpt_20000.pt` (material generator v0; R2 run / validation / weights) | `9abb5c9c7c37dcebfcf61c6d15f2b645f1195d31dbe0fee4fbb6abe1272d829e` |
| `r2p/eval/eval_vlm.json` (photometric ablation; one run) | `505267c50c0d01402bf1921ab776fb4618caad149bf63744fbde38f7595bb548` |
| `r2p/eval/mat_stats.json` (photometric ablation; one run) | `2dd47799300fe1a0350886bc8e4183e6b97c643359fe7931ff7c4912f1680bc2` |
| `r2p/eval/bg.json` (photometric ablation; one run) | `78400c3cb6fda3ab4caf8e33bc73ba6d95c3219e9cf1c1dea58763cc12028654` |
| `r2p/eval/temporal.json` (photometric ablation; one run) | `3eb278d0612c004fc8cb3252ac5aa1a117bc4ad044fde22e8f56ee5687c34789` |
| `r2p/train_stats.json` (training-set statistics used by P1) | `65f9be560fb02ac6d76535243d22caa7b12a35c1084916123c87e3290ec431e9` |
| `v1_base128/eval_val.json` (capacity test; validation only) | `0bbbbe15e1cb41248fc6dc72ef601ed96ae2cd8bc950ef53de5ee7273f0f8bb4` |
| `v1_base128/ckpt_20000.pt` (capacity test; validation only) | `43d47b025502d7bb7ba0e14b4fe56e4aaa16f5841a4bcad91dd21e027c1a238f` |

| `out/l3/v4/verdict_T4.json` (sealed test T4) | `0e3519b411205e12b7cd343e74fe0cede3f54ebedfebece24e097c7ff0bb77c2` |
| `out/l3/v4/cog_T4/analysis_v4nat.json` (T4, natural subset) | `0eff6cf7a4c334a75a8d9d205e3d8ba7b2712ebebb7972505242caf42c1bd7ed` |
| `out/l3/v4/wan_T4/analysis_v4nat.json` (T4, natural subset) | `68f40740753aa32645f63a199b202a7162e28a248b2e762ed54c9778231ba03f` |
| `out/l3/v4/verdict_D4_devcheck_v40.json` (D4, development) | `ae063bedfb35f80f9b23ec6043bdb04953a650947a02f8ea38cd1f72a1b06349` |
| `out/l3/v4/verdict_D4_devcheck_v41.json` (D4, development) | `6bfaff7974c1a1dc61c10039487cf82f37b30276a021f8010a115b68e24ac999` |
| `out/l3/v4/cog_D4/compare_v41.json` (D4, v4 vs v4.1) | `8d0d2fc682785e8b0e32aba999ef1fb6c23a1458526f9ede62e3043f714a9a10` |
| `out/l3/v4/wan_D4/compare_v41.json` (D4, v4 vs v4.1) | `c2469ca2e761209d082ff569c9df89c15613f9e3a7496a3c4a680a41690579ec` |
| `analysis_g.json` (v2, G-syn) | `210a702577db5414586d1d04f52309ff332146b0e607d5945082b5c85b7a59d6` |
| `analysis_n.json` (v2, G-nat2) | `7a6867aab216b6fb3ef7e762bf98d3fabb01612d0882d8a8b32b933298b95910` |
| `analysis_g2.json` (v3, G-syn2) | `226b5983e25e68eeb4eb9a89d250963ec5d7e0a93231869413e75991ec05c594` |
| `analysis_b3.json` (v3, benchmark pseudo-text) | `45fcb567fc086fb99810416d0389fcf863f4c9f86a4825e940d95c3b7b18eb1b` |

## 7. How the paper's numbers are produced

`python tools/paper_numbers.py <dir with the six al_r1 JSONs> manifest/dev_selective_40.txt manifest/dev_reuse_9.txt <out>`
writes `numbers.tex` (copied to `numbers_r1.tex`), `tables/tab_main.tex`, `tables/tab_reuse.tex` and
`tables/tab_supp_comp.tex`; `manifest/cache_stats_r1.json` holds the statistics read from the run's caches (font choices,
rotations, gate decisions). Figures: `figs/plots/make_plots.py <eval_vlm.json> figs/plots <bg.json>` and
`make_paper_cells.py` with `--banksy align/r1/B1` on the picks in `figs/picks.json`.

Glyph-free rounds and the sealed test: `python tools/make_numbers_g4.py --t4 verdict_T4.json --t4nat-cog cog_T4/analysis_v4nat.json --t4nat-wan wan_T4/analysis_v4nat.json --d4v40 verdict_D4_devcheck_v40.json --d4v41 verdict_D4_devcheck_v41.json --cmp-cog cog_D4/compare_v41.json --cmp-wan wan_D4/compare_v41.json --v2syn analysis_g.json --v2nat analysis_n.json --v3syn analysis_g2.json --v3l2 analysis_b3.json` writes `numbers_g4.tex`, `tables/tab_g4.tex` and `tables/tab_g_rounds.tex` (the script records the sha256 of every input in the header of `numbers_g4.tex`).
