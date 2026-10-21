# Banksy: Evidence-Gated Repair of Garbled Text in AI-Generated Videos

CVPR 2027 submission draft, written with the official CVPR author kit (`cvpr.sty`, `ieeenat_fullname.bst`).
The page limit is ignored for now: the main text and the supplement (appended after the references) contain
everything that was run.

## Build

```
latexmk main.tex        # pdflatex + bibtex; Chinese strings use CJKutf8 (gbsn), rare glyphs are images in figs/cjk/
```

## Layout

| path | content |
|---|---|
| `main.tex`, `preamble.tex` | document and macros (`\method`, `\steer`, `\zh{}`, `\TODO{}`, `\pending{}`) |
| `sec/` | sections: 0 abstract, 1 intro, 2 related work, 3 task/benchmark/metrics, 4 method, 5 experiments (5b front end, 5c SAT, 5d selective editing, 5e learned refiner, 5f external benchmarks), 6 analysis and limitations, 7 conclusion, X supplement |
| `tables/` | one file per table; numbers copied from the evaluation outputs of the final 526-clip run unless the caption says otherwise |
| `figs/fig_*.tex` | figure layouts (teaser, overview in TikZ, qualitative grid, edit locality, SAT, failures) |
| `figs/cells/` | photo cells exported at native crop resolution; labels and reader transcriptions are typeset by LaTeX |
| `figs/plots/` | vector charts and the script that draws them (`make_plots.py`) |
| `figs/picks.json`, `figs/cells_meta.json` | which clips appear in each figure and why (the qualitative grid is a fixed-seed random sample) |

## Open items (marked in red in the PDF)

All 49 references were verified against arXiv / proceedings / publisher pages (October 2026).


- 99 two-row clips are being re-run after a compositing fix; Tables 1 and 5, the failure counts and Fig. 4 will be updated.
- Exact rule used to select the 526 benchmark clips; runtime of the final pipeline.
- Selective-editing human study (88 pairs) is still collecting answers.
- External benchmarks (VTE-Bench, VidScribe) were run with the earlier box-policy front end.
