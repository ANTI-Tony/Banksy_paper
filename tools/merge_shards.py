"""Merge sharded eval_vlm.py / eval_vlm2.py outputs (SHARD=k NSHARD=n over the same csv, same environment and GPU type)
into one JSON in the scripts' own format; the summary is recomputed exactly as the scripts do (sim = mean of per-clip sims,
fixed = share of clips read exactly on >= 75% of frames). Fails if shards overlap or a shard is missing.
Usage: python merge_shards.py <out.json> <shard0.json> [<shard1.json> ...]"""
import json, sys
import numpy as np
OUT, parts = sys.argv[1], sys.argv[2:]
res, readers, errs = {}, set(), 0
for p in parts:
    d = json.load(open(p)); readers.add(d.get("reader")); errs += int(d.get("read_errors") or 0)
    for m, D in d["per_clip"].items():
        R = res.setdefault(m, {})
        dup = set(R) & set(D)
        if dup: sys.exit(f"overlap in {m}: {sorted(dup)[:5]}")
        R.update(D)
assert len(readers) == 1, f"shards read with different readers: {readers}"
summ = {m: {"n": len(D), "sim": float(np.mean([v["sim"] for v in D.values()])), "fixed": float(np.mean([v["fixed"] for v in D.values()]))} for m, D in res.items()}
json.dump({"summary": summ, "per_clip": res, "read_errors": errs, "reader": readers.pop(), "shards": len(parts)}, open(OUT, "w"), ensure_ascii=False, indent=1)
print("MERGED", OUT, json.dumps(summ))
