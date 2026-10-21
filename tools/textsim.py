"""Normalized-edit-distance similarity (traditional->simplified folded), dependency-free."""
try:
    from opencc import OpenCC
    fold = OpenCC("t2s").convert
except Exception:
    fold = lambda s: s


def norm(s):
    return "".join(str(s).split()).lower()


def ned_sim(a, b):
    a, b = norm(fold(a)), norm(fold(b))
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    D = list(range(n + 1))
    for i in range(1, m + 1):
        prev, D[0] = D[0], i
        for j in range(1, n + 1):
            cur = min(D[j] + 1, D[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, D[j] = D[j], cur
    return 1.0 - D[n] / max(m, n)


def reading_order(boxes):
    """boxes: [(pts Nx2 array, text)] -> texts in reading order (rows top-down, left-right)."""
    import numpy as np
    if not boxes: return []
    info = [(float(p[:, 1].mean()), float(p[:, 0].min()), float(p[:, 1].max() - p[:, 1].min()), t) for p, t in boxes]
    tol = 0.5 * float(np.median([i[2] for i in info])) + 1e-6
    info.sort(key=lambda i: i[0]); rows, cur = [], [info[0]]
    for i in info[1:]:
        if abs(i[0] - np.mean([c[0] for c in cur])) <= tol: cur.append(i)
        else: rows.append(cur); cur = [i]
    rows.append(cur)
    return [c[3] for r in rows for c in sorted(r, key=lambda c: c[1])]


def best_read_sim(res, target):
    """Similarity of an OCR result to the target where multi-line / multi-box text counts as one
    reading: max over each single box and over the reading-order concatenation of all boxes."""
    import numpy as np
    boxes = []
    for r in (res or []):
        polys = r.get("rec_polys") or r.get("dt_polys") or []
        for p, t in zip(polys, r.get("rec_texts") or []):
            if str(t).strip(): boxes.append((np.asarray(p, np.float32).reshape(-1, 2), str(t)))
    if not boxes: return 0.0
    cands = [t for _, t in boxes] + [" ".join(reading_order(boxes))]
    return max(ned_sim(c, target) for c in cands)
