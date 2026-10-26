"""Relabel the framework figure (user decision 2026-10-27): the material column becomes an EXTERNAL appearance backend
(e.g. SteerVTE; not a Banksy component, used only in the Banksy-V hybrid) plus the proposed constrained fusion (M), drawn as
planned (dashed) -- claimed only if implemented and validated. The trained material generator is a negative result and is no
longer drawn as a pipeline module. A status note marks G as exploratory (it did not pass its sealed test).
Input: the user's original raster (figs/src/fig_overview_framework_user.png); output: figs/fig_overview_framework.png."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent; SRC = HERE / "fig_overview_framework_user.png"; OUT = HERE.parent / "fig_overview_framework.png"
FD = "/System/Library/Fonts/Supplemental/"
K = 4                                                   # supersampling factor for the patch
im = Image.open(SRC).convert("RGB")
X0, Y0, X1, Y1 = 1281, 21, 1436, 566                    # the material column's box in the original raster
W, H = X1 - X0 + 1, Y1 - Y0 + 1
P = Image.new("RGB", (W * K, H * K), (255, 255, 255)); d = ImageDraw.Draw(P)
f = lambda name, px: ImageFont.truetype(FD + name, px * K)
BOLD, REG, IT = "Times New Roman Bold.ttf", "Times New Roman.ttf", "Times New Roman Italic.ttf"
GREY, DGREY, RED, INK = (150, 150, 150), (95, 95, 95), (176, 58, 46), (25, 25, 25)


def rrect(box, fill, outline, dashed=False, r=9, w=1.6):
    x0, y0, x1, y1 = [v * K for v in box]; r *= K; w = int(w * K)
    d.rounded_rectangle((x0, y0, x1, y1), radius=r, fill=fill, outline=None if dashed else outline, width=w)
    if dashed:
        dash, gap = 6 * K, 4 * K
        for (a, b, horiz, fixed) in ((x0 + r, x1 - r, True, y0), (x0 + r, x1 - r, True, y1), (y0 + r, y1 - r, False, x0), (y0 + r, y1 - r, False, x1)):
            t = a
            while t < b:
                e = min(t + dash, b)
                d.line(((t, fixed), (e, fixed)) if horiz else ((fixed, t), (fixed, e)), fill=outline, width=w); t = e + gap
        for (cx, cy, s) in ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90)):
            d.arc((cx - r, cy - r, cx + r, cy + r), s, s + 90, fill=outline, width=w)


def text(x, y, lines, font, fill=INK, lh=1.18):
    for i, s in enumerate(lines):
        d.text((x * K, (y + i * font.size / K * lh) * K), s, font=font, fill=fill)
    return y + len(lines) * font.size / K * lh


# box A: the external backend (grey = not a Banksy component)
rrect((3, 3, W - 4, 268), (243, 243, 243), GREY)
y = text(11, 11, ["Appearance", "Backend"], f(BOLD, 16))
y = text(11, y + 1, ["(e.g., SteerVTE)"], f(IT, 14), DGREY)
y = text(11, y + 9, ["External pretrained", "generative model;", "not a Banksy", "component."], f(REG, 13))
y = text(11, y + 9, ["Used only in the", "Banksy-V hybrid", "(verified selection)."], f(REG, 13))
# arrow A -> B
ax = W // 2; d.line(((ax * K, 270 * K), (ax * K, 289 * K)), fill=DGREY, width=int(1.4 * K))
d.polygon(((ax - 4) * K, 287 * K, (ax + 4) * K, 287 * K, ax * K, 294 * K), fill=DGREY)
# box B: the proposed fusion (dashed = planned)
rrect((3, 297, W - 4, H - 4), (255, 255, 255), DGREY, dashed=True)
y = text(11, 305, ["Constrained", "Fusion (M)"], f(BOLD, 16))
y = text(11, y + 8, ["Banksy fixes content,", "glyphs, geometry and", "edit region; the", "backend supplies", "appearance."], f(REG, 13))
y = text(11, y + 10, ["Planned: claimed only", "if implemented and", "validated."], f(IT, 13), RED)
im.paste(P.resize((W, H), Image.LANCZOS), (X0, Y0))
D = ImageDraw.Draw(im)
D.line(((1270, 323), (1282, 323)), fill=(0, 0, 0), width=2); D.polygon(((1279, 318), (1279, 328), (1287, 323)), fill=(0, 0, 0))   # into box B
D.line(((1433, 323), (1442, 323)), fill=(0, 0, 0), width=2)                                                                         # out of box B
# status note above the G box (white gap between the B1 and G boxes)
note = "G: exploratory (failed its sealed test)"; fn = ImageFont.truetype(FD + IT, 13)
tw = D.textlength(note, font=fn); D.text((1112 - tw, 200), note, font=fn, fill=RED)
im.save(OUT); print("wrote", OUT, im.size)
