"""GlyphLayer data factory v2 — COUNTERFACTUAL PAIRS (10-22, research track, pre-registered in
ideas/2026-10-22_材质生成器_预注册.md): one scene, one material, one camera path, TWO strings (A = garbled, B = target).

Blender 4.2 headless:  blender -b -P factory_v2_pair.py -- --out DIR --seed N --text-a "FRESY BREAD" --text-b "FRESH BREAD" ...

Exports, per pair:
  background/%03d.png      scene with the text hidden (shared by A and B)
  A/composite/%03d.png     string A as a lit scene object (full Cycles GI)        B/composite/%03d.png   same for B
  A/textlayer/%03d.png     string A, flat white emission, RGBA (glyph alpha)      B/textlayer/%03d.png
  A/aux/%03d.exr           normal / depth / emission passes (--passes)            B/aux/%03d.exr
  theta.json               per-frame pixel corners of the board and of both text quads
  meta.json                seed, strings, surface, text style, font, camera

Why one session: every random draw that shapes the scene happens before the text is created, and the camera path is
sampled once with the in-frame check applied to BOTH strings, so A and B differ only in the text. The text is fitted to
the board separately for A and B (same rule), as the deployed system fits the target to the carrier.
v2 adds text styles the generator produces and v1 lacked: extruded (3D) letters with real shadows, bevels, metallic
letters, random text colour, random fonts from a library (--font-dir, CJK strings use --cjk-font-dir), a wood board."""
import argparse, json, math, random, sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--text-a", default=""); ap.add_argument("--text-b", default="")
    ap.add_argument("--pair-json", default="", help="JSON with text_a / text_b (and optionally seed, surface): strings with spaces survive the launcher this way")
    ap.add_argument("--frames", type=int, default=9); ap.add_argument("--res", default="832x480"); ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--font-dir", default=""); ap.add_argument("--cjk-font-dir", default=""); ap.add_argument("--font", default="")
    ap.add_argument("--surface", default="random", choices=["random", "matte", "glossy", "neon", "glass", "paper", "screen", "metal", "wood"])
    ap.add_argument("--style", default="random", choices=["random", "flat", "extruded", "bevel", "metallic", "emissive"])
    ap.add_argument("--passes", action="store_true", help="also write normal/depth/emission EXR passes (4 samples)")
    return ap.parse_args(argv)


A = parse()
if A.pair_json:
    _pj = json.load(open(A.pair_json, encoding="utf-8")); A.text_a, A.text_b = _pj["text_a"], _pj["text_b"]
    A.seed = int(_pj.get("seed", A.seed)); A.surface = _pj.get("surface", A.surface) if A.surface == "random" else A.surface
assert A.text_a and A.text_b, "need --text-a/--text-b or --pair-json"
RNG = random.Random(A.seed)
is_cjk = any(ord(c) > 0x2E7F for c in A.text_a + A.text_b)
if A.surface == "random": A.surface = RNG.choice(["matte", "matte", "glossy", "neon", "glass", "paper", "screen", "metal", "wood"])
if A.style == "random":
    A.style = RNG.choice({"neon": ["emissive", "emissive", "flat"], "screen": ["emissive", "flat"], "paper": ["flat", "flat", "bevel"],
                          "metal": ["metallic", "extruded", "bevel"], "wood": ["extruded", "flat", "bevel"]}.get(A.surface, ["flat", "extruded", "bevel", "metallic"]))
W, H = (int(v) for v in A.res.split("x")); OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene; sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = A.samples; sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = W, H; sc.frame_start, sc.frame_end = 1, A.frames
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"; sc.cycles.seed = A.seed


def mat(name, base=(0.8, 0.8, 0.8, 1), rough=0.7, emit=None, metallic=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = base; b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metallic
    if emit is not None: b.inputs["Emission Color"].default_value = base; b.inputs["Emission Strength"].default_value = emit
    return m


# ---------------------------------------------------------------- scene (every draw here is independent of the strings)
world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
sky = world.node_tree.nodes.new("ShaderNodeTexSky"); sky.sky_type = "NISHITA"
sky.sun_elevation = math.radians(RNG.uniform(15, 70)); sky.sun_rotation = math.radians(RNG.uniform(0, 360))
world.node_tree.links.new(sky.outputs["Color"], world.node_tree.nodes["Background"].inputs["Color"])
world.node_tree.nodes["Background"].inputs["Strength"].default_value = RNG.uniform(0.4, 1.2)
bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0)); ground = bpy.context.object
ground.data.materials.append(mat("ground", (RNG.uniform(.2, .5),) * 3 + (1,), 0.9))
bw, bh = RNG.uniform(4.0, 7.0), RNG.uniform(2.2, 3.5)
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, bh / 2 + RNG.uniform(1.0, 2.5))); bb = bpy.context.object
bb.scale = (bw / 2, bh / 2, 1); bb.rotation_euler = (math.radians(90), 0, math.radians(RNG.uniform(-25, 25)))
dark = RNG.uniform(0.02, 0.12); board_dark = True
if A.surface == "matte": bb.data.materials.append(mat("board", (dark, dark, dark * RNG.uniform(1, 2.5), 1), 0.85))
elif A.surface == "glossy": bb.data.materials.append(mat("board", (dark * 1.5, dark * 1.2, dark, 1), 0.12))
elif A.surface == "neon": bb.data.materials.append(mat("board", (0.01, 0.01, 0.015, 1), 0.3))
elif A.surface == "glass": bb.data.materials.append(mat("board", (0.4, 0.42, 0.45, 1), 0.2)); board_dark = False
elif A.surface == "paper": bb.data.materials.append(mat("board", (0.85, 0.82, 0.75, 1), 0.95)); board_dark = False
elif A.surface == "screen": bb.data.materials.append(mat("board", (0.05, 0.08, 0.2, 1), 0.4, emit=1.5))
elif A.surface == "metal": bb.data.materials.append(mat("board", (0.55, 0.55, 0.58, 1), RNG.uniform(0.2, 0.5), metallic=1.0)); board_dark = RNG.random() < 0.5
elif A.surface == "wood":
    m = bpy.data.materials.new("board"); m.use_nodes = True; nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    noise = nt.nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = RNG.uniform(3, 8); noise.inputs["Detail"].default_value = 6
    ramp = nt.nodes.new("ShaderNodeValToRGB"); ramp.color_ramp.elements[0].color = (0.25, 0.14, 0.06, 1); ramp.color_ramp.elements[1].color = (0.55, 0.35, 0.16, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"]); nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"]); b.inputs["Roughness"].default_value = 0.6
    bb.data.materials.append(m)
bpy.context.view_layer.update()
if A.surface == "glass":
    bpy.ops.mesh.primitive_plane_add(size=1, location=bb.location); pane = bpy.context.object
    pane.scale = (bw / 2 * 1.05, bh / 2 * 1.05, 1); pane.rotation_euler = bb.rotation_euler
    pane.location = bb.location + bb.matrix_world.to_3x3() @ Vector((0, 0, 0.08))
    gm = bpy.data.materials.new("glass"); gm.use_nodes = True; gb = gm.node_tree.nodes["Principled BSDF"]
    gb.inputs["Transmission Weight"].default_value = 1.0; gb.inputs["Roughness"].default_value = 0.02; gb.inputs["IOR"].default_value = 1.45
    pane.data.materials.append(gm); pane.name = "glasspane"
if A.surface == "neon":
    world.node_tree.nodes["Background"].inputs["Strength"].default_value *= 0.08
    bpy.ops.object.light_add(type="POINT", location=(bb.location.x, bb.location.y - 3, bb.location.z + 1))
    bpy.context.object.data.energy = RNG.uniform(50, 200); bpy.context.object.data.color = (0.9, 0.6, 1.0)
for i in range(RNG.randint(2, 4)):
    kind = RNG.choice(["cube", "uv_sphere", "cylinder"])
    getattr(bpy.ops.mesh, f"primitive_{kind}_add")(location=(RNG.uniform(-10, 10), RNG.uniform(3, 12), RNG.uniform(0.5, 1.5)))
    o = bpy.context.object; o.scale = (RNG.uniform(.4, 1.4),) * 3
    o.data.materials.append(mat(f"p{i}", (RNG.random(), RNG.random(), RNG.random(), 1), RNG.uniform(.2, .9)))

# text material and style (drawn once, shared by A and B)
palette_light = [(1, 1, 1, 1), (0.95, 0.9, 0.6, 1), (0.6, 0.9, 1.0, 1), (1.0, 0.6, 0.5, 1), (0.8, 1.0, 0.7, 1)]
palette_dark = [(0.08, 0.08, 0.1, 1), (0.5, 0.05, 0.05, 1), (0.05, 0.15, 0.45, 1), (0.1, 0.3, 0.1, 1), (0.3, 0.1, 0.3, 1)]
if A.style == "emissive":
    col = RNG.choice([(1.0, 0.2, 0.6, 1), (0.2, 0.6, 1.0, 1), (1.0, 0.5, 0.1, 1), (0.3, 1.0, 0.5, 1), (1, 1, 1, 1)]); tmat = mat("textmat", col, 0.5, emit=RNG.uniform(3, 10))
elif A.style == "metallic":
    col = RNG.choice([(0.9, 0.75, 0.3, 1), (0.85, 0.87, 0.9, 1), (0.8, 0.5, 0.3, 1)]); tmat = mat("textmat", col, RNG.uniform(0.15, 0.45), metallic=1.0)
else:
    col = RNG.choice(palette_light if board_dark else palette_dark); tmat = mat("textmat", col, RNG.uniform(0.3, 0.9))
extrude = RNG.uniform(0.03, 0.10) if A.style in ("extruded", "metallic") else (RNG.uniform(0.01, 0.03) if A.style == "bevel" else 0.0)
bevel = RNG.uniform(0.004, 0.012) if A.style in ("bevel", "metallic") else 0.0

# font (one draw; CJK strings take a CJK font)
font_path = A.font
if not font_path:
    fdir = A.cjk_font_dir if is_cjk else A.font_dir
    cands = sorted([str(p) for p in Path(fdir).glob("*") if p.suffix.lower() in (".ttf", ".otf")]) if fdir else []
    if cands: font_path = RNG.choice(cands)
if not font_path:
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf"):
        if Path(cand).exists(): font_path = cand; break
fc = bpy.data.curves.new("txt", type="FONT"); fc.body = A.text_a
if font_path:
    try: fc.font = bpy.data.fonts.load(font_path)
    except Exception as e: print("font load failed, default font:", font_path, e, flush=True); font_path = "blender-default"
fc.align_x = "CENTER"; fc.align_y = "CENTER"; fc.extrude = extrude; fc.bevel_depth = bevel; fc.bevel_resolution = 2
txt = bpy.data.objects.new("text", fc); sc.collection.objects.link(txt); txt.parent = bb; txt.data.materials.append(tmat)


def fit_text(body):
    """same fitting rule for both strings: fill 86% of the board width or 60% of its height, whichever binds"""
    fc.body = body; txt.scale = (1, 1, 1); bpy.context.view_layer.update()
    tb = txt.dimensions; fit = min((2 * 0.86) / max(tb.x, 1e-6), (2 * 0.6) / max(tb.y, 1e-6))
    txt.scale = (fit, fit, 1); txt.location = (0, 0, 0.01); bpy.context.view_layer.update()
    return txt.dimensions.x / 2, txt.dimensions.y / 2, fit


half = {s: fit_text(t) for s, t in (("A", A.text_a), ("B", A.text_b))}

# camera: track-to, keyframed dolly/orbit; the path must keep BOTH text quads inside the frame on every frame
cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam); sc.camera = cam
tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt); tgt.location = bb.location
con = cam.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"


def quads_inside(margin=0.04):
    for s in ("A", "B"):
        tx_, ty_ = half[s][0], half[s][1]; pts = [(-tx_, -ty_, 0.01), (tx_, -ty_, 0.01), (tx_, ty_, 0.01), (-tx_, ty_, 0.01)]
        for f in range(1, A.frames + 1):
            sc.frame_set(f); bpy.context.view_layer.update()
            for p in pts:
                v = world_to_camera_view(sc, cam, bb.matrix_world @ Vector(p))
                if not (margin < v.x < 1 - margin and margin < v.y < 1 - margin and v.z > 0): return False
    return True


for attempt in range(40):
    r0 = RNG.uniform(9, 15); ang0 = math.radians(RNG.uniform(-40, 40)); r1 = r0 * RNG.uniform(0.55, 0.9); ang1 = ang0 + math.radians(RNG.uniform(-25, 25))
    z0, z1 = RNG.uniform(1.2, 3.0), RNG.uniform(1.2, 3.0); cam.animation_data_clear()
    for f, (r, ang, z) in ((1, (r0, ang0, z0)), (A.frames, (r1, ang1, z1))):
        cam.location = (r * math.sin(ang), -r * math.cos(ang), z); cam.keyframe_insert("location", frame=f)
    if quads_inside(): break
else:
    print("WARN: no in-frame camera path found after 40 tries", flush=True)


def corners_px(local_pts, frame):
    sc.frame_set(frame); bpy.context.view_layer.update(); out = []
    for p in local_pts:
        v = world_to_camera_view(sc, cam, bb.matrix_world @ Vector(p)); out.append([round(v.x * W, 2), round((1 - v.y) * H, 2)])
    return out


theta = {"board": {}, "text_A": {}, "text_B": {}}
for f in range(1, A.frames + 1):
    theta["board"][f] = corners_px([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], f)
    for s in ("A", "B"):
        tx, ty = half[s][0], half[s][1]; theta[f"text_{s}"][f] = corners_px([(-tx, -ty, 0.01), (tx, -ty, 0.01), (tx, ty, 0.01), (-tx, ty, 0.01)], f)
(OUT / "theta.json").write_text(json.dumps(theta))


# ---------------------------------------------------------------- renders
def set_flat(on):
    b = tmat.node_tree.nodes["Principled BSDF"]
    if on:
        b.inputs["Emission Color"].default_value = (1, 1, 1, 1); b.inputs["Emission Strength"].default_value = 1.0; b.inputs["Base Color"].default_value = (0, 0, 0, 1); b.inputs["Metallic"].default_value = 0.0
    else:
        b.inputs["Base Color"].default_value = col; b.inputs["Metallic"].default_value = 1.0 if A.style == "metallic" else 0.0
        if A.style == "emissive": b.inputs["Emission Color"].default_value = col
        else: b.inputs["Emission Strength"].default_value = 0.0


def render_frames(d):
    d.mkdir(parents=True, exist_ok=True)
    for f in range(1, A.frames + 1):
        sc.frame_set(f); sc.render.filepath = str(d / f"{f:03d}"); bpy.ops.render.render(write_still=True)


def render_pass(d, hide_text=False, flat_text=False, transparent=False):
    txt.hide_render = hide_text; ground.hide_render = transparent; bb.hide_render = transparent
    for o in sc.objects:
        if o.type == "MESH" and o not in (ground,) and o.name != "text": o.hide_render = transparent
    set_flat(flat_text); sc.render.film_transparent = transparent
    sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"; sc.cycles.samples = A.samples
    render_frames(d); set_flat(False)


def render_aux(d):
    vl = sc.view_layers[0]; vl.use_pass_normal = True; vl.use_pass_z = True; vl.use_pass_emit = True
    txt.hide_render = False; ground.hide_render = False; bb.hide_render = False
    for o in sc.objects:
        if o.type == "MESH": o.hide_render = False
    sc.render.film_transparent = False; sc.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"; sc.render.image_settings.color_depth = "16"; sc.cycles.samples = 4
    render_frames(d)
    sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"; sc.cycles.samples = A.samples
    vl.use_pass_normal = False; vl.use_pass_z = False; vl.use_pass_emit = False


render_pass(OUT / "background", hide_text=True)
for s, body in (("A", A.text_a), ("B", A.text_b)):
    fit_text(body)
    render_pass(OUT / s / "composite")
    if A.passes: render_aux(OUT / s / "aux")
    render_pass(OUT / s / "textlayer", flat_text=True, transparent=True)

(OUT / "meta.json").write_text(json.dumps({
    "seed": A.seed, "text_a": A.text_a, "text_b": A.text_b, "surface": A.surface, "style": A.style, "extrude": extrude, "bevel": bevel, "text_colour": list(col),
    "font": font_path or "blender-default", "res": [W, H], "frames": A.frames, "samples": A.samples, "board_wh": [bw, bh],
    "board_rot_z_deg": math.degrees(bb.rotation_euler[2]), "sun_elev_deg": math.degrees(sky.sun_elevation), "fit": {s: half[s][2] for s in half},
    "cam": {"r0": r0, "r1": r1, "z0": z0, "z1": z1}, "cam_attempt": attempt}, ensure_ascii=False))
print("FACTORY PAIR DONE", OUT, flush=True)
