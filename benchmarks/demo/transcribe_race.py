#!/usr/bin/env python3
r"""transcribe_race - the moss-transcribe.cpp vs PyTorch TRANSCRIBE RACE.

Two engines transcribe the SAME real audio (jfk.wav) side by side: the compact
timestamped, speaker-labelled transcript types out in each pane while a progress
bar fills at the REAL measured proc-time. moss-transcribe.cpp (ggml CPU) finishes
first, both land on the byte-identical transcript, then a LocalAI end card carries
the honest wins: 1.6 to 1.8x faster than PyTorch on CPU, bit-exact, lighter on RAM,
zero Python.

Same house treatment as recorder-for-agents/examples/voice_race: pure Pillow frame
synthesis + ffmpeg palettegen/paletteuse, headless (no docker, no display). All the
numbers come from spec.json (real measured means, not invented). --dilate only sets
playback speed. No em-dashes anywhere.

  python3 transcribe_race.py --spec ./spec.json --fixtures <moss>/tests/fixtures --out ./out

Renders out/transcribe_race.{mp4,gif} (16:9) + out/transcribe_race_square.{mp4,gif} (1:1).
"""
import argparse, json, os, subprocess, tempfile, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
LOGO_PATH = HERE.parent.parent / "assets" / "localai_logo.png"  # repo assets/ copy

BG, PANEL, INK, DIM = (13, 17, 23), (22, 28, 36), (215, 221, 229), (110, 118, 129)
DIMMER = (96, 105, 117)
GRID = (34, 43, 52)
TEAL = (62, 200, 224)
SLATE = (150, 165, 180)
GREEN = (102, 214, 130)
GOLD = (240, 200, 90)
FPS = 20


def fontp(bold):
    return f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"


def fontm(bold):
    return f"/usr/share/fonts/truetype/dejavu/DejaVuSansMono{'-Bold' if bold else ''}.ttf"


def font(sz, bold=True, mono=False):
    try:
        return ImageFont.truetype((fontm if mono else fontp)(bold), sz)
    except Exception:
        return ImageFont.load_default()


def load_wave_env(path, nbins):
    w = wave.open(str(path), "rb")
    n = w.getnframes()
    raw = w.readframes(n)
    w.close()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    edges = np.linspace(0, len(a), nbins + 1).astype(int)
    env = np.array([np.max(np.abs(a[edges[i]:edges[i + 1]])) if edges[i + 1] > edges[i] else 0.0
                    for i in range(nbins)])
    m = env.max() or 1.0
    return env / m


def draw_wave(d, rect, env, color, reveal=1.0):
    x, y, w, h = rect
    mid = y + h // 2
    n = len(env)
    bw = w / n
    cut = int(n * reveal)
    for i in range(n):
        bx = x + i * bw
        amp = env[i] * (h * 0.46)
        c = color if i < cut else GRID
        d.line([(bx, mid - amp), (bx, mid + amp)], fill=c, width=max(1, int(bw) - 1))


def rounded(d, rect, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(rect, r, fill=fill, outline=outline, width=width)


def chunkify(t):
    """Split the raw transcript into atomic display chunks with a kind:
    [1.23] time markers, [S01] speaker tags, and plain word runs."""
    chunks = []
    i = 0
    while i < len(t):
        c = t[i]
        if c == "[":
            j = t.find("]", i)
            if j < 0:
                j = len(t) - 1
            seg = t[i:j + 1]
            kind = "time" if (len(seg) > 1 and seg[1].isdigit()) else "spk"
            chunks.append((seg, kind))
            i = j + 1
        elif c == " ":
            i += 1
        else:
            j = i
            while j < len(t) and t[j] not in "[ ":
                j += 1
            chunks.append((t[i:j], "text"))
            i = j
    return chunks


def layout_transcript(d, chunks, fnt, x0, y0, w, line_h):
    """Greedy word-wrap of the chunks (single space between), returning per-chunk
    (text, kind, x, y, start_char) plus the total revealable char count and the y
    past the block. Wrapping is fixed up front so typing does not reflow."""
    space_w = d.textlength(" ", font=fnt)
    x, y = x0, y0
    items = []
    pos = 0
    for seg, kind in chunks:
        sw = d.textlength(seg, font=fnt)
        if x > x0 and x + sw > x0 + w:
            x = x0
            y += line_h
        items.append((seg, kind, x, y, pos))
        x += sw + space_w
        pos += len(seg) + 1
    return items, pos, y + line_h


TCOLOR = {"time": SLATE, "spk": GOLD, "text": INK}


def draw_transcript(d, items, reveal_chars, fnt):
    last = None
    for seg, kind, x, y, start in items:
        if reveal_chars <= start:
            break
        show = min(len(seg), reveal_chars - start)
        d.text((x, y), seg[:show], fill=TCOLOR[kind], font=fnt)
        cw = d.textlength(seg[:show], font=fnt)
        last = (x + cw, y)
    if last is not None and reveal_chars < 10_000:
        cx, cy = last
        d.rectangle([cx + 2, cy + 2, cx + 4, cy + 18], fill=TEAL)  # caret


def header(cv, W, spec, wave_env):
    d = ImageDraw.Draw(cv)
    fh = font(26)
    ft = font(15, False)
    d.text((40, 22), "moss-transcribe.cpp", fill=TEAL, font=fh)
    x = 40 + d.textlength("moss-transcribe.cpp", font=fh)
    d.text((x + 14, 28), "vs", fill=DIM, font=ft)
    d.text((x + 40, 22), "PyTorch", fill=INK, font=fh)
    note = f"0.9B  ·  {spec['threads']} threads  ·  {spec['dtype']}  ·  bit-exact"
    d.text((W - 40 - d.textlength(note, font=ft), 28), note, fill=DIM, font=ft)
    d.line([40, 60, W - 40, 60], fill=GRID, width=1)
    # shared input strip: one audio, transcribed by both
    fq = font(15, False)
    lab = f"transcribe   {spec['audio']}   ({spec['audio_s']:.0f}s)"
    d.text((40, 74), lab, fill=DIM, font=fq)
    wx = 40 + d.textlength(lab, font=fq) + 20
    draw_wave(d, (wx, 72, W - 40 - wx, 24), wave_env, TEAL, reveal=1.0)


def pane(cv, rect, spec, eng, frac, tfnt):
    d = ImageDraw.Draw(cv)
    ox, oy, pw, ph = rect
    accent = eng["accent"]
    done = frac >= 1.0
    rounded(d, [ox, oy, ox + pw, oy + ph], 12, fill=PANEL,
            outline=(accent if eng["winner"] and done else GRID),
            width=2 if eng["winner"] and done else 1)
    pad = 20
    ix = ox + pad
    iw = pw - 2 * pad
    d.text((ix, oy + 14), eng["label"], fill=accent, font=font(20))
    d.text((ix, oy + 40), eng["device"], fill=DIM, font=font(14, False))
    # transcript typing area
    ty = oy + 74
    items, total, _ = layout_transcript(d, eng["_chunks"], tfnt, ix, ty, iw, 26)
    draw_transcript(d, items, int(total * min(1.0, frac * 1.02)), tfnt)
    # progress bar near the bottom
    pby = oy + ph - 58
    rounded(d, [ix, pby, ix + iw, pby + 9], 4, fill=(34, 41, 50))
    rounded(d, [ix, pby, ix + int(iw * min(1.0, frac)), pby + 9], 4, fill=accent)
    sy = pby + 20
    if done:
        tline = f"transcribed in {eng['proc_s']:.1f} s   (RTF {eng['rtf']:.2f})"
        d.text((ix, sy), tline, fill=accent, font=font(17))
        if eng["winner"]:
            badge = f"{spec['speedup']:.2f}x faster"
            bw = d.textlength(badge, font=font(15)) + 34
            bx = ix + iw - bw
            rounded(d, [bx, sy - 2, bx + bw, sy + 24], 6, outline=accent, width=2)
            d.text((bx + 12, sy + 1), "✔", fill=accent, font=font(15))
            d.text((bx + 30, sy + 1), badge, fill=accent, font=font(15))
    else:
        d.text((ix, sy), f"decoding  {min(frac * eng['proc_s'], eng['proc_s']):.1f} s",
               fill=accent, font=font(17))


def race_frame(W, H, spec, engines, w_elapsed, dilate, wave_env):
    cv = Image.new("RGB", (W, H), BG)
    header(cv, W, spec, wave_env)
    tfnt = font(17, False, mono=True)
    top = 128
    gap = 24
    pw = (W - 80 - gap) // 2
    ph = 430
    t_real = w_elapsed / dilate
    rects = [(40, top, pw, ph), (40 + pw + gap, top, pw, ph)]
    states = []
    for r, e in zip(rects, engines):
        frac = min(1.0, t_real / e["proc_s"])
        pane(cv, r, spec, e, frac, tfnt)
        states.append(frac >= 1.0)
    d = ImageDraw.Draw(cv)
    fy = top + ph + 22
    if all(states):
        foot = (f"identical transcript, timestamps and speaker tags  ·  "
                f"moss-transcribe.cpp is {spec['speedup']:.2f}x faster on CPU and uses ~{spec['ram_ratio']:.1f}x less RAM")
        d.text(((W - d.textlength(foot, font=font(16, False))) // 2, fy), foot, fill=GREEN, font=font(16, False))
    else:
        foot = "same audio, two engines  ·  moss-transcribe.cpp runs the whole model in C++/ggml, no Python"
        d.text(((W - d.textlength(foot, font=font(16, False))) // 2, fy), foot, fill=DIM, font=font(16, False))
    brandline(cv, W, H)
    return cv


def race_frame_square(W, H, spec, engines, w_elapsed, dilate, wave_env):
    cv = Image.new("RGB", (W, H), BG)
    header(cv, W, spec, wave_env)
    tfnt = font(18, False, mono=True)
    top = 118
    gap = 22
    reserve = 96
    ph = (H - top - reserve - gap) // 2
    pw = W - 80
    t_real = w_elapsed / dilate
    rects = [(40, top, pw, ph), (40, top + ph + gap, pw, ph)]
    states = []
    for r, e in zip(rects, engines):
        frac = min(1.0, t_real / e["proc_s"])
        pane(cv, r, spec, e, frac, tfnt)
        states.append(frac >= 1.0)
    d = ImageDraw.Draw(cv)
    fy = top + 2 * ph + gap + 14
    if all(states):
        foot = f"identical transcript  ·  {spec['speedup']:.2f}x faster on CPU  ·  ~{spec['ram_ratio']:.1f}x less RAM"
        d.text(((W - d.textlength(foot, font=font(16, False))) // 2, fy), foot, fill=GREEN, font=font(16, False))
    else:
        foot = "same audio, two engines  ·  full model in C++/ggml, no Python"
        d.text(((W - d.textlength(foot, font=font(16, False))) // 2, fy), foot, fill=DIM, font=font(16, False))
    brandline(cv, W, H)
    return cv


def brandline(cv, W, H, margin=40):
    d = ImageDraw.Draw(cv)
    fb = font(13, False)
    y = H - 28
    d.text((margin + 16, y), "github.com/mudler/moss-transcribe.cpp", font=fb, fill=DIMMER)
    t = "Brought to you by the LocalAI team  ·  localai.io"
    d.text((W - margin - 16 - d.textlength(t, font=fb), y), t, font=fb, fill=DIMMER)


def bars(d, rows, bx0, bw_full, y, fmt, bar_h=16, row_gap=18):
    vmax = max(v for _, v, _ in rows)
    lab_font, val_font = font(16), font(16)
    step = bar_h + row_gap
    for i, (lab, v, c) in enumerate(rows):
        ry = y + i * step
        ty = ry + (bar_h - 17) // 2
        d.text((bx0 - d.textlength(lab, font=lab_font) - 16, ty), lab, fill=c, font=lab_font)
        rounded(d, [bx0, ry, bx0 + bw_full, ry + bar_h], 4, fill=(28, 35, 44))
        fillw = max(4, int(bw_full * v / vmax))
        rounded(d, [bx0, ry, bx0 + fillw, ry + bar_h], 4, fill=c)
        d.text((bx0 + bw_full + 16, ty), fmt(v), fill=INK, font=val_font)
    return y + len(rows) * step - row_gap


def end_card(W, H, spec, square=False):
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    cx = W // 2
    logo = Image.open(LOGO_PATH).convert("RGBA")
    ls = 176 if not square else 190
    logo = logo.resize((ls, ls), Image.LANCZOS)
    ly = int(H * (0.05 if not square else 0.10))
    cv.paste(logo, ((W - ls) // 2, ly), logo)
    y = ly + ls + 6
    team = "from the LocalAI team  ·  localai.io"
    d.text((cx - d.textlength(team, font=font(18, False)) / 2, y), team, fill=DIM, font=font(18, False))
    y += 40
    big = font(38)
    head = f"{spec['speedup_range']} faster than PyTorch on CPU"
    if d.textlength(head, font=big) > W - 80:
        big = font(32)
    d.text((cx - d.textlength(head, font=big) / 2, y), head, fill=TEAL, font=big)
    y += 52
    sub = "byte-identical transcript  ·  timestamps + speaker diarization  ·  one C++ binary, zero Python"
    fsub = font(17, False)
    if d.textlength(sub, font=fsub) > W - 60:
        fsub = font(15, False)
    d.text((cx - d.textlength(sub, font=fsub) / 2, y), sub, fill=INK, font=fsub)
    y += 48
    bx0 = int(W * (0.40 if not square else 0.42))
    bw_full = int(W * (0.26 if not square else 0.30))
    d.text((bx0, y - 20), f"inference, 11s audio, {spec['threads']} threads, {spec['dtype']}, lower is better", fill=DIM, font=font(14, False))
    y = bars(d, [("moss-transcribe.cpp", spec["ggml_infer_s"], TEAL),
                 ("PyTorch", spec["torch_infer_s"], SLATE)],
             bx0, bw_full, y, lambda v: f"{v:.1f} s", bar_h=15, row_gap=16) + 34
    d.text((bx0, y - 20), "peak RAM, one transcribe, lower is better", fill=DIM, font=font(14, False))
    y = bars(d, [("moss-transcribe.cpp", spec["ram_ggml_mb"] / 1024.0, TEAL),
                 ("PyTorch", spec["ram_torch_mb"] / 1024.0, SLATE)],
             bx0, bw_full, y, lambda v: f"{v:.1f} GB", bar_h=18, row_gap=14) + 12
    callout = f"~{spec['ram_ratio']:.1f}x less RAM  ·  bit-exact (cosine 1.000)"
    d.text((cx - d.textlength(callout, font=font(22)) / 2, y), callout, fill=GOLD, font=font(22))
    y += 44
    fl = font(17)
    gapx = 56

    def row(pair, ry):
        widths = [d.textlength(t, font=fl) for t in pair]
        total = sum(widths) + gapx * (len(pair) - 1)
        x = (W - total) // 2
        for t, w in zip(pair, widths):
            d.text((x, ry), t, fill=TEAL, font=fl)
            x += w + gapx

    row(["localai.io", "github.com/mudler/LocalAI"], int(H * (0.86 if not square else 0.88)))
    d.text((cx - d.textlength("github.com/mudler/moss-transcribe.cpp", font=font(16, False)) / 2,
            int(H * (0.91 if not square else 0.93))),
           "github.com/mudler/moss-transcribe.cpp", fill=DIMMER, font=font(16, False))
    return cv


def render(out, W, H, spec, engines, dilate, wall, frame_fn, card_fn, wave_env,
           fps, gif_fps, stills, stills_dir):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        k = 0
        for i in range(int(wall * fps) + 1):
            frame_fn(W, H, spec, engines, i / fps, dilate, wave_env).save(tmp / f"f{k:05d}.png")
            k += 1
        hold = frame_fn(W, H, spec, engines, wall * 1.5, dilate, wave_env)
        for _ in range(int(2.4 * fps)):
            hold.save(tmp / f"f{k:05d}.png")
            k += 1
        card = card_fn(W, H, spec)
        for _ in range(int(3.6 * fps)):
            card.save(tmp / f"f{k:05d}.png")
            k += 1
        if stills:
            sd = Path(stills_dir)
            sd.mkdir(parents=True, exist_ok=True)
            hold.save(sd / f"{out.stem}_still.png")
            card.save(sd / f"{out.stem}_endcard.png")
            frame_fn(W, H, spec, engines, wall * 0.55, dilate, wave_env).save(sd / f"{out.stem}_midrace.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(tmp / "f%05d.png"), "-pix_fmt", "yuv420p", str(out)], check=True)
        pal = tmp / "pal.png"
        gw = 900
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out),
                        "-vf", f"fps={gif_fps},scale={gw}:-1:flags=lanczos,palettegen=stats_mode=diff", str(pal)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-i", str(pal),
                        "-lavfi", f"fps={gif_fps},scale={gw}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                        str(out.with_suffix(".gif"))], check=True)
    print("wrote", out, "+ gif")


def main():
    global LOGO_PATH
    ap = argparse.ArgumentParser(description="moss-transcribe.cpp vs PyTorch transcribe race")
    ap.add_argument("--spec", default=str(HERE / "spec.json"))
    ap.add_argument("--fixtures", default=str(Path.home() / "_git" / "moss-transcribe.cpp" / "tests" / "fixtures"))
    ap.add_argument("--logo", default=str(LOGO_PATH))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--gif-fps", type=int, default=14)
    ap.add_argument("--dilate", type=float, default=0.0, help="playback factor; 0 = auto ~11 s")
    ap.add_argument("--stills", action="store_true")
    a = ap.parse_args()

    LOGO_PATH = Path(a.logo)
    spec = json.loads(Path(a.spec).read_text())
    wave_env = load_wave_env(Path(a.fixtures) / "short.wav", 180)
    chunks = chunkify(spec["transcript"])
    engines = [
        {"label": "moss-transcribe.cpp", "device": "ggml CPU, 8 threads", "accent": TEAL,
         "proc_s": spec["ggml_infer_s"], "rtf": spec["ggml_rtf"], "winner": True, "_chunks": chunks},
        {"label": "PyTorch", "device": "torch CPU, 8 threads", "accent": SLATE,
         "proc_s": spec["torch_infer_s"], "rtf": spec["torch_rtf"], "winner": False, "_chunks": chunks},
    ]
    proc_max = max(e["proc_s"] for e in engines)
    dilate = a.dilate if a.dilate > 0 else max(0.4, 11.0 / proc_max)
    wall = proc_max * dilate
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    render(outdir / "transcribe_race.mp4", 1280, 720, spec, engines, dilate, wall,
           race_frame, lambda W, H, s: end_card(W, H, s, False), wave_env,
           a.fps, a.gif_fps, a.stills, outdir)
    render(outdir / "transcribe_race_square.mp4", 1080, 1080, spec, engines, dilate, wall,
           race_frame_square, lambda W, H, s: end_card(W, H, s, True), wave_env,
           a.fps, a.gif_fps, a.stills, outdir)


if __name__ == "__main__":
    main()
