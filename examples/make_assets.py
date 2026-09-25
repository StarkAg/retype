"""Build the README demo GIF and the GitHub social-preview image from the example outputs.

    python3 examples/make_assets.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, 'examples')
OUT = os.path.join(ROOT, 'assets')
PAIRS = [('typewriter.jpg', 'typewriter_fit_width.png', 'Typewriter'),
         ('printed.jpg', 'printed_same_spacing.png', 'Printed font'),
         ('handwriting.jpg', 'handwriting_same_spacing.png', 'Pen / handwriting')]
BG, FG, MUTED, ACCENT = (250, 248, 244), (28, 28, 30), (110, 108, 104), (214, 72, 44)


def font(size, bold=False):
    for p in ['/System/Library/Fonts/Supplemental/Avenir Next.ttc', '/System/Library/Fonts/Helvetica.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size, index=2 if bold and p.endswith('Avenir Next.ttc') else 0)
            except OSError:
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load(name, sub=''):
    return Image.open(os.path.join(EX, sub, name)).convert('RGB')


def fit(im, w):
    return im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)


def demo_gif():
    W, row_w, pad = 900, 760, 22
    rows = [(fit(load(a), row_w), fit(load(b, 'output'), row_w), label) for a, b, label in PAIRS]
    for i, (a, b, l) in enumerate(rows):      # same size for crossfade
        h = max(a.height, b.height)
        rows[i] = (a.crop((0, 0, row_w, h)), b.crop((0, 0, row_w, h)), l)
    H = pad + sum(r[0].height + 44 for r in rows) + pad
    f_lab, f_tag = font(20, True), font(18, True)
    frames, durs = [], []

    def frame(t, tag):
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        y = pad
        for a, b, label in rows:
            d.text(((W - row_w) // 2, y), label, font=f_lab, fill=MUTED)
            d.text(((W + row_w) // 2, y), tag, font=f_tag, fill=ACCENT if tag == 'AFTER' else MUTED, anchor='ra')
            y += 30
            im.paste(Image.blend(a, b, t), ((W - row_w) // 2, y))
            y += a.height + 14
        return im

    for t, tag, ms in [(0, 'BEFORE', 1600)] + [(i / 6, '', 60) for i in range(1, 6)] + \
                      [(1, 'AFTER', 2200)] + [(1 - i / 6, '', 60) for i in range(1, 6)]:
        frames.append(frame(t, tag)); durs.append(ms)
    frames[0].save(os.path.join(OUT, 'demo.gif'), save_all=True, append_images=frames[1:],
                   duration=durs, loop=0, optimize=True)


def social():
    W, H = 1280, 640
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((64, 52), 'retype', font=font(84, True), fill=FG)
    d.text((64, 160), "Change text in scans & photos using the document's own letters",
           font=font(30), fill=MUTED)
    y = 232
    for a, b, label in PAIRS:
        A, B = fit(load(a), 560), fit(load(b, 'output'), 560)
        im.paste(A, (64, y)); im.paste(B, (656, y))
        y += max(A.height, B.height) + 18
    d.text((64, y + 4), 'BEFORE', font=font(22, True), fill=MUTED)
    d.text((656, y + 4), 'AFTER', font=font(22, True), fill=ACCENT)
    d.text((W - 64, H - 48), 'typewriter · printed · handwriting  —  no GPU',
           font=font(22), fill=MUTED, anchor='ra')
    im.save(os.path.join(OUT, 'social-preview.png'))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    demo_gif(); social()
    print('wrote assets/demo.gif, assets/social-preview.png')
