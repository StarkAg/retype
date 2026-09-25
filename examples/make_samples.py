"""Generate the synthetic sample images used in the README (no real documents involved).

    python3 examples/make_samples.py
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = '/System/Library/Fonts/Supplemental/'


def paper(h, w, rng, tint=(222, 212, 198)):
    base = np.ones((h, w, 3), np.float32) * np.array(tint[::-1], np.float32)
    low = cv2.resize(rng.normal(0, 1, (4, 12)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    fib = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.9)
    return base + (low * 5 + fib * 3.5)[..., None]


def strike(mask, rng, strength, uneven):
    """Per-letter ink with uneven pressure, fuzzy edges and ribbon texture."""
    h, w = mask.shape
    grad = cv2.resize(rng.uniform(1 - uneven, 1, (3, 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    tex = cv2.GaussianBlur(rng.normal(1, 0.25, (h, w)).astype(np.float32), (0, 0), 0.7)
    return cv2.GaussianBlur(mask, (0, 0), 0.9) * grad * np.clip(tex, 0.3, 1.4) * strength


def typeset(text, font, size, h, rng, pitch=None, ink=(70, 60, 62), strength=0.75, uneven=0.45,
            tilt=0.0, wobble=0.0, jitter=1, x0=24):
    f = ImageFont.truetype(font, size)
    w = x0 * 2 + (int(pitch * len(text)) if pitch else int(f.getlength(text) * 1.03))
    img = paper(h, w, rng)
    alpha = np.zeros((h, w), np.float32)
    x = float(x0)
    base = int(h * 0.68)
    for ch in text:
        adv = pitch if pitch else f.getlength(ch)
        if ch != ' ':
            L = Image.new('L', (w, h), 0)
            dy = rng.integers(-jitter, jitter + 1) + (x - x0) * np.tan(np.radians(tilt))
            ImageDraw.Draw(L).text((x + (pitch / 2 if pitch else 0), base + dy), ch, font=f, fill=255,
                                   anchor='ms' if pitch else 'ls')
            m = np.array(L).astype(np.float32) / 255
            if wobble:
                M = cv2.getRotationMatrix2D((x + adv / 2, base), rng.uniform(-wobble, wobble), rng.uniform(0.95, 1.05))
                m = cv2.warpAffine(m, M, (w, h))
            alpha = np.maximum(alpha, strike(m, rng, strength * rng.uniform(0.8, 1.1), uneven))
        x += adv
    ink = np.array(ink[::-1], np.float32)
    out = img * (1 - alpha[..., None]) + ink * alpha[..., None]
    out = cv2.GaussianBlur(out, (0, 0), 0.5)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(3)
    samples = {
        'typewriter.jpg': typeset('RECEIVED WITH THANKS', FONTS + 'Courier New Bold.ttf', 40, 80, rng, pitch=30),
        'printed.jpg': typeset('Certified true copy of the original', FONTS + 'Georgia.ttf', 34, 76, rng,
                               ink=(40, 38, 40), strength=0.85, uneven=0.25, jitter=0),
        'handwriting.jpg': typeset('Balance paid in full', FONTS + 'Bradley Hand Bold.ttf', 44, 96, rng,
                                   ink=(40, 60, 150), strength=0.9, uneven=0.3, tilt=-1.6, wobble=4, jitter=2),
    }
    for name, im in samples.items():
        cv2.imwrite(os.path.join(HERE, name), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print('wrote', name, im.shape[1], 'x', im.shape[0])


if __name__ == '__main__':
    main()
