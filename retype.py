#!/usr/bin/env python3
"""retype - replace a line of text in a photo or scan with new text, built from the
document's own letters, so the result matches the original ink, font and paper.

Works on typewriter text, printed text in any font (monospaced or proportional),
hand-printed pen writing, stamps - dark ink on lighter paper, one line at a time.

    python3 retype.py IMAGE --old "RECEIVED WITH THANKS" --new "RECEIVED WITH REGARDS"

Pipeline
  1. Ink map     paper brightness is estimated with a large median filter; ink = how much
                 darker each pixel is than the paper around it (per colour channel, so blue
                 or red pen keeps its colour).
  2. Straighten  a tilted line is detected from the letter positions and levelled; only
                 changed pixels are rotated back at the end.
  3. Letters     strong-ink columns are grouped into one blob per non-space character of
                 --old (merged/split until the count matches). If the blob centres sit on a
                 regular grid the text is treated as monospaced (typewriter), otherwise as
                 proportional (letter gaps and word gaps are measured instead).
  4. Library     each letter is cut out at the middle of the gaps around it with a soft edge,
                 so faint edges and joining strokes survive.
  5. Paper       the old text is inpainted and real paper grain is laid back on top.
  6. Compose     runs of 2+ letters that already exist in --old are copied as one piece
                 (keeps the real spacing and pen joins); other letters come from the library.
                 Missing letters, in order of preference:
                   flip    mirror of another letter (b/d/p/q, n/u; M/W for monospace)
                   strokes built from real stems and bars of the source (H I L T E F)
                   font    drawn with a fallback font (--font), matched to letter height,
                           stroke weight, blur and ink colour. Always check these by eye.
  7. Finish      ink is evened out, smooth ribbon/pen variation and +-1px baseline jitter
                 are added, and a light sharpen matches the source crispness.

Outputs (in --out-dir, prefixed by --name):
  <name>_same_spacing.png/.jpg  original spacing (image grows to the right if the text is longer)
  <name>_fit_width.png/.jpg     squeezed to the original width (only when the text is longer)
  <name>_comparison.png         original + results stacked, 2x
  <name>_zoom.png               original vs result, 4x, for checking letter quality
  <name>_report.json            layout, measurements and where every output letter came from
With --region the line is cut from a larger page and the result is pasted back into a
full-page copy (<name>_page_same_spacing.png, <name>_page_fit_width.png).
"""
import argparse, json, math, os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FLIPS = {  # missing char: (source char, axis)  v = upside down, h = mirrored
    'M': ('W', 'v'), 'W': ('M', 'v'),
    'b': ('d', 'h'), 'd': ('b', 'h'), 'p': ('q', 'h'), 'q': ('p', 'h'),
    'n': ('u', 'v'), 'u': ('n', 'v'),
}
MONO_ONLY_FLIPS = set('MW')
STEM_SOURCES = 'LEFBDPRKHNTI'
BOTTOM_BAR_SOURCES = 'LEZ'
TOP_BAR_SOURCES = 'TEFZ'
DESCENDERS = set('gjpqyQ,;')
NARROW = set('IJijlft1.,:;!|\'"-')
MONO_FONTS = ['/System/Library/Fonts/Supplemental/Andale Mono.ttf',
              '/System/Library/Fonts/Supplemental/Courier New.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
              'C:/Windows/Fonts/cour.ttf']
PROP_FONTS = ['/System/Library/Fonts/Supplemental/Arial.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              'C:/Windows/Fonts/arial.ttf']


def odd(n):
    n = int(n)
    return n if n % 2 else n + 1


def shift(a, dy=0, dx=0):
    return np.roll(np.roll(a, int(dy), 0), int(dx), 1)


def runs_of(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append([s, i - 1]); s = None
    if s is not None:
        out.append([s, len(mask) - 1])
    return out


class Retype:
    def __init__(self, crop, old, line_w, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.src = crop.astype(np.float32)
        self.H, self.W = self.src.shape[:2]
        self.line_w = line_w
        self.old = old
        self.log = []
        self.angle = 0.0
        self.img = self.src
        self._ink_map(); self._find_letters()
        if not args.no_deskew and len(self.chars) >= 4:
            ang = self._tilt()
            if abs(ang) > 0.5:
                self.angle = ang
                self.img = self._rotate(self.src, ang)
                self._ink_map(); self._find_letters()
                self.log.append(f'line was tilted {ang:.2f} deg; straightened while editing')
        self._paper()
        self._library()

    # ------------------------------------------------------------------ geometry
    def _rotate(self, im, ang):
        m = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), ang, 1)
        return cv2.warpAffine(im, m, (im.shape[1], im.shape[0]), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _tilt(self):
        """Slope of the letters' box centres (Theil-Sen, so odd letters don't skew it)."""
        b0, b1 = self.band
        pts = []
        for a, b in self.runs:
            rows = np.where((self.d_line[b0:b1, a:b + 1] > self.strong).any(1))[0]
            if len(rows):
                pts.append(((a + b) / 2, b0 + (rows.min() + rows.max()) / 2))
        if len(pts) < 4:
            return 0.0
        slopes = [(y2 - y1) / (x2 - x1) for i, (x1, y1) in enumerate(pts)
                  for x2, y2 in pts[i + 1:] if x2 - x1 > 2 * self.cap]
        return math.degrees(math.atan(np.median(slopes))) if slopes else 0.0

    def unlevel(self, result, base):
        """Rotate only the changed pixels back into the original (tilted) image."""
        if self.angle == 0:
            return result
        lev = self._rotate(base.astype(np.float32), self.angle)
        diff = np.abs(result.astype(np.float32) - lev).mean(2) > 2
        m = cv2.dilate(diff.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), 1.5)
        back = self._rotate(result.astype(np.float32), -self.angle)
        mb = np.clip(self._rotate(m, -self.angle), 0, 1)[..., None]
        return np.clip(base * (1 - mb) + back * mb, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ analysis
    def _ink_map(self):
        g = self.img.mean(2)
        bgl = cv2.medianBlur(g.astype(np.uint8), 31).astype(np.float32)
        d = cv2.GaussianBlur(bgl - g, (0, 0), 0.7)
        d[:, self.line_w:] = 0
        strong = max(12.0, 0.35 * np.percentile(d, 99.5))
        rows = runs_of((d > strong).sum(1) > 0)
        if not rows:
            sys.exit('No text found in the image/region.')
        merged = [rows[0]]
        for r in rows[1:]:
            if r[0] - merged[-1][1] <= max(3, 0.25 * (merged[-1][1] - merged[-1][0])):
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        top, bot = max(merged, key=lambda r: (d[r[0]:r[1] + 1] > strong).sum())
        if len(merged) > 1:
            self.log.append(f'{len(merged)} ink bands found, using rows {top}-{bot}; '
                            'pass --region if that is the wrong line')
        cap = bot - top + 1
        bgl = cv2.medianBlur(g.astype(np.uint8), min(odd(max(31, 1.6 * cap)), 255)).astype(np.float32)
        self.d = cv2.GaussianBlur(bgl - g, (0, 0), 0.7)
        self.d_line = self.d.copy(); self.d_line[:, self.line_w:] = 0
        self.strong = strong
        self.top, self.bot, self.cap = top, bot, cap
        self.band = (max(0, top - int(0.35 * cap)), min(self.H, bot + 1 + int(0.35 * cap)))

    def _find_letters(self):
        chars = [(i, c) for i, c in enumerate(self.old) if c != ' ']
        n = len(chars)
        b0, b1 = self.band
        # letters = 2-D ink shapes (not column runs), so overlapping/kerned letters stay apart
        mk = (self.d_line > self.strong).astype(np.uint8)
        mk[:b0] = 0; mk[b1:] = 0
        ncomp, lab, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
        areas = st[1:, cv2.CC_STAT_AREA]
        med = float(np.median(areas)) if len(areas) else 0
        L = np.zeros_like(lab)
        ids = []
        for i in range(1, ncomp):
            if st[i, cv2.CC_STAT_AREA] >= max(3, 0.02 * med):
                L[lab == i] = i; ids.append(i)
        cols = np.arange(self.W)[None, :].repeat(self.H, 0)

        def ext(g):
            xs = cols[L == g]
            return int(xs.min()), int(xs.max())

        def ordered():
            return sorted(ids, key=lambda g: sum(ext(g)) / 2)

        merged = True                              # stacked pieces: i/j dots, broken strokes
        while merged:
            merged = False
            gs = ordered()
            for a_, b_ in zip(gs, gs[1:]):
                (a0, a1), (c0, c1) = ext(a_), ext(b_)
                ov = min(a1, c1) - max(a0, c0) + 1
                if ov >= 0.5 * min(a1 - a0 + 1, c1 - c0 + 1):
                    L[L == b_] = a_; ids.remove(b_); merged = True
                    break
        gs = ordered()
        plan = self._align([ext(g) for g in gs], chars)
        nxt = int(L.max()) + 1
        new_ids = []
        for kind, pis, ncs, cuts in plan:
            if kind == 'group':                    # several shapes make one letter
                g0 = gs[pis[0]]
                for p in pis[1:]:
                    L[L == gs[p]] = g0
                new_ids.append(g0)
            else:                                  # one shape holds several touching letters
                g = gs[pis[0]]
                prev = g
                new_ids.append(g)
                for c in cuts:
                    L[(L == prev) & (cols > c)] = nxt
                    prev = nxt; new_ids.append(nxt); nxt += 1
        ids = new_ids
        gs = ordered()
        self.letter_label = np.zeros_like(L)
        for k, g in enumerate(gs):
            self.letter_label[L == g] = k + 1
        runs = [list(ext(g)) for g in gs]
        self.runs, self.chars = runs, chars
        self.pos2k = {i: k for k, (i, _) in enumerate(chars)}
        widths = np.array([b - a + 1 for a, b in runs], np.float32)
        centres = np.array([(a + b) / 2 for a, b in runs])
        idx = np.array([i for i, _ in chars], np.float32)

        lgaps, wgaps, adv = [], [], []
        for k in range(n - 1):
            step = chars[k + 1][0] - chars[k][0]
            gap = runs[k + 1][0] - runs[k][1] - 1
            if step == 1:
                lgaps.append(gap); adv.append(centres[k + 1] - centres[k])
            else:
                wgaps.append(gap / (step - 1) if step > 2 else gap)
        mw = float(np.median(widths))
        self.letter_gap = float(np.median(lgaps)) if lgaps else 0.15 * mw
        self.word_gap = float(np.median(wgaps)) if wgaps else 2 * self.letter_gap + 0.5 * mw
        self.advance = float(np.median(adv)) if adv else mw + self.letter_gap

        mono, p, x0 = False, self.advance, centres[0]
        if n >= 3:
            p, x0 = np.polyfit(idx, centres, 1)
            resid = np.abs(centres - (x0 + p * idx)).max()
            mono = resid < 0.22 * p
        if self.args.layout != 'auto':
            mono = self.args.layout == 'mono'
        self.mono = mono
        self.pitch = float(p) if mono else self.advance
        self.x0c = float(x0)
        self.scale = max(0.4, self.cap / 32.0)
        self.GW = int(np.ceil((max(widths.max(), 1.4 * self.pitch if mono else 0) + 0.8 * self.advance + 8) / 2) * 2)
        self.left_edge = runs[0][0]
        self.right_edge = runs[-1][1]
        if mono:
            self.right_margin = max(0.0, self.line_w - (self.x0c + p * (len(self.old) - 1) + p / 2))
        else:
            self.right_margin = max(0.0, self.line_w - 1 - self.right_edge)

    def _align(self, pieces, chars):
        """Match ink shapes to the letters of --old with dynamic programming.

        A letter may be made of several shapes (broken strokes) and one shape may hold
        several touching letters. Costs compare each candidate's width with the letter's
        typical width (from a reference font) and check that word spaces fall on big gaps.
        Returns [(kind, piece indices, n chars, cut columns)] in reading order.
        """
        m, n = len(pieces), len(chars)
        path = next((f for f in PROP_FONTS if os.path.exists(f)), None)
        try:
            f = ImageFont.truetype(path, 100) if path else ImageFont.load_default(100)
            ew = [max(8.0, f.getbbox(c)[2] - f.getbbox(c)[0]) for _, c in chars]
        except (OSError, TypeError):
            ew = [1.0] * n
        aw = [b - a + 1 for a, b in pieces]
        gaps = [pieces[i + 1][0] - pieces[i][1] - 1 for i in range(m - 1)]
        tgap = float(np.median(gaps)) if gaps else 0.0
        sp = [chars[j + 1][0] - chars[j][0] > 1 for j in range(n - 1)]
        cols_prof = (self.d_line > self.strong).sum(0)

        def solve(scale):
            mw = scale * float(np.median(ew))
            INF = 1e18
            dp = np.full((m + 1, n + 1), INF); dp[0, 0] = 0
            back = {}
            for i in range(m + 1):
                for j in range(n + 1):
                    if dp[i, j] >= INF or (i == m and j == n):
                        continue
                    if i == m or j == n:
                        continue
                    base = dp[i, j]
                    if i > 0 and j > 0:            # gap before this letter
                        g = gaps[i - 1]
                        if sp[j - 1]:
                            base += 4 * max(0.0, (tgap + 0.25 * mw - g) / mw) ** 2
                        else:
                            base += 4 * max(0.0, (g - tgap - 0.6 * mw) / mw) ** 2
                    for a in (1, 2, 3):            # a shapes -> 1 letter
                        if i + a > m:
                            break
                        x0, x1 = pieces[i][0], max(p[1] for p in pieces[i:i + a])
                        inner = sum(4 * max(0.0, (gaps[i + t] - 0.25 * mw) / mw) ** 2 for t in range(a - 1))
                        c = np.log((x1 - x0 + 1) / (scale * ew[j])) ** 2 + 0.15 * (a - 1) + inner
                        if base + c < dp[i + a, j + 1]:
                            dp[i + a, j + 1] = base + c; back[(i + a, j + 1)] = (i, j, 'group', a)
                    for b in (2, 3):               # 1 shape -> b touching letters
                        if j + b > n or any(sp[j + t] for t in range(b - 1)):
                            continue
                        c = np.log(aw[i] / (scale * sum(ew[j:j + b]))) ** 2 + 0.35 * (b - 1)
                        if base + c < dp[i + 1, j + b]:
                            dp[i + 1, j + b] = base + c; back[(i + 1, j + b)] = (i, j, 'split', b)
            if dp[m, n] >= INF:
                return None, INF
            out, ij = [], (m, n)
            while ij != (0, 0):
                i, j, kind, k = back[ij]
                out.append((kind, i, j, k)); ij = (i, j)
            return out[::-1], dp[m, n]

        scale = sum(aw) / sum(ew)
        plan, cost = solve(scale)
        if plan:
            ratios = [(pieces[i + k - 1][1] - pieces[i][0] + 1) / ew[j] for kind, i, j, k in plan
                      if kind == 'group']
            if ratios:
                p2, c2 = solve(float(np.median(ratios)))
                if p2:
                    plan, cost = p2, c2
        if m == n:                                 # keep the plain 1:1 reading unless clearly worse
            ident = sum(np.log(aw[t] / (scale * ew[t])) ** 2 for t in range(n))
            if plan is None or ident <= cost + 1.0:
                return [('group', [t], 1, []) for t in range(n)]
        if plan is None:
            sys.exit(f'Found {m} ink shapes but --old has {n} letters. Check --old / --region.')
        result = []
        for kind, i, j, k in plan:
            if kind == 'group':
                result.append(('group', list(range(i, i + k)), 1, []))
            else:
                x0, x1 = pieces[i]
                w = [ew[j + t] for t in range(k)]
                cuts, acc = [], 0.0
                for t in range(k - 1):
                    acc += w[t]
                    guess = x0 + (x1 - x0) * acc / sum(w)
                    r = max(2, int(0.25 * (x1 - x0) * w[t] / sum(w)))
                    lo, hi = int(guess - r), int(guess + r)
                    cuts.append(lo + int(np.argmin(cols_prof[lo:hi + 1])))
                result.append(('split', [i], k, cuts))
        return result

    def _paper(self):
        s = self.scale
        g = self.img.mean(2)
        bgl = cv2.medianBlur(g.astype(np.uint8), min(odd(max(31, 1.6 * self.cap)), 255)).astype(np.float32)
        ink = ((bgl - g) > 5).astype(np.uint8)
        hole = cv2.dilate(ink, np.ones((odd(9 * s),) * 2, np.uint8))
        low = cv2.inpaint(np.clip(self.img, 0, 255).astype(np.uint8), hole * 255,
                          max(3, int(9 * s)), cv2.INPAINT_TELEA).astype(np.float32)
        hf = hole[..., None].astype(np.float32)
        low = cv2.GaussianBlur(low, (0, 0), 3 * s) * hf + low * (1 - hf)
        resid = self.img - cv2.GaussianBlur(self.img, (0, 0), 3 * s)
        b0, b1 = self.band
        lo, hi = max([(0, b0), (b1, self.H)], key=lambda r: r[1] - r[0])
        paper = low.copy()
        noise = 2.0
        if hi - lo >= 4:
            src = resid[lo:hi]
            reps = int(np.ceil(self.H / (2 * (hi - lo)))) + 1
            grain = np.concatenate([src, src[::-1]] * reps, 0)[:self.H]
            paper += grain * cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2)[..., None]
            noise = float(resid[lo:hi].mean(2).std())
        self.low, self.paper = low, paper
        floor = max(2.0, 1.5 * noise + 1)
        D = np.clip(cv2.GaussianBlur(low, (0, 0), 1) - self.img - floor, 0, None)
        D[:self.band[0]] = 0; D[self.band[1]:] = 0; D[:, self.line_w:] = 0
        self.D = D

    def _library(self):
        W, GW, runs, adv = self.W, self.GW, self.runs, self.advance
        # every ink pixel belongs to the nearest letter shape; soft edges between neighbours
        Lb = self.letter_label
        dist, lab = cv2.distanceTransformWithLabels((Lb == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(int(lab.max()) + 1, np.int32)
        lut[lab[Lb > 0]] = Lb[Lb > 0]
        assign = lut[lab]
        assign[dist > 0.45 * adv] = 0
        self.occ = []
        for k, (_, ch) in enumerate(self.chars):
            a, b = runs[k]
            cx = int(round((a + b) / 2))
            m = cv2.GaussianBlur((assign == k + 1).astype(np.float32), (0, 0), 1.0 * self.scale)
            buf = np.zeros((self.H, GW, 3), np.float32)
            xs = np.arange(cx - GW // 2, cx + GW // 2)
            ok = (xs >= 0) & (xs < W)
            buf[:, ok] = self.D[:, xs[ok]] * m[:, xs[ok]][..., None]
            buf = self._despeck(buf)
            self.occ.append({'buf': buf, 'x0': cx - GW // 2, 'ch': ch})
        self.lib = {}
        for k, (_, ch) in enumerate(self.chars):
            self.lib.setdefault(ch, []).append(k)
        self.used = {}

        bufs = [o['buf'] for o in self.occ]
        self.target = float(np.median([self.depth(b) for b in bufs]))
        self.thr = 0.35 * self.target
        boxes = [self.bbox(b) for b in bufs]
        chs = [c for _, c in self.chars]
        bots = [bx[3] for bx, c in zip(boxes, chs) if c not in DESCENDERS]
        self.baseline = int(np.median(bots or [bx[3] for bx in boxes]))
        tops = ([bx[2] for bx, c in zip(boxes, chs) if c.isupper() or c.isdigit()] or
                [bx[2] for bx, c in zip(boxes, chs) if c in 'bdfhklt'] or [bx[2] for bx in boxes])
        self.cap_top = int(np.median(tops))
        self.has_caps = any(c.isupper() for c in chs)
        wide = [bx[1] - bx[0] + 1 for bx, c in zip(boxes, chs) if c not in NARROW and (c.isupper() or not self.has_caps)]
        self.wref = float(np.median(wide)) if wide else 0.65 * self.advance
        self.heights = {}
        for bx, c in zip(boxes, chs):
            self.heights.setdefault(c, []).append(bx[3] - bx[2] + 1)
        strong = np.concatenate([b[b.mean(2) > 0.6 * self.target] for b in bufs])
        col = strong.mean(0)
        self.ink_rgb = col / max(col.mean(), 1e-3)
        self.edge_sigma = 0.9 * self.scale
        self.stem = self._stroke_piece()
        self.bars = {'bottom': self._bar_piece('bottom'), 'top': self._bar_piece('top')}
        if self.mono and 'M' in self.lib and 'W' in self.lib and not self.args.no_reinforce:
            wf = self.occ[self.lib['W'][0]]['buf'][::-1].copy()
            for j, k in enumerate(self.lib['M']):
                m = self.occ[k]['buf']
                f = self._align_to(wf, m)
                self.occ[k]['buf'] = (np.maximum(self.norm(m), self.norm(f) * 0.95) if j % 2 == 0 else
                                      np.maximum(self.norm(m) * 0.9, self.norm(f)))
            self.log.append('M: weak strokes filled in with an upside-down W')

    # ------------------------------------------------------------------ helpers
    def _despeck(self, buf):
        """Remove small dirt specks that are not attached to the letter."""
        v = buf.mean(2)
        mk = (v > 0.12 * max(60.0, np.percentile(v[v > 0], 99) if (v > 0).any() else 20)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
        if n <= 2:
            return buf
        big = st[1:, cv2.CC_STAT_AREA].max()
        drop = np.zeros_like(mk)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 0.06 * big:
                drop[lab == i] = 1
        if not drop.any():
            return buf
        keep = 1 - cv2.dilate(drop, np.ones((5, 5), np.uint8)) * (1 - cv2.dilate(mk - drop, np.ones((3, 3), np.uint8)))
        keep = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 0.8)
        return buf * np.minimum(keep, 1)[..., None]

    def depth(self, a):
        v = a.mean(2)
        return float(np.percentile(v[v > 8], 85)) if (v > 8).any() else 1.0

    def norm(self, a, t=None):
        return a * ((t or self.target) / self.depth(a))

    def bbox(self, a, thr=None):
        v = a.mean(2) > (thr if thr is not None else max(10.0, 0.3 * getattr(self, 'target', 40)))
        cols, rows = np.where(v.any(0))[0], np.where(v.any(1))[0]
        if not len(cols):
            return 0, a.shape[1] - 1, 0, a.shape[0] - 1
        return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())

    def _align_to(self, a, ref):
        c0, c1, r0, r1 = self.bbox(a)
        d0, d1, s0, s1 = self.bbox(ref)
        return shift(a, round((s0 + s1) / 2 - (r0 + r1) / 2), round((d0 + d1) / 2 - (c0 + c1) / 2))

    def _stroke_piece(self):
        """A real vertical stem (column run covering >=70% of an upper-case letter's height)."""
        for ch in STEM_SOURCES:
            for k in self.lib.get(ch, []):
                buf = self.occ[k]['buf']
                c0, c1, r0, r1 = self.bbox(buf, self.thr)
                cov = (buf[r0:r1 + 1].mean(2) > self.thr).mean(0)
                for a, b in runs_of(cov >= 0.7):
                    if 0.08 * self.advance <= b - a + 1 <= 0.35 * self.advance:
                        piece = np.zeros_like(buf)
                        piece[:, a - 1:b + 2] = buf[:, a - 1:b + 2]
                        return {'buf': piece, 'x0': a, 'x1': b, 'r0': r0, 'r1': r1, 'from': ch}
        return None

    def _bar_piece(self, where):
        for ch in (BOTTOM_BAR_SOURCES if where == 'bottom' else TOP_BAR_SOURCES):
            for k in self.lib.get(ch, []):
                buf = self.occ[k]['buf']
                c0, c1, r0, r1 = self.bbox(buf, self.thr)
                cov = (buf[:, c0:c1 + 1].mean(2) > self.thr).mean(1)
                rr = runs_of(cov >= 0.6)
                if not rr:
                    continue
                a, b = rr[-1] if where == 'bottom' else rr[0]
                if b - a + 1 > 0.35 * self.cap:
                    continue
                colcov = (buf[r0:r1 + 1].mean(2) > self.thr).mean(0)
                free = [x for x in range(c0, c1 + 1) if colcov[x] < 0.6]
                x0, x1 = (min(free), max(free)) if free else (c0, c1)
                return {'buf': buf[a - 1:b + 2, x0:x1 + 1], 'from': ch}
        return None

    def _place_bar(self, out, cx0, cx1, y_centre):
        bar = self.bars['bottom'] or self.bars['top']
        strip = cv2.resize(bar['buf'], (max(2, int(cx1 - cx0 + 1)), bar['buf'].shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        h, w = strip.shape[:2]
        y, x = int(round(y_centre - h / 2)), int(cx0)
        out[y:y + h, x:x + w] = np.maximum(out[y:y + h, x:x + w], self.norm(strip) * 0.95)

    def _place_stem(self, out, x_left, y0=None, y1=None):
        st = self.stem
        y0 = self.cap_top if y0 is None else int(y0)
        y1 = self.baseline if y1 is None else int(y1)
        col = st['buf'][st['r0']:st['r1'] + 1, st['x0'] - 1:st['x1'] + 2]
        col = cv2.resize(col, (col.shape[1], max(2, y1 - y0 + 1)), interpolation=cv2.INTER_LINEAR)
        x = int(round(x_left)) - 1
        out[y0:y0 + col.shape[0], x:x + col.shape[1]] = np.maximum(
            out[y0:y0 + col.shape[0], x:x + col.shape[1]], self.norm(col))

    def build_strokes(self, ch):
        if (ch not in 'HILTEF' or not self.has_caps or self.stem is None or
                (self.bars['bottom'] is None and self.bars['top'] is None)):
            return None
        GW, wref = self.GW, self.wref
        out = np.zeros((self.H, GW, 3), np.float32)
        sw = self.stem['x1'] - self.stem['x0'] + 1
        left = GW / 2 - wref / 2
        right = left + wref - sw
        mid = self.cap_top + 0.52 * (self.baseline - self.cap_top)
        bh = (self.bars['bottom'] or self.bars['top'])['buf'].shape[0]
        top_c, bot_c = self.cap_top + bh / 2 - 1, self.baseline - bh / 2 + 1
        if ch == 'H':
            self._place_stem(out, left); self._place_stem(out, right)
            self._place_bar(out, left + sw - 1, right, mid)
        elif ch == 'I':
            self._place_stem(out, GW / 2 - sw / 2)
            if self.mono and not self.args.plain_i:       # typewriter I has serifs
                half = 0.28 * wref
                self._place_bar(out, GW / 2 - half, GW / 2 + half, top_c)
                self._place_bar(out, GW / 2 - half, GW / 2 + half, bot_c)
        elif ch == 'L':
            self._place_stem(out, left); self._place_bar(out, left, left + wref - 1, bot_c)
        elif ch == 'T':
            self._place_bar(out, left, left + wref - 1, top_c)
            self._place_stem(out, GW / 2 - sw / 2, top_c)
        else:  # E, F
            self._place_stem(out, left)
            self._place_bar(out, left, left + 0.95 * wref, top_c)
            self._place_bar(out, left, left + 0.8 * wref, mid)
            if ch == 'E':
                self._place_bar(out, left, left + 0.95 * wref, bot_c)
        return out

    def _font(self):
        if getattr(self, '_fontcache', None):
            return self._fontcache
        path = self.args.font or next((f for f in (MONO_FONTS if self.mono else PROP_FONTS)
                                       if os.path.exists(f)), None)
        if path is None:
            self._fontcache = (None, None, None)
            return self._fontcache
        up, ref = 4, 100
        font = ImageFont.truetype(path, ref)
        ratios = []                      # match the font's letter heights to the source's
        for c, hs in self.heights.items():
            bb = font.getbbox(c)
            if bb[3] - bb[1] > 0:
                ratios.append(np.median(hs) * up / (bb[3] - bb[1]))
        size = max(4, int(ref * np.median(ratios))) if ratios else ref
        self._fontcache = (ImageFont.truetype(path, size), up, os.path.basename(path))
        return self._fontcache

    def build_font(self, ch):
        font, up, name = self._font()
        if font is None:
            return None, None
        Wc, Hc = self.GW * up, self.H * up
        im = Image.new('L', (Wc, Hc), 0)
        ImageDraw.Draw(im).text((Wc / 2, (self.baseline + 1) * up), ch, fill=255, font=font, anchor='ms')
        m = np.array(im).astype(np.float32) / 255
        if m.max() == 0:
            return None, None
        want = ((self.stem['x1'] - self.stem['x0'] + 1) if self.stem else self._stroke_width()) * up
        dist = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        have = 2 * np.percentile(dist[dist > 0], 90) if (dist > 0).any() else want
        k = int(round((want - have) / 2))
        if k > 0:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2))
        elif k < 0:
            m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-2 * k + 1,) * 2))
        m = cv2.resize(m, (self.GW, self.H), interpolation=cv2.INTER_AREA)
        m = cv2.GaussianBlur(m, (0, 0), self.edge_sigma)
        tex = cv2.GaussianBlur(self.rng.normal(1, 0.18, m.shape).astype(np.float32), (0, 0), 0.8)
        return (m * tex)[..., None] * self.target * self.ink_rgb[None, None, :], name

    def _stroke_width(self):
        if getattr(self, '_sw', None) is None:
            ws = []
            for o in self.occ:
                mk = (o['buf'].mean(2) > 0.5 * self.target).astype(np.uint8)
                dist = cv2.distanceTransform(mk, cv2.DIST_L2, 3)
                if (dist > 0).any():
                    ws.append(2 * np.percentile(dist[dist > 0], 90))
            self._sw = float(np.median(ws)) if ws else 0.12 * self.cap
        return self._sw

    def pick(self, ch):
        ks = self.lib[ch]
        k = min(ks, key=lambda k: (self.used.get(k, 0), ks.index(k)))
        self.used[k] = self.used.get(k, 0) + 1
        return k

    def glyph(self, ch):
        if ch in self.lib:
            k = self.pick(ch)
            return self.occ[k]['buf'], f'copied (source letter #{k + 1})'
        if ch in FLIPS and FLIPS[ch][0] in self.lib and (self.mono or ch not in MONO_ONLY_FLIPS):
            src, ax = FLIPS[ch]
            b = self.occ[self.pick(src)]['buf']
            f = b[::-1].copy() if ax == 'v' else b[:, ::-1].copy()
            if ax == 'v':      # keep the baseline where it was
                c0, c1, r0, r1 = self.bbox(f); _, _, s0, s1 = self.bbox(b)
                f = shift(f, s1 - r1, 0)
            return f, f'flip of {src} ({"upside down" if ax == "v" else "mirrored"}) - CHECK'
        s = self.build_strokes(ch)
        if s is not None:
            bar = (self.bars['bottom'] or self.bars['top'])['from']
            return s, f'built from strokes (stem of {self.stem["from"]}, bar of {bar}) - CHECK'
        f, name = self.build_font(ch)
        if f is not None:
            return f, f'FONT FALLBACK ({name}) - check closely, try --font'
        return None, 'MISSING'

    def plan(self, text):
        """Split text into spaces, multi-letter chunks copied whole, and single letters."""
        segs, i, uses = [], 0, {}
        while i < len(text):
            if text[i] == ' ':
                segs.append(('space',)); i += 1; continue
            best = None
            if not self.args.no_chunks:
                L = 0
                while i + L < len(text) and text[i + L] != ' ':
                    L += 1
                for L in range(L, 1, -1):
                    sub = text[i:i + L]
                    hits = [j for j in range(len(self.old) - L + 1) if self.old[j:j + L] == sub]
                    if hits:
                        j = min(hits, key=lambda j: uses.get(j, 0))
                        uses[j] = uses.get(j, 0) + 1
                        best = (L, j); break
            if best:
                L, j = best
                ks = [self.pos2k[j + t] for t in range(L)]
                for k in ks:
                    self.used[k] = self.used.get(k, 0) + 1
                segs.append(('chunk', text[i:i + L], ks)); i += L
            else:
                segs.append(('glyph', text[i])); i += 1
        return segs

    def ribbon(self, shape):
        n = cv2.resize(self.rng.uniform(0.88, 1.1, (6, 3)).astype(np.float32), (shape[1], shape[0]),
                       interpolation=cv2.INTER_CUBIC)
        return n[..., None]

    def finish(self, a, t):
        a = self.norm(a, t)
        return t * np.power(np.clip(a / t, 0, None), self.args.gamma) * self.ribbon(a.shape)

    # ------------------------------------------------------------------ render
    def layout(self, text, pitch=None, gapscale=1.0):
        """Returns pieces [(buf, x0)], report, ink right edge. Coordinates are image columns."""
        self.used = {}
        segs = self.plan(text)
        pitch = pitch or self.pitch
        GW = self.GW
        pieces, report = [], []
        cursor, spaces, n = None, 0, 0
        jit = max(1, round(self.scale))
        for seg in segs:
            if seg[0] == 'space':
                spaces += 1; n += 1; continue
            t = self.target * self.args.darkness * self.rng.uniform(0.94, 1.05)
            dy = int(self.rng.integers(-jit, jit + 1))
            if seg[0] == 'chunk':
                _, sub, ks = seg
                bufs = [shift(self.finish(self.occ[k]['buf'], t), dy, 0) for k in ks]
                origin = [self.occ[k]['x0'] for k in ks]
                c0 = self.bbox(bufs[0])[0]
                for ch, k in zip(sub, ks):
                    report.append({'pos': n, 'char': ch, 'source': f'copied with neighbours (source letter #{k + 1})'})
                    n += 1
                if self.mono:
                    base = self.x0c + pitch * (n - len(sub)) - (self.occ[ks[0]]['x0'] + GW // 2)
                    offs = [o + base for o in origin]
                else:
                    gap = self.letter_gap if spaces == 0 else self.word_gap * spaces
                    left = self.left_edge if cursor is None else cursor + 1 + gap * gapscale
                    base = left - (origin[0] + c0)
                    offs = [o + base for o in origin]
                for b, x in zip(bufs, offs):
                    pieces.append((b, x))
                cursor = offs[-1] + self.bbox(bufs[-1])[1]
            else:
                ch = seg[1]
                a, how = self.glyph(ch)
                report.append({'pos': n, 'char': ch, 'source': how})
                n += 1
                if a is None:
                    continue
                a = shift(self.finish(a, t), dy, 0)
                c0, c1, _, _ = self.bbox(a)
                if self.mono:
                    a = shift(a, 0, round(GW / 2 - (c0 + c1) / 2 + self.rng.uniform(-0.7, 0.7) * self.scale))
                    x = self.x0c + pitch * (n - 1) - GW / 2
                    cursor = x + self.bbox(a)[1]
                else:
                    gap = self.letter_gap if spaces == 0 else self.word_gap * spaces
                    left = self.left_edge if cursor is None else cursor + 1 + gap * gapscale
                    x = left - c0
                    cursor = x + c1
                pieces.append((a, x))
            spaces = 0
        return pieces, report, cursor if cursor is not None else self.left_edge

    def render(self, text, fit=False):
        pitch, gapscale = self.pitch, 1.0
        avail_right = self.line_w - 1 - self.right_margin
        pieces, report, right = self.layout(text)
        squeeze = None
        if fit and right > avail_right:
            if self.mono:
                start = self.x0c - self.pitch / 2
                pitch = (self.line_w - self.right_margin - start) / len(text)
                self.rng = np.random.default_rng(self.args.seed)
                pieces, report, right = self.layout(text, pitch=pitch)
            else:
                ngaps = max(1, sum(1 for i in range(1, len(text)) if text[i] != ' '))
                gapscale = max(0.35, 1 - (right - avail_right) / max(1.0, self.letter_gap * ngaps + self.word_gap * text.count(' ')))
                self.rng = np.random.default_rng(self.args.seed)
                pieces, report, right = self.layout(text, gapscale=gapscale)
            limit = self.line_w - 2 if self.mono else avail_right + 2   # mono: new pitch already fits
            if right > limit:
                squeeze = (self.left_edge, right, avail_right)
        shift_x = 0.0
        if self.args.align != 'left' and squeeze is None:
            first = min(x + self.bbox(b)[0] for b, x in pieces) if pieces else self.left_edge
            if self.args.align == 'center':
                shift_x = (self.left_edge + self.right_edge) / 2 - (first + right) / 2
            else:
                shift_x = self.right_edge - right
            right += shift_x
        width = self.line_w if fit else max(self.line_w, int(math.ceil(right + 1 + self.right_margin)))
        pad = self.GW * 2
        acc = np.zeros((self.H, width + 2 * pad, 3), np.float32)
        for b, x in pieces:
            xi = int(round(x + shift_x)) + pad
            if 0 <= xi and xi + b.shape[1] <= acc.shape[1]:
                acc[:, xi:xi + b.shape[1]] = np.maximum(acc[:, xi:xi + b.shape[1]], b)
        if squeeze:
            l, r, a = squeeze
            seg = acc[:, pad + l:pad + int(r) + 1]
            acc[:, pad + l:] = 0
            acc[:, pad + l:pad + int(a) + 1] = cv2.resize(seg, (int(a) - l + 1, self.H), interpolation=cv2.INTER_AREA)
            self.log.append(f'fit_width: letters squeezed horizontally by {100 * (1 - (a - l) / (r - l)):.0f}% to fit')
        acc = acc[:, pad:pad + width]
        acc = np.clip(1.35 * acc - 0.35 * cv2.GaussianBlur(acc, (0, 0), 1.2 * self.scale), 0, None)
        paper = self.paper
        while paper.shape[1] < width:
            paper = np.concatenate([paper, paper[:, ::-1]], 1)
        out = np.clip(paper[:, :width] - acc, 0, 255).astype(np.uint8)
        if fit and not squeeze:
            self.log.append(f'fit_width: ' + (f'pitch {pitch:.1f}px instead of {self.pitch:.1f}px' if self.mono
                                              else f'letter/word gaps at {100 * gapscale:.0f}%'))
        return out, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image')
    ap.add_argument('--old', required=True, help='text currently in the image, exactly, with spaces')
    ap.add_argument('--new', required=True, help='replacement text')
    ap.add_argument('--region', help='x0,y0,x1,y1 of the text line inside a larger page')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--name', help='output filename prefix (default: new text, snake_case)')
    ap.add_argument('--layout', choices=['auto', 'mono', 'proportional'], default='auto',
                    help='mono = typewriter/fixed-width, proportional = normal fonts and handwriting')
    ap.add_argument('--align', choices=['left', 'center', 'right'], default='left')
    ap.add_argument('--font', help='fallback .ttf/.ttc for letters missing from the source '
                                   '(pick one close to the document, e.g. a handwriting font for pen)')
    ap.add_argument('--darkness', type=float, default=1.0, help='ink strength multiplier, e.g. 1.15')
    ap.add_argument('--gamma', type=float, default=0.85, help='<1 fills in weak strokes more, 1 = untouched')
    ap.add_argument('--seed', type=int, default=11, help='change for different random ink/jitter')
    ap.add_argument('--plain-i', action='store_true', help='build I without typewriter serifs')
    ap.add_argument('--no-chunks', action='store_true', help='place every letter individually')
    ap.add_argument('--no-reinforce', action='store_true', help="don't fill weak M strokes with a flipped W")
    ap.add_argument('--no-deskew', action='store_true', help="don't straighten tilted lines")
    ap.add_argument('--jpeg-quality', type=int, default=88)
    args = ap.parse_args()

    page = cv2.imread(args.image)
    if page is None:
        sys.exit(f'cannot read {args.image}')
    name = args.name or ''.join(c if c.isalnum() else '_' for c in args.new.lower()).strip('_')
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda s: os.path.join(args.out_dir, f'{name}_{s}')

    x0, y0, x1, y1 = (map(int, args.region.split(',')) if args.region else
                      (0, 0, page.shape[1], page.shape[0]))
    line_w = x1 - x0
    extra = int(max(0, len(args.new) - len(args.old)) * 1.2 * (y1 - y0))
    crop = page[y0:y1, x0:min(page.shape[1], x1 + extra)].copy()
    rt = Retype(crop, args.old, line_w, args)

    results = {}
    results['same_spacing'], rep = rt.render(args.new)
    if results['same_spacing'].shape[1] > line_w:
        results['fit_width'], _ = rt.render(args.new, fit=True)

    for k, v in list(results.items()):
        base = crop
        while base.shape[1] < v.shape[1]:
            base = np.concatenate([base, base[:, ::-1]], 1)
        v = rt.unlevel(v, base[:, :v.shape[1]].astype(np.float32))
        results[k] = v
        cv2.imwrite(out(f'{k}.png'), v)
        cv2.imwrite(out(f'{k}.jpg'), v, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if args.region:
            full = page.copy()
            w = min(v.shape[1], full.shape[1] - x0)
            if w < v.shape[1]:
                rt.log.append(f'page too narrow for {k}; clipped at the right edge')
            full[y0:y1, x0:x0 + w] = v[:, :w]
            cv2.imwrite(out(f'page_{k}.png'), full)

    orig = crop[:, :line_w]
    widest = max(v.shape[1] for v in results.values())
    padw = lambda im: np.pad(im, ((0, 4), (0, widest - im.shape[1]), (0, 0)), constant_values=255)
    cmp = np.vstack([padw(orig)] + [padw(v) for v in results.values()])
    cv2.imwrite(out('comparison.png'), cv2.resize(cmp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    b0, b1 = rt.band
    b0, b1 = max(0, b0 - 4), min(rt.H, b1 + 4)
    z = lambda im: cv2.resize(im[b0:b1], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    zo, zn = z(orig), z(results['same_spacing'])
    wz = max(zo.shape[1], zn.shape[1])
    zp = lambda im: np.pad(im, ((0, 6), (0, wz - im.shape[1]), (0, 0)), constant_values=255)
    cv2.imwrite(out('zoom.png'), np.vstack([zp(zo), zp(zn)]))

    info = {'old': args.old, 'new': args.new,
            'layout': 'mono' if rt.mono else 'proportional',
            'pitch_or_advance_px': round(rt.pitch, 2), 'letter_height_px': int(rt.baseline - rt.cap_top + 1),
            'tilt_deg': round(rt.angle, 2), 'letters': rep, 'notes': rt.log,
            'outputs': [os.path.basename(out(f'{k}.png')) for k in results]}
    with open(out('report.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f'layout {info["layout"]}, {"pitch" if rt.mono else "advance"} {rt.pitch:.1f}px, '
          f'letter height {info["letter_height_px"]}px')
    for r in rep:
        if not r['source'].startswith('copied'):
            print(f"  {r['char']!r} at {r['pos']}: {r['source']}")
    for n in rt.log:
        print('  ' + n)
    print('wrote', ', '.join(info['outputs']), '+ comparison, zoom, report in', os.path.abspath(args.out_dir))


if __name__ == '__main__':
    main()
