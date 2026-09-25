---
name: retype
description: Replace a line of text in a photo or scan (typewriter, printed font, pen/handwriting, stamp) with new text built from the document's own letters, so it matches the original ink, font and paper. Use when the user shares an image of text and asks to change, correct, or replace the words in it.
---

# retype: replace text in an image using its own letters

The script is `retype.py`, in this skill folder (`~/.claude/skills/retype/retype.py`).
Requirements: `pip install opencv-python numpy pillow`.

## Workflow

1. **Get the inputs.**
   - The image path.
   - `--old`: exactly what the image says, including spaces and letter case. Read it from the image
     yourself, then confirm it with the user if anything is unclear.
   - `--new`: the replacement text.
   - If the image is a full page, find the pixel box of the line (`--region x0,y0,x1,y1`), or crop
     the line first.
2. **Run it** into a scratch/output folder:
   ```bash
   python3 ~/.claude/skills/retype/retype.py IMAGE --old "..." --new "..." --out-dir OUT
   ```
3. **Read the console summary.** It lists every letter that was *not* copied from the original
   (flip / strokes / FONT FALLBACK), plus notes about tilt, layout and fit.
4. **Inspect `OUT/<name>_zoom.png`** (original vs result at 4×). Look for:
   - broken or clipped letters, or wrong letters (a bad `--old` or letter alignment);
   - ghosts of the old text, or specks;
   - fallback letters that don't match the style: pass `--font` with a closer font
     (Courier for typewriters, the right serif/sans for print, a handwriting font for pen);
   - ink too light or too dark next to the rest of the page: `--darkness 0.9..1.2`;
   - spacing: `--layout mono|proportional`, `--align`.
5. **Iterate** until it looks right, then give the user `same_spacing` and/or `fit_width`
   (fit_width is only produced when the new text is longer). Copy them wherever the user asks.

## Notes

- The more of the new text that already appears in `--old`, the better. Groups of letters are
  copied whole, which keeps real spacing and pen joins.
- Only dark ink on light paper, one line per run.
- Don't use it to alter official records, IDs, certificates, contracts, or similar documents to
  deceive someone. If the request looks like that, decline.
