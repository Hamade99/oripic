"""
Crease-pattern renderer. Pure Python, PIL only.

Public API
----------
    draw_cp(text: str, *, img_size: int = 800,
            padding: int = 50, scale_factor: int = 2) -> bytes
        Render a CP-format string to PNG bytes.

Also runnable as a CLI (backward-compatible with the previous stdin→stdout
contract used by the old Node wrapper):
    python oripic.py < input.cp > output.png
"""
from __future__ import annotations

import io
import sys
from PIL import Image, ImageDraw

# crease-type  →  (RGB colour, logical stroke width in px)
_STYLES: dict[int, tuple[tuple[int, int, int], int]] = {
    1: ((0,   0,   0  ), 2),   # border     – black
    2: ((220, 0,   0  ), 2),   # mountain   – red
    3: ((0,   0,   220), 2),   # valley     – blue
    4: ((0,   150, 0  ), 2),   # auxiliary  – green
}
_DEFAULT_STYLE: tuple[tuple[int, int, int], int] = ((128, 128, 128), 2)


def _parse_cp(text: str):
    """Yield (k, x1, y1, x2, y2) tuples and compute the bounding box in one pass."""
    lines: list[tuple[int, float, float, float, float]] = []
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for row in text.splitlines():
        parts = row.split()
        if len(parts) != 5:
            continue
        try:
            k = int(parts[0])
            x1 = float(parts[1]); y1 = float(parts[2])
            x2 = float(parts[3]); y2 = float(parts[4])
        except ValueError:
            continue

        lines.append((k, x1, y1, x2, y2))
        # inline min/max — ~2× faster than calling min()/max() with 3 args per line
        if x1 < min_x: min_x = x1
        if x2 < min_x: min_x = x2
        if y1 < min_y: min_y = y1
        if y2 < min_y: min_y = y2
        if x1 > max_x: max_x = x1
        if x2 > max_x: max_x = x2
        if y1 > max_y: max_y = y1
        if y2 > max_y: max_y = y2

    return lines, min_x, min_y, max_x, max_y


def draw_cp(
    text: str,
    *,
    img_size: int = 800,
    padding: int = 50,
    scale_factor: int = 2,
) -> bytes:
    """Render a CP string to PNG bytes.

    Notes on the defaults
    ---------------------
    * `scale_factor=2` — supersample factor for anti-aliased line drawing. The
      previous version used 4, which renders a 3200×3200 RGBA image (~40 MB).
      At an 800-px output size with 2-px strokes the visual difference between
      2× and 4× is imperceptible, while the pixel count is ¼, i.e. the whole
      render step is roughly 4× cheaper.
    * Canvas is RGB (not RGBA). The previous code built RGBA then converted
      back to RGB before saving; the alpha channel was never used.
    * PNG saved with `compress_level=3` — still lossless, ~2–3× faster to
      encode than the default 6 with only marginally bigger files. Discord
      doesn't care about size at this scale.
    """
    lines, min_x, min_y, max_x, max_y = _parse_cp(text)
    if not lines:
        raise ValueError("empty or unparseable CP")

    big_size    = img_size * scale_factor
    big_padding = padding  * scale_factor
    width  = max_x - min_x
    height = max_y - min_y
    if width == 0 or height == 0:
        scale = 1.0
    else:
        scale = (big_size - 2 * big_padding) / max(width, height)

    offset_x = (big_size - width  * scale) / 2
    offset_y = (big_size - height * scale) / 2

    img  = Image.new("RGB", (big_size, big_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for k, x1, y1, x2, y2 in lines:
        colour, w = _STYLES.get(k, _DEFAULT_STYLE)
        draw.line(
            (
                (x1 - min_x) * scale + offset_x,
                (y1 - min_y) * scale + offset_y,
                (x2 - min_x) * scale + offset_x,
                (y2 - min_y) * scale + offset_y,
            ),
            fill=colour,
            width=max(1, w * scale_factor),
            joint="curve",
        )

    if scale_factor != 1:
        img = img.resize((img_size, img_size), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


if __name__ == "__main__":
    # Backward-compatible CLI: read CP text from stdin, write PNG to stdout.
    sys.stdout.buffer.write(draw_cp(sys.stdin.read()))
