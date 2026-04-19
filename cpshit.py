"""
cpshit.py — convert any image into a (probably nonsensical) .cp file.

Pipeline:
  1. Load image, convert to grayscale, downscale if huge.
  2. Bilateral smooth (preserves edges, flattens textured regions).
  3. Canny edge detection.
  4. findContours → approxPolyDP: trace each edge as a polyline,
     simplify to a small set of line segments per curve.
  5. Optional tonal shading: sweep parallel hatch lines through regions
     whose local darkness exceeds per-level cutoffs. Extra levels
     crosshatch at different angles for tonal gradation.
  6. Write each segment as a .cp line (edges as mountain folds,
     hatches as auxiliary).

The output won't be flat-foldable or even sensible — it's meant as novelty,
not a real origami design.

Public API
----------
    image_to_cp(path, ...)  -> str   # read from disk
    bytes_to_cp(data, ...)  -> str   # decode from memory (no disk I/O)

Usage (CLI):
  python cpshit.py input.jpg                         # defaults (crosshatch)
  python cpshit.py input.jpg --shade-levels 0        # edges only
  python cpshit.py input.jpg --shade-levels 3        # triple-hatch darkest
  python cpshit.py input.jpg --epsilon-frac 0.003    # finer polylines

Install:
  pip install opencv-python numpy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import math
import numpy as np


# Hatching directions used, in the order each extra level is added.
# Level 1: primary diagonal.
# Level 2: add opposite diagonal → crosshatch in mid-dark regions.
# Level 3: add vertical → triple-hatch only in the very darkest spots.
_HATCH_ANGLES_DEG: tuple[float, ...] = (45.0, 135.0, 90.0)

# Darkness cutoff per level, expressed as "top X% darkest pixels after smoothing".
# Percentile-based (not absolute): an overall-dark photo and a bright one both
# get proportional shading rather than "everything hatched" vs "nothing hatched".
_HATCH_PERCENTILES: tuple[float, ...] = (40.0, 18.0, 7.0)


def _shade_lines(
    img: np.ndarray,
    *,
    levels: int,
    spacing: int,
    blur_ksize: int,
    line_types: tuple[int, ...],
    min_segment_px: int = 4,
) -> list[tuple[int, float, float, float, float]]:
    """Generate parallel hatch segments across dark regions.

    For each level, parallel scan lines are swept across the image at a chosen
    angle. Runs of pixels whose smoothed darkness exceeds the level's
    percentile cutoff are emitted as segments, using the level's own crease
    type (`line_types[i]`). Extra levels crosshatch on top of earlier ones
    (different angles, tighter darkness cutoffs), producing tonal gradations:
    nothing → single hatch → crosshatch → triple hatch.

    The darkness map is the Gaussian-blurred inverse of the gray image. Blurring
    makes the map reflect regional tone rather than per-pixel detail, so we
    get tonal shading rather than edge-following.
    """
    if levels <= 0:
        return []
    h, w = img.shape[:2]
    k = blur_ksize | 1  # odd kernel size required
    dark_map = 255 - cv2.GaussianBlur(img, (k, k), 0)

    lines: list[tuple[int, float, float, float, float]] = []
    diag = int(math.hypot(w, h)) + spacing

    for lvl in range(min(levels, len(_HATCH_ANGLES_DEG), len(line_types))):
        angle_rad = math.radians(_HATCH_ANGLES_DEG[lvl])
        threshold = float(np.percentile(dark_map, 100 - _HATCH_PERCENTILES[lvl]))
        line_type = line_types[lvl]
        dx, dy = math.cos(angle_rad), math.sin(angle_rad)
        nx, ny = -dy, dx  # perpendicular unit vector — we step along this

        min_sq = min_segment_px * min_segment_px

        for offset in range(-diag, diag + 1, spacing):
            # Anchor for this scan line: center of image + offset along the
            # perpendicular. Scan extends `diag` in each direction along (dx,dy).
            cx = w / 2 + offset * nx
            cy = h / 2 + offset * ny
            run_start: tuple[int, int] | None = None
            run_end: tuple[int, int] | None = None
            for t in range(-diag, diag + 1):
                x = int(round(cx + t * dx))
                y = int(round(cy + t * dy))
                if not (0 <= x < w and 0 <= y < h):
                    continue
                if dark_map[y, x] > threshold:
                    if run_start is None:
                        run_start = (x, y)
                    run_end = (x, y)
                elif run_start is not None:
                    # End of dark run → emit if long enough to matter.
                    ex, ey = run_end  # type: ignore[misc]
                    sx, sy = run_start
                    if (ex - sx) ** 2 + (ey - sy) ** 2 >= min_sq:
                        lines.append((line_type, float(sx), float(sy),
                                      float(ex), float(ey)))
                    run_start = run_end = None
            # Trailing run that continued to image edge.
            if run_start is not None and run_end is not None:
                sx, sy = run_start
                ex, ey = run_end
                if (ex - sx) ** 2 + (ey - sy) ** 2 >= min_sq:
                    lines.append((line_type, float(sx), float(sy),
                                  float(ex), float(ey)))
    return lines


def _gray_array_to_cp(
    img: np.ndarray,
    *,
    canny_low: int,
    canny_high: int,
    epsilon_frac: float,
    min_contour_len: int,
    line_type: int,
    max_dim: int,
    shade_levels: int,
    shade_spacing: int | None,
    shade_line_types: tuple[int, ...],
) -> str:
    """Core pipeline. Expects a 2-D grayscale uint8 array.

    Two passes combined:
      1. Edge pass: bilateral-smooth → Canny → findContours → approxPolyDP,
         giving one polyline per real edge in the image.
      2. Shading pass (optional, `shade_levels > 0`): sweep parallel hatch
         lines through regions whose local darkness exceeds per-level cutoffs.
         Extra levels crosshatch at different angles for tonal gradation.

    Why not Hough? Hough only detects locally-straight segments. On any content
    with curves (anime linework, faces, organic shapes) it produces a cloud of
    short tangential fragments that don't follow the actual edges. Contour
    tracing + Douglas-Peucker polyline simplification handles both straight
    edges and curves uniformly.
    """
    h, w = img.shape[:2]

    # Downscale huge images — the pipeline gets slow and noisy on 4000×4000
    # photos, and the extra resolution doesn't buy anything for this use.
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Bilateral filter smooths flat regions (skin, background) so Canny
    # doesn't fire on per-pixel noise, but preserves strong edges. Sigmas
    # kept low (30, not OpenCV's common ~75): wider sigmas over-smooth
    # subtle interior features (eyes, lips, nose contours) because those are
    # local low-contrast edges relative to the surrounding face.
    smoothed = cv2.bilateralFilter(img, d=7, sigmaColor=30, sigmaSpace=30)
    edges = cv2.Canny(smoothed, canny_low, canny_high)

    # RETR_LIST: we want interior features (eyes, buttons, text) as well as
    # outlines. CHAIN_APPROX_NONE: keep every contour point so approxPolyDP
    # can do the simplification with a uniform tolerance.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    lines: list[tuple[int, float, float, float, float]] = []
    for c in contours:
        arc = cv2.arcLength(c, closed=False)
        if arc < min_contour_len:
            continue
        # epsilon is a distance tolerance in pixels. We scale it by the
        # contour's own length so the same `epsilon_frac` produces a
        # consistent level of detail across contours of wildly different
        # sizes (hair strand vs. whole-body outline).
        approx = cv2.approxPolyDP(c, epsilon_frac * arc, closed=False)
        pts = approx.reshape(-1, 2)
        # No Y-flip: image coords and our renderer (oripic) both use Y-down,
        # so passing photo coords through as-is gives a right-side-up render.
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            lines.append((line_type, float(x1), float(y1),
                                      float(x2), float(y2)))

    # Tonal shading pass. Uses `shade_line_types` (aux / mountain / valley by
    # default) so each darkness level renders in its own colour, keeping the
    # tonal layers visually distinct from each other and from the edges.
    if shade_levels > 0:
        # Blur kernel scaled to image size so shading reflects regional tone,
        # not per-pixel detail. 1/20 of the long side is a good default.
        blur_k = max(5, max(h, w) // 20)
        # Spacing scales with post-downscale image size so hatch density looks
        # the same across a 300px thumbnail and a 1200px photo. Target ~60
        # hatch lines across the long side; clamp to ≥4 so we don't flood
        # tiny images.
        spacing = shade_spacing if shade_spacing is not None else max(4, max(h, w) // 60)
        lines.extend(_shade_lines(
            img,
            levels=shade_levels,
            spacing=spacing,
            blur_ksize=blur_k,
            line_types=shade_line_types,
        ))

    cp_lines = [f"{k} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
                for k, x1, y1, x2, y2 in lines]
    return "\n".join(cp_lines) + ("\n" if cp_lines else "")


def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """cv2.imdecode with IMREAD_GRAYSCALE already gives 2-D, but be defensive
    in case a caller hands us a 3-channel array."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] in (3, 4):
        return cv2.cvtColor(
            img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        )
    raise ValueError(f"unexpected image shape: {img.shape}")


def image_to_cp(
    image_path: Path,
    *,
    canny_low: int = 30,
    canny_high: int = 100,
    epsilon_frac: float = 0.008,
    min_contour_len: int = 30,
    line_type: int = 1,       # 1=border, 2=mountain, 3=valley, 4=aux
    max_dim: int = 1200,
    shade_levels: int = 3,
    shade_spacing: int | None = None,  # None → auto-scale to image size
    shade_line_types: tuple[int, ...] = (4, 2, 3),  # aux, mountain, valley
) -> str:
    """Read an image from disk, return .cp file contents."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return _gray_array_to_cp(
        img,
        canny_low=canny_low,
        canny_high=canny_high,
        epsilon_frac=epsilon_frac,
        min_contour_len=min_contour_len,
        line_type=line_type,
        max_dim=max_dim,
        shade_levels=shade_levels,
        shade_spacing=shade_spacing,
        shade_line_types=shade_line_types,
    )


def bytes_to_cp(
    image_bytes: bytes,
    *,
    canny_low: int = 30,
    canny_high: int = 100,
    epsilon_frac: float = 0.008,
    min_contour_len: int = 30,
    line_type: int = 1,
    max_dim: int = 1200,
    shade_levels: int = 3,
    shade_spacing: int | None = None,
    shade_line_types: tuple[int, ...] = (4, 2, 3),
) -> str:
    """Decode an image from a bytes buffer (no disk I/O), return .cp file contents."""
    if not image_bytes:
        raise ValueError("empty image buffer")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not decode image bytes (unsupported format or corrupt data)")
    img = _ensure_grayscale(img)
    return _gray_array_to_cp(
        img,
        canny_low=canny_low,
        canny_high=canny_high,
        epsilon_frac=epsilon_frac,
        min_contour_len=min_contour_len,
        line_type=line_type,
        max_dim=max_dim,
        shade_levels=shade_levels,
        shade_spacing=shade_spacing,
        shade_line_types=shade_line_types,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Novelty image → .cp converter (Canny + contour polylines).",
    )
    p.add_argument("image", type=Path, help="Input image (any format OpenCV reads).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output .cp path (default: <input>.cp next to the image).")
    p.add_argument("--low", dest="canny_low", type=int, default=30,
                   help="Canny lower threshold (default 30). Lower = more edges.")
    p.add_argument("--high", dest="canny_high", type=int, default=100,
                   help="Canny upper threshold (default 100). Lower = more edges.")
    p.add_argument("--epsilon-frac", type=float, default=0.008,
                   help="Polyline simplification tolerance as a fraction of "
                        "contour length (default 0.008). Smaller = more "
                        "segments, finer detail; larger = coarser, fewer lines.")
    p.add_argument("--min-contour-len", type=int, default=30,
                   help="Drop contours shorter than this in pixels (default 30). "
                        "Raise to suppress noise speckles.")
    p.add_argument("--type", dest="line_type", type=int, default=1,
                   choices=[1, 2, 3, 4],
                   help="Crease type for edge lines (1=border, 2=mountain, "
                        "3=valley, 4=aux). Default 1 (border / black).")
    p.add_argument("--max-dim", type=int, default=1200,
                   help="Downscale longest side to this many px (default 1200).")
    p.add_argument("--shade-levels", type=int, default=3, choices=[0, 1, 2, 3],
                   help="Hatch shading levels (default 3). 0 = off, 1 = single "
                        "diagonal, 2 = crosshatch, 3 = triple hatch.")
    p.add_argument("--shade-spacing", type=int, default=None,
                   help="Pixels between hatch lines. Default: auto (~long-side / 60) "
                        "so density looks the same across image sizes.")
    args = p.parse_args(argv)

    if not args.image.exists():
        print(f"error: {args.image} does not exist", file=sys.stderr)
        return 1

    out_path = args.output or args.image.with_suffix(".cp")

    cp = image_to_cp(
        args.image,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        epsilon_frac=args.epsilon_frac,
        min_contour_len=args.min_contour_len,
        line_type=args.line_type,
        max_dim=args.max_dim,
        shade_levels=args.shade_levels,
        shade_spacing=args.shade_spacing,
    )

    out_path.write_text(cp)
    num_lines = cp.count("\n")
    print(f"wrote {num_lines} line(s) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())