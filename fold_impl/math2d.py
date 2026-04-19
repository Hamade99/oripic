# 2D geometry for flat folding. Scalar ops stay as tuples (tight loops in conversion);
# bulk ops use NumPy where it wins (areas, boxes, batches of points).
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Line = Tuple[Point, Point]

# float64: stable for crease-pattern coordinates; avoids extra casts in hot paths
_DT = np.float64


def _to_xy(P: Sequence[Point]) -> np.ndarray:
    """(n, 2) float64; zero-copy when P is already a contiguous (n,2) float array."""
    if isinstance(P, np.ndarray) and P.dtype == _DT and P.ndim == 2 and P.shape[1] == 2:
        return P
    return np.asarray(P, dtype=_DT).reshape(-1, 2)


class _M:
    EPS = 300
    FLOAT_EPS = 10 ** (-16)

    @staticmethod
    def near_zero(a: float) -> bool:
        return abs(a) < _M.FLOAT_EPS

    @staticmethod
    def encode(A: Iterable[int]) -> str:
        B: List[str] = []
        for a in A:
            if a >= 0x8000:
                if a >= 0x80000000:
                    raise ValueError("Integers must be < 2^31 for encoding")
                B.append(chr(0x8000 + (a >> 16)))
            B.append(chr(a & 0xFFFF))
        return "".join(B)

    @staticmethod
    def encode_order_pair(t: Tuple[int, int]) -> str:
        a, b = t
        return _M.encode((a, b) if a < b else (b, a))

    @staticmethod
    def decode(S: str) -> List[int]:
        B: List[int] = []
        i = 0
        n = len(S)
        while i < n:
            a = ord(S[i])
            if a >= 0x8000:
                i += 1
                if i >= n:
                    break
                a = ((a - 0x8000) << 16) + ord(S[i])
            B.append(a)
            i += 1
        return B

    @staticmethod
    def expand(F: Sequence[int], V: Sequence[Point]) -> List[Point]:
        if not F:
            return []
        V_ar = np.asarray(V, dtype=_DT).reshape(-1, 2)
        idx = np.asarray(F, dtype=np.intp)
        return list(map(tuple, V_ar[idx]))

    @staticmethod
    def mul(v: Point, s: float) -> Point:
        return (s * v[0], s * v[1])

    @staticmethod
    def div(v: Point, s: float) -> Point:
        return _M.mul(v, 1.0 / s)

    @staticmethod
    def add(a: Point, b: Point) -> Point:
        return (a[0] + b[0], a[1] + b[1])

    @staticmethod
    def sub(a: Point, b: Point) -> Point:
        return (a[0] - b[0], a[1] - b[1])

    @staticmethod
    def dot(a: Point, b: Point) -> float:
        return a[0] * b[0] + a[1] * b[1]

    @staticmethod
    def magsq(v: Point) -> float:
        return _M.dot(v, v)

    @staticmethod
    def mag(v: Point) -> float:
        return math.sqrt(_M.magsq(v))

    @staticmethod
    def unit(v: Point) -> Point:
        m = _M.mag(v)
        return (v[0] / m, v[1] / m)

    @staticmethod
    def perp(v: Point) -> Point:
        return (v[1], -v[0])

    @staticmethod
    def refX(v: Point) -> Point:
        return (-v[0], v[1])

    @staticmethod
    def refY(v: Point) -> Point:
        return (v[0], -v[1])

    @staticmethod
    def distsq(v1: Point, v2: Point) -> float:
        return _M.magsq(_M.sub(v2, v1))

    @staticmethod
    def dist(v1: Point, v2: Point) -> float:
        return _M.mag(_M.sub(v2, v1))

    @staticmethod
    def close(v1: Point, v2: Point, eps: float) -> bool:
        return abs(v1[0] - v2[0]) < eps and abs(v1[1] - v2[1]) < eps

    @staticmethod
    def area2(p1: Point, p2: Point, p3: Point) -> float:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    @staticmethod
    def angle(v: Point) -> float:
        ang = math.atan2(v[1], v[0])
        return ang + (2 * math.pi if ang < 0 else 0)

    @staticmethod
    def rotate_cos_sin(v: Point, c: float, s: float) -> Point:
        x, y = v
        return (x * c - y * s, x * s + y * c)

    @staticmethod
    def centroid(P: Sequence[Point]) -> Point:
        G = _to_xy(P).mean(axis=0)
        return (float(G[0]), float(G[1]))

    @staticmethod
    def previous_in_list(A: Sequence, v):
        for i, x in enumerate(A):
            if x == v:
                return A[-1] if i == 0 else A[i - 1]
        return None

    @staticmethod
    def min_line_length(lines) -> float:
        """Each line may be [p, q] or [p, q, assignment] like CP_2_L output."""
        if not lines:
            return 0.0
        n = len(lines)
        p0 = np.empty(n, dtype=_DT)
        p1 = np.empty(n, dtype=_DT)
        q0 = np.empty(n, dtype=_DT)
        q1 = np.empty(n, dtype=_DT)
        for i, ln in enumerate(lines):
            a, b = ln[0], ln[1]
            p0[i], p1[i] = a[0], a[1]
            q0[i], q1[i] = b[0], b[1]
        dx, dy = q0 - p0, q1 - p1
        return float(np.sqrt(np.min(dx * dx + dy * dy)))

    @staticmethod
    def sort_faces(FV: List[List[int]], V: Sequence[Point]) -> None:
        V_ar = np.asarray(V, dtype=_DT).reshape(-1, 2)

        def face_area2(f: List[int]) -> float:
            if not f:
                return 0.0
            P = V_ar[np.asarray(f, dtype=np.intp)]
            x, y = P[:, 0], P[:, 1]
            # Match legacy shoelace: sum (x[i-1]+x[i])*(y[i]-y[i-1])
            return float(np.sum((np.roll(x, 1) + x) * (y - np.roll(y, 1))))

        FV.sort(key=lambda f: -face_area2(f))

    @staticmethod
    def image(F: Sequence[Point], F_: Sequence[Point], P: Sequence[Point]) -> List[Point]:
        longest = 0.0
        i1 = i2 = 0
        for i in range(len(F)):
            j = (i + 1) % len(F)
            d = _M.distsq(F[i], F[j])
            if d > longest:
                longest = d
                i1, i2 = i, j
        x = np.array(_M.unit(_M.sub(F[i2], F[i1])), dtype=_DT)
        y = np.array(_M.perp(tuple(x)), dtype=_DT)
        x_ = np.array(
            _M.mul(
                _M.unit(_M.sub(F_[i2], F_[i1])),
                _M.dist(F_[i2], F_[i1]) / _M.dist(F[i2], F[i1]),
            ),
            dtype=_DT,
        )
        same_winding = (_M.polygon_area2(F) < 0) == (_M.polygon_area2(F_) < 0)
        y_ = np.array(_M.perp(tuple(x_)) if same_winding else _M.mul(tuple(x_), -1), dtype=_DT)
        P_arr = _to_xy(P)
        origin = np.array(F[i1], dtype=_DT)
        F1_ = np.array(F_[i1], dtype=_DT)
        Pm = P_arr - origin
        proj_x = Pm @ x
        proj_y = Pm @ y
        # Legacy JS also multiplies proj_y by a parity flag `s`; fold path never calls `image`.
        out = proj_x[:, None] * x_[None, :] + proj_y[:, None] * y_[None, :] + F1_
        return list(map(tuple, out))

    @staticmethod
    def bounding_box(P: Sequence[Point]) -> Tuple[Point, Point]:
        if not P:
            return (0.0, 0.0), (0.0, 0.0)
        A = _to_xy(P)
        mn = A.min(axis=0)
        mx = A.max(axis=0)
        return (float(mn[0]), float(mn[1])), (float(mx[0]), float(mx[1]))

    @staticmethod
    def center_points_on(P: Sequence[Point], c: Point) -> List[Point]:
        A = _to_xy(P)
        p_min = A.min(axis=0)
        p_max = A.max(axis=0)
        off = np.array(c, dtype=_DT) - (p_min + p_max) / 2.0
        return list(map(tuple, A + off))

    @staticmethod
    def normalize_points_xy(P: Sequence[Point]) -> np.ndarray:
        """Same layout as normalize_points but (n, 2) float64 — no tuple list."""
        A = _to_xy(P)
        p_min = A.min(axis=0)
        p_max = A.max(axis=0)
        x_diff, y_diff = p_max - p_min
        is_tall = x_diff < y_diff
        diff = y_diff if is_tall else x_diff
        if diff == 0.0:
            diff = 1.0
        off = np.array((0.5, 0.5), dtype=_DT) - np.array((x_diff, y_diff), dtype=_DT) / (
            2.0 * diff
        )
        return (A - p_min) / diff + off

    @staticmethod
    def normalize_points(P: Sequence[Point]) -> List[Point]:
        Q = _M.normalize_points_xy(P)
        return list(map(tuple, Q))

    @staticmethod
    def interior_point(P_: Sequence[Point]) -> Point:
        P = list(P_)
        if _M.polygon_area2(P_) < 0:
            P.reverse()
        n = len(P)
        largest_ear = None
        max_area = float("-inf")
        p1, p2 = P[n - 2], P[n - 1]
        for p3 in P:
            a = _M.area2(p1, p2, p3)
            if a <= 0:
                p1, p2 = p2, p3
                continue
            found = True
            for p in P:
                if p is not p1 and p is not p2 and p is not p3:
                    if (
                        _M.area2(p1, p2, p) >= 0
                        and _M.area2(p2, p3, p) >= 0
                        and _M.area2(p3, p1, p) >= 0
                    ):
                        found = False
                        break
            if found:
                if max_area < a:
                    max_area = a
                    largest_ear = [p1, p2, p3]
            p1, p2 = p2, p3
        if largest_ear is None:
            raise RuntimeError("interior_point: no ear found")
        return _M.centroid(largest_ear)

    @staticmethod
    def on_segment(a: Point, b: Point, c: Point, eps: float) -> bool:
        v = _M.sub(b, a)
        pa, pb, pc = (_M.dot(a, v), _M.dot(b, v), _M.dot(c, v))
        if (pc < pa) == (pc < pb):
            return False
        d = _M.dot(_M.unit(_M.perp(v)), _M.sub(c, a))
        return abs(d) <= eps

    @staticmethod
    def polygon_area2(P: Sequence[Point]) -> float:
        if not P:
            return 0.0
        A = _to_xy(P)
        if A.shape[0] < 2:
            return 0.0
        x, y = A[:, 0], A[:, 1]
        return float(np.sum((np.roll(x, 1) + x) * (y - np.roll(y, 1))))

    @staticmethod
    def intersect(seg1: Line, seg2: Line, eps: float):
        a, b = seg1
        c, d = seg2
        if (
            _M.close(a, c, eps)
            or _M.close(a, d, eps)
            or _M.close(b, c, eps)
            or _M.close(b, d, eps)
            or _M.on_segment(a, b, c, eps)
            or _M.on_segment(a, b, d, eps)
            or _M.on_segment(c, d, a, eps)
            or _M.on_segment(c, d, b, eps)
        ):
            return None
        denom = (
            a[0] * (d[1] - c[1])
            + b[0] * (c[1] - d[1])
            + d[0] * (b[1] - a[1])
            + c[0] * (a[1] - b[1])
        )
        if _M.near_zero(denom):
            return None
        s_num = a[0] * (d[1] - c[1]) + c[0] * (a[1] - d[1]) + d[0] * (c[1] - a[1])
        if _M.near_zero(s_num) or _M.near_zero(s_num - denom):
            return None
        t_num = -(
            a[0] * (c[1] - b[1]) + b[0] * (a[1] - c[1]) + c[0] * (b[1] - a[1])
        )
        if _M.near_zero(t_num) or _M.near_zero(t_num - denom):
            return None
        s = s_num / denom
        t = t_num / denom
        if s < 0 or 1 < s or t < 0 or 1 < t:
            return None
        p = (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]))
        if (
            _M.close(a, p, eps)
            or _M.close(b, p, eps)
            or _M.close(c, p, eps)
            or _M.close(d, p, eps)
        ):
            return None
        return p

    @staticmethod
    def bit_encode(A: Sequence[int]) -> str:
        B: List[int] = []
        i = 0
        while i < len(A):
            bite = 0
            for j in range(8):
                if i + j < len(A):
                    b = A[i + j] - 1
                    bite |= b << j
            B.append(bite)
            i += 8
        return _M.encode(B)

    @staticmethod
    def bit_decode(B: str, n: int) -> List[int]:
        if n == 0:
            return []
        A: List[int] = []
        for bite in _M.decode(B):
            for j in range(8):
                A.append(((bite >> j) & 1) + 1)
                if len(A) == n:
                    return A
        raise RuntimeError("bit_decode: input shorter than requested length")


M = _M()
