# Port of src/avl.js — AVL tree used by line arrangement.
from __future__ import annotations

from typing import Any, Callable, List, Optional

_L, _R, _X, _H, _N = 0, 1, 2, 3, 4


class AVL:
    __slots__ = ("_comp", "_root", "_free", "_n", "_A")

    def __init__(self, comp: Callable[[Any, Any], int] = lambda a, b: a - b) -> None:
        self._comp = comp
        self._root: Optional[int] = None
        self._free: Optional[int] = None
        self._n = 0
        self._A: List[Any] = []

    def _get(self, off: int, i: int) -> Any:
        return self._A[i * _N + off]

    def _set(self, off: int, i: int, v: Any = None) -> None:
        self._A[i * _N + off] = v

    def _H(self, i: Optional[int]) -> int:
        return 0 if i is None else self._get(_H, i)

    def _skew(self, i: int) -> int:
        return self._H(self._get(_R, i)) - self._H(self._get(_L, i))

    def _update(self, i: int) -> None:
        hL = self._H(self._get(_L, i))
        hR = self._H(self._get(_R, i))
        self._set(_H, i, 1 + (hR if hL < hR else hL))

    def _obtain(self) -> int:
        self._n += 1
        i = len(self._A) // _N
        if self._free is not None:
            i = self._free
            self._free = self._get(_R, i)
            self._set(_R, i)
        else:
            self._A.extend([None] * _N)
        return i

    def _release(self, i: int) -> None:
        self._n -= 1
        for k in range(_N):
            self._set(k, i)
        self._set(_R, i, self._free)
        self._free = i

    def _rotate(self, D: int, r: int, l: int) -> None:
        B = self._get(l, D)
        E = self._get(r, D)
        d = self._get(_X, D)
        A = self._get(l, B)
        C = self._get(r, B)
        b = self._get(_X, B)
        self._set(l, B, C)
        self._set(r, B, E)
        self._set(_X, B, d)
        self._set(l, D, A)
        self._set(r, D, B)
        self._set(_X, D, b)
        self._update(B)
        self._update(D)

    def _maintain(self, P: List[int]) -> None:
        while P:
            i = P.pop()
            self._update(i)
            s = self._skew(i)
            for t, r, l in ((1, _R, _L), (-1, _L, _R)):
                if s != 2 * t:
                    continue
                j = self._get(r, i)
                if self._skew(j) == -t:
                    self._rotate(j, r, l)
                self._rotate(i, l, r)

    def _path(self, x: Any) -> List[int]:
        if self._root is None:
            return []
        P: List[int] = []
        i = self._root
        while True:
            P.append(i)
            c = self._comp(x, self._get(_X, i))
            nxt = self._get(_L if c < 0 else _R, i)
            if c == 0 or nxt is None:
                break
            i = nxt
        return P

    def _adj(self, P: List[int], r: int) -> None:
        i = P[-1]
        j = self._get(r, i)
        if j is None:
            P.pop()
            while P:
                p = P.pop()
                if self._get(r, p) != i:
                    P.append(p)
                    break
                i = p
        else:
            while j is not None:
                P.append(j)
                j = self._get(r ^ 1, j)

    def _seq(self) -> List[int]:
        out: List[int] = []

        def dfs(i: Optional[int]) -> None:
            if i is None:
                return
            dfs(self._get(_L, i))
            out.append(i)
            dfs(self._get(_R, i))

        dfs(self._root)
        return out

    @property
    def length(self) -> int:
        return self._n

    def insert(self, x: Any) -> Any:
        i = self._obtain()
        P = self._path(x)
        if not P:
            self._root = i
        else:
            p = P[-1]
            x_ = self._get(_X, p)
            c = self._comp(x, x_)
            if c == 0:
                self._release(i)
                return x_
            self._set(_L if c < 0 else _R, p, i)
        self._set(_X, i, x)
        P.append(i)
        self._maintain(P)
        return None

    def _remove(self, P: List[int]) -> None:
        i = P[-1]
        r = _L
        c = self._get(_L, i)
        if c is None:
            r = _R
            c = self._get(_R, i)
        while c is not None:
            while c is not None:
                P.append(c)
                c = self._get(r ^ 1, c)
            c = P[-1]
            self._set(_X, i, self._get(_X, c))
            i = c
            c = self._get(r, i)
        P.pop()
        self._release(i)
        p = P[-1] if P else None
        if p is None:
            self._root = None
        else:
            self._set(_L if self._get(_L, p) == i else _R, p)
            self._maintain(P)

    def remove(self, x: Any) -> Any:
        if self._root is None:
            return None
        P = self._path(x)
        i = P[-1]
        x_ = self._get(_X, i)
        if self._comp(x, x_) != 0:
            return None
        self._remove(P)
        return x_

    def next(self, x: Any, r: int = _R) -> Any:
        if self._root is None:
            return None
        P = self._path(x)
        i = P[-1]
        c = self._comp(x, self._get(_X, i))
        if c != 0 and ((r == _R) == (c < 0)):
            return self._get(_X, i)
        self._adj(P, r)
        i = P.pop() if P else None
        return None if i is None else self._get(_X, i)

    def prev(self, x: Any) -> Any:
        return self.next(x, _L)

    def remove_next(self, x: Any, r: int = _R) -> Any:
        if self._root is None:
            return None
        P = self._path(x)
        i = P[-1]
        c = self._comp(x, self._get(_X, i))
        if not (c != 0 and ((r == _R) == (c < 0))):
            self._adj(P, r)
            i = P.pop() if P else None
            if i is None:
                return None
            P.append(i)
        x_ = self._get(_X, i)
        self._remove(P)
        return x_

    def remove_prev(self, x: Any) -> Any:
        return self.remove_next(x, _L)

    def iter(self) -> List[Any]:
        return [self._get(_X, i) for i in self._seq()]
