# Port of src/constraints.js
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List


class _CON_cls:
    T = SimpleNamespace(
        taco_taco=0,
        taco_tortilla=1,
        tortilla_tortilla=2,
        transitivity=3,
    )
    names = ["taco-taco", "taco-tortilla", "tortilla-tortilla", "transitivity"]
    types = [0, 1, 2, 3]

    @staticmethod
    def pair_maps_0(q):
        A, B, C, D = q
        return [[A, B], [C, D], [C, B], [A, D], [A, C], [B, D]]

    @staticmethod
    def pair_maps_1(q):
        A, B, C = q
        return [[A, C], [C, B]]

    @staticmethod
    def pair_maps_2(q):
        A, B, C, D = q
        return [[A, C], [B, D]]

    @staticmethod
    def pair_maps_3(q):
        A, B, C = q
        return [[A, B], [B, C], [C, A]]

    @classmethod
    def type_F_2_pairs(cls, type_: int, F):
        # Call via cls — do not store staticmethods in a list (they are not callable).
        fns = (cls.pair_maps_0, cls.pair_maps_1, cls.pair_maps_2, cls.pair_maps_3)
        return fns[type_](F)

    valid = [
        [
            "111112",
            "111121",
            "111222",
            "112111",
            "121112",
            "121222",
            "122111",
            "122212",
            "211121",
            "211222",
            "212111",
            "212221",
            "221222",
            "222111",
            "222212",
            "222221",
        ],
        ["12", "21"],
        ["11", "22"],
        ["112", "121", "122", "211", "212", "221"],
    ]
    implied: List[Dict[str, Any]] = []
    state = SimpleNamespace(conflict=0, alive=1, dead=2)

    @classmethod
    def build(cls) -> None:
        cls.implied = []
        for _ in cls.types:
            cls.implied.append({})
        for type_ in cls.types:
            vvalid = cls.valid[type_]
            n = len(vvalid[0])
            I: List[Dict[str, Any]] = [{} for _ in range(n + 1)]
            for i in range(3**n):
                k, num_zeros = i, 0
                A: List[int] = []
                for j in range(n):
                    val = k % 3
                    num_zeros += 1 if val == 0 else 0
                    A.append(val)
                    k = (k - A[j]) // 3
                I[num_zeros]["".join(str(x) for x in A)] = cls.state.conflict
            for kk in vvalid:
                I[0][kk] = cls.state.dead
            for ii in range(1, n + 1):
                for kstr in list(I[ii].keys()):
                    A = list(kstr)
                    implied_list: List[List[int]] = []
                    conflict, dead = True, True
                    for j in range(n):
                        if A[j] != "0":
                            continue
                        possible = 0
                        for c in ("1", "2"):
                            A[j] = c
                            st = I[ii - 1].get("".join(A))
                            if st != cls.state.dead:
                                dead = False
                            if st != cls.state.conflict:
                                possible |= int(c)
                        A[j] = "0"
                        if possible:
                            conflict = False
                            if possible < 3:
                                implied_list.append([j, possible])
                    st = (
                        cls.state.conflict
                        if conflict
                        else (
                            implied_list
                            if implied_list
                            else (cls.state.dead if dead else cls.state.alive)
                        )
                    )
                    I[ii][kstr] = st
            for ii in range(n, -1, -1):
                for k, v in I[ii].items():
                    cls.implied[type_][k] = v


CON = _CON_cls
