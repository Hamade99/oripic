# Port of src/solver.js — constraint solver (SOLVER).
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constraints import CON
from . import conversion as conv
from .math2d import M
from .note import NOTE

X = conv.X

_flip = [[0, 1, 2], [0, 2, 1]]


class _SOLVER:
    @staticmethod
    def infer(typ, F, BI, BA):
        pairs = CON.type_F_2_pairs(typ, F)
        tup = []
        for x, y in pairs:
            a = BA[BI[M.encode_order_pair([x, y])]]
            tup.append(_flip[1 if y < x else 0][a])
        key = "".join(str(t) for t in tup)
        I = CON.implied[typ].get(key)
        if not isinstance(I, list):
            return I
        out = []
        for idx, a in I:
            x, y = pairs[idx]
            bi = BI[M.encode_order_pair([x, y])]
            out.append([bi, _flip[1 if y < x else 0][a]])
        return out

    @staticmethod
    def propagate(bi, a, BI, BF, BT, BA, FC, CF, CC):
        B = [bi]
        BA[bi] = a
        idx = 0
        while idx < len(B):
            i = B[idx]
            f1, f2 = M.decode(BF[i])
            C = BT[i]
            for typ in CON.types:
                TF = list(_SOLVER.unpack_cons(C, typ, f1, f2, FC, CF, CC, None))
                for F in TF:
                    I = _SOLVER.infer(typ, F, BI, BA)
                    if I == CON.state.conflict:
                        for j in B:
                            BA[j] = 0
                        return []
                    if not isinstance(I, list):
                        continue
                    conflict = False
                    for j, s in I:
                        if BA[j] == 0:
                            B.append(j)
                            BA[j] = s
                        elif BA[j] != s:
                            conflict = True
                            break
                    if conflict:
                        for j in B:
                            BA[j] = 0
                        return []
                if typ == CON.T.transitivity:
                    TF.clear()
            idx += 1
        return B

    @staticmethod
    def get_components(BI, BF, BT, BA, FC, CF, CC, trans_count):
        B0 = [i for i, _ in enumerate(BA) if BA[i] != 0]
        GB = []
        seen = set()
        NOTE.start_check("variable", BF)
        for bi, a in enumerate(BA):
            if bi not in seen and a == 0:
                stack = [bi]
                seen.add(bi)
                si = 0
                while si < len(stack):
                    bi_ = stack[si]
                    C = BT[bi_]
                    f1, f2 = M.decode(BF[bi_])
                    for typ in CON.types:
                        TF = list(
                            _SOLVER.unpack_cons(C, typ, f1, f2, FC, CF, CC, trans_count)
                        )
                        for F in TF:
                            I = _SOLVER.infer(typ, F, BI, BA)
                            if I == CON.state.dead:
                                continue
                            for k__ in (
                                M.encode_order_pair(p) for p in CON.type_F_2_pairs(typ, F)
                            ):
                                bi__ = BI[k__]
                                if bi__ not in seen and BA[bi__] == 0:
                                    stack.append(bi__)
                                    seen.add(bi__)
                                NOTE.check(len(seen))
                        if typ == CON.T.transitivity:
                            TF.clear()
                    si += 1
                GB.append(stack)
        GB.sort(key=len)
        GB.insert(0, B0)
        return GB

    @staticmethod
    def unpack_cons(C, typ, f1, f2, FC, CF, CC, trans_count):
        if typ == CON.T.transitivity:
            return [
                [f1, f2, f3]
                for f3 in X.FC_CF_CC_Bf_2_Bt3(FC, CF, CC, [f1, f2], trans_count)
            ]
        return [M.decode(k) for k in C[typ]]

    @staticmethod
    def guess_vars(G, BI, BF, BT, BA, FC, CF, CC, lim):
        guesses = []
        A = []
        sol = [0] * len(G)
        idx = 0
        backtracking = False
        NOTE.start_check("state")
        while True:
            NOTE.check(len(A))
            for i in range(idx):
                if BA[G[i]] == 0:
                    raise RuntimeError("guess_vars invariant")
            if backtracking:
                if len(guesses) == 0:
                    break
                guess = guesses.pop()
                a = BA[guess[0]]
                while G[idx] != guess[0]:
                    idx -= 1
                    if idx < 0:
                        raise RuntimeError("guess_vars idx")
                for i in guess:
                    BA[i] = 0
                if a == 1:
                    B = _SOLVER.propagate(G[idx], 2, BI, BF, BT, BA, FC, CF, CC)
                    if len(B) > 0:
                        guesses.append(B)
                        backtracking = False
                        idx += 1
                    else:
                        guesses.append([G[idx]])
                        BA[G[idx]] = 2
            else:
                if idx == len(G):
                    for i, gi in enumerate(G):
                        sol[i] = BA[gi]
                    A.append(M.bit_encode(sol))
                    if len(A) >= lim:
                        return A
                    backtracking = True
                else:
                    if BA[G[idx]] == 0:
                        B = _SOLVER.propagate(G[idx], 1, BI, BF, BT, BA, FC, CF, CC)
                        if len(B) > 0:
                            guesses.append(B)
                        else:
                            guesses.append([G[idx]])
                            BA[G[idx]] = 1
                            backtracking = True
                idx += 1
        return A

    @staticmethod
    def EF_EA_Ff_BF_BI_2_BA0(EF, EA, Ff, BF, BI):
        BA0 = [0] * len(BF)
        for i, a in enumerate(EA):
            if a in ("M", "V"):
                k = M.encode_order_pair(EF[i])
                f1, f2 = M.decode(k)
                o = (
                    2
                    if ((not Ff[f1] and a == "M") or (Ff[f1] and a == "V"))
                    else 1
                )
                BA0[BI[k]] = o
        return BA0

    @staticmethod
    def initial_assignment(BA, BF, BT, BI, FC, CF, CC, trans_count):
        BP: List[Any] = [None] * len(BA)
        level = {i: a for i, a in enumerate(BA) if a != 0}
        new_level: Dict[int, int] = {}
        for i in level:
            BP[i] = []
        count, depth = 0, 0
        NOTE.start_check("variable", BA)
        while len(level) > 0:
            if NOTE.show:
                NOTE.log(f"   - {len(level)} orders assigned at depth {depth}")
            for i, a in level.items():
                BA[i] = a
            new_level.clear()
            for i, a in level.items():
                NOTE.check(count)
                count += 1
                f1, f2 = M.decode(BF[i])
                C = BT[i]
                for typ in CON.types:
                    TF = list(
                        _SOLVER.unpack_cons(C, typ, f1, f2, FC, CF, CC, trans_count)
                    )
                    for ci, F in enumerate(TF):
                        I = _SOLVER.infer(typ, F, BI, BA)
                        if I == CON.state.conflict:
                            E = _SOLVER.error_faces(typ, F, BF, BT, BI, BA, BP, FC, CF, CC)
                            return [typ, F, E]
                        if I == CON.state.alive or I == CON.state.dead:
                            continue
                        for i_, a_ in I:
                            a__ = new_level.get(i_)
                            if a__ is None:
                                BP[i_] = [typ, i, ci]
                                new_level[i_] = a_
                            elif a_ != a__:
                                E = _SOLVER.error_faces(typ, F, BF, BT, BI, BA, BP, FC, CF, CC)
                                return [typ, F, E]
                    if typ == CON.T.transitivity:
                        TF.clear()
            level, new_level = new_level, level
            depth += 1
        return BA

    @staticmethod
    def solve(BI, BF, BT, BA, GB, FC, CF, CC, lim):
        B0 = [i for i, _ in enumerate(BA) if BA[i] != 0]
        GA = [[M.bit_encode([BA[i] for i in B0])]]
        for i, B in enumerate(GB):
            if i == 0:
                continue
            NOTE.time(f"Solving component {i}/{len(GB) - 1} with size {len(B)}")
            A = _SOLVER.guess_vars(B, BI, BF, BT, BA, FC, CF, CC, lim)
            NOTE.count(len(A), "assignments")
            if len(A) == 0:
                return i
            GA.append(A)
        return GA

    @staticmethod
    def error_faces(typ, F, BF, BT, BI, BA, BP, FC, CF, CC):
        stack = [BI[M.encode_order_pair(pair)] for pair in CON.type_F_2_pairs(typ, F)]
        seen = set()
        while stack:
            i = stack.pop()
            seen.add(i)
            a = BA[i]
            if a != 0:
                par = BP[i]
                if par:
                    ptyp, bi, pci = par
                    f1, f2 = M.decode(BF[bi])
                    F2 = _SOLVER.unpack_cons(BT[bi], ptyp, f1, f2, FC, CF, CC, None)[pci]
                    for pair in CON.type_F_2_pairs(ptyp, F2):
                        i_ = BI[M.encode_order_pair(pair)]
                        if i_ not in seen:
                            stack.append(i_)
        E_set = set()
        for i in seen:
            for f in M.decode(BF[i]):
                E_set.add(f)
        return sorted(E_set)


SOLVER = _SOLVER
