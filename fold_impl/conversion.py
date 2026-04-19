# Port of src/conversion.js — flat-fold geometry (X).
from __future__ import annotations

import math
from functools import cmp_to_key
from typing import Any, Dict, List, Optional, Set, Tuple

from .avl import AVL
from .constraints import CON
from .math2d import M
from .note import NOTE


class _X:
    @staticmethod
    def L_2_V_EV_EL(L):
        d = M.min_line_length(L)
        k = 3
        N = 25
        k_, i_, nV, nE, count = 0, 1, 0, 0, 0
        for i in range(3, N + 3):
            eps = d / (2**i)
            if eps < M.FLOAT_EPS:
                break
            V, EV, EL = _X.L_eps_2_V_EV_EL(L, eps)
            if len(V) == 0:
                nV = nE = count = 0
                continue
            count = count + 1 if (len(V) == nV and len(EV) == nE) else 1
            nV, nE = len(V), len(EV)
            if count <= k_:
                continue
            k_, i_ = count, i
            if k_ == k:
                break
        eps_i = i_ - k_ + 1
        eps = d / (2**eps_i)
        V, EV, EL = _X.L_eps_2_V_EV_EL(L, eps)
        return [V, EV, EL, eps_i]

    @staticmethod
    def L_eps_2_V_EV_EL(L, eps):
        def point_comp(p, q):
            x1, y1 = p
            x2, y2 = q
            dx, dy = x1 - x2, y1 - y2
            if abs(dy) > eps:
                return -1 if dy < 0 else 1
            if abs(dx) > eps:
                return -1 if dx < 0 else 1
            return 0

        def line_intersect(a, b, c, d_):
            x1, y1 = a
            x2, y2 = b
            x3, y3 = c
            x4, y4 = d_
            dx12, dx34 = x1 - x2, x3 - x4
            dy12, dy34 = y1 - y2, y3 - y4
            denom = dx12 * dy34 - dx34 * dy12
            if abs(denom) < eps * eps:
                return None
            x = ((x1 * y2 - y1 * x2) * dx34 - (x3 * y4 - y3 * x4) * dx12) / denom
            y = ((x1 * y2 - y1 * x2) * dy34 - (x3 * y4 - y3 * x4) * dy12) / denom
            return (x, y)

        V = [[float("-inf"), float("-inf")]]
        VL: List[List[int]] = [[]]
        LV: List[List[int]] = []
        LU = []
        LA = []
        LD = []
        Q = AVL(lambda vi, vj: point_comp(V[vi], V[vj]))
        for li, ln in enumerate(L):
            p, q = ln[0], ln[1]
            if point_comp(p, q) > 0:
                p, q = q, p

            def map_pt(v):
                vn = len(V)
                V.append(v)
                VL.append([])
                j = Q.insert(vn)
                if j is None:
                    return vn
                V.pop()
                VL.pop()
                return j

            vi, vj = map_pt(p), map_pt(q)
            LV.append([vi, vj])
            dvec = M.sub(V[vj], V[vi])
            u = M.unit(dvec)
            LU.append(u)
            LA.append(0 if dvec[1] < eps else M.angle(dvec))
            LD.append(M.dot(M.perp(u), V[vj]))
            VL[vi].append(li)

        bb = M.bounding_box(V[1:])
        x_min, y_min = bb[0]
        x_max, y_max = bb[1]
        HEIGHT, WIDTH = y_max - y_min, x_max - x_min
        SCALE = WIDTH if HEIGHT < WIDTH else HEIGHT
        SV = [[None, None]]
        SU = [[-1, 0]]
        SA = [float("inf")]
        SD = [None]
        SL: List[List[int]] = [[]]

        def point_seg_dist(vi, si):
            return SD[si] - M.dot(M.perp(SU[si]), V[vi])

        def on_line(vi, si):
            return abs(point_seg_dist(vi, si)) < eps

        curr_ref = [0]

        def seg_comp(si, sj):
            dj = point_seg_dist(curr_ref[0], sj)
            if abs(dj) < eps:
                pi = SV[si][1]
                if pi is not None and on_line(pi, sj):
                    return 0
                return 1 if SA[sj] - SA[si] > 0 else -1
            return 1 if (-dj) > 0 else -1

        T = AVL(seg_comp)
        VP: Dict[int, int] = {}
        P: List = []
        while Q.length > 0:
            vi = Q.remove_next(0)
            curr_ref[0] = vi
            v = V[vi]
            S1 = []
            SV[0][0] = vi
            SD[0] = M.dot(M.perp(SU[0]), v)
            pr = T.prev(0)
            if pr is not None and SA[pr] == 0:
                S1.append(T.remove_prev(0))
            while True:
                si = T.next(0)
                if si is None or not on_line(vi, si):
                    break
                S1.append(T.remove_next(0))
            if len(VL[vi]) == 0 and len(S1) < 2:
                if len(S1) == 0:
                    continue
                si = S1[0]
                ends = False
                for li in SL[si]:
                    if point_comp(V[LV[li][1]], V[vi]) <= 0:
                        ends = True
                        break
                if not ends:
                    T.insert(si)
                    continue
            if len(S1) == 1:
                s0 = S1[0]
                all_parallel = True
                for ll in VL[vi]:
                    if not on_line(LV[ll][1], s0):
                        all_parallel = False
                        break
                if all_parallel:
                    T.insert(s0)
                    for ll in VL[vi]:
                        SL[s0].append(ll)
                    continue
            VP[vi] = len(P)
            P.append(v)
            for si in S1:
                SV[si][1] = vi
                for li in SL[si]:
                    if point_comp(V[LV[li][1]], V[vi]) <= 0:
                        continue
                    VL[vi].append(li)
            VL[vi].sort(
                key=lambda i: (M.distsq(v, V[LV[i][1]]), i)
            )
            for li in VL[vi]:
                si = len(SV)
                SV.append([vi, LV[li][1]])
                SU.append(LU[li])
                SA.append(LA[li])
                SD.append(LD[li])
                SL.append([li])
                sj = T.insert(si)
                if sj is not None:
                    SV.pop()
                    SU.pop()
                    SA.pop()
                    SD.pop()
                    SL.pop()
                    SL[sj].append(li)
                else:
                    SV[si][1] = None
            pairs = []
            SA[0] = -1
            pairs.append([T.prev(0), T.next(0)])
            SA[0] = float("inf")
            pairs.append([T.prev(0), T.next(0)])
            for l, r in pairs:
                if l is None or r is None:
                    continue
                vl, al = SV[l][0], SA[l]
                vr, ar = SV[r][0], SA[r]
                x = line_intersect(
                    V[vl],
                    M.add(V[vl], M.mul(SU[l], SCALE)),
                    V[vr],
                    M.add(V[vr], M.mul(SU[r], SCALE)),
                )
                if x is None:
                    continue
                c = point_comp(x, v)
                if c == 0:
                    continue
                if c < 0 and (x[1] - v[1]) < eps:
                    continue
                V.append(list(x))
                vx = len(V) - 1
                vj = Q.insert(vx)
                if vj is not None:
                    V.pop()
                else:
                    VL.append([])

        if T.length != 0:
            return [[], [], []]

        X_list = []
        X_map: Dict[str, int] = {}
        for si in range(1, len(SV)):
            pp = [VP[SV[si][0]], VP[SV[si][1]]]
            if pp[1] < pp[0]:
                pp.reverse()
            k = M.encode(pp)
            xi = X_map.get(k)
            if xi is None:
                xi = len(X_list)
                X_list.append([pp, []])
                X_map[k] = xi
            X_list[xi][1].extend(SL[si])
        X_list.sort(key=lambda xi: (xi[0][0], xi[0][1]))
        XP = [xi[0] for xi in X_list]
        XL = [sorted(xi[1]) for xi in X_list]
        return [P, XP, XL]

    @staticmethod
    def V_EV_2_VV_FV(V, EV):
        adj = [[] for _ in V]
        for pi, qi in EV:
            adj[pi].append(qi)
            adj[qi].append(pi)
        VV = []
        for i, v0 in enumerate(V):
            A = [[vi, M.angle(M.sub(V[vi], v0))] for vi in adj[i]]
            A.sort(key=lambda t: t[1])
            VV.append([t[0] for t in A])
        FV = []
        seen: Set[str] = set()
        for v1, A in enumerate(VV):
            for v2 in A:
                key = M.encode([v1, v2])
                if key not in seen:
                    seen.add(key)
                    F = [v1]
                    i, j = v1, v2
                    while j != v1:
                        F.append(j)
                        i, j = j, M.previous_in_list(VV[j], i)
                        seen.add(M.encode([i, j]))
                    if len(F) > 2:
                        FV.append(F)
        M.sort_faces(FV, V)
        if FV:
            FV.pop()
        return [VV, FV]

    @staticmethod
    def V_FV_2_VV(V, FV):
        nxt = [{} for _ in V]
        prv = [{} for _ in V]
        for fi, face in enumerate(FV):
            v1, v2 = face[-2], face[-1]
            for v3 in face:
                nxt[v2][v1] = v3
                prv[v2][v3] = v1
                v1, v2 = v2, v3
        VV = [[] for _ in V]
        for i, _Adj in enumerate(VV):
            v0 = next(iter(nxt[i].keys()), None)
            v, v_ = v0, None
            v1 = prv[i].get(v0) if v0 is not None else None
            if v1 is not None:
                v, v_ = v1, prv[i].get(v)
                while v_ is not None and v_ != v0:
                    v, v_ = v_, prv[i].get(v_)
            start = v
            _Adj.append(start)
            v = nxt[i].get(v) if start is not None else None
            while v is not None and v != start:
                _Adj.append(v)
                v = nxt[i].get(v)
        return VV

    @staticmethod
    def V_VV_EV_EA_2_VK(V, VV, EV, EA):
        VVA_map = {}
        for i, (v1, v2) in enumerate(EV):
            a = EA[i]
            VVA_map[M.encode([v1, v2])] = a
            VVA_map[M.encode([v2, v1])] = a
        VK = []
        for i, A in enumerate(VV):
            adj = []
            boundary = False
            count_M = count_V = count_U = 0
            for j in A:
                a = VVA_map.get(M.encode([i, j]))
                if a == "B":
                    boundary = True
                    break
                if a in ("V", "M", "U"):
                    adj.append(j)
                if a == "M":
                    count_M += 1
                if a == "V":
                    count_V += 1
                if a == "U":
                    count_U += 1
            if boundary or len(adj) == 0:
                VK.append(0)
            elif ((count_U == 0 and abs(count_M - count_V) != 2) or (len(adj) % 2 != 0)):
                VK.append(1)
            else:
                angles = [M.angle(M.sub(V[j], V[i])) for j in adj]
                angles.sort()
                kawasaki = 0.0
                for j in range(0, len(angles), 2):
                    kawasaki += angles[j + 1] - angles[j]
                VK.append(abs(kawasaki - math.pi))
        return VK

    @staticmethod
    def V_FV_EV_EA_2_Vf_Ff(V, FV, EV, EA):
        EA_map = {}
        for i, vs in enumerate(EV):
            EA_map[M.encode_order_pair(vs)] = EA[i]
        EF_map = {}
        for i, F in enumerate(FV):
            for j, v1 in enumerate(F):
                v2 = F[(j + 1) % len(F)]
                EF_map[M.encode([v2, v1])] = i
        Vf = [None] * len(V)
        seen: Set[int] = set()
        v1, v2 = FV[0][0], FV[0][1]
        for i in (v1, v2):
            Vf[i] = tuple(V[i])
        Ff = [None] * len(FV)
        Q = [[0, v1, v2, float("inf"), True]]
        nxt = 0
        while nxt < len(Q):
            fi, i1, i2, l, s = Q[nxt]
            nxt += 1
            if fi in seen:
                continue
            seen.add(fi)
            Ff[fi] = not s
            F = FV[fi]
            x = M.unit(M.sub(V[i2], V[i1]))
            y = M.perp(x)
            xf = M.unit(M.sub(Vf[i2], Vf[i1]))
            yf = M.perp(xf)
            vi = F[-1]
            for vj in F:
                if Vf[vj] is None:
                    v = M.sub(V[vj], V[i1])
                    dx = M.mul(xf, M.dot(v, x))
                    dy = M.mul(yf, M.dot(v, y) * (1 if s else -1))
                    Vf[vj] = M.add(M.add(dx, dy), Vf[i1])
                ln = M.distsq(V[vi], V[vj])
                f = EF_map.get(M.encode([vi, vj]))
                a = EA_map.get(M.encode_order_pair([vi, vj]))
                new_s = (not s) if a in ("M", "V", "U") else s
                if f is not None and f not in seen:
                    Q.append([f, vi, vj, ln, new_s])
                    for i in range(len(Q) - 1, nxt, -1):
                        len_ = Q[i - 1][3]
                        if len_ < ln:
                            Q[i], Q[i - 1] = Q[i - 1], Q[i]
                        else:
                            break
                vi = vj
        for p in Vf:
            if p is None:
                raise RuntimeError("V_FV_EV_EA_2_Vf_Ff: incomplete Vf")
        return [Vf, Ff]

    @staticmethod
    def EV_FV_2_EF_FE(EV, FV):
        EV_map = {}
        for i, e in enumerate(EV):
            EV_map[M.encode(e)] = i
        EF = [[None, None] for _ in EV]
        for i, F in enumerate(FV):
            for j, v1 in enumerate(F):
                v2 = F[(j + 1) % len(F)]
                ei = EV_map.get(M.encode_order_pair([v1, v2]))
                c = 0 if v2 < v1 else 1
                EF[ei][c] = i
        for i, F in enumerate(EF):
            c = 1 if F[0] is None else (0 if F[1] is None else None)
            if c is not None:
                EF[i] = [F[c]]
        FE = []
        for face in FV:
            E = []
            v1 = face[0]
            for i in range(1, len(face)):
                v2 = face[i]
                E.append(EV_map[M.encode_order_pair([v1, v2])])
                v1 = v2
            E.append(EV_map[M.encode_order_pair([v1, face[0]])])
            FE.append(E)
        return [EF, FE]

    @staticmethod
    def f_FC_CF_2_fB_set(f, FC, CF):
        fB = set()
        for c in FC[f]:
            for ff in CF[c]:
                fB.add(ff)
        return fB

    @staticmethod
    def EF_FV_P_SP_SE_CP_SC_2_CF_FC(EF, FV, P, SP, SE, CP, SC):
        SF_map = {}
        for i, sP in enumerate(SP):
            sF = []
            for ei in SE[i]:
                for f in EF[ei]:
                    sF.append(f)
            SF_map[M.encode_order_pair(sP)] = sF
        SC_map = {}
        for i, cP in enumerate(CP):
            p1 = cP[-1]
            for p2 in cP:
                SC_map[M.encode([p2, p1])] = i
                p1 = p2
        CF = [[] for _ in CP]
        Q = []
        si = ci = None
        d = -1.0
        for i, sC in enumerate(SC):
            if len(sC) == 1:
                p1, p2 = SP[i]
                p1, p2 = P[p1], P[p2]
                d_ = M.distsq(p1, p2)
                if d_ > d:
                    si, ci, d = i, sC[0], d_
        CF[ci] = list(SF_map[M.encode_order_pair(SP[si])])
        Q.append([ci, d])
        nxt = 0
        seen: Set[int] = set()
        while nxt < len(Q):
            ci = Q[nxt][0]
            nxt += 1
            if ci in seen:
                continue
            seen.add(ci)
            cP = CP[ci]
            p1 = cP[-1]
            for p2 in cP:
                cj = SC_map.get(M.encode([p1, p2]))
                if cj is not None and cj not in seen:
                    d = M.distsq(P[p1], P[p2])
                    Q.append([cj, d])
                    for i in range(len(Q) - 2, nxt - 1, -1):
                        d_ = Q[i][1]
                        if d_ < d:
                            Q[i + 1], Q[i] = Q[i], Q[i + 1]
                    cF = set(CF[ci])
                    k = M.encode_order_pair([p1, p2])
                    for f in SF_map[k]:
                        if f in cF:
                            cF.remove(f)
                        else:
                            cF.add(f)
                    CF[cj] = list(cF)
                p1 = p2
        FC = [[] for _ in FV]
        for ci, cF in enumerate(CF):
            cF.sort()
            for f in cF:
                FC[f].append(ci)
        return [CF, FC]

    @staticmethod
    def CF_2_BF(CF):
        BF_set = set()
        NOTE.start_check("cell", CF)
        for i, F in enumerate(CF):
            NOTE.check(i)
            for j, f1 in enumerate(F):
                for k in range(j + 1, len(F)):
                    f2 = F[k]
                    BF_set.add(M.encode([f1, f2]))
        return sorted(BF_set)

    @staticmethod
    def EF_SP_SE_CP_CF_2_BF(EF, SP, SE, CP, CF):
        SF_map = {}
        for i, vs in enumerate(SP):
            Fs = []
            for ei in SE[i]:
                for f in EF[ei]:
                    Fs.append(f)
            SF_map[M.encode_order_pair(vs)] = Fs
        SC_map = {}
        for i, C in enumerate(CP):
            v1 = C[-1]
            for v2 in C:
                SC_map[M.encode([v2, v1])] = i
                v1 = v2
        BF_set = set()
        seen: Set[int] = set()
        Q = [0]
        F0 = CF[0]
        for j in range(1, len(F0)):
            for i in range(j):
                BF_set.add(M.encode_order_pair([F0[i], F0[j]]))
        nxt = 0
        CF_set = [set(f) for f in CF]
        while nxt < len(Q):
            ci = Q[nxt]
            nxt += 1
            C = CP[ci]
            v1 = C[-1]
            for v2 in C:
                cj = SC_map.get(M.encode([v1, v2]))
                if cj is not None and cj not in seen:
                    Q.append(cj)
                    seen.add(cj)
                    Fi_set = CF_set[ci]
                    Fj_set = CF_set[cj]
                    k = M.encode_order_pair([v1, v2])
                    for fi in SF_map[k]:
                        if fi in Fi_set or fi not in Fj_set:
                            continue
                        for fj in Fj_set:
                            if fi == fj:
                                continue
                            BF_set.add(M.encode_order_pair([fi, fj]))
                v1 = v2
        return sorted(BF_set)

    @staticmethod
    def check_overlap(p, BI):
        return 1 if M.encode_order_pair(p) in BI else 0

    @staticmethod
    def add_constraint(T, BI, BT):
        if T is None:
            return
        typ, F = T
        pairs = CON.type_F_2_pairs(typ, F)
        for p in pairs:
            i = BI[M.encode_order_pair(p)]
            BT[i][typ].append(M.encode(F))

    @staticmethod
    def ExE_fill_BT(BT, BI, EF, SE):
        ExE = [set() for _ in EF]
        for edges in SE:
            for j, v1 in enumerate(edges):
                for k in range(j + 1, len(edges)):
                    v2 = edges[k]
                    a, b = (v1, v2) if v1 < v2 else (v2, v1)
                    ExE[a].add(b)
        NOTE.start_check("edge", ExE)
        for e1, E in enumerate(ExE):
            NOTE.check(e1)
            for e2 in E:
                if len(EF[e1]) != 2 or len(EF[e2]) != 2:
                    continue
                f1, f2 = EF[e1]
                f3, f4 = EF[e2]
                f1f2 = _X.check_overlap([f1, f2], BI)
                f1f3 = _X.check_overlap([f1, f3], BI)
                f1f4 = _X.check_overlap([f1, f4], BI)
                cons = None
                choice = (f1f2 << 2) | (f1f3 << 1) | f1f4
                if choice == 4:
                    continue
                if choice == 0:
                    cons = (CON.T.taco_tortilla, [f3, f4, f2])
                elif choice == 1:
                    cons = (CON.T.tortilla_tortilla, [f1, f2, f4, f3])
                elif choice == 2:
                    cons = (CON.T.tortilla_tortilla, [f1, f2, f3, f4])
                elif choice == 3:
                    cons = (CON.T.taco_tortilla, [f3, f4, f1])
                elif choice == 5:
                    cons = (CON.T.taco_tortilla, [f1, f2, f4])
                elif choice == 6:
                    cons = (CON.T.taco_tortilla, [f1, f2, f3])
                elif choice == 7:
                    cons = (CON.T.taco_taco, [f1, f2, f3, f4])
                _X.add_constraint(cons, BI, BT)
            E.clear()
        del ExE[:]

    @staticmethod
    def ExF_fill_BT(BT, BI, EF, SE, CF, SC):
        ExF = [set() for _ in EF]
        for i, C in enumerate(SC):
            if len(C) == 2:
                E = SE[i]
                c1, c2 = C
                F = []
                F1 = set(CF[c1])
                for fi in CF[c2]:
                    if fi in F1:
                        F.append(fi)
                for ei in E:
                    for fi in F:
                        ExF[ei].add(fi)
        NOTE.start_check("edge", ExF)
        for e, F in enumerate(ExF):
            NOTE.check(e)
            for f3 in F:
                if len(EF[e]) != 2:
                    continue
                f1, f2 = EF[e]
                if f1 == f3 or f2 == f3:
                    continue
                f1f2 = _X.check_overlap([f1, f2], BI)
                if f1f2 == 1:
                    cons = (CON.T.taco_tortilla, [f1, f2, f3])
                else:
                    cons = (CON.T.tortilla_tortilla, [f1, f2, f3, f3])
                _X.add_constraint(cons, BI, BT)
            F.clear()
        del ExF[:]

    @staticmethod
    def BF_BI_EF_SE_CF_SC_2_BT(BF, BI, EF, SE, CF, SC):
        BT = [[[], [], []] for _ in BF]
        NOTE.time("Computing from edge-edge intersections")
        _X.ExE_fill_BT(BT, BI, EF, SE)
        NOTE.time("Computing from edge-face intersections")
        _X.ExF_fill_BT(BT, BI, EF, SE, CF, SC)
        return BT

    @staticmethod
    def FC_BF_BI_BT_2_CC(FC, BF, BI, BT):
        FG = [{} for _ in FC]
        NOTE.start_check("taco-tortilla", BT)
        for i, T in enumerate(BT):
            NOTE.check(i)
            for k in T[CON.T.taco_tortilla]:
                a, b, c = M.decode(k)
                G = FG[c]
                G.setdefault(a, set()).add(b)
                G.setdefault(b, set()).add(a)
        CC = []
        NOTE.start_check("face", FG)
        for c, G in enumerate(FG):
            NOTE.check(c)
            C = {}
            ci = 0
            for F in G:
                if F in C:
                    continue
                Q = [F]
                C[F] = ci
                ii = 0
                while ii < len(Q):
                    for f in G[Q[ii]]:
                        if f in C:
                            continue
                        Q.append(f)
                        C[f] = ci
                    ii += 1
                ci += 1
            CC.append(C)
        return CC

    @staticmethod
    def FC_CF_CC_Bf_2_Bt3(FC, CF, CC, f12, trans_count=None):
        f1, f2 = f12
        C = set(FC[f1])
        T = set()
        for c in FC[f2]:
            if c not in C:
                continue
            for f3 in CF[c]:
                T.add(f3)
            T.discard(f1)
            T.discard(f2)
        if trans_count is not None:
            trans_count["all"] += len(T)
        CC1, CC2 = CC[f1], CC[f2]
        c12 = CC1.get(f2)
        c21 = CC2.get(f1)

        def keep(f3):
            CC3 = CC[f3]
            c31 = CC3.get(f1)
            return not (
                (c12 is not None and c12 == CC1.get(f3))
                or (c21 is not None and c21 == CC2.get(f3))
                or (c31 is not None and c31 == CC3.get(f2))
            )

        out = [f3 for f3 in T if keep(f3)]
        if trans_count is not None:
            trans_count["reduced"] += len(out)
        return out

    @staticmethod
    def BF_GB_GA_GI_2_edges(BF, GB, GA, GI):
        edges = []
        for i, B in enumerate(GB):
            orders = M.bit_decode(GA[i][GI[i]], len(B))
            for j, F in enumerate(B):
                f1, f2 = M.decode(BF[F])
                o = orders[j]
                edges.append(M.encode((f1, f2) if o == 1 else (f2, f1)))
        return edges

    @staticmethod
    def edges_Ff_2_FO(edges, Ff):
        out = []
        for k in edges:
            f1, f2 = M.decode(k)
            out.append([f1, f2, 1 if Ff[f2] else -1])
        return out

    @staticmethod
    def CF_edges_2_CD(CF, edges):
        edge_map = set(edges)

        def cmp_faces(a, b):
            return 1 if M.encode([a, b]) in edge_map else -1

        out = []
        for F in CF:
            S = list(F)
            S.sort(key=cmp_to_key(cmp_faces))
            out.append(S)
        return out

    @staticmethod
    def Ctop_SC_SE_EF_Ff_2_SD(Ctop, SC, SE, EF, Ff):
        EF_set = {M.encode_order_pair(F) for F in EF if len(F) == 2}
        SD = []
        for si, C in enumerate(SC):
            F = [Ctop[ci] for ci in C]
            if len(F) < 2:
                SD.append("B")
                continue
            if F[0] == F[1]:
                SD.append("N")
            elif F[0] is None or F[1] is None:
                SD.append("B")
            else:
                flips = [Ff[fi] for fi in F]
                if flips[0] == flips[1] and M.encode_order_pair(F) in EF_set:
                    SD.append("C")
                else:
                    left = right = False
                    for ei in SE[si]:
                        ef_e = EF[ei]
                        if len(ef_e) < 2:
                            continue
                        fi, fj = ef_e[0], ef_e[1]
                        if Ff[fi] == Ff[fj]:
                            continue
                        if fi == F[0] or fj == F[0]:
                            left = True
                        if fi == F[1] or fj == F[1]:
                            right = True
                    if left == right:
                        SD.append("B")
                    else:
                        SD.append("BL" if left else "BR")
        return SD

    @staticmethod
    def Ctop_CP_SC_SD_Ff_P_2_RP_Rf(Ctop, CP, SC, SD, Ff, P):
        cn = len(Ctop)
        CC = [[] for _ in range(cn)]
        for si, C in enumerate(SC):
            if len(C) != 2:
                continue
            if SD[si][0] == "B":
                continue
            ci, cj = C
            CC[ci].append(cj)
            CC[cj].append(ci)
        RP = []
        Rf = []
        seen = [False] * cn
        for ci in range(cn):
            fi = Ctop[ci]
            if fi is None or seen[ci]:
                continue
            seen[ci] = True
            C = [ci]
            i = 0
            while i < len(C):
                for cj in CC[C[i]]:
                    if not seen[cj]:
                        seen[cj] = True
                        C.append(cj)
                i += 1
            Adj = [set() for _ in P]
            for cii in C:
                P_ = CP[cii]
                pi = P_[-1]
                for pj in P_:
                    A = Adj[pj]
                    if pi in A:
                        A.remove(pi)
                    else:
                        Adj[pi].add(pj)
                    pi = pj
            start = None
            for i in range(len(Adj)):
                if len(Adj[i]) == 1:
                    start = i
                    break
            Q = []
            u = start
            while True:
                Q.append(u)
                v = next(iter(Adj[u]))
                u = v
                if u == start:
                    break
            RP.append(Q)
            Rf.append(Ff[fi])
        return [RP, Rf]


X = _X


