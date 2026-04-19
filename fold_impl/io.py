# Port of src/io.js — CP / FOLD ingestion (export / SVG / OPX omitted).
from __future__ import annotations

from typing import List, Tuple

from . import conversion as conv
from .math2d import M
from .note import NOTE

X = conv.X


def CP_2_L(doc: str) -> List:
    cmap = ["U", "B", "M", "V", "F"]
    L = []
    for raw in doc.split("\n"):
        line = raw.strip()
        if not line:
            continue
        parts = [t.strip() for t in line.split()]
        a, x1, y1, x2, y2 = parts[:5]
        typ = cmap[int(a)] if int(a) < len(cmap) else "U"
        L.append([[float(x1), float(y1)], [float(x2), float(y2)], typ])
    return L


def FOLD_2_V_EV_EA_VV_FV(doc: str):
    import json

    ex = json.loads(doc)
    if "vertices_coords" not in ex:
        NOTE.time("FOLD file does not contain vertices_coords")
        return []
    V = ex["vertices_coords"]
    if "edges_vertices" not in ex:
        NOTE.time("FOLD file does not contain edges_vertices")
        return []
    EV = [[v1, v2] if v1 < v2 else [v2, v1] for v1, v2 in ex["edges_vertices"]]
    if "edges_assignment" in ex:
        EA = ex["edges_assignment"]
    else:
        NOTE.time("FOLD file does not contain edges_assignments")
        NOTE.time("   - assuming all unassigned")
        EA = ["U"] * len(EV)
    if "faces_vertices" in ex:
        FV = ex["faces_vertices"]
        M.sort_faces(FV, V)
        VV = X.V_FV_2_VV(V, FV)
    else:
        L = [M.expand(e, V) for e in EV]
        V, EV, EL, eps_i = X.L_2_V_EV_EL(L)
        EA_prev = EA
        EA = []
        for eL in EL:
            a = "F"
            for l in eL:
                if EA_prev[l] != "F":
                    a = EA_prev[l]
                    break
            EA.append(a)
    return [V, EV, EA, VV, FV]


def doc_type_side_2_V_VV_EV_EA_EF_FV_FE(doc: str, typ: str, side: bool):
    eps_i = None
    if typ == "fold":
        parsed = FOLD_2_V_EV_EA_VV_FV(doc)
        if not parsed or parsed[0] is None:
            return []
        V, EV, EA, VV, FV = parsed
    else:
        if typ == "cp":
            L = CP_2_L(doc)
        else:
            NOTE.time(f"ERROR: File extension .{typ} not supported!")
            NOTE.time("       Please use from [.fold, .svg, .cp, .opx]")
            return []
        NOTE.annotate(L, "lines")
        NOTE.lap()
        NOTE.time("Constructing FOLD from lines")
        V, EV, EL, eps_i = X.L_2_V_EV_EL(L)
        eps = M.min_line_length(L) / (2**eps_i)
        NOTE.time(f"Used eps: {2**eps_i} | {eps}")
        EA = []
        for eL in EL:
            a = "F"
            for l in eL:
                if L[l][2] != "F":
                    a = L[l][2]
                    break
            EA.append(a)

    def flip_EA(EA):
        return ["V" if a == "M" else "M" if a == "V" else a for a in EA]

    def flip_Y(V):
        return [[x, -y + 1] for x, y in V]

    def reverse_FV(FV):
        for F in FV:
            F.reverse()

    if typ != "fold":
        if side:
            EA = flip_EA(EA)
        else:
            V = flip_Y(V)
        VV, FV = X.V_EV_2_VV_FV(V, EV)
    else:
        if M.polygon_area2(M.expand(FV[0], V)) < 0:
            EA = flip_EA(EA)
            reverse_FV(FV)
        if not side:
            EA = flip_EA(EA)
            reverse_FV(FV)
            V = flip_Y(V)
    EF, FE = X.EV_FV_2_EF_FE(EV, FV)
    if len(FV) > 1:
        FV = [F for i, F in enumerate(FV) if not all(EA[e] == "B" for e in FE[i])]
    if len(FV) != len(FE):
        EF, FE = X.EV_FV_2_EF_FE(EV, FV)
    for i, F in enumerate(EF):
        if len(F) == 1:
            EA[i] = "B"
    return [V, VV, EV, EA, EF, FV, FE]


class _IO:
    CP_2_L = staticmethod(CP_2_L)
    FOLD_2_V_EV_EA_VV_FV = staticmethod(FOLD_2_V_EV_EA_VV_FV)
    doc_type_side_2_V_VV_EV_EA_EF_FV_FE = staticmethod(doc_type_side_2_V_VV_EV_EA_EF_FV_FE)


IO = _IO()
