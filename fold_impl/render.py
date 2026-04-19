# Pure-Python folded-state raster (replaces fold_service.mjs + sharp).
from __future__ import annotations

import io
from typing import List

import numpy as np
from PIL import Image, ImageDraw

from .constraints import CON
from .io import IO
from .math2d import M
from .note import NOTE
from . import conversion as conv
from .solver_mod import SOLVER

X = conv.X

SCALE = 800
PADDING = 40
DRAW = SCALE - 2 * PADDING

_built = False


def _ensure_constraints() -> None:
    global _built
    if not _built:
        CON.build()
        _built = True


def render_cp_folded_png(doc: str) -> bytes:
    prev_show = NOTE.show
    NOTE.show = False
    try:
        _ensure_constraints()
        parsed = IO.doc_type_side_2_V_VV_EV_EA_EF_FV_FE(doc, "cp", False)
        if not parsed or not parsed[0]:
            raise ValueError("failed to parse CP")
        V, VV, EV, EA, EF, FV, FE = parsed
        Vf, Ff = X.V_FV_EV_EA_2_Vf_Ff(V, FV, EV, EA)
        for i in range(len(Vf)):
            if Vf[i] is None:
                Vf[i] = tuple(V[i])
        L: List = [M.expand(e, Vf) for e in EV]
        P, SP, SE, _eps_i = X.L_2_V_EV_EL(L)
        if len(P) == 0:
            raise ValueError("precision error: could not build stable overlap graph")
        _vv, CP = X.V_EV_2_VV_FV(P, SP)
        SC, _cs = X.EV_FV_2_EF_FE(SP, CP)
        CF, FC = X.EF_FV_P_SP_SE_CP_SC_2_CF_FC(EF, FV, P, SP, SE, CP, SC)
        BF = X.EF_SP_SE_CP_CF_2_BF(EF, SP, SE, CP, CF)
        BI = {F: i for i, F in enumerate(BF)}
        if len(BF) == 0:
            Ctop = [stack[-1] for stack in CF]
            return _rasterize(Ctop, CP, SC, SE, EF, Ff, P)
        BT = X.BF_BI_EF_SE_CF_SC_2_BT(BF, BI, EF, SE, CF, SC)
        CC = X.FC_BF_BI_BT_2_CC(FC, BF, BI, BT)
        BA0 = SOLVER.EF_EA_Ff_BF_BI_2_BA0(EF, EA, Ff, BF, BI)
        trans_count = {"all": 0, "reduced": 0}
        out = SOLVER.initial_assignment(BA0, BF, BT, BI, FC, CF, CC, trans_count)
        # Conflict path returns [type, F, E]; success returns BA (ints only).
        if isinstance(out, list) and len(out) == 3 and isinstance(out[1], list):
            raise ValueError("constraint conflict — CP may not be flat-foldable")
        BA = out
        GB = SOLVER.get_components(BI, BF, BT, BA, FC, CF, CC, trans_count)
        GA = SOLVER.solve(BI, BF, BT, BA, GB, FC, CF, CC, 1)
        if isinstance(GA, int):
            raise ValueError(f"could not resolve component {GA}")
        Gi = [0] * len(GA)
        edges = X.BF_GB_GA_GI_2_edges(BF, GB, GA, Gi)
        CD = X.CF_edges_2_CD(CF, edges)
        Ctop = [stack[-1] for stack in CD]
        return _rasterize(Ctop, CP, SC, SE, EF, Ff, P)
    finally:
        NOTE.show = prev_show


def _rasterize(Ctop, CP, SC, SE, EF, Ff, P):
    Pn_xy = M.normalize_points_xy(P)

    SD = X.Ctop_SC_SE_EF_Ff_2_SD(Ctop, SC, SE, EF, Ff)
    RP, Rf = X.Ctop_CP_SC_SD_Ff_P_2_RP_Rf(Ctop, CP, SC, SD, Ff, P)
    img = Image.new("RGB", (SCALE, SCALE), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    for ri, ptIdxs in enumerate(RP):
        fill = (170, 170, 170) if Rf[ri] else (255, 255, 255)
        idx = np.asarray(ptIdxs, dtype=np.intp)
        # Flat [x0,y0,...] avoids per-vertex tuples; width=1 skips Pillow's wide-outline mask path.
        xy = (PADDING + DRAW * Pn_xy[idx]).ravel(order="C").tolist()
        draw.polygon(xy, fill=fill, outline=(0, 0, 0), width=1)
    buf = io.BytesIO()
    # compress_level=1 encodes faster than default/6; slightly larger files (fine for Discord).
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()
