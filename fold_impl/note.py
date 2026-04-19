# Port of src/note.js — logging / timing (browser bits stripped).
from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, List, Optional


class _TIME:
    main_start = 0.0
    main_lap = 0.0
    est_start = 0.0
    est_lap = 0.0
    est_lim: Optional[int] = None

    @staticmethod
    def start_main() -> None:
        _TIME.main_start = time.time() * 1000
        _TIME.main_lap = _TIME.main_start

    @staticmethod
    def read_time() -> float:
        return time.time() * 1000 - _TIME.main_start

    @staticmethod
    def lap() -> float:
        stop = time.time() * 1000
        t = stop - _TIME.main_lap
        _TIME.main_lap = stop
        return t

    @staticmethod
    def read_est() -> float:
        return time.time() * 1000 - _TIME.est_lap

    @staticmethod
    def start_est(lim: Optional[int]) -> None:
        _TIME.est_start = time.time() * 1000
        _TIME.est_lap = _TIME.est_start
        _TIME.est_lim = lim

    @staticmethod
    def lap_est() -> None:
        _TIME.est_lap = time.time() * 1000

    @staticmethod
    def remaining(i: int) -> str:
        return _TIME.str(
            (time.time() * 1000 - _TIME.est_start) * (_TIME.est_lim / i - 1)
            if _TIME.est_lim and i
            else 0
        )

    @staticmethod
    def str(ms: float) -> str:
        if ms < 1000:
            return f"{int(math.ceil(ms))} millisecs"
        if ms < 60000:
            return f"{int(math.ceil(ms / 1000))} secs"
        mins = int(ms // 60000)
        secs = int(math.ceil((ms - mins * 60000) / 1000))
        return f"{mins} mins {secs} secs"


class _NOTE:
    # `show` is instance-only so NOTE.show = False (see render) is visible here; do not read _NOTE.show.
    lines: List[str] = []
    console = None
    check_interval = 5000
    check_label = ""

    def __init__(self) -> None:
        self.show = True

    @staticmethod
    def start(label: Optional[str] = None) -> None:
        _TIME.start_main()
        if label is not None:
            _NOTE.time(label)

    @staticmethod
    def lap() -> float:
        if not NOTE.show:
            return 0.0
        t = _TIME.lap()
        _NOTE.log(f"   - Time elapsed: {_TIME.str(t)}")
        return t

    @staticmethod
    def start_check(label: str, A: Any = None, interval: int = 5000) -> None:
        if not NOTE.show:
            return
        lim = None if A is None else len(A)
        _TIME.start_est(lim)
        _NOTE.check_interval = interval
        _NOTE.check_label = label

    @staticmethod
    def check(i: int) -> None:
        if not NOTE.show:
            return
        if _TIME.read_est() > _NOTE.check_interval:
            if _TIME.est_lim is not None:
                _NOTE.log(
                    f"    On {_NOTE.check_label} {i} out of "
                    f"{_TIME.est_lim}, est time left: {_TIME.remaining(i)}"
                )
            else:
                _NOTE.log(f"    On {_NOTE.check_label} {i} of unknown")
            _TIME.lap_est()

    @staticmethod
    def annotate(A: List, label: str) -> None:
        if not NOTE.show:
            return
        main = f"   - Found {len(A)} {label}"
        detail = "" if len(A) == 0 else f"[0] = {A[0]!r}"
        _NOTE.log(main + detail)

    @staticmethod
    def time(label: str) -> None:
        if not NOTE.show:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        _NOTE.log(f"{ts} | {label}")

    @staticmethod
    def end() -> float:
        t = _TIME.read_time()
        if NOTE.show:
            _NOTE.log(f"*** Total Time elapsed: {_TIME.str(t)} ***")
            _NOTE.log("")
        return t

    @staticmethod
    def count(A: Any, label: str, div: int = 1) -> int:
        n = _NOTE.count_subarrays(A) // div if isinstance(A, list) else int(A)
        if NOTE.show:
            _NOTE.log(f"   - Found {n} {label}")
        return n

    @staticmethod
    def log(s: str) -> None:
        if NOTE.show:
            print(s)
            _NOTE.lines.append(s)

    @staticmethod
    def clear_log() -> None:
        _NOTE.lines.clear()

    @staticmethod
    def count_subarrays(A: List) -> int:
        return sum(len(adj) for adj in A)


NOTE = _NOTE()
