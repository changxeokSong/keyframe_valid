"""콘솔(run.bat/run.sh 창)과 파일에 동시에 남는 로그 설정.

병목 디버깅용으로 각 단계 소요시간을 찍을 수 있게 log_time() 컨텍스트 매니저도 제공한다.

사용법:
    from .logging_setup import setup_logging, log, log_time
    setup_logging()
    log.info("뭔가 시작")
    with log_time("동영상 다운로드"):
        ...
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

from .paths import REPORTS_DIR

LOG_PATH = REPORTS_DIR / "ksl_validator.log"

log = logging.getLogger("ksl_validator")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    if log.handlers:
        return log  # 이미 설정됨 (여러 번 불려도 핸들러 중복 안 되게)

    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    log.addHandler(console)

    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(fmt)
        log.addHandler(file_handler)
    except OSError:
        pass  # 로그 파일을 못 쓰는 환경이어도 콘솔 로깅은 계속 되게

    log.propagate = False
    return log


@contextmanager
def log_time(label: str, level: int = logging.INFO):
    """with log_time("사전 동영상 다운로드"): ... 형태로 감싸면 소요시간을 자동으로 찍는다."""
    t0 = time.perf_counter()
    log.log(level, f"▶ {label} 시작")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        log.log(level, f"■ {label} 완료 ({elapsed:.3f}초)")
