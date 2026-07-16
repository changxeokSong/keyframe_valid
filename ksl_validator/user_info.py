"""getpass.getuser()는 일부 Windows 환경(특히 USERNAME 환경변수가 없거나
서비스/자동화 컨텍스트로 실행될 때)에서 예외를 던질 수 있다. 실제로 사용자
환경에서 GUI 시작 중 여기서 죽는 걸 확인해서, 절대 실패하지 않는 안전한
버전으로 감싼다.
"""

from __future__ import annotations

import getpass

FALLBACK_USERNAME = "user"


def safe_getuser() -> str:
    try:
        name = getpass.getuser()
        return name or FALLBACK_USERNAME
    except Exception:  # noqa: BLE001 - 사용자 이름 하나 때문에 앱이 죽으면 안 됨
        return FALLBACK_USERNAME
