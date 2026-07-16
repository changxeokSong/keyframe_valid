"""이 컴퓨터에서 마지막으로 쓴 경로들을 기억해두는 머신별 로컬 설정.

NAS 마운트 위치(윈도우 드라이브 문자, macOS /Volumes 경로 등)는 컴퓨터마다
다르고 dataset.json만으로는 자동으로 못 찾는 경우가 많다(예: 매핑된 드라이브).
그래서 사용자가 GUI에서 한 번 수동으로 지정하면 여기에 저장해두고, 다음 실행부터는
매번 다시 지정하지 않아도 자동으로 불러오게 한다.

git에는 올리지 않는다(.gitignore) — 컴퓨터마다 값이 다르기 때문.
"""

from __future__ import annotations

import json

from .paths import LOCAL_SETTINGS_PATH

SETTINGS_PATH = LOCAL_SETTINGS_PATH


def load() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - 손상된 로컬 설정 파일이 앱 실행을 막으면 안 됨
        return {}


def save(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def update(**kwargs) -> None:
    settings = load()
    for k, v in kwargs.items():
        if v is None:
            continue
        settings[k] = str(v)
    save(settings)
