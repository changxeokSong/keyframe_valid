"""tools/tagging/config/dataset.json 을 읽어 NAS 경로들을 자동으로 채운다.

dataset.json에는 Windows UNC 경로(\\\\mldisk2\\nfs_shared\\...)로 적혀 있는데,
Mac에서 같은 NAS를 SMB로 마운트하면 보통 /Volumes/<공유이름>/... 형태가 된다.
그래서 UNC 경로를 몇 가지 흔한 Mac 마운트 위치로 변환 시도해보고,
실제로 존재하는 경로를 찾으면 그걸 쓰고, 없으면 '미마운트' 상태로 원본 경로를 보여준다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_PATH = Path(
    "online-sign-keyframe-detection-transformers/tools/tagging/config/dataset.json"
)


@dataclass
class ResolvedPath:
    raw: str                    # dataset.json에 적힌 원본 경로 (UNC 등)
    resolved: Optional[Path]    # 실제로 존재해서 쓸 수 있는 로컬 경로 (없으면 None)

    @property
    def mounted(self) -> bool:
        return self.resolved is not None


@dataclass
class DatasetPaths:
    name: str
    dataset_root: ResolvedPath
    metadata_file: ResolvedPath
    handshape_image_dir: ResolvedPath
    excel_file: ResolvedPath


def _unc_candidates(unc_or_path: str) -> list[Path]:
    """UNC(\\\\server\\share\\...) 또는 //server/share/... 경로를 Mac 마운트 후보들로 변환."""
    if not unc_or_path:
        return []

    norm = unc_or_path.replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    if not parts:
        return []

    # \\mldisk2\nfs_shared\abd\... -> server=mldisk2, share=nfs_shared, rest=abd/...
    server = parts[0]
    rest = parts[1:]

    candidates = [
        Path("/Volumes", *rest),                 # /Volumes/nfs_shared/abd/...
        Path("/Volumes", server, *rest),          # /Volumes/mldisk2/nfs_shared/abd/...
        Path("/media/mmlab", *rest),              # 사용자가 실제 언급한 리눅스 마운트 경로 계열
        Path("/", *rest),                         # 이미 절대경로로 마운트된 경우
    ]
    return candidates


def resolve_path(raw: str) -> ResolvedPath:
    if not raw:
        return ResolvedPath(raw=raw, resolved=None)

    # 이미 로컬에 존재하는 절대경로면 그대로 사용
    direct = Path(raw)
    if direct.exists():
        return ResolvedPath(raw=raw, resolved=direct)

    for cand in _unc_candidates(raw):
        if cand.exists():
            return ResolvedPath(raw=raw, resolved=cand)

    return ResolvedPath(raw=raw, resolved=None)


def load_dataset_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Optional[DatasetPaths]:
    config_path = Path(config_path)
    if not config_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    active = data.get("active")
    cfg = (data.get("configs") or {}).get(active)
    if not cfg:
        return None

    return DatasetPaths(
        name=active,
        dataset_root=resolve_path(cfg.get("dataset_root", "")),
        metadata_file=resolve_path(cfg.get("metadata_file", "")),
        handshape_image_dir=resolve_path(cfg.get("handshape_image_dir", "")),
        excel_file=resolve_path(cfg.get("excel_file", "")),
    )
