"""ETRI 한국수어사전 데이터셋 메타데이터 로더.

두 가지 소스를 지원한다.
1) xlsx (Sign_Gloss 시트) — 온라인수어 tagging 프로젝트의
   sample/etri_ksl_db/ETRI_KSL_Dictionary_*.xlsx 와 동일 포맷.
   Origin_Number, Gloss_Name, KeyFrames(프레임 인덱스)만 있으면 되므로
   NAS 마운트 없이도 라벨 검증에 바로 쓸 수 있다.
2) metadata.csv (online-sign-keyframe-detection-transformers의
   src/etri_metadata_builder.py 산출물) — video_path가 있어 NAS가
   마운트되어 있으면 실제 원본 비디오에서 키프레임을 추출할 수 있다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import openpyxl

from .logging_setup import log


@dataclass
class DatasetEntry:
    origin_no: str
    gloss_name: str
    keyframes: list[int] = field(default_factory=list)
    video_rel_path: Optional[str] = None  # metadata.csv에만 존재 (NAS 상대경로)
    video_id: Optional[str] = None  # metadata.csv에만 존재 (예: "003/1.003.C")
    gloss_description: str = ""


def entry_key(entry: DatasetEntry) -> str:
    """entry의 유일 식별자. origin_no(글로스 번호)만으로는 부족하다 - 같은 글로스를
    004/009/011처럼 여러 사람(subset)이 각자 따로 촬영한 경우가 흔한데, origin_no만
    키로 쓰면 그 사람들의 서로 다른 영상이 전부 한 항목으로 뭉개진다(직접 확인된 버그:
    검증 결과/예외처리 사유/정답 영상이 다른 사람 것으로 뒤바뀌어 보임). 그래서
    origin_no+video_id를 합쳐서 인스턴스 단위로 구분한다."""
    return f"{entry.origin_no}#{entry.video_id or ''}"


def load_from_excel(xlsx_path: Path) -> Iterator[DatasetEntry]:
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb["Sign_Gloss"]
    seen: set[str] = set()
    n_dup = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        origin = row[1]
        name = row[2]
        if origin is None or not name:
            continue
        origin_no = str(int(origin))
        if origin_no in seen:
            n_dup += 1
            continue
        seen.add(origin_no)
        kf_str = str(row[8] or "").strip()
        keyframes = [int(t) for t in kf_str.split() if t.strip().isdigit()]
        yield DatasetEntry(
            origin_no=origin_no,
            gloss_name=str(name).strip(),
            keyframes=keyframes,
            gloss_description=str(row[3] or "").strip(),
        )
    wb.close()
    if n_dup:
        log.info(f"[metadata] xlsx 중복 행 {n_dup}개 건너뜀 (origin_no 기준)")


def load_from_metadata_csv(csv_path: Path) -> Iterator[DatasetEntry]:
    seen: set[tuple] = set()
    n_dup = 0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            origin = (row.get("gloss") or "").strip()  # etri_metadata_builder는 origin_no를 'gloss' 컬럼에 저장
            name = (row.get("gloss_name") or "").strip()
            if not origin or not name:
                continue
            video_id = (row.get("video_id") or "").strip() or None
            video_rel_path = (row.get("video_path") or "").strip() or None
            # video_id(예: "011/1.011.C")는 이 gloss를 촬영한 "사람"(subset) 식별자라서,
            # 같은 origin_no라도 video_id가 다르면 서로 다른 사람의 서로 다른 영상이라
            # 별개 항목으로 남겨야 한다. 반대로 origin_no+video_id가 완전히 같은 행이
            # 여러 번 있으면(직접 확인: 사용자 데이터에서 동일 행 3연속) 소스 CSV 쪽
            # 중복이므로 첫 번째만 남긴다.
            dedup_key = (origin, video_id or video_rel_path or name)
            if dedup_key in seen:
                n_dup += 1
                continue
            seen.add(dedup_key)
            kf_str = (row.get("keyframes") or "").strip()
            keyframes = [int(t) for t in kf_str.split() if t.strip().isdigit()]
            yield DatasetEntry(
                origin_no=origin,
                gloss_name=name,
                keyframes=keyframes,
                video_rel_path=video_rel_path,
                video_id=video_id,
                gloss_description=(row.get("gloss_description") or "").strip(),
            )
    if n_dup:
        log.info(f"[metadata] metadata.csv 중복 행 {n_dup}개 건너뜀 (origin_no+video_id 기준)")


def load_dataset(path: Path) -> list[DatasetEntry]:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return list(load_from_excel(path))
    if path.suffix.lower() == ".csv":
        return list(load_from_metadata_csv(path))
    raise ValueError(f"지원하지 않는 메타데이터 형식: {path}")
