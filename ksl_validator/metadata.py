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


@dataclass
class DatasetEntry:
    origin_no: str
    gloss_name: str
    keyframes: list[int] = field(default_factory=list)
    video_rel_path: Optional[str] = None  # metadata.csv에만 존재 (NAS 상대경로)
    video_id: Optional[str] = None  # metadata.csv에만 존재 (예: "003/1.003.C")
    gloss_description: str = ""


def load_from_excel(xlsx_path: Path) -> Iterator[DatasetEntry]:
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb["Sign_Gloss"]
    for row in ws.iter_rows(min_row=2, values_only=True):
        origin = row[1]
        name = row[2]
        if origin is None or not name:
            continue
        kf_str = str(row[8] or "").strip()
        keyframes = [int(t) for t in kf_str.split() if t.strip().isdigit()]
        yield DatasetEntry(
            origin_no=str(int(origin)),
            gloss_name=str(name).strip(),
            keyframes=keyframes,
            gloss_description=str(row[3] or "").strip(),
        )
    wb.close()


def load_from_metadata_csv(csv_path: Path) -> Iterator[DatasetEntry]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            origin = (row.get("gloss") or "").strip()  # etri_metadata_builder는 origin_no를 'gloss' 컬럼에 저장
            name = (row.get("gloss_name") or "").strip()
            if not origin or not name:
                continue
            kf_str = (row.get("keyframes") or "").strip()
            keyframes = [int(t) for t in kf_str.split() if t.strip().isdigit()]
            yield DatasetEntry(
                origin_no=origin,
                gloss_name=name,
                keyframes=keyframes,
                video_rel_path=(row.get("video_path") or "").strip() or None,
                video_id=(row.get("video_id") or "").strip() or None,
                gloss_description=(row.get("gloss_description") or "").strip(),
            )


def load_dataset(path: Path) -> list[DatasetEntry]:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return list(load_from_excel(path))
    if path.suffix.lower() == ".csv":
        return list(load_from_metadata_csv(path))
    raise ValueError(f"지원하지 않는 메타데이터 형식: {path}")
