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
    origin_no+video_id를 합쳐서 인스턴스 단위로 구분한다.

    video_id가 비어있는 행(사람 구분 정보가 없는 데이터)이 같은 origin_no로 여러 개
    있으면 origin_no만으로는 여전히 서로 겹친다. 이때는 video_rel_path로, 그것도
    없으면 keyframes/설명으로 최대한 구분한다(완전히 똑같은 내용이면 겹쳐도 화면에
    보이는 내용 자체는 같으므로 실질적인 영향은 적다)."""
    if entry.video_id:
        return f"{entry.origin_no}#vid:{entry.video_id}"
    if entry.video_rel_path:
        return f"{entry.origin_no}#path:{entry.video_rel_path}"
    kf = ",".join(str(k) for k in entry.keyframes)
    return f"{entry.origin_no}#kf:{kf}#{entry.gloss_description}"


def load_from_excel(xlsx_path: Path) -> Iterator[DatasetEntry]:
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb["Sign_Gloss"]
    seen: set[tuple] = set()
    n_dup = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        origin = row[1]
        name = row[2]
        if origin is None or not name:
            continue
        origin_no = str(int(origin))
        gloss_name = str(name).strip()
        kf_str = str(row[8] or "").strip()
        keyframes = [int(t) for t in kf_str.split() if t.strip().isdigit()]
        gloss_description = str(row[3] or "").strip()

        # 모든 필드가 전부 같은 행만 "완전 중복"으로 본다 (아래 csv 로더와 동일한
        # 이유 - origin_no만으로 판단하면 실제로는 다른 내용인데 지워버릴 위험이 있다).
        dedup_key = (origin_no, gloss_name, tuple(keyframes), gloss_description)
        if dedup_key in seen:
            n_dup += 1
            continue
        seen.add(dedup_key)
        yield DatasetEntry(
            origin_no=origin_no,
            gloss_name=gloss_name,
            keyframes=keyframes,
            gloss_description=gloss_description,
        )
    wb.close()
    if n_dup:
        log.info(f"[metadata] xlsx 완전 중복 행 {n_dup}개 건너뜀 (모든 필드 동일)")


def load_from_metadata_csv(csv_path: Path) -> Iterator[DatasetEntry]:
    """origin_no+video_id 하나(=한 사람의 한 영상)를 metadata.csv에서 여러 행에
    나눠서 싣는 경우가 실제로 있다(직접 확인: keyframes만 다르고 나머지는 똑같은
    행들). 처음엔 이걸 "중복"으로 보고 origin_no+video_id 기준으로 뒤에 오는 행을
    버렸는데, 그러면 그 행에만 있던 키프레임 정보가 통째로 사라진다(직접 확인:
    실제 데이터에서 21114개 행이 이렇게 잘못 걸러짐 - 예외처리된 항목이 로딩
    단계에서부터 아예 안 보이던 원인). 그래서 버리지 않고 같은 영상(instance_key)을
    가리키는 행들의 keyframes를 순서 유지하며 합친다. video_id도 video_path도
    둘 다 없어서 어느 영상인지 구분할 수 없는 행은 절대 합치지 않고 각자 독립된
    항목으로 남긴다."""
    merged: dict[tuple, DatasetEntry] = {}
    order: list[tuple] = []
    n_merged_rows = 0
    n_malformed = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # gloss_description처럼 자유 텍스트에 큰따옴표가 잘못 들어가 있으면
            # 그 지점부터 csv 파싱이 밀려서 이후 모든 행의 컬럼이 엉뚱하게 섞일 수
            # 있다(직접 확인: 로그에는 있다고 나온 origin_no를 실제 파일에서 검색하면
            # 없는 사례 발생). DictReader는 헤더보다 필드가 적으면 값이 None, 많으면
            # None 키에 나머지를 몰아넣으므로 그걸로 밀린 행을 미리 잡아낸다.
            if None in row or any(v is None for v in row.values()):
                n_malformed += 1
                if n_malformed <= 5:
                    log.warning(
                        f"[metadata] metadata.csv {i + 2}번째 줄의 필드 개수가 헤더와 안 맞음 "
                        f"(따옴표 처리 문제로 이 지점부터 뒤의 행이 밀렸을 수 있음) - "
                        f"읽힌 값: gloss={row.get('gloss')!r}, video_id={row.get('video_id')!r}"
                    )
                continue

            origin = (row.get("gloss") or "").strip()  # etri_metadata_builder는 origin_no를 'gloss' 컬럼에 저장
            name = (row.get("gloss_name") or "").strip()
            if not origin or not name:
                continue
            video_id = (row.get("video_id") or "").strip() or None
            video_rel_path = (row.get("video_path") or "").strip() or None
            kf_str = (row.get("keyframes") or "").strip()
            keyframes = [int(t) for t in kf_str.split() if t.strip().isdigit()]
            gloss_description = (row.get("gloss_description") or "").strip()

            if video_id:
                instance_key = (origin, "vid", video_id)
            elif video_rel_path:
                instance_key = (origin, "path", video_rel_path)
            else:
                instance_key = (origin, "row", i)  # 구분 신호가 없으면 매 행을 독립 항목으로

            existing = merged.get(instance_key)
            if existing is not None:
                for kf in keyframes:
                    if kf not in existing.keyframes:
                        existing.keyframes.append(kf)
                n_merged_rows += 1
                continue

            entry = DatasetEntry(
                origin_no=origin,
                gloss_name=name,
                keyframes=keyframes,
                video_rel_path=video_rel_path,
                video_id=video_id,
                gloss_description=gloss_description,
            )
            merged[instance_key] = entry
            order.append(instance_key)

    if n_malformed:
        log.warning(
            f"[metadata] metadata.csv: 필드 개수가 헤더와 안 맞는 행 {n_malformed}개 발견 - "
            f"CSV 파일이 어딘가에서 깨졌을 가능성이 있습니다(위 경고의 줄 번호 근처를 확인해보세요)"
        )
    if n_merged_rows:
        log.info(
            f"[metadata] metadata.csv: 같은 영상(origin_no+video_id)을 가리키는 행 "
            f"{n_merged_rows}개를 키프레임 기준으로 병합함 (버리지 않음)"
        )

    for key in order:
        yield merged[key]


def load_dataset(path: Path) -> list[DatasetEntry]:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return list(load_from_excel(path))
    if path.suffix.lower() == ".csv":
        return list(load_from_metadata_csv(path))
    raise ValueError(f"지원하지 않는 메타데이터 형식: {path}")
