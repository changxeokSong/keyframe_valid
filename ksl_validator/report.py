"""검증 세션 결과 + 예외처리 이력을 사람이 읽기 좋은 마크다운 리포트로 정리.

"뭐가 이렇게 됐고(검증 상태별 집계) 그래서 처리를 이렇게 했다(이번 세션에서
예외처리/복구/정상확인 한 목록)" 형태로 정리해서 검토 결과를 공유하기 쉽게 한다.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from .exception_store import ExceptionStore
from .metadata import DatasetEntry
from .paths import SUMMARY_REPORT_PATH
from .pipeline import ValidationResult

DEFAULT_REPORT_MD_PATH = SUMMARY_REPORT_PATH


def _load_review_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_report(
    entries: list[DatasetEntry],
    results: dict[str, ValidationResult],
    exception_store: Optional[ExceptionStore],
    review_log_path: Path,
    dataset_root: Optional[Path],
    keyframe_images_dir: Optional[Path],
    output_path: Path = DEFAULT_REPORT_MD_PATH,
) -> Path:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    review_rows = _load_review_log(review_log_path)

    by_status: dict[str, list[DatasetEntry]] = {}
    for e in entries:
        r = results.get(e.origin_no)
        status = r.status if r else "미검증"
        by_status.setdefault(status, []).append(e)

    exceptions_this_session = [r for r in review_rows if r["decision"] == "EXCEPTION"]
    restored_this_session = [r for r in review_rows if r["decision"] == "RESTORED"]
    confirmed_ok_this_session = [r for r in review_rows if r["decision"] == "OK"]

    lines: list[str] = []
    lines.append("# KSL Validator 검증 요약 리포트")
    lines.append(f"생성 시각: {now}")
    lines.append("")

    lines.append("## 데이터 소스")
    lines.append(f"- 전체 항목 수: {len(entries)}")
    lines.append(f"- NAS 데이터셋 루트: {dataset_root or '미지정'}")
    lines.append(f"- 키프레임 사진 폴더: {keyframe_images_dir or '미지정'}")
    lines.append(f"- 예외처리 CSV: {exception_store.csv_path if exception_store else '미지정'}")
    lines.append("")

    lines.append("## 검증 상태별 개수")
    for status, es in sorted(by_status.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- {status}: {len(es)}개")
    lines.append("")

    suspects = by_status.get("SUSPECT", [])
    if suspects:
        lines.append("## ⚠ 라벨 불일치 의심(SUSPECT) — 점수 낮은 순(검토 우선순위)")
        lines.append("")
        lines.append("| origin_no | gloss_name | 점수 | 키프레임매칭 | 정답비교 | 손모양반영 | 최고매칭프레임 | 비고 |")
        lines.append("|---|---|---|---|---|---|---|---|")

        def score_of(e: DatasetEntry) -> float:
            r = results.get(e.origin_no)
            return r.best_score if r and r.best_score is not None else 0.0

        for e in sorted(suspects, key=score_of):
            r = results[e.origin_no]
            score_txt = f"{r.best_score:.4f}" if r.best_score is not None else "-"
            ref_txt = f"{r.reference_score:.4f}" if r.reference_score is not None else "-"
            lines.append(
                f"| {e.origin_no} | {e.gloss_name} | {score_txt} | "
                f"{r.keyframe_matched}/{r.keyframe_total} | {ref_txt} | "
                f"{'Y' if r.hand_used else 'N'} | {r.best_frame_idx} | {r.note} |"
            )
        lines.append("")
        lines.append("※ 점수는 자동 판별의 참고 우선순위일 뿐입니다. GUI에서 실제 프레임을 "
                      "눈으로 비교해 최종 판단해야 합니다.")
        lines.append("")

    if exceptions_this_session:
        lines.append("## 이번 세션에서 새로 예외처리한 항목")
        for row in exceptions_this_session:
            reason_txt = f", 사유: {row['reason']}" if row.get("reason") else ""
            lines.append(
                f"- origin_no={row['origin_no']} ({row['gloss_name']}), "
                f"점수={row['best_score'] or '-'}, 시각={row['reviewed_at']}{reason_txt}"
            )
        lines.append("")

    if restored_this_session:
        lines.append("## 이번 세션에서 예외 해제(정상 복구)한 항목")
        for row in restored_this_session:
            lines.append(f"- origin_no={row['origin_no']} ({row['gloss_name']}), 시각={row['reviewed_at']}")
        lines.append("")

    if confirmed_ok_this_session:
        lines.append("## 이번 세션에서 정상 확인한 항목")
        for row in confirmed_ok_this_session:
            lines.append(f"- origin_no={row['origin_no']} ({row['gloss_name']}), 시각={row['reviewed_at']}")
        lines.append("")

    failure_statuses = ("DOWNLOAD_FAILED", "NO_MY_KEYFRAME", "NO_POSE_DETECTED", "ERROR")
    failures = [e for s in failure_statuses for e in by_status.get(s, [])]
    if failures:
        lines.append("## 처리 실패/보류 항목 (재확인 필요)")
        for e in failures:
            r = results.get(e.origin_no)
            lines.append(f"- origin_no={e.origin_no} ({e.gloss_name}): {r.status if r else '-'} — {r.note if r else ''}")
        lines.append("")

    unvalidated = by_status.get("미검증", [])
    if unvalidated:
        lines.append(f"## 아직 검증하지 않은 항목: {len(unvalidated)}개")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
