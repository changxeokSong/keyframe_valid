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
from .metadata import DatasetEntry, entry_key
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
        r = results.get(entry_key(e))
        status = r.status if r else "미검증"
        by_status.setdefault(status, []).append(e)

    exceptions_this_session = [r for r in review_rows if r["decision"] == "EXCEPTION"]
    restored_this_session = [r for r in review_rows if r["decision"] == "RESTORED"]
    confirmed_ok_this_session = [r for r in review_rows if r["decision"] == "OK"]

    # origin_no(글로스) 하나를 여러 사람(video_id/subset)이 각자 촬영한 경우가
    # 많아서, "이 origin_no에 서로 다른 영상이 몇 개 있는지" 알아야 리포트를
    # 오해 없이 읽을 수 있다 (예: 같은 글로스라도 004/009/011번 사람이 각자
    # 따로 촬영한 별개 영상 - 하나가 SUSPECT라고 나머지도 문제라는 뜻은 아님).
    signers_per_gloss: dict[str, set] = {}
    for e in entries:
        signers_per_gloss.setdefault(e.origin_no, set()).add(e.video_id or "(사람 구분 없음)")
    multi_signer_glosses = sum(1 for v in signers_per_gloss.values() if len(v) > 1)

    lines: list[str] = []
    lines.append("# KSL Validator 검증 요약 리포트")
    lines.append(f"생성 시각: {now}")
    lines.append("")

    lines.append("## 이 리포트 보는 법")
    lines.append(
        "- **origin_no**: 수어 단어(글로스) 식별번호. **video_id(사람)**: 그 단어를 실제로 "
        "촬영한 사람/subset 구분 번호(예: 004, 009, 011...). **같은 origin_no라도 video_id가 "
        "다르면 서로 다른 사람이 각자 찍은 별개 영상**이라서, 한 사람 영상이 SUSPECT라고 "
        "같은 글로스의 다른 사람 영상까지 문제라는 뜻은 아닙니다."
    )
    lines.append(
        "- **검증 상태**: MATCH=태깅된 키프레임 전부가 사전 공식 영상과 임계값 이상 "
        "비슷함(라벨 정상 가능성 높음) / SUSPECT=하나 이상 안 비슷함(라벨 오류 의심, 사람 "
        "검토 필요) / NO_MY_KEYFRAME=이 영상에서 사람/손을 못 찾음 / DOWNLOAD_FAILED=사전 "
        "사이트 영상을 못 받음 / NO_POSE_DETECTED=사전 영상 어디서도 비교할 포즈를 못 찾음."
    )
    lines.append(
        "- **키프레임매칭**: 태깅된 키프레임 중 몇 개가 사전 영상과 임계값 이상 비슷했는지 "
        "(예: 2/3 → 3개 중 2개만 통과, 그래서 SUSPECT). **정답비교**: 사전 동영상과 별개로, "
        "같은 글로스 공통 기준사진과도 비교한 두 번째 독립 신호(둘 다 낮으면 라벨 오류일 "
        "확률이 더 높음)."
    )
    lines.append(
        "- ⚠ 이 점수들은 전부 **검토 우선순위를 정하는 참고용**입니다. 최종 판단은 GUI에서 "
        "실제 프레임을 눈으로 비교해서 내려야 합니다."
    )
    lines.append("")

    lines.append("## 데이터 소스")
    lines.append(f"- 전체 항목(영상) 수: {len(entries)}개, 서로 다른 글로스(origin_no) 수: {len(signers_per_gloss)}개")
    lines.append(f"- 여러 사람이 같이 촬영한 글로스: {multi_signer_glosses}개")
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
        lines.append("| origin_no | gloss_name | video_id(사람) | 점수 | 키프레임매칭 | 정답비교 | 손모양반영 | 최고매칭프레임 | 비고 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        def score_of(e: DatasetEntry) -> float:
            r = results.get(entry_key(e))
            return r.best_score if r and r.best_score is not None else 0.0

        for e in sorted(suspects, key=score_of):
            r = results[entry_key(e)]
            score_txt = f"{r.best_score:.4f}" if r.best_score is not None else "-"
            ref_txt = f"{r.reference_score:.4f}" if r.reference_score is not None else "-"
            lines.append(
                f"| {e.origin_no} | {e.gloss_name} | {e.video_id or '-'} | {score_txt} | "
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
            video_txt = f", video_id={row['video_id']}" if row.get("video_id") else ""
            lines.append(
                f"- origin_no={row['origin_no']} ({row['gloss_name']}){video_txt}, "
                f"점수={row['best_score'] or '-'}, 시각={row['reviewed_at']}{reason_txt}"
            )
        lines.append("")

    if restored_this_session:
        lines.append("## 이번 세션에서 예외 해제(정상 복구)한 항목")
        for row in restored_this_session:
            video_txt = f", video_id={row['video_id']}" if row.get("video_id") else ""
            lines.append(f"- origin_no={row['origin_no']} ({row['gloss_name']}){video_txt}, 시각={row['reviewed_at']}")
        lines.append("")

    if confirmed_ok_this_session:
        lines.append("## 이번 세션에서 정상 확인한 항목")
        for row in confirmed_ok_this_session:
            video_txt = f", video_id={row['video_id']}" if row.get("video_id") else ""
            lines.append(f"- origin_no={row['origin_no']} ({row['gloss_name']}){video_txt}, 시각={row['reviewed_at']}")
        lines.append("")

    failure_statuses = ("DOWNLOAD_FAILED", "NO_MY_KEYFRAME", "NO_POSE_DETECTED", "ERROR")
    failures = [e for s in failure_statuses for e in by_status.get(s, [])]
    if failures:
        lines.append("## 처리 실패/보류 항목 (재확인 필요)")
        for e in failures:
            r = results.get(entry_key(e))
            lines.append(
                f"- origin_no={e.origin_no} ({e.gloss_name}, video_id={e.video_id or '-'}): "
                f"{r.status if r else '-'} — {r.note if r else ''}"
            )
        lines.append("")

    unvalidated = by_status.get("미검증", [])
    if unvalidated:
        lines.append(f"## 아직 검증하지 않은 항목: {len(unvalidated)}개")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
