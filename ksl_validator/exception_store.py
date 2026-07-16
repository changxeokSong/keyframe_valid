"""tools/tagging 예외처리 CSV들과 호환되는 예외처리 저장소.

⚠ 안전 원칙: 이 모듈은 NAS(또는 사용자가 지정한 원본 exception_*.csv)에
**절대 쓰기(write)를 하지 않는다.** 오직 읽기만 한다.

실제 NAS(etri_ksl_db 루트)에는 검토자별로 파일이 나뉘어 있다:
  exception_{이름}.csv          — 현재 예외처리 상태
  exception_history_{이름}.csv  — append-only 이력 로그

GUI에서 "예외처리" / "예외 해제(정상 복구)"를 눌러도 이 원본 파일들은 절대
수정되지 않고, 대신 로컬 스테이징 폴더(기본: reports/exception_staging/)에
'덮어쓰는 게 아니라 얹는' 방식(overlay)으로만 기록된다:
  - mark_exception  -> staging/exception_{reviewer}.csv 에 추가
  - restore
      - 로컬에서만 예외처리했던 항목이면 staging에서 그냥 제거
      - 원본(NAS 등)에 이미 있던 항목이면 원본은 건드리지 않고,
        staging/restore_requests.csv 에 "복구 요청됨(NAS 미반영)"만 기록
        (실제로 NAS 파일에서 지우는 건 사람이 직접 해야 함)
모든 로컬 변경 이력은 staging/exception_history_{reviewer}.csv 에 남는다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .paths import EXCEPTION_STAGING_DIR
from .user_info import safe_getuser

REQUIRED_FIELDS = ["video_id", "original_gloss", "original_gloss_name", "moved_at", "reason"]

ORIGIN_KEYS = ("original_gloss", "origin_no")
NAME_KEYS = ("original_gloss_name", "gloss_name")
MOVED_AT_KEYS = ("moved_at", "timestamp", "reviewed_at")
REASON_KEYS = ("reason", "note", "사유", "비고", "comment")

DEFAULT_STAGING_DIR = EXCEPTION_STAGING_DIR


@dataclass
class ExceptionRow:
    video_id: str
    origin_no: str
    gloss_name: str
    moved_at: str
    reason: str = ""
    source: str = "source"  # "source"(원본, 읽기전용) | "staging"(로컬 대기중)
    raw: dict = field(default_factory=dict)


def _first(row: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v).strip()
    return ""


def _load_csv_rows(path: Path) -> tuple[dict[str, ExceptionRow], list[str]]:
    rows: dict[str, ExceptionRow] = {}
    fieldnames = list(REQUIRED_FIELDS)
    if not path.exists():
        return rows, fieldnames
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or fieldnames
        for row in reader:
            origin = _first(row, ORIGIN_KEYS)
            if not origin:
                continue
            rows[origin] = ExceptionRow(
                video_id=_first(row, ("video_id",)),
                origin_no=origin,
                gloss_name=_first(row, NAME_KEYS),
                moved_at=_first(row, MOVED_AT_KEYS),
                reason=_first(row, REASON_KEYS),
                raw=row,
            )
    return rows, fieldnames


class ExceptionStore:
    """원본(NAS 등, 읽기전용) + 로컬 스테이징(쓰기 가능)을 합쳐서 보여주는 뷰.

    source_path: 단일 CSV 파일 또는 exception_*.csv들이 있는 디렉토리. **읽기만 함.**
    staging_dir: 로컬 변경사항을 저장하는 디렉토리. 기본 reports/exception_staging/.
    """

    def __init__(self, source_path: Path, reviewer: str = "", staging_dir: Path = DEFAULT_STAGING_DIR):
        self.path = Path(source_path)          # 표시용/호환용 이름 유지 (읽기전용 원본)
        self.dir_mode = self.path.is_dir()
        self.reviewer = reviewer or safe_getuser()
        self.staging_dir = Path(staging_dir)

        # reviewer -> {origin_no: ExceptionRow}  (원본, 읽기전용)
        self._source_by_reviewer: dict[str, dict[str, ExceptionRow]] = {}
        # reviewer -> {origin_no: ExceptionRow}  (로컬 스테이징)
        self._staging_by_reviewer: dict[str, dict[str, ExceptionRow]] = {}
        # 원본에 있던 항목의 "복구 요청됨" 표시: {origin_no: (requested_by, requested_at)}
        self._restore_requests: dict[str, tuple[str, str]] = {}

        self.load()

    # ── 원본 파일 목록 (읽기 전용) ────────────────────────────────────
    def _source_files(self) -> dict[str, Path]:
        if not self.dir_mode:
            return {self.reviewer: self.path}
        files: dict[str, Path] = {}
        for p in sorted(self.path.glob("exception_*.csv")):
            if p.stem.startswith("exception_history"):
                continue
            name = p.stem[len("exception_"):]
            files[name] = p
        return files

    def _staging_exception_path(self, reviewer: str) -> Path:
        return self.staging_dir / f"exception_{reviewer}.csv"

    def _staging_history_path(self, reviewer: str) -> Path:
        return self.staging_dir / f"exception_history_{reviewer}.csv"

    def _restore_requests_path(self) -> Path:
        return self.staging_dir / "restore_requests.csv"

    # ── 로드 (원본은 읽기만, 스테이징도 로드) ──────────────────────────
    def load(self) -> None:
        self._source_by_reviewer.clear()
        for reviewer, path in self._source_files().items():
            rows, _ = _load_csv_rows(path)
            for row in rows.values():
                row.source = "source"
            self._source_by_reviewer[reviewer] = rows

        self._staging_by_reviewer.clear()
        if self.staging_dir.exists():
            for p in sorted(self.staging_dir.glob("exception_*.csv")):
                if p.stem.startswith("exception_history"):
                    continue
                name = p.stem[len("exception_"):]
                rows, _ = _load_csv_rows(p)
                for row in rows.values():
                    row.source = "staging"
                self._staging_by_reviewer[name] = rows

        self._restore_requests.clear()
        rr_path = self._restore_requests_path()
        if rr_path.exists():
            with open(rr_path, "r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    origin = (row.get("origin_no") or "").strip()
                    if origin:
                        self._restore_requests[origin] = (
                            (row.get("requested_by") or "").strip(),
                            (row.get("requested_at") or "").strip(),
                        )

    # ── 조회 (원본 ∪ 스테이징, 복구요청은 원본 쪽만 마스킹) ─────────────
    def _all_reviewers_raw(self, origin_no: str) -> list[tuple[str, ExceptionRow]]:
        out = []
        for reviewer, rows in self._source_by_reviewer.items():
            if origin_no in rows and origin_no not in self._restore_requests:
                out.append((reviewer, rows[origin_no]))
        for reviewer, rows in self._staging_by_reviewer.items():
            if origin_no in rows:
                out.append((f"{reviewer}(로컬대기)", rows[origin_no]))
        return out

    def is_exception(self, origin_no: str) -> bool:
        return len(self._all_reviewers_raw(origin_no)) > 0

    def is_restore_requested(self, origin_no: str) -> bool:
        return origin_no in self._restore_requests

    def get(self, origin_no: str) -> ExceptionRow | None:
        rows = self._all_reviewers_raw(origin_no)
        return rows[0][1] if rows else None

    def get_all_reviewers(self, origin_no: str) -> list[tuple[str, ExceptionRow]]:
        return self._all_reviewers_raw(origin_no)

    @property
    def csv_path(self) -> Path:
        return self.path

    def _reviewer_files(self) -> dict[str, Path]:
        """호환용: 상태표시 등에서 '원본에 검토자가 몇 명 있나' 보여줄 때 사용."""
        return self._source_files()

    # ── 로컬 스테이징 저장 (원본 절대 안 건드림) ────────────────────────
    def _save_staging_reviewer(self, reviewer: str) -> None:
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        path = self._staging_exception_path(reviewer)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            for row in self._staging_by_reviewer.get(reviewer, {}).values():
                writer.writerow({
                    "video_id": row.video_id,
                    "original_gloss": row.origin_no,
                    "original_gloss_name": row.gloss_name,
                    "moved_at": row.moved_at,
                    "reason": row.reason,
                })

    def _save_restore_requests(self) -> None:
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        with open(self._restore_requests_path(), "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["origin_no", "requested_by", "requested_at"])
            for origin, (by, at) in self._restore_requests.items():
                writer.writerow([origin, by, at])

    def _append_history(self, reviewer: str, origin_no: str, gloss_name: str, video_id: str, action: str,
                         reason: str = "") -> None:
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        hist_path = self._staging_history_path(reviewer)
        is_new = not hist_path.exists()
        with open(hist_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["video_id", "original_gloss", "original_gloss_name", "action", "reason", "at"])
            writer.writerow([
                video_id, origin_no, gloss_name, action, reason,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ])

    # ── 변경 (전부 로컬 스테이징에만 기록, 원본 파일 write 없음) ─────────
    def mark_exception(self, origin_no: str, gloss_name: str, video_id: str = "", reviewer: str | None = None,
                        reason: str = "") -> None:
        reviewer = reviewer or self.reviewer
        self._staging_by_reviewer.setdefault(reviewer, {})[origin_no] = ExceptionRow(
            video_id=video_id or f"(video_id 불명, origin_no={origin_no}로 대체)",
            origin_no=origin_no,
            gloss_name=gloss_name,
            moved_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason=reason,
            source="staging",
        )
        self._save_staging_reviewer(reviewer)
        self._append_history(reviewer, origin_no, gloss_name, video_id, "EXCEPTION", reason)

    def restore(self, origin_no: str, reviewer: str | None = None) -> bool:
        """로컬 스테이징에만 있던 항목이면 그냥 제거. 원본(NAS 등)에 있던 항목이면
        원본은 절대 건드리지 않고 '복구 요청됨' 마커만 로컬에 남긴다.
        """
        reviewer = reviewer or self.reviewer
        changed = False

        for rev, rows in list(self._staging_by_reviewer.items()):
            if origin_no in rows:
                gloss_name = rows[origin_no].gloss_name
                video_id = rows[origin_no].video_id
                del rows[origin_no]
                self._save_staging_reviewer(rev)
                self._append_history(rev, origin_no, gloss_name, video_id, "RESTORED_LOCAL")
                changed = True

        still_in_source = any(origin_no in rows for rows in self._source_by_reviewer.values())
        if still_in_source and origin_no not in self._restore_requests:
            self._restore_requests[origin_no] = (reviewer, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self._save_restore_requests()
            self._append_history(reviewer, origin_no, "", "", "RESTORE_REQUESTED(NAS 미반영)")
            changed = True

        return changed
