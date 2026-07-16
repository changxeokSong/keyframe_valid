"""KSL Validator 메인 윈도우.

메타데이터(xlsx/metadata.csv)를 불러와 각 항목을 한국수어사전 공식 동영상과
pose 비교 검증하고, 결과를 눈으로 직접 확인하면서 예외처리(exception_videos.csv)를
추가하거나 해제할 수 있는 검토용 GUI.
"""

from __future__ import annotations

import csv
import getpass
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from .. import sldict_client
from ..dataset_config import DEFAULT_CONFIG_PATH, load_dataset_config
from ..exception_store import DEFAULT_STAGING_DIR, ExceptionStore
from ..keyframe_images import find_keyframe_images
from ..metadata import DatasetEntry, load_dataset
from ..pipeline import ValidationResult
from ..report import build_report, DEFAULT_REPORT_MD_PATH
from .worker import ValidationWorker

IMG_W, IMG_H = 320, 240
DEFAULT_CACHE_DIR = Path("cache/videos")
DEFAULT_HANDSHAPE_DIR = Path("cache/handshape_images")
DEFAULT_REVIEW_LOG = Path("reports/gui_review.csv")
REPO_EXCEPTION_CSV_GUESS = Path(
    "online-sign-keyframe-detection-transformers/tools/tagging/config/etri_ksl_db/exception_videos.csv"
)

STATUS_COLORS = {
    "MATCH": QColor(215, 245, 215),
    "SUSPECT": QColor(255, 220, 200),
    "NO_MY_KEYFRAME": QColor(235, 235, 235),
    "DOWNLOAD_FAILED": QColor(255, 235, 180),
    "NO_POSE_DETECTED": QColor(235, 235, 235),
    "ERROR": QColor(255, 200, 200),
    "": QColor(255, 255, 255),
}

COLS = ["origin_no", "gloss_name", "기존상태", "검증상태", "점수", "손모양", "프레임", "비고"]


def cv2_to_pixmap(frame_bgr: np.ndarray, size=(IMG_W, IMG_H)) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(*size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def placeholder_pixmap(text: str, size=(IMG_W, IMG_H)) -> QPixmap:
    pix = QPixmap(*size)
    pix.fill(QColor(230, 230, 230))
    return pix


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KSL Validator — 한국수어사전 라벨 검증 도구")
        self.resize(1360, 860)

        self.entries: list[DatasetEntry] = []
        self.entry_by_origin: dict[str, DatasetEntry] = {}
        self.results: dict[str, ValidationResult] = {}
        self.manual_keyframe: dict[str, Path] = {}
        self.dataset_root: Optional[Path] = None
        self.keyframe_images_dir: Optional[Path] = None
        self.exception_store: Optional[ExceptionStore] = None
        self.cache_dir = DEFAULT_CACHE_DIR
        self.handshape_dir = DEFAULT_HANDSHAPE_DIR
        self.http_session = requests.Session()
        self.worker: Optional[ValidationWorker] = None
        self._current_video_cap: Optional[cv2.VideoCapture] = None
        self._current_video_total = 0
        self._handshape_paths: list[Path] = []
        self._handshape_idx = 0

        self._init_ui()
        self._try_autoload_exception_csv()
        self._try_autoload_dataset_config()

    # ── UI 구성 ────────────────────────────────────────────────────────
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addLayout(self._build_toolbar())

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._build_table())
        splitter.addWidget(self._build_detail_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("대기 중")
        root.addWidget(self.progress_bar)

    def _build_toolbar(self) -> QVBoxLayout:
        box = QVBoxLayout()

        # ── 1행: 데이터 소스 ──────────────────────────────
        row1 = QHBoxLayout()

        btn_cfg = QPushButton("설정파일(dataset.json) 자동 불러오기")
        btn_cfg.clicked.connect(self._try_autoload_dataset_config)
        row1.addWidget(btn_cfg)

        btn_meta = QPushButton("메타데이터 열기 (xlsx/csv)")
        btn_meta.clicked.connect(self._open_metadata)
        row1.addWidget(btn_meta)

        self.meta_label = QLabel("메타데이터 없음")
        self.meta_label.setStyleSheet("color:#666;")
        row1.addWidget(self.meta_label)

        btn_nas = QPushButton("NAS 데이터셋 루트 지정")
        btn_nas.clicked.connect(self._open_dataset_root)
        row1.addWidget(btn_nas)

        btn_kf_dir = QPushButton("키프레임 사진 폴더 지정")
        btn_kf_dir.clicked.connect(self._open_keyframe_images_dir)
        row1.addWidget(btn_kf_dir)

        btn_exc_dir = QPushButton("예외처리 폴더 열기 (NAS, 읽기전용)")
        btn_exc_dir.setToolTip("이 폴더는 읽기만 합니다. 실제 결정은 로컬 스테이징에만 저장돼요.")
        btn_exc_dir.clicked.connect(self._open_exception_dir)
        row1.addWidget(btn_exc_dir)

        btn_exc = QPushButton("예외처리 CSV 열기 (단일 파일, 읽기전용)")
        btn_exc.clicked.connect(self._open_exception_csv)
        row1.addWidget(btn_exc)

        row1.addWidget(QLabel("검토자:"))
        self.reviewer_edit = QLineEdit(getpass.getuser())
        self.reviewer_edit.setFixedWidth(90)
        self.reviewer_edit.setToolTip("로컬 스테이징에 exception_{검토자}.csv로 기록됩니다 (원본은 안 건드림).")
        self.reviewer_edit.editingFinished.connect(self._on_reviewer_changed)
        row1.addWidget(self.reviewer_edit)

        self.only_exceptions_cb = QCheckBox("예외 항목만 보기")
        self.only_exceptions_cb.stateChanged.connect(lambda _checked: self._refresh_table())
        row1.addWidget(self.only_exceptions_cb)

        row1.addStretch()
        box.addLayout(row1)

        # ── 2행: 소스 상태 표시 ──────────────────────────
        self.source_status_label = QLabel(
            "NAS 데이터셋 루트: 미지정 | 키프레임 사진 폴더: 미지정 | 예외처리(읽기전용): 미지정 | "
            f"로컬 스테이징(쓰기): {DEFAULT_STAGING_DIR}"
        )
        self.source_status_label.setStyleSheet("color:#444; padding:2px;")
        self.source_status_label.setWordWrap(True)
        box.addWidget(self.source_status_label)

        # ── 3행: 검증 실행 컨트롤 ──────────────────────────
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("threshold"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.7)
        row3.addWidget(self.threshold_spin)

        row3.addWidget(QLabel("stride"))
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 20)
        self.stride_spin.setValue(3)
        row3.addWidget(self.stride_spin)

        self.btn_validate_selected = QPushButton("선택 항목 검증")
        self.btn_validate_selected.clicked.connect(lambda: self._run_validation(selected_only=True))
        row3.addWidget(self.btn_validate_selected)

        self.btn_validate_all = QPushButton("전체 검증 실행")
        self.btn_validate_all.clicked.connect(lambda: self._run_validation(selected_only=False))
        row3.addWidget(self.btn_validate_all)

        self.btn_stop = QPushButton("중지")
        self.btn_stop.clicked.connect(self._stop_validation)
        self.btn_stop.setEnabled(False)
        row3.addWidget(self.btn_stop)

        row3.addStretch()

        btn_report = QPushButton("📄 요약 리포트 생성")
        btn_report.clicked.connect(self._generate_report)
        row3.addWidget(btn_report)

        box.addLayout(row3)
        return box

    def _build_table(self) -> QTableWidget:
        self.table = QTableWidget()
        self.table.setColumnCount(len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        return self.table

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)

        # 내 키프레임
        left_box = QVBoxLayout()
        left_box.addWidget(QLabel("<b>내 키프레임</b>"))
        self.my_kf_label = QLabel()
        self.my_kf_label.setFixedSize(IMG_W, IMG_H)
        self.my_kf_label.setStyleSheet("background:#eee; border:1px solid #ccc;")
        self.my_kf_label.setAlignment(Qt.AlignCenter)
        left_box.addWidget(self.my_kf_label)
        btn_manual_kf = QPushButton("키프레임 이미지 직접 지정...")
        btn_manual_kf.clicked.connect(self._pick_manual_keyframe)
        left_box.addWidget(btn_manual_kf)
        left_box.addStretch()
        layout.addLayout(left_box)

        # 사전 사이트 프레임 (스크러버)
        mid_box = QVBoxLayout()
        mid_box.addWidget(QLabel("<b>사전 사이트 동영상 (최고매칭 프레임으로 이동됨)</b>"))
        self.site_frame_label = QLabel()
        self.site_frame_label.setFixedSize(IMG_W, IMG_H)
        self.site_frame_label.setStyleSheet("background:#eee; border:1px solid #ccc;")
        self.site_frame_label.setAlignment(Qt.AlignCenter)
        mid_box.addWidget(self.site_frame_label)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self._on_slider_moved)
        mid_box.addWidget(self.frame_slider)
        self.frame_idx_label = QLabel("프레임: -")
        mid_box.addWidget(self.frame_idx_label)
        btn_open_site = QPushButton("사전 사이트에서 직접 열기")
        btn_open_site.clicked.connect(self._open_in_browser)
        mid_box.addWidget(btn_open_site)
        mid_box.addStretch()
        layout.addLayout(mid_box)

        # 사전 공식 참고 이미지 (수형사진 - 일러스트, 사람 눈으로 참고용)
        ref_box = QVBoxLayout()
        ref_box.addWidget(QLabel("<b>사전 공식 참고(수형사진, 일러스트)</b>"))
        self.handshape_label = QLabel()
        self.handshape_label.setFixedSize(IMG_W, IMG_H)
        self.handshape_label.setStyleSheet("background:#eee; border:1px solid #ccc;")
        self.handshape_label.setAlignment(Qt.AlignCenter)
        self.handshape_label.setWordWrap(True)
        ref_box.addWidget(self.handshape_label)
        btn_next_handshape = QPushButton("다음 이미지")
        btn_next_handshape.clicked.connect(self._show_next_handshape)
        ref_box.addWidget(btn_next_handshape)
        ref_box.addStretch()
        layout.addLayout(ref_box)

        # 점수/판정/액션
        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("<b>검증 정보 및 판정</b>"))
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        right_box.addWidget(self.detail_text)

        self.btn_confirm_ok = QPushButton("✔ 정상 확인")
        self.btn_confirm_ok.clicked.connect(self._confirm_ok)
        right_box.addWidget(self.btn_confirm_ok)

        self.btn_mark_exception = QPushButton("⚠ 라벨 불일치 → 예외처리")
        self.btn_mark_exception.clicked.connect(self._mark_exception)
        right_box.addWidget(self.btn_mark_exception)

        self.btn_restore = QPushButton("↺ 예외 해제(정상으로 복구)")
        self.btn_restore.clicked.connect(self._restore_exception)
        right_box.addWidget(self.btn_restore)

        layout.addLayout(right_box)
        return panel

    # ── 파일 열기 ──────────────────────────────────────────────────────
    def _open_metadata(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "메타데이터 선택", "", "Metadata (*.xlsx *.xls *.csv)"
        )
        if not path:
            return
        try:
            self.entries = load_dataset(Path(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "오류", f"메타데이터 로드 실패:\n{e}")
            return
        self.entry_by_origin = {e.origin_no: e for e in self.entries}
        self.meta_label.setText(f"{Path(path).name} ({len(self.entries)}개 항목)")
        self._refresh_table()

    def _open_dataset_root(self):
        path = QFileDialog.getExistingDirectory(self, "NAS 데이터셋 루트 선택")
        if path:
            self.dataset_root = Path(path)
            self._update_source_status_label()

    def _open_keyframe_images_dir(self):
        path = QFileDialog.getExistingDirectory(self, "키프레임 사진 폴더 선택 (keyframe_images)")
        if path:
            self.keyframe_images_dir = Path(path)
            self._update_source_status_label()

    def _open_exception_csv(self):
        """단일 파일 모드 (로컬 git 샘플 등, 검토자 구분 없는 파일 하나)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "예외처리 CSV 선택",
            str(REPO_EXCEPTION_CSV_GUESS if REPO_EXCEPTION_CSV_GUESS.exists() else Path(".")),
            "CSV (*.csv)",
        )
        if not path:
            return
        self.exception_store = ExceptionStore(Path(path), reviewer=self.reviewer_edit.text().strip())
        self._update_source_status_label()
        self._refresh_table()

    def _open_exception_dir(self):
        """디렉토리 모드 (NAS 실제 구조: exception_{이름}.csv 여러 개가 한 폴더에 있음)."""
        path = QFileDialog.getExistingDirectory(
            self, "exception_*.csv 들이 있는 폴더 선택 (보통 etri_ksl_db 루트)"
        )
        if not path:
            return
        self.exception_store = ExceptionStore(Path(path), reviewer=self.reviewer_edit.text().strip())
        self._update_source_status_label()
        self._refresh_table()

    def _on_reviewer_changed(self):
        if self.exception_store is not None:
            self.exception_store.reviewer = self.reviewer_edit.text().strip()

    def _try_autoload_exception_csv(self):
        if REPO_EXCEPTION_CSV_GUESS.exists():
            self.exception_store = ExceptionStore(REPO_EXCEPTION_CSV_GUESS, reviewer=getpass.getuser())

    def _try_autoload_dataset_config(self):
        """tools/tagging/config/dataset.json 을 읽어 NAS 경로들을 자동으로 채운다.
        NAS가 마운트 안 돼 있으면 경로만 채우고 상태 라벨에 '미마운트'로 표시한다.
        exception_*.csv / exception_history_*.csv 는 dataset_root와 같은 폴더에
        검토자별로 흩어져 있으므로, dataset_root가 마운트되면 그 폴더를
        예외처리 디렉토리로 자동 지정한다(단, 사용자가 이미 다른 소스를 지정했으면 덮어쓰지 않음).
        """
        cfg = load_dataset_config(DEFAULT_CONFIG_PATH)
        if cfg is None:
            self._update_source_status_label()
            return

        if cfg.dataset_root.mounted:
            self.dataset_root = cfg.dataset_root.resolved
            if self.exception_store is None or self.exception_store.path == REPO_EXCEPTION_CSV_GUESS:
                self.exception_store = ExceptionStore(
                    cfg.dataset_root.resolved, reviewer=self.reviewer_edit.text().strip()
                )
        if cfg.handshape_image_dir.mounted:
            self.keyframe_images_dir = cfg.handshape_image_dir.resolved

        # 메타데이터가 아직 없을 때만 자동 로드 시도 (사용자가 이미 연 걸 덮어쓰지 않음)
        if not self.entries:
            meta_candidate = None
            if cfg.metadata_file.mounted:
                meta_candidate = cfg.metadata_file.resolved
            elif cfg.excel_file.mounted:
                meta_candidate = cfg.excel_file.resolved
            if meta_candidate is not None:
                try:
                    self.entries = load_dataset(meta_candidate)
                    self.entry_by_origin = {e.origin_no: e for e in self.entries}
                    self.meta_label.setText(f"{meta_candidate.name} ({len(self.entries)}개 항목, 자동)")
                    self._refresh_table()
                except Exception:  # noqa: BLE001
                    pass

        self._last_dataset_config = cfg
        self._update_source_status_label()

    def _update_source_status_label(self):
        def fmt(label: str, resolved: Optional[Path], raw: str = "") -> str:
            if resolved is not None:
                return f"{label}: ✔ {resolved}"
            if raw:
                return f"{label}: ✘ 미마운트 ({raw})"
            return f"{label}: 미지정"

        cfg = getattr(self, "_last_dataset_config", None)
        exc_desc = None
        if self.exception_store:
            n_reviewers = len(self.exception_store._reviewer_files())
            mode = f"디렉토리, 검토자 {n_reviewers}명" if self.exception_store.dir_mode else "단일 파일"
            exc_desc = f"{self.exception_store.path} ({mode}, 읽기전용)"
        parts = [
            fmt("NAS 데이터셋 루트", self.dataset_root, cfg.dataset_root.raw if cfg else ""),
            fmt("키프레임 사진 폴더", self.keyframe_images_dir, cfg.handshape_image_dir.raw if cfg else ""),
            f"예외처리 원본: {exc_desc}" if exc_desc else "예외처리 원본: 미지정",
            f"로컬 스테이징(쓰기): {self.exception_store.staging_dir if self.exception_store else DEFAULT_STAGING_DIR}",
        ]
        self.source_status_label.setText(" | ".join(parts))

    def _pick_manual_keyframe(self):
        entry = self._selected_entry()
        if entry is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "키프레임 이미지 선택", "", "Images (*.jpg *.jpeg *.png)")
        if path:
            self.manual_keyframe[entry.origin_no] = Path(path)
            self._show_detail_for(entry)

    # ── 테이블 ─────────────────────────────────────────────────────────
    def _visible_entries(self) -> list[DatasetEntry]:
        if self.only_exceptions_cb.isChecked() and self.exception_store is not None:
            return [e for e in self.entries if self.exception_store.is_exception(e.origin_no)]
        return self.entries

    def _refresh_table(self):
        self.table.setSortingEnabled(False)
        visible = self._visible_entries()
        self.table.setRowCount(len(visible))
        for row, entry in enumerate(visible):
            self._fill_row(row, entry)
        self.table.setSortingEnabled(True)

    def _fill_row(self, row: int, entry: DatasetEntry):
        result = self.results.get(entry.origin_no)
        is_exc = self.exception_store.is_exception(entry.origin_no) if self.exception_store else False

        status = result.status if result else ""
        score = f"{result.best_score:.4f}" if result and result.best_score is not None else ""
        hand = ("Y" if result.hand_used else "N") if result else ""
        frame = str(result.best_frame_idx) if result and result.best_frame_idx is not None else ""
        note = result.note if result else ""

        values = [
            entry.origin_no, entry.gloss_name,
            "예외" if is_exc else "정상", status, score, hand, frame, note,
        ]
        bg = STATUS_COLORS.get(status, STATUS_COLORS[""])
        if is_exc:
            bg = QColor(230, 220, 255)

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setBackground(bg)
            self.table.setItem(row, col, item)

    def _selected_entry(self) -> Optional[DatasetEntry]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        origin_no = self.table.item(rows[0].row(), 0).text()
        return self.entry_by_origin.get(origin_no)

    def _selected_entries(self) -> list[DatasetEntry]:
        rows = self.table.selectionModel().selectedRows()
        out = []
        for r in rows:
            origin_no = self.table.item(r.row(), 0).text()
            e = self.entry_by_origin.get(origin_no)
            if e:
                out.append(e)
        return out

    # ── 검증 실행 ──────────────────────────────────────────────────────
    def _run_validation(self, selected_only: bool):
        if not self.entries:
            QMessageBox.warning(self, "안내", "먼저 메타데이터를 열어주세요.")
            return
        entries = self._selected_entries() if selected_only else self.entries
        if not entries:
            QMessageBox.warning(self, "안내", "선택된 항목이 없습니다.")
            return

        self.worker = ValidationWorker(
            entries, self.cache_dir, dataset_root=self.dataset_root,
            keyframe_images_dir=self.keyframe_images_dir,
            threshold=self.threshold_spin.value(), stride=self.stride_spin.value(),
        )
        self.worker.result_ready.connect(self._on_result_ready)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_all.connect(self._on_validation_finished)
        self.worker.log.connect(lambda msg: QMessageBox.warning(self, "안내", msg))

        self.btn_validate_all.setEnabled(False)
        self.btn_validate_selected.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setFormat("검증 중... %v/%m")
        self.progress_bar.setRange(0, len(entries))
        self.progress_bar.setValue(0)
        self.worker.start()

    def _stop_validation(self):
        if self.worker:
            self.worker.stop()

    def _on_result_ready(self, origin_no: str, result: ValidationResult):
        self.results[origin_no] = result
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == origin_no:
                self._fill_row(row, self.entry_by_origin[origin_no])
                break
        entry = self._selected_entry()
        if entry and entry.origin_no == origin_no:
            self._show_detail_for(entry)

    def _on_progress(self, done: int, total: int):
        self.progress_bar.setValue(done)

    def _on_validation_finished(self):
        self.btn_validate_all.setEnabled(True)
        self.btn_validate_selected.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setFormat("완료")

    # ── 상세 패널 ──────────────────────────────────────────────────────
    def _on_row_selected(self):
        entry = self._selected_entry()
        if entry:
            self._show_detail_for(entry)

    def _release_current_cap(self):
        if self._current_video_cap is not None:
            self._current_video_cap.release()
            self._current_video_cap = None

    def _show_detail_for(self, entry: DatasetEntry):
        self._release_current_cap()
        result = self.results.get(entry.origin_no)
        is_exc = self.exception_store.is_exception(entry.origin_no) if self.exception_store else False

        # 내 키프레임 표시
        my_frame = self._load_my_keyframe_frame(entry)
        if my_frame is not None:
            self.my_kf_label.setPixmap(cv2_to_pixmap(my_frame))
        else:
            self.my_kf_label.setText("(키프레임 없음 - NAS 미마운트\n또는 직접 지정 필요)")
            self.my_kf_label.setPixmap(QPixmap())

        # 사전 사이트 프레임 표시 + 슬라이더 세팅
        if result and result.video_path and Path(result.video_path).exists():
            self._current_video_cap = cv2.VideoCapture(result.video_path)
            self._current_video_total = int(self._current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, max(0, self._current_video_total - 1))
            self.frame_slider.setValue(result.best_frame_idx or 0)
            self.frame_slider.blockSignals(False)
            self._show_video_frame(result.best_frame_idx or 0)
        else:
            self.site_frame_label.setText("(아직 검증 안 됨)")
            self.site_frame_label.setPixmap(QPixmap())
            self.frame_slider.setRange(0, 0)
            self.frame_idx_label.setText("프레임: -")

        # 사전 공식 참고 이미지(수형사진)
        self._load_handshape_images(entry)

        # 텍스트 요약
        if is_exc:
            reviewers = self.exception_store.get_all_reviewers(entry.origin_no)
            exc_desc = "예외 (" + "; ".join(f"{rev}: {row.moved_at}" for rev, row in reviewers) + ")"
        else:
            exc_desc = "정상"
        lines = [
            f"origin_no: {entry.origin_no}",
            f"gloss_name: {entry.gloss_name}",
            f"gloss_description: {entry.gloss_description}",
            f"metadata keyframes: {entry.keyframes}",
            f"기존 예외처리 상태: {exc_desc}",
        ]
        if result:
            lines += [
                "",
                f"검증 상태: {result.status}",
                f"최고 유사도: {result.best_score:.4f}" if result.best_score is not None else "최고 유사도: -",
                f"최고매칭 프레임: {result.best_frame_idx}",
                f"손모양 신호 반영: {'예' if result.hand_used else '아니오 (body pose만 사용 - 신뢰도 낮음)'}",
                f"스캔한 프레임 수: {result.frames_scanned}",
                f"비고: {result.note}",
                "",
                "※ 점수는 참고용 우선순위입니다. 반드시 좌/우 이미지를 눈으로 비교해 최종 판단하세요.",
            ]
        else:
            lines.append("\n(아직 검증되지 않음 — '선택 항목 검증'을 눌러주세요)")

        self.detail_text.setPlainText("\n".join(lines))
        self.btn_restore.setEnabled(is_exc)
        self.btn_mark_exception.setEnabled(not is_exc)

    def _load_my_keyframe_frame(self, entry: DatasetEntry) -> Optional[np.ndarray]:
        """'내 키프레임' 표시용 원본 프레임 로드. 우선순위:
        수동 지정 이미지 > NAS keyframe_images 실제 사진 > NAS 원본 비디오 프레임 탐색
        (pipeline.validate_entry 와 동일한 우선순위를 사용해 화면과 채점이 일치하게 한다)
        """
        override = self.manual_keyframe.get(entry.origin_no)
        if override is not None:
            frame = cv2.imread(str(override))
            if frame is not None:
                return frame

        if self.keyframe_images_dir is not None:
            matches = find_keyframe_images(self.keyframe_images_dir, entry.origin_no)
            if matches:
                frame = cv2.imread(str(matches[0]))
                if frame is not None:
                    return frame

        if self.dataset_root is not None and entry.video_rel_path and entry.keyframes:
            video_path = self.dataset_root / entry.video_rel_path
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, entry.keyframes[0])
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return frame
        return None

    def _load_handshape_images(self, entry: DatasetEntry):
        """사전 사이트의 '수형 사진'(일러스트) 로드. 참고용이며 자동 채점엔 안 씀."""
        self._handshape_paths = []
        self._handshape_idx = 0
        try:
            self._handshape_paths = sldict_client.download_handshape_images(
                self.http_session, entry.origin_no, self.handshape_dir
            )
        except Exception as e:  # noqa: BLE001 - 오프라인/사이트 오류 시 조용히 비워둠
            self.handshape_label.setText(f"(불러오기 실패)\n{e}")
            self.handshape_label.setPixmap(QPixmap())
            return

        if not self._handshape_paths:
            self.handshape_label.setText("(참고 이미지 없음)")
            self.handshape_label.setPixmap(QPixmap())
            return
        self._show_handshape_at(0)

    def _show_handshape_at(self, idx: int):
        if not self._handshape_paths:
            return
        idx = idx % len(self._handshape_paths)
        self._handshape_idx = idx
        frame = cv2.imread(str(self._handshape_paths[idx]))
        if frame is not None:
            self.handshape_label.setPixmap(cv2_to_pixmap(frame))
            if len(self._handshape_paths) > 1:
                self.handshape_label.setToolTip(f"{idx + 1}/{len(self._handshape_paths)}")

    def _show_next_handshape(self):
        if self._handshape_paths:
            self._show_handshape_at(self._handshape_idx + 1)

    def _on_slider_moved(self, value: int):
        self.frame_idx_label.setText(f"프레임: {value} / {self._current_video_total - 1}")
        self._show_video_frame(value)

    def _show_video_frame(self, idx: int):
        if self._current_video_cap is None:
            return
        self._current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._current_video_cap.read()
        if ret:
            self.site_frame_label.setPixmap(cv2_to_pixmap(frame))
        self.frame_idx_label.setText(f"프레임: {idx} / {self._current_video_total - 1}")

    # ── 리뷰 액션 ──────────────────────────────────────────────────────
    def _confirm_ok(self):
        entry = self._selected_entry()
        if entry is None:
            return
        self._append_review_log(entry, "OK")
        QMessageBox.information(self, "기록됨", f"{entry.gloss_name}(origin_no={entry.origin_no}) 정상 확인 기록됨")

    def _ensure_exception_store(self):
        """staging에는 항상 쓸 수 있으므로 원본(source)이 없어도(=파일이 없어도) 빈 상태로 시작 가능."""
        if self.exception_store is None:
            self.exception_store = ExceptionStore(
                REPO_EXCEPTION_CSV_GUESS, reviewer=self.reviewer_edit.text().strip(),
            )

    def _mark_exception(self):
        """선택된 여러 항목을 한꺼번에 예외처리. 이미 누군가(원본이든 로컬이든)
        예외처리한 항목은 중복 방지를 위해 자동으로 건너뛴다."""
        entries = self._selected_entries()
        if not entries:
            return
        self._ensure_exception_store()

        already = [e for e in entries if self.exception_store.is_exception(e.origin_no)]
        targets = [e for e in entries if e not in already]

        if not targets:
            QMessageBox.information(self, "안내", f"선택한 {len(entries)}개 항목 모두 이미 예외처리되어 있습니다 (중복 방지로 건너뜀).")
            return

        msg = f"{len(targets)}개 항목을 라벨 불일치로 예외처리 하시겠습니까?"
        if already:
            msg += f"\n(이미 예외처리된 {len(already)}개는 중복이라 건너뜁니다)"
        reply = QMessageBox.question(self, "예외처리 확인(일괄)", msg)
        if reply != QMessageBox.Yes:
            return

        for entry in targets:
            self.exception_store.mark_exception(entry.origin_no, entry.gloss_name, entry.video_id or "")
            self._append_review_log(entry, "EXCEPTION")
            self._fill_row_by_origin(entry.origin_no)

        sel = self._selected_entry()
        if sel:
            self._show_detail_for(sel)

    def _restore_exception(self):
        """선택된 여러 항목을 한꺼번에 복구. 로컬 전용 예외는 바로 제거되고,
        원본(NAS 등)에 있던 항목은 원본을 건드리지 않고 '복구 요청'만 로컬에 남긴다."""
        entries = self._selected_entries()
        if not entries or self.exception_store is None:
            return

        targets = [e for e in entries if self.exception_store.is_exception(e.origin_no)]
        if not targets:
            QMessageBox.information(self, "안내", "선택한 항목 중 예외처리된 것이 없습니다.")
            return

        reply = QMessageBox.question(
            self, "예외 해제 확인(일괄)",
            f"{len(targets)}개 항목을 예외처리에서 해제(정상으로 복구)하시겠습니까?\n"
            f"※ 원본(NAS 등) 파일에 있던 항목은 원본을 직접 지우지 않고 "
            f"'복구 요청됨'으로만 로컬에 표시됩니다.",
        )
        if reply != QMessageBox.Yes:
            return

        for entry in targets:
            if self.exception_store.restore(entry.origin_no):
                self._append_review_log(entry, "RESTORED")
            self._fill_row_by_origin(entry.origin_no)

        sel = self._selected_entry()
        if sel:
            self._show_detail_for(sel)
        self._fill_row_by_origin(entry.origin_no)
        self._show_detail_for(entry)

    def _fill_row_by_origin(self, origin_no: str):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == origin_no:
                self._fill_row(row, self.entry_by_origin[origin_no])
                break

    def _append_review_log(self, entry: DatasetEntry, decision: str):
        result = self.results.get(entry.origin_no)
        DEFAULT_REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
        is_new = not DEFAULT_REVIEW_LOG.exists()
        with open(DEFAULT_REVIEW_LOG, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["origin_no", "gloss_name", "decision", "best_score", "reviewed_at"])
            writer.writerow([
                entry.origin_no, entry.gloss_name, decision,
                f"{result.best_score:.4f}" if result and result.best_score is not None else "",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ])

    def _open_in_browser(self):
        entry = self._selected_entry()
        if entry is None:
            return
        webbrowser.open(f"https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no={entry.origin_no}")

    # ── 리포트 ─────────────────────────────────────────────────────────
    def _generate_report(self):
        if not self.entries:
            QMessageBox.warning(self, "안내", "먼저 메타데이터를 열고 검증을 실행해주세요.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "요약 리포트 저장", str(DEFAULT_REPORT_MD_PATH), "Markdown (*.md)"
        )
        if not path:
            return
        out_path = build_report(
            self.entries, self.results, self.exception_store, DEFAULT_REVIEW_LOG,
            self.dataset_root, self.keyframe_images_dir, output_path=Path(path),
        )
        reply = QMessageBox.information(
            self, "리포트 생성 완료", f"저장됨: {out_path}\n\n지금 열어볼까요?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            webbrowser.open(out_path.resolve().as_uri())

    def closeEvent(self, event):
        self._release_current_cap()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        super().closeEvent(event)
