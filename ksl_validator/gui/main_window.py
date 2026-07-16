"""KSL Validator 메인 윈도우.

메타데이터(xlsx/metadata.csv)를 불러와 각 항목을 한국수어사전 공식 동영상과
pose 비교 검증하고, 결과를 눈으로 직접 확인하면서 예외처리(exception_videos.csv)를
추가하거나 해제할 수 있는 검토용 GUI.
"""

from __future__ import annotations

import csv
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from .. import local_settings
from ..dataset_config import DEFAULT_CONFIG_PATH, load_dataset_config
from ..exception_store import DEFAULT_STAGING_DIR, ExceptionStore
from ..logging_setup import log, log_time
from ..metadata import DatasetEntry, load_dataset
from ..paths import HANDSHAPE_CACHE_DIR, REVIEW_LOG_PATH, SAMPLE_EXCEPTION_CSV_PATH, VIDEOS_CACHE_DIR
from ..pipeline import ValidationResult
from ..compare import combined_similarity
from ..pipeline import Extractors
from ..report import build_report, DEFAULT_REPORT_MD_PATH
from ..user_info import safe_getuser
from .worker import HandshapeFetchThread, KeyframeLoadThread, ValidationWorker

IMG_W, IMG_H = 320, 240

# YOLO COCO 17-keypoint 스켈레톤 연결선 (팔/어깨/골반/다리만 - 얼굴은 판별에 안 씀)
BODY_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
    (5, 6), (5, 11), (6, 12), (11, 12),  # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16),  # 다리
]
# MediaPipe 21점 손가락 체인 (엄지/검지/중지/약지/새끼)
HAND_CHAINS = [
    (0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20),
]

DEFAULT_CACHE_DIR = VIDEOS_CACHE_DIR
DEFAULT_HANDSHAPE_DIR = HANDSHAPE_CACHE_DIR
DEFAULT_REVIEW_LOG = REVIEW_LOG_PATH
REPO_EXCEPTION_CSV_GUESS = SAMPLE_EXCEPTION_CSV_PATH

STATUS_COLORS = {
    "MATCH": QColor(215, 245, 215),
    "SUSPECT": QColor(255, 220, 200),
    "NO_MY_KEYFRAME": QColor(235, 235, 235),
    "DOWNLOAD_FAILED": QColor(255, 235, 180),
    "NO_POSE_DETECTED": QColor(235, 235, 235),
    "ERROR": QColor(255, 200, 200),
    "": QColor(255, 255, 255),
}

COLS = ["origin_no", "gloss_name", "video_id(subset/사람)", "기존상태", "검증상태", "점수", "손모양", "프레임", "비고"]


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
        self._entries_by_gloss: dict[str, list[DatasetEntry]] = {}  # gloss_name -> entries (기준 이미지 찾기용)
        self._item_by_origin: dict[str, QTableWidgetItem] = {}  # origin_no -> col0 item (O(1) 행 찾기용)
        self.results: dict[str, ValidationResult] = {}
        self.manual_keyframe: dict[str, Path] = {}
        self.dataset_root: Optional[Path] = None
        self.keyframe_images_dir: Optional[Path] = None
        self.exception_store: Optional[ExceptionStore] = None
        self.cache_dir = DEFAULT_CACHE_DIR
        self.handshape_dir = DEFAULT_HANDSHAPE_DIR
        self.worker: Optional[ValidationWorker] = None
        self._current_video_cap: Optional[cv2.VideoCapture] = None
        self._current_video_total = 0
        self._handshape_paths: list[Path] = []
        self._handshape_frames: list[np.ndarray] = []
        self._handshape_cache: dict[str, list[Path]] = {}  # origin_no -> paths (빈 리스트 = 확인함, 없음)
        self._handshape_thread: Optional[HandshapeFetchThread] = None
        self._my_kf_frames: list[np.ndarray] = []
        self._my_kf_cache: dict[str, list] = {}  # origin_no -> frames, NAS 재읽기 방지
        self._ref_kf_frames: list[np.ndarray] = []
        self._ref_kf_sources: list[str] = []
        self._ref_kf_cache: dict[str, tuple] = {}  # gloss_name -> (frames, sources), NAS 재읽기 방지
        self._kf_thread: Optional[KeyframeLoadThread] = None
        self._extractors: Optional[Extractors] = None  # 지연 로딩 (첫 비교 때 한 번만 모델 로드)
        self._validating_origin_no: Optional[str] = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        self._init_ui()
        self._try_autoload_exception_csv()
        self._try_load_local_settings()
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

        root.addLayout(self._build_live_progress_row())

        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("대기 중")
        root.addWidget(self.progress_bar)

    def _build_live_progress_row(self) -> QHBoxLayout:
        """검증(YOLO+MediaPipe pose 비교) 진행 중 지금 훑고 있는 프레임을 실시간으로 보여준다."""
        row = QHBoxLayout()
        self.live_preview_label = QLabel()
        self.live_preview_label.setFixedSize(160, 120)
        self.live_preview_label.setStyleSheet("background:#111; border:1px solid #ccc;")
        self.live_preview_label.setAlignment(Qt.AlignCenter)
        row.addWidget(self.live_preview_label)

        self.live_preview_text = QLabel("검증 대기 중")
        self.live_preview_text.setStyleSheet("color:#444;")
        row.addWidget(self.live_preview_text, 1)
        return row

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
        self.reviewer_edit = QLineEdit(safe_getuser())
        self.reviewer_edit.setFixedWidth(90)
        self.reviewer_edit.setToolTip("로컬 스테이징에 exception_{검토자}.csv로 기록됩니다 (원본은 안 건드림).")
        self.reviewer_edit.editingFinished.connect(self._on_reviewer_changed)
        row1.addWidget(self.reviewer_edit)

        self.only_exceptions_cb = QCheckBox("예외 항목만 보기")
        self.only_exceptions_cb.stateChanged.connect(lambda _checked: self._refresh_table())
        row1.addWidget(self.only_exceptions_cb)

        self.show_pose_cb = QCheckBox("포즈/손 keypoint 시각적으로 표시")
        self.show_pose_cb.setToolTip(
            "검토대상/정답 키프레임 위에 YOLO body pose + MediaPipe 손가락 keypoint를 그려서 보여줍니다. "
            "실제로 뭘 검출하는지 눈으로 바로 확인할 수 있어요 (첫 사용시 모델 로딩으로 잠깐 걸릴 수 있음)."
        )
        self.show_pose_cb.stateChanged.connect(self._on_show_pose_toggled)
        row1.addWidget(self.show_pose_cb)

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
        self.table.setColumnWidth(1, 180)  # gloss_name
        self.table.setColumnWidth(2, 180)  # video_id(subset/사람) - gloss_name과 같은 폭
        self.table.horizontalHeader().setSectionResizeMode(len(COLS) - 1, QHeaderView.Stretch)  # 비고
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        return self.table

    def _build_slider_panel(self, title: str, on_slide) -> tuple[QVBoxLayout, QLabel, QSlider, QLabel]:
        """이미지/프레임이 여러 개일 수 있는 패널의 공통 뼈대.
        내 키프레임 / 기준 이미지 / 사전 동영상 / 수형사진 전부 이 슬라이더
        패턴 하나로 통일한다 (일관된 조작감).
        """
        box = QVBoxLayout()
        box.addWidget(QLabel(f"<b>{title}</b>"))

        img_label = QLabel()
        img_label.setFixedSize(IMG_W, IMG_H)
        img_label.setStyleSheet("background:#eee; border:1px solid #ccc;")
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setWordWrap(True)
        box.addWidget(img_label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 0)
        slider.valueChanged.connect(on_slide)
        box.addWidget(slider)

        idx_label = QLabel("- / -")
        box.addWidget(idx_label)

        return box, img_label, slider, idx_label

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)

        # 검토 대상 키프레임 (지금 선택한 행 - 예외처리된 항목이면 그 문제의 영상 키프레임이 여기 뜸)
        left_box, self.my_kf_label, self.my_kf_slider, self.my_kf_idx_label = self._build_slider_panel(
            "검토 대상 키프레임 (지금 선택한 항목 — 예외처리된 항목이면 그 영상)",
            self._on_my_kf_slider_moved,
        )
        btn_manual_kf = QPushButton("키프레임 이미지 직접 지정...")
        btn_manual_kf.clicked.connect(self._pick_manual_keyframe)
        left_box.addWidget(btn_manual_kf)
        left_box.addStretch()
        layout.addLayout(left_box)

        # 정답 기준 이미지 (같은 글로스의 다른 정상(비예외) 영상 키프레임)
        ref_ex_box, self.ref_kf_label, self.ref_kf_slider, self.ref_kf_idx_label = self._build_slider_panel(
            "정답 키프레임 (같은 글로스의 다른 정상 영상)", self._on_ref_kf_slider_moved
        )
        self.ref_kf_source_label = QLabel("")
        self.ref_kf_source_label.setWordWrap(True)
        self.ref_kf_source_label.setStyleSheet("color:#666;")
        ref_ex_box.addWidget(self.ref_kf_source_label)
        self.my_vs_ref_label = QLabel("검토대상 vs 정답 직접비교: -")
        self.my_vs_ref_label.setWordWrap(True)
        self.my_vs_ref_label.setStyleSheet("font-weight:bold;")
        ref_ex_box.addWidget(self.my_vs_ref_label)
        btn_compare = QPushButton("지금 보이는 프레임끼리 직접 비교")
        btn_compare.setToolTip("첫 사용시 모델 로딩으로 1~2초 걸릴 수 있습니다.")
        btn_compare.clicked.connect(self._update_my_vs_ref_score)
        ref_ex_box.addWidget(btn_compare)
        ref_ex_box.addStretch()
        layout.addLayout(ref_ex_box)

        # 사전 사이트 동영상 (최고매칭 프레임 - 재생도 가능)
        mid_box, self.site_frame_label, self.frame_slider, self.frame_idx_label = self._build_slider_panel(
            "사전 사이트 동영상 (최고매칭 프레임으로 이동됨)", self._on_slider_moved
        )
        play_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ 재생")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_play.setEnabled(False)
        play_row.addWidget(self.btn_play)
        mid_box.addLayout(play_row)
        btn_open_site = QPushButton("사전 사이트에서 직접 열기")
        btn_open_site.clicked.connect(self._open_in_browser)
        mid_box.addWidget(btn_open_site)
        mid_box.addStretch()
        layout.addLayout(mid_box)

        # 사전 공식 참고 이미지 (수형사진 - 일러스트, 사람 눈으로 참고용)
        ref_box, self.handshape_label, self.handshape_slider, self.handshape_idx_label = self._build_slider_panel(
            "사전 공식 참고(수형사진, 일러스트)", self._on_handshape_slider_moved
        )
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
        self._index_entries()
        self.meta_label.setText(f"{Path(path).name} ({len(self.entries)}개 항목)")
        self._refresh_table()
        local_settings.update(metadata_path=path)

    def _open_dataset_root(self):
        path = QFileDialog.getExistingDirectory(self, "NAS 데이터셋 루트 선택")
        if path:
            self.dataset_root = Path(path)
            self._update_source_status_label()
            local_settings.update(dataset_root=path)

    def _open_keyframe_images_dir(self):
        path = QFileDialog.getExistingDirectory(self, "키프레임 사진 폴더 선택 (keyframe_images)")
        if path:
            self.keyframe_images_dir = Path(path)
            self._update_source_status_label()
            local_settings.update(keyframe_images_dir=path)

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
        local_settings.update(exception_source=path)

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
        local_settings.update(exception_source=path)

    def _on_reviewer_changed(self):
        reviewer = self.reviewer_edit.text().strip()
        if self.exception_store is not None:
            self.exception_store.reviewer = reviewer
        local_settings.update(reviewer=reviewer)

    def _try_autoload_exception_csv(self):
        if REPO_EXCEPTION_CSV_GUESS.exists():
            self.exception_store = ExceptionStore(REPO_EXCEPTION_CSV_GUESS, reviewer=safe_getuser())

    def _try_load_local_settings(self):
        """이 컴퓨터에서 이전에 수동으로 지정해뒀던 경로들을 불러온다.
        (NAS 마운트 위치는 컴퓨터마다 달라서 dataset.json 자동추측보다 우선한다)
        """
        settings = local_settings.load()
        if not settings:
            return

        if settings.get("dataset_root"):
            p = Path(settings["dataset_root"])
            if p.exists():
                self.dataset_root = p
        if settings.get("keyframe_images_dir"):
            p = Path(settings["keyframe_images_dir"])
            if p.exists():
                self.keyframe_images_dir = p
        if settings.get("exception_source"):
            p = Path(settings["exception_source"])
            if p.exists():
                self.exception_store = ExceptionStore(p, reviewer=settings.get("reviewer", "") or safe_getuser())
        if settings.get("reviewer"):
            self.reviewer_edit.setText(settings["reviewer"])
        if settings.get("metadata_path") and not self.entries:
            p = Path(settings["metadata_path"])
            if p.exists():
                try:
                    self.entries = load_dataset(p)
                    self._index_entries()
                    self.meta_label.setText(f"{p.name} ({len(self.entries)}개 항목, 이전 설정 기억)")
                    self._refresh_table()
                except Exception:  # noqa: BLE001
                    pass

        self._update_source_status_label()

    def _try_autoload_dataset_config(self):
        """tools/tagging/config/dataset.json 을 읽어 NAS 경로들을 자동으로 채운다.
        NAS가 마운트 안 돼 있으면 경로만 채우고 상태 라벨에 '미마운트'로 표시한다.
        exception_*.csv / exception_history_*.csv 는 dataset_root와 같은 폴더에
        검토자별로 흩어져 있으므로, dataset_root가 마운트되면 그 폴더를
        예외처리 디렉토리로 자동 지정한다(단, 사용자가 이미 다른 소스를 지정했으면 덮어쓰지 않음).
        로컬에 저장된 이전 지정값이 있으면 그게 우선이고, 이 자동추측은 빈 곳만 채운다.
        """
        cfg = load_dataset_config(DEFAULT_CONFIG_PATH)
        if cfg is None:
            self._update_source_status_label()
            return

        if self.dataset_root is not None:
            pass  # 로컬 설정으로 이미 채워짐 - 자동추측으로 덮어쓰지 않음
        elif cfg.dataset_root.mounted:
            self.dataset_root = cfg.dataset_root.resolved
            if self.exception_store is None or self.exception_store.path == REPO_EXCEPTION_CSV_GUESS:
                self.exception_store = ExceptionStore(
                    cfg.dataset_root.resolved, reviewer=self.reviewer_edit.text().strip()
                )
            # 처음으로 자동 탐색에 성공한 경로는 로컬에 저장해서, 다음 실행부터는
            # (특히 윈도우에서 드라이브 문자 A~Z를 훑는) 재탐색 없이 바로 불러오게 한다
            local_settings.update(
                dataset_root=str(cfg.dataset_root.resolved),
                exception_source=str(cfg.dataset_root.resolved),
            )
        if self.keyframe_images_dir is None and cfg.handshape_image_dir.mounted:
            self.keyframe_images_dir = cfg.handshape_image_dir.resolved
            local_settings.update(keyframe_images_dir=str(cfg.handshape_image_dir.resolved))

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
                    self._index_entries()
                    self.meta_label.setText(f"{meta_candidate.name} ({len(self.entries)}개 항목, 자동)")
                    self._refresh_table()
                    local_settings.update(metadata_path=str(meta_candidate))
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
            self._my_kf_cache.pop(entry.origin_no, None)  # 캐시된 값 대신 새로 지정한 이미지를 쓰게
            self._show_detail_for(entry)

    def _index_entries(self):
        """entries가 새로 로드될 때 한 번만 인덱싱해둔다. origin_no 조회와
        같은 gloss_name 찾기(기준 이미지용)를 매번 전체 스캔하면 데이터가
        많을 때(수천 개) 느려지므로, 여기서 미리 dict로 묶어둔다."""
        self.entry_by_origin = {e.origin_no: e for e in self.entries}
        self._entries_by_gloss: dict[str, list[DatasetEntry]] = {}
        for e in self.entries:
            self._entries_by_gloss.setdefault(e.gloss_name, []).append(e)
        self._ref_kf_cache.clear()
        self._my_kf_cache.clear()

    # ── 테이블 ─────────────────────────────────────────────────────────
    def _visible_entries(self) -> list[DatasetEntry]:
        if self.only_exceptions_cb.isChecked() and self.exception_store is not None:
            return [e for e in self.entries if self.exception_store.is_exception(e.origin_no)]
        return self.entries

    def _refresh_table(self):
        self.table.setSortingEnabled(False)
        self._item_by_origin.clear()
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
            entry.origin_no, entry.gloss_name, entry.video_id or "",
            "예외" if is_exc else "정상", status, score, hand, frame, note,
        ]
        bg = STATUS_COLORS.get(status, STATUS_COLORS[""])
        if is_exc:
            bg = QColor(230, 220, 255)

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setBackground(bg)
            self.table.setItem(row, col, item)
            if col == 0:
                # origin_no -> item 참조를 저장해두면, 정렬로 행 순서가 바뀌어도
                # item.row()로 항상 O(1)에 현재 행을 찾을 수 있다 (테이블 전체를
                # 매번 훑는 건 데이터가 많을 때(수천 개) 눈에 띄게 느려짐)
                self._item_by_origin[entry.origin_no] = item

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
        self.worker.frame_progress.connect(self._on_frame_progress)
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

    def _row_of(self, origin_no: str) -> Optional[int]:
        """origin_no의 현재 테이블 행 번호를 O(1)에 찾는다 (정렬돼도 안전).
        데이터가 수천 개일 때 매번 테이블 전체를 훑으면 눈에 띄게 느려진다."""
        item = self._item_by_origin.get(origin_no)
        return item.row() if item is not None else None

    def _on_result_ready(self, origin_no: str, result: ValidationResult):
        self.results[origin_no] = result
        row = self._row_of(origin_no)
        if row is not None:
            self._fill_row(row, self.entry_by_origin[origin_no])
        entry = self._selected_entry()
        if entry and entry.origin_no == origin_no:
            self._show_detail_for(entry)

    def _on_progress(self, done: int, total: int):
        self.progress_bar.setValue(done)

    def _on_frame_progress(self, origin_no: str, frame: np.ndarray, score: Optional[float]):
        """검증 중인 프레임을 실시간으로 보여준다 (YOLO+MediaPipe가 실제로
        사람을 잡고 있는지 눈으로 바로 확인 가능). 처리 중인 항목이 바뀌면
        테이블에서도 그 행을 자동으로 선택/스크롤해서, 상세 패널(내 키프레임/
        사전 동영상/수형사진)이 지금 검증 중인 글로스를 같이 보여주게 한다.
        """
        self.live_preview_label.setPixmap(cv2_to_pixmap(frame, size=(160, 120)))
        gloss = self.entry_by_origin.get(origin_no)
        gloss_name = gloss.gloss_name if gloss else ""
        score_txt = f"{score:.4f}" if score is not None else "검출 안 됨"
        self.live_preview_text.setText(
            f"검증 중: origin_no={origin_no} ({gloss_name})  |  현재 프레임 유사도: {score_txt}"
        )

        if origin_no != self._validating_origin_no:
            self._validating_origin_no = origin_no
            row = self._row_of(origin_no)
            if row is not None:
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, 0))

    def _on_validation_finished(self):
        self.btn_validate_all.setEnabled(True)
        self.btn_validate_selected.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setFormat("완료")
        self.live_preview_text.setText("검증 완료")

    # ── 상세 패널 ──────────────────────────────────────────────────────
    def _on_row_selected(self):
        entry = self._selected_entry()
        if entry:
            self._show_detail_for(entry)

    def _release_current_cap(self):
        self._play_timer.stop()
        self.btn_play.setText("▶ 재생")
        if self._current_video_cap is not None:
            self._current_video_cap.release()
            self._current_video_cap = None

    def _show_detail_for(self, entry: DatasetEntry):
        t_total = time.perf_counter()
        log.info(f"[gui] 행 선택: origin_no={entry.origin_no} ({entry.gloss_name})")
        self._release_current_cap()
        self.my_vs_ref_label.setText("검토대상 vs 정답 직접비교: -")
        result = self.results.get(entry.origin_no)
        is_exc = self.exception_store.is_exception(entry.origin_no) if self.exception_store else False

        # 내 키프레임 + 기준 이미지 표시. NAS 읽기가 항목당 4~5초씩 걸리는 게
        # 확인돼서, 캐시에 없으면 메인 스레드를 막지 않게 백그라운드로 돌린다.
        origin_no = entry.origin_no
        my_cached = self._my_kf_cache.get(origin_no)
        ref_cached = self._ref_kf_cache.get(entry.gloss_name)

        if my_cached is not None:
            self._my_kf_frames = my_cached
            self._set_slider_frames(self.my_kf_slider, self.my_kf_label, self.my_kf_idx_label,
                                     self._my_kf_frames, "(이 영상 인스턴스 키프레임 없음 - NAS 데이터셋 루트\n미마운트 또는 직접 지정 필요. keyframe_images는\n글로스 공통이라 여기엔 안 씀)")
            log.debug(f"[gui]   내 키프레임: 캐시 사용, {len(self._my_kf_frames)}장")
        else:
            self.my_kf_label.setText("(불러오는 중... NAS 읽기라 몇 초 걸릴 수 있음)")
            self.my_kf_slider.setRange(0, 0)
            self.my_kf_idx_label.setText("- / -")

        if ref_cached is not None:
            self._ref_kf_frames, ref_sources = ref_cached
            self._ref_kf_sources = ref_sources
            self._set_slider_frames(self.ref_kf_slider, self.ref_kf_label, self.ref_kf_idx_label,
                                     self._ref_kf_frames, "(같은 글로스의 다른 정상 영상을 못 찾음)")
            self.ref_kf_source_label.setText(ref_sources[0] if ref_sources else "")
            log.debug(f"[gui]   기준 이미지: 캐시 사용, {len(self._ref_kf_frames)}장")

        if my_cached is None or ref_cached is None:
            if self._kf_thread is not None and self._kf_thread.isRunning():
                self._kf_thread.wait(0)  # 이전 요청은 그냥 흘려보냄(콜백에서 origin_no로 걸러짐)
            ref_entries = self._find_reference_entries(entry)
            manual_override = self.manual_keyframe.get(origin_no)
            self._kf_thread = KeyframeLoadThread(
                entry, ref_entries, self.keyframe_images_dir, self.dataset_root, manual_override
            )
            self._kf_thread.finished_loading.connect(self._on_keyframe_loaded)
            self._kf_thread.start()

        # 사전 사이트 프레임 표시 + 슬라이더 세팅
        t0 = time.perf_counter()
        if result and result.video_path and Path(result.video_path).exists():
            self._current_video_cap = cv2.VideoCapture(result.video_path)
            self._current_video_total = int(self._current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, max(0, self._current_video_total - 1))
            self.frame_slider.setValue(result.best_frame_idx or 0)
            self.frame_slider.blockSignals(False)
            self._show_video_frame(result.best_frame_idx or 0)
            self.btn_play.setEnabled(True)
            log.debug(f"[gui]   사전 동영상(로컬 캐시) 열기: {time.perf_counter() - t0:.3f}초")
        else:
            self.btn_play.setEnabled(False)
            self.site_frame_label.setText("(아직 검증 안 됨)")
            self.site_frame_label.setPixmap(QPixmap())
            self.frame_slider.setRange(0, 0)
            self.frame_idx_label.setText("프레임: -")

        # 사전 공식 참고 이미지(수형사진) - 캐시 없으면 백그라운드 스레드에서 비동기로 진행됨
        self._load_handshape_images(entry)
        log.info(f"[gui] 행 선택 처리 완료: {time.perf_counter() - t_total:.3f}초 (수형사진은 비동기라 미포함)")

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
            per_kf = ""
            if result.per_keyframe_scores:
                per_kf = " (" + ", ".join(
                    f"{s:.3f}" if s is not None else "-" for s in result.per_keyframe_scores
                ) + ")"
            lines += [
                "",
                f"검증 상태: {result.status}",
                f"최고 유사도: {result.best_score:.4f}" if result.best_score is not None else "최고 유사도: -",
                f"최고매칭 프레임: {result.best_frame_idx}",
                f"metadata keyframes 매칭: {result.keyframe_matched}/{result.keyframe_total}개{per_kf}",
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

    def _on_keyframe_loaded(self, origin_no: str, my_frames: list, gloss_name: str,
                             ref_frames: list, ref_sources: list):
        """KeyframeLoadThread가 NAS에서 다 읽어온 뒤 호출됨. 캐시에 저장해두고,
        지금도 그 행/글로스를 보고 있으면 화면을 갱신한다(그 사이 다른 행을
        클릭했으면 화면은 안 건드리고 캐시만 채워둠)."""
        self._my_kf_cache[origin_no] = my_frames
        self._ref_kf_cache[gloss_name] = (ref_frames, ref_sources)

        entry = self._selected_entry()
        if entry is None:
            return
        if entry.origin_no == origin_no:
            self._my_kf_frames = my_frames
            self._set_slider_frames(self.my_kf_slider, self.my_kf_label, self.my_kf_idx_label,
                                     my_frames, "(이 영상 인스턴스 키프레임 없음 - NAS 데이터셋 루트\n미마운트 또는 직접 지정 필요. keyframe_images는\n글로스 공통이라 여기엔 안 씀)")
        if entry.gloss_name == gloss_name:
            self._ref_kf_frames = ref_frames
            self._ref_kf_sources = ref_sources
            self._set_slider_frames(self.ref_kf_slider, self.ref_kf_label, self.ref_kf_idx_label,
                                     ref_frames, "(같은 글로스의 다른 정상 영상을 못 찾음)")
            self.ref_kf_source_label.setText(ref_sources[0] if ref_sources else "")

    def _set_slider_frames(self, slider: QSlider, img_label: QLabel, idx_label: QLabel,
                            frames: list, empty_text: str):
        """내 키프레임/기준 이미지/수형사진 패널 공통: frames 리스트를 슬라이더에 연결하고
        첫 프레임을 보여준다. 전부 이 패턴 하나로 통일해서 조작감을 동일하게 맞춘다."""
        slider.blockSignals(True)
        slider.setRange(0, max(0, len(frames) - 1))
        slider.setValue(0)
        slider.blockSignals(False)
        if frames:
            self._display_frame(img_label, frames[0])
            idx_label.setText(f"1 / {len(frames)}")
        else:
            img_label.setText(empty_text)
            idx_label.setText("- / -")

    def _display_frame(self, img_label: QLabel, frame: np.ndarray):
        """포즈 시각화 체크박스가 켜져 있으면 keypoint를 그려서 보여준다."""
        if self.show_pose_cb.isChecked():
            frame = self._draw_pose_overlay(frame)
        img_label.setPixmap(cv2_to_pixmap(frame))

    def _get_extractors(self) -> Extractors:
        """YOLO+MediaPipe 모델을 첫 사용 시 한 번만 로드해서 재사용 (로딩에 1~2초 걸림)."""
        if self._extractors is None:
            log.info("[gui] pose 시각화/직접비교용 모델 로딩 중...")
            t0 = time.perf_counter()
            self._extractors = Extractors()
            log.info(f"[gui] 모델 로딩 완료: {time.perf_counter() - t0:.3f}초")
        return self._extractors

    def _draw_pose_overlay(self, frame: np.ndarray) -> np.ndarray:
        """YOLO body keypoint(초록 선/점) + MediaPipe 손가락 keypoint(왼손 파랑/오른손 빨강)를
        프레임 위에 그려서, 실제로 뭘 검출하고 있는지 눈으로 확인할 수 있게 한다."""
        try:
            extractors = self._get_extractors()
            sig = extractors.signature(frame)
        except Exception as e:  # noqa: BLE001 - 시각화 실패해도 원본은 보여줘야 함
            log.debug(f"[gui] pose 오버레이 실패: {e}")
            return frame

        vis = frame.copy()
        if sig.body is not None:
            body = sig.body
            for a, b in BODY_EDGES:
                if body[a][2] > 0.3 and body[b][2] > 0.3:
                    pa = (int(body[a][0]), int(body[a][1]))
                    pb = (int(body[b][0]), int(body[b][1]))
                    cv2.line(vis, pa, pb, (0, 255, 0), 2)
            for x, y, c in body:
                if c > 0.3:
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)

        h, w = frame.shape[:2]
        for label, pts in sig.hands.items():
            color = (255, 0, 0) if label == "Left" else (0, 0, 255)
            px = [(int(x * w), int(y * h)) for x, y in pts]
            for chain in HAND_CHAINS:
                for i in range(len(chain) - 1):
                    cv2.line(vis, px[chain[i]], px[chain[i + 1]], color, 1)
            for p in px:
                cv2.circle(vis, p, 2, color, -1)
        return vis

    def _on_show_pose_toggled(self, _checked):
        """체크박스를 바꾸면 지금 보이는 프레임을 다시 그린다(다시 클릭할 필요 없게)."""
        if self._my_kf_frames:
            self._display_frame(self.my_kf_label, self._my_kf_frames[self.my_kf_slider.value()])
        if self._ref_kf_frames:
            self._display_frame(self.ref_kf_label, self._ref_kf_frames[self.ref_kf_slider.value()])

    def _update_my_vs_ref_score(self):
        """'검토대상 키프레임'과 '정답 키프레임' 중 지금 슬라이더로 보고 있는
        프레임끼리 직접 pose 비교 점수를 계산해서 보여준다."""
        if not self._my_kf_frames or not self._ref_kf_frames:
            self.my_vs_ref_label.setText("검토대상 vs 정답 직접비교: 두 이미지가 다 있어야 계산 가능")
            return
        try:
            extractors = self._get_extractors()
            a = extractors.signature(self._my_kf_frames[self.my_kf_slider.value()])
            b = extractors.signature(self._ref_kf_frames[self.ref_kf_slider.value()])
            score, _detail = combined_similarity(a.body, b.body, a.hands, b.hands)
        except Exception as e:  # noqa: BLE001
            log.debug(f"[gui] 직접비교 실패: {e}")
            self.my_vs_ref_label.setText("검토대상 vs 정답 직접비교: 계산 실패")
            return

        if score is None:
            self.my_vs_ref_label.setText("검토대상 vs 정답 직접비교: 사람/손 검출 실패")
            return
        verdict = "일치 가능성 높음" if score >= self.threshold_spin.value() else "불일치 의심"
        self.my_vs_ref_label.setText(f"검토대상 vs 정답 직접비교: {score:.4f} ({verdict})")

    def _on_my_kf_slider_moved(self, idx: int):
        if not self._my_kf_frames:
            return
        self._display_frame(self.my_kf_label, self._my_kf_frames[idx])
        self.my_kf_idx_label.setText(f"{idx + 1} / {len(self._my_kf_frames)}")

    def _find_reference_entries(self, entry: DatasetEntry) -> list[DatasetEntry]:
        """같은 gloss_name을 가진 다른 항목들 - "정답" 예시로 쓸 후보.
        예외처리 안 된(정상) 항목을 우선한다. 정상이 하나도 없으면 예외 항목이라도 보여준다.
        미리 만들어둔 gloss_name 인덱스를 쓰므로 전체 항목을 다시 훑지 않는다."""
        same_gloss = self._entries_by_gloss.get(entry.gloss_name, [])
        others = [e for e in same_gloss if e.origin_no != entry.origin_no]
        if self.exception_store is not None:
            normal = [e for e in others if not self.exception_store.is_exception(e.origin_no)]
            if normal:
                return normal
        return others

    def _on_ref_kf_slider_moved(self, idx: int):
        if not self._ref_kf_frames:
            return
        self._display_frame(self.ref_kf_label, self._ref_kf_frames[idx])
        self.ref_kf_idx_label.setText(f"{idx + 1} / {len(self._ref_kf_frames)}")
        if idx < len(self._ref_kf_sources):
            self.ref_kf_source_label.setText(self._ref_kf_sources[idx])

    def _load_handshape_images(self, entry: DatasetEntry):
        """사전 사이트의 '수형 사진'(일러스트) 로드. 참고용이며 자동 채점엔 안 씀.

        origin_no마다 매번 네트워크로 다시 긁어오면 행을 클릭할 때마다 렉이 걸리므로,
        메모리 캐시 -> 디스크 캐시 -> (그래도 없으면) 백그라운드 스레드로 네트워크 요청
        순서로 확인한다. UI 스레드를 절대 막지 않는다.
        """
        self._handshape_paths = []
        self._handshape_frames = []
        origin_no = entry.origin_no

        # 1) 메모리 캐시 (이미 이번 세션에 한 번이라도 확인했으면 즉시 표시)
        if origin_no in self._handshape_cache:
            self._handshape_paths = self._handshape_cache[origin_no]
            self._display_handshape_result()
            return

        # 2) 디스크 캐시 (이전 실행에서 이미 받아둔 파일이 있으면 네트워크 요청 자체를 생략)
        cached = sorted(self.handshape_dir.glob(f"{origin_no}_*.jpg")) if self.handshape_dir.exists() else []
        if cached:
            self._handshape_cache[origin_no] = cached
            self._handshape_paths = cached
            self._display_handshape_result()
            return

        # 3) 진짜 처음 보는 항목 -> 백그라운드 스레드로 네트워크 요청 (UI 안 멈춤)
        # setText()가 이전 pixmap을 알아서 지우므로 setPixmap(QPixmap())을 따로 부르지 않는다
        # (부르면 방금 설정한 텍스트가 바로 지워짐)
        self.handshape_label.setText("(불러오는 중...)")

        if self._handshape_thread is not None and self._handshape_thread.isRunning():
            self._handshape_thread.wait(0)  # 이전 요청 결과는 무시(콜백에서 origin_no로 걸러짐)

        self._handshape_thread = HandshapeFetchThread(origin_no, self.handshape_dir)
        self._handshape_thread.fetched.connect(self._on_handshape_fetched)
        self._handshape_thread.failed.connect(self._on_handshape_failed)
        self._handshape_thread.start()

    def _on_handshape_fetched(self, origin_no: str, paths: list):
        self._handshape_cache[origin_no] = paths
        entry = self._selected_entry()
        if entry is None or entry.origin_no != origin_no:
            return  # 그 사이에 사용자가 다른 행을 클릭함 - 지금 화면과 무관
        self._handshape_paths = paths
        self._display_handshape_result()

    def _on_handshape_failed(self, origin_no: str, error: str):
        self._handshape_cache[origin_no] = []
        entry = self._selected_entry()
        if entry is None or entry.origin_no != origin_no:
            return
        self.handshape_label.setText(f"(불러오기 실패)\n{error}")
        self.handshape_label.setPixmap(QPixmap())

    def _display_handshape_result(self):
        frames = []
        for p in self._handshape_paths:
            frame = cv2.imread(str(p))
            if frame is not None:
                frames.append(frame)
        self._handshape_frames = frames
        self._set_slider_frames(self.handshape_slider, self.handshape_label, self.handshape_idx_label,
                                 frames, "(참고 이미지 없음)")

    def _on_handshape_slider_moved(self, idx: int):
        if not self._handshape_frames:
            return
        self.handshape_label.setPixmap(cv2_to_pixmap(self._handshape_frames[idx]))
        self.handshape_idx_label.setText(f"{idx + 1} / {len(self._handshape_frames)}")

    def _on_slider_moved(self, value: int):
        self.frame_idx_label.setText(f"프레임: {value} / {self._current_video_total - 1}")
        self._show_video_frame(value)

    def _show_video_frame(self, idx: int):
        if self._current_video_cap is None:
            return
        self._current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._current_video_cap.read()
        if ret:
            # 포즈 표시 체크박스 켰을 때만 오버레이 비용이 붙는다 - 기본(꺼짐)은 그대로 빠름
            self._display_frame(self.site_frame_label, frame)
        self.frame_idx_label.setText(f"프레임: {idx} / {self._current_video_total - 1}")

    def _toggle_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶ 재생")
            return
        if self._current_video_cap is None:
            return
        fps = self._current_video_cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval_ms = max(1, int(1000 / fps))
        if self.frame_slider.value() >= self._current_video_total - 1:
            self.frame_slider.setValue(0)  # 끝까지 봤으면 처음부터 다시 재생
        self._play_timer.start(interval_ms)
        self.btn_play.setText("⏸ 정지")

    def _on_play_tick(self):
        next_idx = self.frame_slider.value() + 1
        if next_idx > self._current_video_total - 1:
            self._play_timer.stop()
            self.btn_play.setText("▶ 재생")
            return
        self.frame_slider.setValue(next_idx)  # valueChanged가 _on_slider_moved를 호출해 화면 갱신

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
        row = self._row_of(origin_no)
        if row is not None:
            self._fill_row(row, self.entry_by_origin[origin_no])

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
        if self._handshape_thread and self._handshape_thread.isRunning():
            self._handshape_thread.wait(2000)
        if self._kf_thread and self._kf_thread.isRunning():
            self._kf_thread.wait(2000)
        super().closeEvent(event)
