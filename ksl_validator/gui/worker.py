"""검증 파이프라인을 UI를 멈추지 않고 돌리기 위한 백그라운드 QThread."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests
from PyQt5.QtCore import QThread, pyqtSignal

from .. import sldict_client
from ..metadata import DatasetEntry
from ..pipeline import Extractors, ValidationResult, validate_entry


class ValidationWorker(QThread):
    result_ready = pyqtSignal(str, object)   # origin_no, ValidationResult
    progress = pyqtSignal(int, int)          # done, total
    frame_progress = pyqtSignal(str, object, object)  # origin_no, frame(ndarray), score(float|None)
    log = pyqtSignal(str)
    finished_all = pyqtSignal()

    def __init__(
        self,
        entries: list[DatasetEntry],
        cache_dir: Path,
        dataset_root: Optional[Path] = None,
        keyframe_images_dir: Optional[Path] = None,
        threshold: float = 0.7,
        stride: int = 3,
        parent=None,
    ):
        super().__init__(parent)
        self.entries = entries
        self.cache_dir = cache_dir
        self.dataset_root = dataset_root
        self.keyframe_images_dir = keyframe_images_dir
        self.threshold = threshold
        self.stride = stride
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            extractors = Extractors()
        except Exception as e:  # noqa: BLE001
            self.log.emit(f"모델 로딩 실패: {e}")
            self.finished_all.emit()
            return

        session = requests.Session()
        total = len(self.entries)

        for i, entry in enumerate(self.entries):
            if self._stop_requested:
                break
            try:
                result = validate_entry(
                    entry, extractors, self.cache_dir,
                    dataset_root=self.dataset_root,
                    keyframe_images_dir=self.keyframe_images_dir,
                    threshold=self.threshold, stride=self.stride,
                    session=session,
                    on_frame=lambda idx, frame, score, o=entry.origin_no: (
                        self.frame_progress.emit(o, frame, score)
                    ),
                )
            except Exception as e:  # noqa: BLE001 - 개별 항목 실패로 전체가 멈추면 안 됨
                result = ValidationResult(entry.origin_no, entry.gloss_name, "ERROR", note=str(e))

            self.result_ready.emit(entry.origin_no, result)
            self.progress.emit(i + 1, total)

        self.finished_all.emit()


class HandshapeFetchThread(QThread):
    """사전 사이트의 '수형 사진'(참고용 일러스트)을 네트워크로 가져오는 동안
    UI가 멈추지 않도록 백그라운드에서 실행. 행을 클릭할 때마다 매번 이걸
    동기로 부르면 클릭할 때마다 렉이 걸려서 분리했다."""

    fetched = pyqtSignal(str, list)  # origin_no, [Path, ...]
    failed = pyqtSignal(str, str)    # origin_no, error message

    def __init__(self, origin_no: str, dest_dir: Path, parent=None):
        super().__init__(parent)
        self.origin_no = origin_no
        self.dest_dir = dest_dir

    def run(self):
        try:
            paths = sldict_client.download_handshape_images(
                requests.Session(), self.origin_no, self.dest_dir
            )
            self.fetched.emit(self.origin_no, paths)
        except Exception as e:  # noqa: BLE001 - 오프라인/사이트 오류는 조용히 실패 처리
            self.failed.emit(self.origin_no, str(e))
