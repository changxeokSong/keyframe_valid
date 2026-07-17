"""검증 파이프라인을 UI를 멈추지 않고 돌리기 위한 백그라운드 QThread."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import requests
from PyQt5.QtCore import QThread, pyqtSignal

from .. import sldict_client
from ..keyframe_images import (
    build_keyframe_images_index,
    find_exception_video_path,
    load_gloss_reference_images,
    load_instance_keyframes,
    resolve_instance_video_path,
)
from ..metadata import DatasetEntry, entry_key
from ..pipeline import Extractors, ValidationResult, validate_entry


class ValidationWorker(QThread):
    result_ready = pyqtSignal(str, object)   # entry_key(origin_no+video_id), ValidationResult
    progress = pyqtSignal(int, int)          # done, total
    frame_progress = pyqtSignal(str, object, object)  # entry_key, frame(ndarray), score(float|None)
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
                    on_frame=lambda idx, frame, score, k=entry_key(entry): (
                        self.frame_progress.emit(k, frame, score)
                    ),
                )
            except Exception as e:  # noqa: BLE001 - 개별 항목 실패로 전체가 멈추면 안 됨
                result = ValidationResult(entry.origin_no, entry.gloss_name, "ERROR", note=str(e))

            self.result_ready.emit(entry_key(entry), result)
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


class KeyframeLoadThread(QThread):
    """'검토 대상 키프레임'과 '정답 기준 이미지'를 NAS에서 읽어오는 동안
    UI가 멈추지 않도록 백그라운드에서 처리한다.

    NAS 네트워크 지연이 항목당 4~5초씩 걸리는 게 직접 확인됐는데, 이걸
    GUI 메인 스레드에서 그대로 하면 행을 클릭할 때마다 그만큼 전체 화면이
    멈춘다. 그래서 분리했다.
    """

    finished_loading = pyqtSignal(str, list, str, str, list, list, list)
    # entry_key(origin_no+video_id), my_kf_frames, my_video_path, gloss_name, ref_kf_frames,
    # ref_kf_sources, ref_video_paths (ref_kf_frames와 같은 길이 - 각 후보 이미지마다 재생할 영상, 없으면 "")

    def __init__(
        self,
        entry: DatasetEntry,
        ref_entries: list[DatasetEntry],
        keyframe_images_dir,
        dataset_root,
        manual_override,
        parent=None,
    ):
        super().__init__(parent)
        self.entry = entry
        self.ref_entries = ref_entries
        self.keyframe_images_dir = keyframe_images_dir
        self.dataset_root = dataset_root
        self.manual_override = manual_override

    def run(self):
        # 검토 대상: 지금 선택된 그 인스턴스(사람/subset) 자체의 실제 키프레임 + 그 영상 자체
        # (재생용). keyframe_images는 글로스 공통(=정답)이라 여기서는 절대 안 쓴다 -
        # 안 그러면 어느 행을 클릭해도 같은 "정답"만 보여서 검토가 무의미해진다.
        my_frames = load_instance_keyframes(self.entry, self.dataset_root, self.manual_override)
        my_video_path = resolve_instance_video_path(self.entry, self.dataset_root)

        # 정답: 이 글로스의 공통 기준사진(keyframe_images)을 먼저 보여주고,
        # 그 다음에 같은 글로스의 다른 정상 영상들의 실제 키프레임을 추가로 붙인다.
        # 각 후보 이미지가 "자기 자신의" 영상을 갖고 있는지 하나씩 따로 기록한다 -
        # 안 그러면 슬라이더로 다른 사진을 보고 있는데 엉뚱한 영상이 재생되거나,
        # 재생 가능한 후보가 있는데도 재생 버튼이 꺼져있는 문제가 생긴다.
        ref_frames = list(load_gloss_reference_images(self.entry, self.keyframe_images_dir))
        ref_sources = [f"출처: 글로스 공통 기준사진 (origin_no={self.entry.origin_no})" for _ in ref_frames]
        ref_video_paths = ["" for _ in ref_frames]  # 글로스 공통 사진은 대응하는 영상이 없음

        for e in self.ref_entries[:5]:
            candidates = load_instance_keyframes(e, self.dataset_root, None)
            if candidates:
                ref_frames.append(candidates[0])
                ref_sources.append(f"출처: 다른 정상 영상 origin_no={e.origin_no} ({e.gloss_name})")
                v = resolve_instance_video_path(e, self.dataset_root)
                ref_video_paths.append(str(v) if v else "")

        self.finished_loading.emit(
            entry_key(self.entry), my_frames, str(my_video_path) if my_video_path else "",
            self.entry.gloss_name, ref_frames, ref_sources, ref_video_paths,
        )


class KeyframeIndexThread(QThread):
    """keyframe_images_dir(글로스 공통 사진 폴더) 전체를 한 번 스캔해 origin_no별
    인덱스를 만든다. 이걸 안 하면 항목을 클릭할 때마다 NAS 디렉토리 전체 목록을
    다시 조회해야 해서 (직접 확인: 항목당 25~40초) 검토가 사실상 불가능할 정도로
    느려진다. 폴더가 지정될 때 딱 한 번만 백그라운드로 돌리면 그 뒤로는 즉시 조회된다."""

    finished_index = pyqtSignal(int, float)  # origin_no 개수, 걸린 시간(초)

    def __init__(self, keyframe_images_dir: Path, parent=None):
        super().__init__(parent)
        self.keyframe_images_dir = keyframe_images_dir

    def run(self):
        t0 = time.perf_counter()
        index = build_keyframe_images_index(self.keyframe_images_dir)
        self.finished_index.emit(len(index), time.perf_counter() - t0)


class MissingVideoRecoveryThread(QThread):
    """metadata.csv에 행이 없어서 안 보이던 예외 항목을, 예외처리 기록에 이미
    적혀있는 실제 video_id로 NAS에서 영상 파일을 직접 찾아 검토 목록에 복구한다.

    원본 태깅 도구를 확인해보니 영상을 예외처리하면 EXCEPTION/{subset}/... 폴더로
    옮겨져서 metadata.csv를 다시 만들어도 거기 없는 게 정상 동작이었다(버그가
    아니었음). 후보 개수가 수백~수천 개일 수 있어 NAS 파일 존재 확인을 전부
    백그라운드에서 한다."""

    progress = pyqtSignal(int, int)          # done, total
    found = pyqtSignal(object)               # DatasetEntry (찾은 것마다 하나씩)
    finished_scan = pyqtSignal(int, int)     # 찾은 개수, 전체 후보 개수

    def __init__(self, dataset_root: Path, candidates: list[tuple[str, str, str]], parent=None):
        # candidates: [(origin_no, gloss_name, video_id), ...]
        super().__init__(parent)
        self.dataset_root = dataset_root
        self.candidates = candidates
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        total = len(self.candidates)
        n_found = 0
        for i, (origin_no, gloss_name, video_id) in enumerate(self.candidates):
            if self._stop_requested:
                break
            video_path = find_exception_video_path(self.dataset_root, video_id)
            if video_path is not None:
                rel_path = video_path.relative_to(self.dataset_root)
                entry = DatasetEntry(
                    origin_no=origin_no,
                    gloss_name=gloss_name,
                    keyframes=[],
                    video_rel_path=str(rel_path),
                    video_id=video_id,
                )
                n_found += 1
                self.found.emit(entry)
            self.progress.emit(i + 1, total)
        self.finished_scan.emit(n_found, total)
