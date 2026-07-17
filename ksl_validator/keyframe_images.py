"""NAS의 실제 키프레임 사진 폴더 접근.

경로 (dataset.json 기준): \\mldisk2\\nfs_shared\\abd\\dataset\\sl\\etri_ksl_db\\keyframe_images
파일명 패턴: {origin_no}_{gloss_name}_{index}.jpg   (예: 1_파나마_1.jpg)

sldict 사이트의 '수형 사진'은 일러스트(선화)라 MediaPipe/YOLO가 인식을 못 하지만,
이 폴더의 사진은 실제 촬영본이라 자동 채점(pose 비교)의 '내 키프레임' 소스로 쓸 수 있다.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .logging_setup import log
from .paths import NAS_VIDEO_CACHE_DIR

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
NAS_VIDEO_CACHE_MAX_FILES = 15  # 무제한으로 쌓이면 부담되므로 최근 것만 유지 (LRU)

# keyframe_images 폴더는 보통 전체 글로스 사진이 한 폴더에(수만 장) 들어있다.
# Path.glob("{origin_no}_*")는 매번 그 폴더 전체를 네트워크로 다시 나열해야 해서,
# NAS에서 항목 하나 조회하는 데도 실측 25~40초씩 걸렸다(디렉토리 목록 조회 자체가
# 병목이지 개별 파일 읽기는 아님). 그래서 폴더 전체를 한 번만 훑어 메모리에
# origin_no -> 파일목록 인덱스를 만들어두고, 그 뒤로는 dict 조회로 즉시 끝낸다.
_INDEX_CACHE: dict[str, dict[str, list[Path]]] = {}


def _index_sort_key(p: Path):
    # 파일명 끝의 _{index} 를 정수로 정렬 (문자열 정렬시 1,10,2 순서가 되는 것 방지)
    stem = p.stem
    tail = stem.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


def build_keyframe_images_index(keyframe_images_dir: Path) -> dict[str, list[Path]]:
    """keyframe_images_dir 전체를 한 번 스캔해서 origin_no -> 파일목록 인덱스를 만든다.
    GUI에서 폴더가 지정될 때 백그라운드 스레드에서 한 번만 호출하면 된다
    (worker.KeyframeIndexThread). 폴더 전체 목록 조회라 NAS에서는 최대 1분 정도
    걸릴 수 있지만, 세션 내내 딱 한 번만 하면 되므로 매 항목 클릭마다 걸리던
    25~40초 지연이 사라진다."""
    keyframe_images_dir = Path(keyframe_images_dir)
    index: dict[str, list[Path]] = {}
    if not keyframe_images_dir.exists():
        _INDEX_CACHE[str(keyframe_images_dir)] = index
        return index

    t0 = time.perf_counter()
    n_files = 0
    for p in keyframe_images_dir.iterdir():
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        n_files += 1
        origin_no = p.stem.split("_", 1)[0]
        index.setdefault(origin_no, []).append(p)
    for paths in index.values():
        paths.sort(key=_index_sort_key)

    log.info(
        f"[keyframe_images] 폴더 인덱싱 완료: 파일 {n_files}개 -> origin_no {len(index)}개, "
        f"{time.perf_counter() - t0:.1f}초"
    )
    _INDEX_CACHE[str(keyframe_images_dir)] = index
    return index


def _get_local_video_copy(nas_path: Path) -> Path:
    """NAS 동영상을 로컬(cache/nas_videos/)에 통째로 한 번 복사해두고 그 경로를 반환.

    압축 동영상은 임의 프레임으로 바로 못 가고 가까운 키프레임부터 디코딩해야 해서,
    seek 한 번에도 파일 여러 지점을 오가며 읽는다. 로컬 디스크에선 거의 공짜지만
    네트워크 공유 폴더(NAS)에서는 그 하나하나가 왕복 지연이라 느리다(직접 확인:
    항목당 몇 초). 수어 영상은 보통 짧아서(수 MB) 파일 전체를 한 번에 순차로
    복사하는 게, 여러 번 흩어져서 seek하는 것보다 오히려 더 빠르다.

    디스크가 무한정 쌓이지 않도록 최근 사용한 NAS_VIDEO_CACHE_MAX_FILES개만 유지하고
    오래된 것부터 지운다(LRU).
    """
    NAS_VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = NAS_VIDEO_CACHE_DIR / nas_path.name

    nas_size = nas_path.stat().st_size
    if local_path.exists() and local_path.stat().st_size == nas_size:
        local_path.touch()  # LRU 최신화
        return local_path

    t0 = time.perf_counter()
    # 같은 영상을 서로 다른 행에서 거의 동시에 열면(예: 검토대상/정답 패널이
    # 동시에 같은 영상을 참조) 두 스레드가 동시에 이 함수에 들어올 수 있다.
    # 임시 파일명에 스레드 id를 넣어 서로 다른 스레드가 같은 .part 파일에
    # 동시에 쓰는 걸 막고, os.replace로 원자적 교체를 써서 Windows에서
    # 목적지 파일이 이미 있으면 rename이 실패하는 문제(직접 확인된 크래시)도 피한다.
    tmp_path = local_path.with_suffix(f"{local_path.suffix}.{threading.get_ident()}.part")
    try:
        shutil.copyfile(nas_path, tmp_path)
        os.replace(tmp_path, local_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    log.info(
        f"[keyframe_images] NAS 영상 로컬 캐시 복사: {nas_path.name} "
        f"({nas_size / 1024:.0f}KB), {time.perf_counter() - t0:.3f}초"
    )

    _evict_old_cached_videos()
    return local_path


def _evict_old_cached_videos() -> None:
    files = sorted(
        (p for p in NAS_VIDEO_CACHE_DIR.glob("*") if p.is_file() and not p.name.endswith(".part")),
        key=lambda p: p.stat().st_mtime,
    )
    excess = len(files) - NAS_VIDEO_CACHE_MAX_FILES
    for p in files[:max(0, excess)]:
        try:
            p.unlink()
        except OSError:
            pass


def find_keyframe_images(keyframe_images_dir: Path, origin_no: str) -> list[Path]:
    """origin_no로 시작하는 키프레임 사진들을 인덱스 순으로 반환.

    파일명이 '{origin_no}_...' 형태이므로 '{origin_no}_*' 로 매칭한다.
    (origin_no가 다른 번호의 접두어가 되는 오매칭은 '_' 구분자 덕분에 발생하지 않는다.
     예: origin_no='1' 은 '1_파나마_1.jpg' 는 매칭하지만 '10_...' 은 매칭하지 않음.)
    """
    keyframe_images_dir = Path(keyframe_images_dir)
    index = _INDEX_CACHE.get(str(keyframe_images_dir))
    if index is not None:
        return index.get(origin_no, [])

    if not keyframe_images_dir.exists():
        return []

    matches: list[Path] = []
    for ext in IMAGE_EXTS:
        matches.extend(keyframe_images_dir.glob(f"{origin_no}_*{ext}"))
    return sorted(matches, key=_index_sort_key)


def load_gloss_reference_images(entry, keyframe_images_dir: Optional[Path] = None) -> list[np.ndarray]:
    """keyframe_images/{origin_no}_{gloss}_{idx}.jpg — origin_no(글로스) 단위로만
    저장돼 있고 사람/subset(003/004/005/009/010/011/012) 구분이 없다. 즉 이 사진은
    "이 글로스가 정상적으로는 이렇게 생겼다"는 글로스 공통 기준(=정답)이지,
    특정 영상 인스턴스의 실제 내용이 아니다. '정답' 패널 전용으로만 써야 한다.
    """
    if keyframe_images_dir is None:
        return []
    t0 = time.perf_counter()
    matches = find_keyframe_images(keyframe_images_dir, entry.origin_no)
    frames = [f for f in (cv2.imread(str(m)) for m in matches) if f is not None]
    log.debug(
        f"[keyframe_images] NAS keyframe_images(글로스 공통) 읽기 origin_no={entry.origin_no}: "
        f"{time.perf_counter() - t0:.3f}초, {len(frames)}장"
    )
    return frames


def resolve_instance_video_path(entry, dataset_root: Optional[Path]) -> Optional[Path]:
    """entry가 가리키는 그 특정 인스턴스의 실제 영상 파일 경로 (재생용).
    NAS 원본이 있으면 로컬 캐시로 복사해서 그 경로를 준다(재생/스크럽이 빠르게).
    gloss_reference_images(글로스 공통 사진)에는 대응하는 영상이 없으므로 None."""
    if dataset_root is not None and getattr(entry, "video_rel_path", None):
        p = Path(dataset_root) / entry.video_rel_path
        if p.exists():
            return _get_local_video_copy(p)
    return None


def load_instance_keyframes(entry, dataset_root: Optional[Path] = None,
                             manual_override: Optional[Path] = None) -> list[np.ndarray]:
    """'검토 대상' 전용: 지금 선택된 그 특정 영상 인스턴스(사람/subset) 자체의
    실제 키프레임. metadata.csv의 video_rel_path/keyframes는 그 행이 가리키는
    영상 파일 하나에 묶여 있으므로, keyframe_images(글로스 공통)와 달리 인스턴스별로
    다른 내용을 정확히 보여준다. 예외 의심 영상이면 그 잘못된 내용이 그대로 나와야
    검토가 의미 있다 - 그래서 keyframe_images는 여기서 아예 안 쓴다.

    NAS 파일 읽기는 네트워크 지연으로 몇 초씩 걸릴 수 있으므로(직접 확인됨),
    이 함수는 GUI에서 호출할 때 반드시 백그라운드 스레드(worker.KeyframeLoadThread)
    에서만 불러서 메인 스레드가 멈추지 않게 해야 한다.
    """
    if manual_override is not None:
        frame = cv2.imread(str(manual_override))
        if frame is not None:
            return [frame]

    if dataset_root is not None and getattr(entry, "video_rel_path", None) and entry.keyframes:
        nas_video_path = Path(dataset_root) / entry.video_rel_path
        if nas_video_path.exists():
            # 먼저 로컬로 통째 복사(또는 이미 캐시돼 있으면 재사용)한 뒤, 그 로컬 파일에서
            # seek한다 - NAS에서 직접 여러 번 seek하는 것보다 빠르다(위 함수 설명 참고).
            video_path = _get_local_video_copy(nas_video_path)
            t0 = time.perf_counter()
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            for kf_idx in entry.keyframes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, kf_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
            log.debug(
                f"[keyframe_images] 로컬 캐시 비디오(인스턴스 전용) 프레임 읽기 {video_path.name}: "
                f"{time.perf_counter() - t0:.3f}초, {len(frames)}장"
            )
            if frames:
                return frames
    return []


def find_exception_video_path(dataset_root: Path, video_id: str) -> Optional[Path]:
    """metadata.csv에 그 예외처리된 영상의 행이 없어도, video_id로 실제 파일을
    NAS에서 직접 찾는다.

    원본 태깅 도구(online-sign-keyframe-detection-transformers/tools/tagging)의
    코드를 확인한 결과, 영상을 예외처리하면 원래 위치({subset}/MP4/...)가 아니라
    별도의 EXCEPTION/{subset}/{MP4,MKV,VIDEO}/{stem}{확장자} 폴더로 옮겨진다
    (실제로 NAS에도 EXCEPTION 폴더가 있는 것으로 확인됨). 그래서 metadata.csv를
    아무리 다시 만들어도 이미 예외처리된 영상은 거기 없는 게 정상이다 - 이건
    metadata.csv가 못 따라간 게 아니라 원래 그렇게 동작하는 것.

    video_id의 "/" 앞부분이 subset, 마지막 부분이 실제 파일명(stem, 확장자 제외)
    이다(원본 도구도 항상 이렇게 파싱함 - 중간에 세그먼트가 더 있어도 무시).
    EXCEPTION 폴더를 먼저 찾고, 혹시 옮겨지기 전이면 원래 위치도 확인한다."""
    if "/" not in video_id:
        return None
    subset = video_id.split("/")[0].strip()
    stem = video_id.rsplit("/", 1)[-1].strip()
    if not subset or not stem:
        return None

    exts = (".mp4", ".mkv", ".avi", ".mov")
    for base in (dataset_root / "EXCEPTION" / subset, dataset_root / subset):
        for subdir in ("MP4", "MKV", "VIDEO"):
            for ext in exts:
                candidate = base / subdir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
        for ext in exts:
            candidate = base / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None
