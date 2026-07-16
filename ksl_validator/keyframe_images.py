"""NAS의 실제 키프레임 사진 폴더 접근.

경로 (dataset.json 기준): \\mldisk2\\nfs_shared\\abd\\dataset\\sl\\etri_ksl_db\\keyframe_images
파일명 패턴: {origin_no}_{gloss_name}_{index}.jpg   (예: 1_파나마_1.jpg)

sldict 사이트의 '수형 사진'은 일러스트(선화)라 MediaPipe/YOLO가 인식을 못 하지만,
이 폴더의 사진은 실제 촬영본이라 자동 채점(pose 비교)의 '내 키프레임' 소스로 쓸 수 있다.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .logging_setup import log

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def find_keyframe_images(keyframe_images_dir: Path, origin_no: str) -> list[Path]:
    """origin_no로 시작하는 키프레임 사진들을 인덱스 순으로 반환.

    파일명이 '{origin_no}_...' 형태이므로 '{origin_no}_*' 로 매칭한다.
    (origin_no가 다른 번호의 접두어가 되는 오매칭은 '_' 구분자 덕분에 발생하지 않는다.
     예: origin_no='1' 은 '1_파나마_1.jpg' 는 매칭하지만 '10_...' 은 매칭하지 않음.)
    """
    keyframe_images_dir = Path(keyframe_images_dir)
    if not keyframe_images_dir.exists():
        return []

    matches: list[Path] = []
    for ext in IMAGE_EXTS:
        matches.extend(keyframe_images_dir.glob(f"{origin_no}_*{ext}"))

    def sort_key(p: Path):
        # 파일명 끝의 _{index} 를 정수로 정렬 (문자열 정렬시 1,10,2 순서가 되는 것 방지)
        stem = p.stem
        tail = stem.rsplit("_", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return 0

    return sorted(matches, key=sort_key)


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
    gloss_reference_images(글로스 공통 사진)에는 대응하는 영상이 없으므로 None."""
    if dataset_root is not None and getattr(entry, "video_rel_path", None):
        p = Path(dataset_root) / entry.video_rel_path
        if p.exists():
            return p
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
        video_path = Path(dataset_root) / entry.video_rel_path
        if video_path.exists():
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
                f"[keyframe_images] NAS 원본 비디오(인스턴스 전용) 프레임 읽기 {video_path.name}: "
                f"{time.perf_counter() - t0:.3f}초, {len(frames)}장"
            )
            if frames:
                return frames
    return []
