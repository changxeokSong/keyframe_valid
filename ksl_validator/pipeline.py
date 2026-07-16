"""엔드투엔드 라벨 검증 파이프라인.

각 데이터셋 항목(origin_no, gloss_name, keyframes)에 대해:
1. '내 키프레임'의 body pose + 손모양(signature)을 구한다
   (NAS 원본 비디오 또는 단일 이미지에서).
2. 한국수어사전 사이트에서 같은 origin_no의 공식 동영상을 내려받는다.
3. 그 동영상의 모든(혹은 stride 간격) 프레임에 대해 signature를 뽑아
   내 키프레임과 가장 비슷한 프레임과 점수를 찾는다.

주의: 이 점수는 "동영상 전체 중 가장 비슷했던 한 프레임"의 유사도라서
완전 자동 합격/불합격 판정으로 쓰기엔 잡음이 있다(우연히 스쳐가는 비슷한
손모양에도 점수가 높게 나올 수 있음). 그래서 SUSPECT/MATCH 분류는
"사람이 검토할 우선순위를 정하는 용도"로 취급하고, 최종 판단은
GUI에서 실제 프레임을 눈으로 보고 내리는 걸 전제로 한다.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import requests

from . import sldict_client
from .compare import combined_similarity
from .hands import HandPoseExtractor
from .keyframe_images import find_keyframe_images
from .logging_setup import log
from .metadata import DatasetEntry
from .pose import PoseExtractor

STATUS_MATCH = "MATCH"
STATUS_SUSPECT = "SUSPECT"
STATUS_NO_MY_KEYFRAME = "NO_MY_KEYFRAME"
STATUS_DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
STATUS_NO_POSE_DETECTED = "NO_POSE_DETECTED"

REPORT_FIELDS = [
    "origin_no", "gloss_name", "status", "best_score", "best_frame_idx",
    "keyframe_matched", "keyframe_total", "hand_used", "frames_scanned", "video_path", "note",
]


@dataclass
class Signature:
    """한 프레임(이미지)의 pose 특징: body keypoints + 검출된 손들."""
    body: object  # (17,3) ndarray | None
    hands: dict = field(default_factory=dict)  # {"Left"/"Right": (21,2) ndarray}


class Extractors:
    """YOLO pose + MediaPipe hands 모델을 한 번만 로드해서 재사용."""

    def __init__(self, pose_model: str | Path = None):
        t0 = time.perf_counter()
        self.pose = PoseExtractor(pose_model) if pose_model else PoseExtractor()
        self.hands = HandPoseExtractor()
        log.info(f"[pipeline] YOLO+MediaPipe 모델 로딩: {time.perf_counter() - t0:.3f}초")

    def signature(self, frame) -> Signature:
        return Signature(body=self.pose.extract(frame), hands=self.hands.extract(frame))


@dataclass
class ValidationResult:
    origin_no: str
    gloss_name: str
    status: str
    best_score: Optional[float] = None
    best_frame_idx: Optional[int] = None
    hand_used: bool = False
    frames_scanned: int = 0
    video_path: Optional[str] = None
    note: str = ""
    # metadata에 태깅된 키프레임이 여러 개일 때, 그 각각이 사전 동영상에서
    # 얼마나 잘 매칭됐는지 개수로 집계 (예: 3개 중 2개 매칭)
    keyframe_total: int = 0
    keyframe_matched: int = 0
    per_keyframe_scores: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "origin_no": self.origin_no,
            "gloss_name": self.gloss_name,
            "status": self.status,
            "best_score": f"{self.best_score:.4f}" if self.best_score is not None else "",
            "best_frame_idx": self.best_frame_idx if self.best_frame_idx is not None else "",
            "keyframe_matched": self.keyframe_matched,
            "keyframe_total": self.keyframe_total,
            "hand_used": "Y" if self.hand_used else "N",
            "frames_scanned": self.frames_scanned,
            "video_path": self.video_path or "",
            "note": self.note,
        }


def extract_my_keyframe_signatures_from_nas(
    entry: DatasetEntry, dataset_root: Path, extractors: Extractors
) -> tuple[list[Signature], str]:
    """entry.keyframes에 태깅된 프레임 인덱스 '전부'에 대해 signature를 뽑는다
    (첫 번째만 쓰면 검증 정확도를 개수 기준으로 보고할 수 없다)."""
    if not entry.video_rel_path:
        return [], "metadata에 video_path 없음"
    if not entry.keyframes:
        return [], "metadata에 keyframes 없음"

    video_path = dataset_root / entry.video_rel_path
    if not video_path.exists():
        return [], f"NAS 비디오 없음(마운트 확인 필요): {video_path}"

    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], f"비디오를 열 수 없음: {video_path}"

    sigs = []
    for kf_idx in entry.keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        sig = extractors.signature(frame)
        if sig.body is not None or sig.hands:
            sigs.append(sig)
    cap.release()
    log.debug(
        f"[pipeline] NAS 비디오 태깅 키프레임 {len(entry.keyframes)}개 중 {len(sigs)}개 signature 추출: "
        f"{time.perf_counter() - t0:.3f}초"
    )
    if not sigs:
        return [], "태깅된 키프레임에서 사람/손 검출 실패"
    return sigs, ""


def extract_my_keyframe_signatures_from_images_dir(
    entry: DatasetEntry, keyframe_images_dir: Path, extractors: Extractors
) -> tuple[list[Signature], str]:
    """keyframe_images의 origin_no로 매칭되는 사진 '전부'에 대해 signature를 뽑는다."""
    t0 = time.perf_counter()
    matches = find_keyframe_images(keyframe_images_dir, entry.origin_no)
    if not matches:
        return [], f"keyframe_images에 origin_no={entry.origin_no} 사진 없음(NAS 마운트 확인)"

    sigs = []
    for m in matches:
        frame = cv2.imread(str(m))
        if frame is None:
            continue
        sig = extractors.signature(frame)
        if sig.body is not None or sig.hands:
            sigs.append(sig)
    log.debug(
        f"[pipeline] NAS keyframe_images {len(matches)}장 중 {len(sigs)}개 signature 추출: "
        f"{time.perf_counter() - t0:.3f}초"
    )
    if not sigs:
        return [], "사진들에서 사람/손 검출 실패"
    return sigs, ""


def extract_keyframe_from_image(image_path: Path, extractors: Extractors):
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None, f"이미지를 읽을 수 없음: {image_path}"
    sig = extractors.signature(frame)
    if sig.body is None and not sig.hands:
        return None, f"이미지에서 사람/손 검출 실패: {image_path}"
    return sig, ""


def scan_video_for_best_match(
    video_path: Path, target_sig: Signature, extractors: Extractors, stride: int = 1,
    on_frame=None,
) -> tuple[Optional[float], Optional[int], bool, int]:
    """반환: (best_score, best_frame_idx, hand_used_at_best, scanned_frame_count)

    on_frame: (frame_idx, frame_ndarray, current_score_or_None) -> None
    지정하면 스캔한 프레임마다 호출된다 (GUI에서 진행 상황을 실시간으로 보여주는 용도).
    """
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, False, 0

    best_score = None
    best_idx = None
    best_hand_used = False
    frame_idx = 0
    scanned = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            sig = extractors.signature(frame)
            score, detail = combined_similarity(target_sig.body, sig.body, target_sig.hands, sig.hands)
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_idx = frame_idx
                best_hand_used = detail["hand_used"]
            scanned += 1
            if on_frame is not None:
                on_frame(frame_idx, frame, score)
        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t0
    rate = scanned / elapsed if elapsed > 0 else 0
    log.info(
        f"[pipeline] scan_video_for_best_match {Path(video_path).name}: "
        f"{scanned}프레임 스캔, {elapsed:.3f}초 ({rate:.1f} fps 처리 속도), best_score={best_score}"
    )
    return best_score, best_idx, best_hand_used, scanned


def scan_video_for_all_targets(
    video_path: Path, target_sigs: list[Signature], extractors: Extractors, stride: int = 1,
    on_frame=None,
) -> tuple[list[tuple[Optional[float], Optional[int], bool]], int]:
    """태깅된 키프레임 signature '전부'에 대해, 동영상을 한 번만 스캔하면서
    각각의 최고매칭 프레임을 동시에 찾는다 (target마다 따로 스캔하면 N배
    느려지므로 한 번의 디코딩 패스에서 전부 비교한다).

    반환: ([(best_score, best_frame_idx, hand_used), ...] target_sigs와 같은 순서, scanned_frame_count)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [(None, None, False) for _ in target_sigs], 0

    best = [[None, None, False] for _ in target_sigs]
    frame_idx = 0
    scanned = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            sig = extractors.signature(frame)
            for i, target_sig in enumerate(target_sigs):
                score, detail = combined_similarity(target_sig.body, sig.body, target_sig.hands, sig.hands)
                if score is not None and (best[i][0] is None or score > best[i][0]):
                    best[i][0] = score
                    best[i][1] = frame_idx
                    best[i][2] = detail["hand_used"]
            scanned += 1
            if on_frame is not None:
                on_frame(frame_idx, frame, best[0][0] if best else None)
        frame_idx += 1

    cap.release()
    return [tuple(b) for b in best], scanned


def validate_entry(
    entry: DatasetEntry,
    extractors: Extractors,
    cache_dir: Path,
    dataset_root: Optional[Path] = None,
    keyframe_images_dir: Optional[Path] = None,
    my_keyframe_image: Optional[Path] = None,
    threshold: float = 0.7,
    stride: int = 3,
    session: Optional[requests.Session] = None,
    category: str = "",
    on_frame=None,
) -> ValidationResult:
    session = session or requests.Session()

    # 1) 내 키프레임 signature 확보 - 태깅된 키프레임 '전부' (metadata keyframes 개수만큼).
    #    우선순위: 수동 지정 이미지(항상 1개) > NAS keyframe_images 실제 사진들 > NAS 원본 비디오 프레임들
    if my_keyframe_image is not None:
        sig, note = extract_keyframe_from_image(my_keyframe_image, extractors)
        my_sigs = [sig] if sig is not None else []
    elif keyframe_images_dir is not None:
        my_sigs, note = extract_my_keyframe_signatures_from_images_dir(entry, keyframe_images_dir, extractors)
    elif dataset_root is not None:
        my_sigs, note = extract_my_keyframe_signatures_from_nas(entry, dataset_root, extractors)
    else:
        my_sigs, note = [], "keyframe_images_dir/dataset_root/my_keyframe_image 모두 지정 안 됨"

    if not my_sigs:
        return ValidationResult(entry.origin_no, entry.gloss_name, STATUS_NO_MY_KEYFRAME, note=note)

    # 2) 사전 사이트에서 동영상 다운로드
    try:
        video_path = sldict_client.fetch_and_download(
            entry.origin_no, cache_dir, category=category, session=session
        )
    except Exception as e:  # noqa: BLE001 - 네트워크/사이트 응답 오류를 모두 리포트에 남긴다
        return ValidationResult(
            entry.origin_no, entry.gloss_name, STATUS_DOWNLOAD_FAILED, note=str(e)
        )

    # 3) 프레임 전수 스캔하며, 태깅된 키프레임 각각의 최고 유사 프레임을 동시에 탐색
    per_target, scanned = scan_video_for_all_targets(
        video_path, my_sigs, extractors, stride=stride, on_frame=on_frame
    )
    scores = [s for s, _, _ in per_target if s is not None]

    if not scores:
        return ValidationResult(
            entry.origin_no, entry.gloss_name, STATUS_NO_POSE_DETECTED,
            frames_scanned=scanned, video_path=str(video_path),
            keyframe_total=len(my_sigs),
        )

    # 대표(headline) 점수는 태깅된 키프레임들 중 가장 잘 맞은 것 - 기존 동작과 호환
    best_i = max(range(len(per_target)), key=lambda i: (per_target[i][0] if per_target[i][0] is not None else -1))
    best_score, best_idx, hand_used = per_target[best_i]
    keyframe_matched = sum(1 for s in scores if s >= threshold)
    per_keyframe_scores = [s for s, _, _ in per_target]

    status = STATUS_MATCH if best_score >= threshold else STATUS_SUSPECT
    log.info(
        f"[pipeline] origin_no={entry.origin_no} 키프레임 매칭: "
        f"{keyframe_matched}/{len(my_sigs)}개 (threshold={threshold})"
    )
    return ValidationResult(
        entry.origin_no, entry.gloss_name, status,
        best_score=best_score, best_frame_idx=best_idx, hand_used=hand_used,
        frames_scanned=scanned, video_path=str(video_path),
        keyframe_total=len(my_sigs), keyframe_matched=keyframe_matched,
        per_keyframe_scores=per_keyframe_scores,
    )


def run_validation(
    entries: list[DatasetEntry],
    report_path: Path,
    cache_dir: Path,
    dataset_root: Optional[Path] = None,
    keyframe_images_dir: Optional[Path] = None,
    threshold: float = 0.7,
    stride: int = 3,
    model_name: str | Path = None,
    progress_cb=None,
) -> list[ValidationResult]:
    extractors = Extractors(model_name)
    session = requests.Session()
    results: list[ValidationResult] = []

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()

        for i, entry in enumerate(entries):
            result = validate_entry(
                entry, extractors, cache_dir,
                dataset_root=dataset_root, keyframe_images_dir=keyframe_images_dir,
                threshold=threshold, stride=stride,
                session=session,
            )
            results.append(result)
            writer.writerow(result.as_row())
            f.flush()
            if progress_cb:
                progress_cb(i + 1, len(entries), result)

    return results
