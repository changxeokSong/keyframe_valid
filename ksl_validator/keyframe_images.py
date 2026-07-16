"""NAS의 실제 키프레임 사진 폴더 접근.

경로 (dataset.json 기준): \\mldisk2\\nfs_shared\\abd\\dataset\\sl\\etri_ksl_db\\keyframe_images
파일명 패턴: {origin_no}_{gloss_name}_{index}.jpg   (예: 1_파나마_1.jpg)

sldict 사이트의 '수형 사진'은 일러스트(선화)라 MediaPipe/YOLO가 인식을 못 하지만,
이 폴더의 사진은 실제 촬영본이라 자동 채점(pose 비교)의 '내 키프레임' 소스로 쓸 수 있다.
"""

from __future__ import annotations

from pathlib import Path

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
