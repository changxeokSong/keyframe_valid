"""ultralytics YOLO pose 모델로 프레임에서 keypoint를 뽑는 얇은 래퍼."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from .logging_setup import log

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "yolo11n-pose.pt"


def select_device() -> str:
    """CUDA(NVIDIA) 있으면 그걸, 없으면 MPS(Apple Silicon), 그것도 없으면 CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# COCO 17-keypoint pose. 얼굴 점(0-4)은 수어 판별에 거의 도움이 안 되므로
# 상체/팔/손목 위주(5-10)에 가중치를 더 준다. (손가락 관절까지는 커버 못 함 — 한계점)
KEYPOINT_WEIGHTS = np.array(
    [0.2, 0.2, 0.2, 0.2, 0.2,  # 코/눈/귀
     1.0, 1.0,                 # 어깨
     1.5, 1.5,                 # 팔꿈치
     2.0, 2.0,                 # 손목 (수어에서 가장 중요)
     0.5, 0.5,                 # 골반
     0.3, 0.3,                 # 무릎
     0.2, 0.2],                # 발목
    dtype=np.float32,
)


class PoseExtractor:
    def __init__(self, model_name: str | Path = DEFAULT_MODEL_PATH):
        self.model = YOLO(str(model_name))
        self.device = select_device()
        log.info(f"[pose] YOLO 추론 디바이스: {self.device}")

    def extract(self, frame: np.ndarray) -> np.ndarray | None:
        """frame(BGR ndarray)에서 가장 confidence 높은 사람의 keypoint를 반환.

        반환값: (17, 3) ndarray [x, y, conf] (정규화 전, 원본 픽셀 좌표) 또는
        사람이 검출되지 않으면 None.
        """
        results = self.model(frame, verbose=False, device=self.device)
        if not results:
            return None
        r = results[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None

        # 여러 명이 잡히면 bbox 면적이 가장 큰(=카메라에 가장 가까운) 사람을 사용
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            areas = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).cpu().numpy()
            idx = int(np.argmax(areas))
        else:
            idx = 0

        kpts = r.keypoints.data[idx].cpu().numpy()  # (17, 3)
        return kpts
