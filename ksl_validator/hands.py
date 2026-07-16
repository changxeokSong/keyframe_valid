"""MediaPipe HandLandmarker 래퍼 — 손가락 관절(21점 x 최대 2손) 추출.

YOLO body pose는 손목 2점만 찍어서 손모양(핸드셰이프)을 구분하지 못한다.
수어는 손모양이 의미를 가르는 핵심 신호이므로 이 모듈로 보강한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"


class HandPoseExtractor:
    def __init__(self, model_path: Path = MODEL_PATH, min_confidence: float = 0.3):
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE,
            min_hand_detection_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)

    def extract(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """{'Left': (21,2), 'Right': (21,2)} 정규화 좌표(0~1). 미검출 손은 키 없음."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        out: dict[str, np.ndarray] = {}
        for lm_list, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name  # "Left" | "Right"
            pts = np.array([[p.x, p.y] for p in lm_list], dtype=np.float32)
            # 같은 라벨이 두 번 잡히면(오검출) confidence 높은 쪽만 유지
            if label not in out:
                out[label] = pts
        return out
