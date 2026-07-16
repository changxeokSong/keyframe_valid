"""두 keypoint 세트 사이의 자세 유사도 스코어링.

카메라 거리/사람 위치에 영향받지 않도록 어깨 중심으로 평행이동하고
어깨 폭으로 스케일을 정규화한 뒤, keypoint별 가중 코사인 유사도를 낸다.
"""

from __future__ import annotations

import numpy as np

from .pose import KEYPOINT_WEIGHTS

CONF_THRESHOLD = 0.3
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray | None:
    """(17,3) -> 어깨중심 원점, 어깨폭=1로 정규화된 (17,2). 실패 시 None."""
    if kpts is None:
        return None
    ls, rs = kpts[LEFT_SHOULDER], kpts[RIGHT_SHOULDER]
    if ls[2] < CONF_THRESHOLD or rs[2] < CONF_THRESHOLD:
        return None

    center = (ls[:2] + rs[:2]) / 2.0
    scale = np.linalg.norm(ls[:2] - rs[:2])
    if scale < 1e-6:
        return None

    xy = (kpts[:, :2] - center) / scale
    conf = kpts[:, 2]
    xy[conf < CONF_THRESHOLD] = np.nan
    return xy


def pose_similarity(kpts_a: np.ndarray | None, kpts_b: np.ndarray | None) -> float | None:
    """0~1 유사도 점수. 어느 한쪽이라도 정규화 불가하면 None."""
    a = normalize_keypoints(kpts_a)
    b = normalize_keypoints(kpts_b)
    if a is None or b is None:
        return None

    valid = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    if valid.sum() < 4:  # 비교 가능한 점이 너무 적음
        return None

    diff = np.linalg.norm(a[valid] - b[valid], axis=1)  # 정규화 좌표계에서의 거리
    weights = KEYPOINT_WEIGHTS[valid]
    weighted_dist = float(np.average(diff, weights=weights))

    # 거리(작을수록 유사) -> 0~1 유사도로 변환. 어깨폭 기준 거리이므로
    # weighted_dist=0 -> 1.0, weighted_dist>=1.5(어깨폭 1.5배 이상 벌어짐) -> 0.0 근방
    score = max(0.0, 1.0 - weighted_dist / 1.5)
    return score


# ── 손모양(핸드셰이프) 비교 ──────────────────────────────────────────────
# body pose는 어깨/팔꿈치/손목 위치까지만 보므로 손가락을 접었는지 폈는지
# 구분하지 못한다. 수어는 이 손모양 차이가 뜻을 가르는 핵심이라 별도로 비교한다.

WRIST, MIDDLE_MCP = 0, 9
FINGERTIPS = (4, 8, 12, 16, 20)
HAND_WEIGHTS = np.ones(21, dtype=np.float32)
HAND_WEIGHTS[list(FINGERTIPS)] = 2.0


def normalize_hand(landmarks_xy: np.ndarray) -> np.ndarray | None:
    """(21,2) 정규화좌표 -> 손목 원점/손바닥 길이=1/회전 정렬된 (21,2). 실패 시 None."""
    origin = landmarks_xy[WRIST]
    ref = landmarks_xy[MIDDLE_MCP] - origin
    scale = float(np.linalg.norm(ref))
    if scale < 1e-6:
        return None

    angle = np.arctan2(ref[1], ref[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    pts = (landmarks_xy - origin) / scale
    return pts @ rot.T


def hand_shape_similarity(a_xy: np.ndarray | None, b_xy: np.ndarray | None) -> float | None:
    if a_xy is None or b_xy is None:
        return None
    na = normalize_hand(a_xy)
    nb = normalize_hand(b_xy)
    if na is None or nb is None:
        return None

    dist = np.linalg.norm(na - nb, axis=1)
    weighted_dist = float(np.average(dist, weights=HAND_WEIGHTS))
    # 손목 기준 정규화 좌표계 거리. weighted_dist=0 -> 1.0, >=1.2 -> 0.0 근방
    return max(0.0, 1.0 - weighted_dist / 1.2)


def hands_similarity(hands_a: dict, hands_b: dict) -> float | None:
    """손 라벨(Left/Right)이 겹치는 손들만 비교해서 평균낸다. 겹치는 손 없으면 None."""
    common = set(hands_a) & set(hands_b)
    if not common:
        return None
    scores = []
    for label in common:
        s = hand_shape_similarity(hands_a[label], hands_b[label])
        if s is not None:
            scores.append(s)
    return float(np.mean(scores)) if scores else None


# ── 종합 스코어 (body pose + 손모양) ────────────────────────────────────
BODY_WEIGHT_WITH_HANDS = 0.3
HAND_WEIGHT = 0.7


def combined_similarity(
    body_a: np.ndarray | None, body_b: np.ndarray | None,
    hands_a: dict, hands_b: dict,
) -> tuple[float | None, dict]:
    """(종합점수, {'body': .., 'hand': .., 'hand_used': bool}) 반환.

    손이 양쪽에서 겹쳐 잡히면 body+hand 가중합, 아니면 body 점수만 사용하되
    hand_used=False로 표시해 신뢰도가 낮음을 알 수 있게 한다.
    """
    body_score = pose_similarity(body_a, body_b)
    hand_score = hands_similarity(hands_a, hands_b)

    detail = {"body": body_score, "hand": hand_score, "hand_used": hand_score is not None}

    if body_score is None and hand_score is None:
        return None, detail
    if hand_score is None:
        return body_score, detail
    if body_score is None:
        return hand_score, detail

    combined = BODY_WEIGHT_WITH_HANDS * body_score + HAND_WEIGHT * hand_score
    return combined, detail
