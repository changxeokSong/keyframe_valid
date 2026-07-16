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


# ── 손 위치(어디 근처인지) 비교 ──────────────────────────────────────────
# hand_shape_similarity는 손목을 원점으로 다시 정규화하기 때문에 완전히
# 위치 무관(position-invariant)하다 - 손이 눈 앞이든 가슴 앞이든 손모양만
# 같으면 똑같이 취급된다. 그런데 수어는 "손을 어디 근처에 두는지"(눈/코/입/
# 어깨/가슴 등)도 뜻을 가르는 핵심 요소라서, 손목과 주요 신체 랜드마크
# 사이의 거리를 정규화 좌표계에서 명시적으로 비교하는 신호를 따로 둔다.
PROXIMITY_LANDMARKS = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_hip": 11, "right_hip": 12,
}
WRIST_INDICES = {"left": 9, "right": 10}


def _wrist_proximity_features(norm_kpts: np.ndarray) -> dict[str, float]:
    """정규화된 (17,2) 좌표에서 '손목 -> 주요 랜드마크' 거리들을 계산한다."""
    feats: dict[str, float] = {}
    for wrist_name, wrist_idx in WRIST_INDICES.items():
        wrist = norm_kpts[wrist_idx]
        if np.isnan(wrist).any():
            continue
        for lm_name, lm_idx in PROXIMITY_LANDMARKS.items():
            lm = norm_kpts[lm_idx]
            if np.isnan(lm).any():
                continue
            feats[f"{wrist_name}_wrist_to_{lm_name}"] = float(np.linalg.norm(wrist - lm))
    return feats


def hand_position_similarity(kpts_a: np.ndarray | None, kpts_b: np.ndarray | None) -> float | None:
    """손목이 눈/코/어깨/가슴(골반) 등에서 얼마나 가까운지가 양쪽에서 비슷한지 비교.
    0~1, 겹치는 특징이 없으면 None."""
    a = normalize_keypoints(kpts_a)
    b = normalize_keypoints(kpts_b)
    if a is None or b is None:
        return None

    fa = _wrist_proximity_features(a)
    fb = _wrist_proximity_features(b)
    common = set(fa) & set(fb)
    if not common:
        return None

    avg_diff = float(np.mean([abs(fa[k] - fb[k]) for k in common]))
    # 어깨폭 기준 거리차. 0 -> 1.0, >=1.0(어깨폭만큼 벌어짐) -> 0.0 근방
    return max(0.0, 1.0 - avg_diff / 1.0)


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


# ── 종합 스코어 (body pose + 손모양 + 손 위치) ───────────────────────────
# 손모양(어떤 모양인지)이 여전히 가장 중요하지만, 손 위치(눈 앞/코 앞/가슴 앞 등)도
# 뜻을 가르는 핵심이라 무시할 수 없어서 별도 가중치를 준다.
BODY_WEIGHT = 0.15
HAND_SHAPE_WEIGHT = 0.55
HAND_POSITION_WEIGHT = 0.30


def combined_similarity(
    body_a: np.ndarray | None, body_b: np.ndarray | None,
    hands_a: dict, hands_b: dict,
) -> tuple[float | None, dict]:
    """(종합점수, {'body', 'hand', 'hand_position', 'hand_used'}) 반환.

    사용 가능한 신호만 가중합하고(없는 신호는 그 가중치를 나머지에 재분배),
    손모양이 양쪽에서 겹쳐 잡혀야 hand_used=True로 표시해 신뢰도를 알 수 있게 한다.
    """
    body_score = pose_similarity(body_a, body_b)
    hand_score = hands_similarity(hands_a, hands_b)
    position_score = hand_position_similarity(body_a, body_b)

    detail = {
        "body": body_score, "hand": hand_score, "hand_position": position_score,
        "hand_used": hand_score is not None,
    }

    weighted = [
        (body_score, BODY_WEIGHT),
        (hand_score, HAND_SHAPE_WEIGHT),
        (position_score, HAND_POSITION_WEIGHT),
    ]
    available = [(s, w) for s, w in weighted if s is not None]
    if not available:
        return None, detail

    total_w = sum(w for _, w in available)
    combined = sum(s * w for s, w in available) / total_w
    return combined, detail
