"""steps-to-go 라벨을 만드는 단 하나의 규칙.

기준선(RobomimicSequenceDataset 경로)과 DINO/VIP(특징 캐시 경로)가 각자 라벨을 계산하면
인코더를 비교한다면서 실제로는 라벨 처리를 비교하게 된다. 두 경로 모두 여기를 호출한다.

라벨의 두 가지 조정(둘 다 기본은 꺼짐 — 끄면 예전 동작 그대로다):

label_horizon(H)
    '남은 스텝 수' 대신 '남은 비율 x H'. 사람이 얼마나 빨리 몰았는지에 불변이라, 길이가
    들쭉날쭉한 텔레옵 데이터에서 같은 장면에 정답이 여러 개가 되는 걸 막는다.

preintv
    카운트다운 라벨은 사람이 곧 개입하는 구간에서도 매 스텝 1씩 줄어 "좋아지는 중"이라고
    말한다. 'flat'은 구간 내내 시작점의 참값으로 고정(진전 0), 'rise'는 스텝마다 1씩
    올린다(후퇴). 'drop'은 라벨을 건드리지 않고 마스크만 돌려주며, 그 프레임을 train에서
    뺄지는 부르는 쪽이 정한다(val에서 빼면 held-out 비교가 깨진다).
"""

import numpy as np

PREINTV_MODES = ("none", "drop", "flat", "rise")


def build_labels(names, ts, lengths, success, modes=None, fail_bin=None,
                 label_horizon=None, preintv="none"):
    """(N,) int64 라벨과 PREINTV 마스크를 만든다.

    Args:
        names: (N,) 샘플별 데모 이름.
        ts: (N,) 샘플별 데모 내 프레임 인덱스.
        lengths: {데모 이름: 프레임 수}.
        success: {데모 이름: bool}.
        modes: {데모 이름: (L,) action_mode}. preintv != "none"이면 필수.
        fail_bin: 실패 데모에 붙일 클래스. 실패 데모가 있는데 None이면 죽는다.
        label_horizon: 위 설명 참고. None이면 남은 스텝 수 그대로.
        preintv: PREINTV_MODES 중 하나.

    Returns:
        (labels, preintv_mask): 각각 (N,) int64, (N,) bool.
    """
    if preintv not in PREINTV_MODES:
        raise ValueError(f"preintv={preintv!r}는 {PREINTV_MODES} 중 하나여야 한다")
    names, ts = np.asarray(names), np.asarray(ts)
    if preintv != "none" and not modes:
        raise ValueError("preintv 처리를 쓰려면 action_mode(modes)가 필요하다")

    remaining = np.array([lengths[n] - 1 - t for n, t in zip(names, ts)], dtype=np.float64)
    mask = _preintv_mask(names, ts, modes)

    if preintv in ("flat", "rise"):
        # 구간의 수준은 시작점의 참값으로 두고 기울기만 바꾼다 — 없는 크기를 지어내지 않는다.
        for name, a, b in _runs(names, ts, modes, success):
            base = lengths[name] - 1 - a
            sel = (names == name) & (ts >= a) & (ts <= b)
            offset = (ts[sel] - a) if preintv == "rise" else 0
            remaining[sel] = base + offset

    if label_horizon is not None:
        span = np.array([max(lengths[n] - 1, 1) for n in names], dtype=np.float64)
        remaining = np.round(remaining / span * label_horizon)

    labels = remaining.astype(np.int64)
    fails = np.array([not success[n] for n in names])
    if fails.any():
        if fail_bin is None:
            raise ValueError(
                "실패 데모가 섞여 있는데 fail_bin이 없다 — 실패 transition에 성공 기준 "
                "steps-to-go 라벨을 붙이는 건 범주 오류다(train_dstg_failaware.py 참고)"
            )
        labels[fails] = fail_bin
    return labels, mask


def _preintv_mask(names, ts, modes):
    from square_assembly.datasets.labels import LABEL_PREINTV
    if not modes:
        return np.zeros(len(names), dtype=bool)
    return np.array([n in modes and modes[n][t] == LABEL_PREINTV for n, t in zip(names, ts)])


def _runs(names, ts, modes, success):
    """성공 데모 안의 연속된 PREINTV 구간을 (데모 이름, 시작 t, 끝 t)로 돌려준다."""
    from square_assembly.datasets.labels import LABEL_PREINTV
    for name, mode in modes.items():
        if not success.get(name, False):
            continue
        idx = np.flatnonzero(np.asarray(mode) == LABEL_PREINTV)
        if len(idx) == 0:
            continue
        brk = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([[idx[0]], idx[brk + 1]])
        ends = np.concatenate([idx[brk], [idx[-1]]])
        for a, b in zip(starts, ends):
            yield name, int(a), int(b)
