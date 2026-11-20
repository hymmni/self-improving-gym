import json

import numpy as np

from square_assembly.scripts.branch_rollout import delay_points, is_hacky


def _attrs(t_press, t0, max_delay, events):
    return {"t_press": t_press, "t0": t0, "max_delay": max_delay, "delay_events": json.dumps(events)}


def test_delay_points_reads_the_recorded_takeover():
    a = _attrs(107, 295, 414, [{"t_press": 107, "d": 188, "t0": 295, "max_delay": 414, "status": "taken"}])
    modes = np.r_[np.zeros(295), np.ones(10)]
    assert delay_points(a, modes) == (107, 295, 522)


def test_delay_points_recovers_a_takeover_lost_by_the_rewind_bug():
    # 0e61624 이전: t0로 정확히 되감으면 taken 기록이 rewound로 바뀌고 attrs는 -1로 남았다
    ev = [{"t_press": 249, "d": 88, "t0": 337, "max_delay": 272, "status": "cancelled"},
          {"t_press": 185, "d": 157, "t0": 342, "max_delay": 336, "status": "rewound"}]
    modes = np.r_[np.zeros(342), np.ones(5)]
    assert delay_points(_attrs(-1, -1, -1, ev), modes) == (185, 342, 522)


def test_delay_points_is_none_without_a_takeover():
    ev = [{"t_press": 50, "d": 9, "t0": 59, "max_delay": 400, "status": "cancelled"}]
    assert delay_points(_attrs(-1, -1, -1, ev), np.zeros(300)) is None


def test_hacky_means_the_last_grasp_before_success_was_not_the_handle():
    assert is_hacky(True, [None, "handle", None, "body", "body", None])
    assert not is_hacky(True, ["body", None, "handle", None])
    assert not is_hacky(False, ["body"])          # 실패는 야매 성공이 아니다
    assert not is_hacky(True, [None, None])       # 쥔 적이 없으면 판정하지 않는다
