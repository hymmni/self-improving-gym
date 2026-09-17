"""ScriptedFailureIntervention 상태 머신 + 오라클 자세 정렬 계산 검증.

cv2/robosuite 없이 돌아야 한다 - render()의 그리기 로직과 오라클의 sim 읽기(_state)는
실제 시뮬레이터가 있어야 의미가 있으므로 여기선 빼고, 키 입력을 상태로 바꾸는
_handle_key()/__call__ 위임과, 오라클에서 유일하게 까다로운 계산인 _rot()만 검증한다.
(오라클 전체의 성공률은 서버에서 실측한다 - square_oracle.py 상단 참고.)
"""

import numpy as np

from square_assembly.runners.scripted_intervention import ScriptedFailureIntervention
from square_assembly.runners.square_oracle import SquareAssemblyOracle


class _StubOracle:
    """호출될 때마다 고정 액션을 돌려주는 가짜 오라클."""

    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.phase = "stub"

    def reset(self):
        self.resets += 1

    def __call__(self, step, obs_raw):
        self.calls += 1
        return np.ones(7, dtype=np.float32)


def _make(**kwargs):
    oracle = _StubOracle()
    return ScriptedFailureIntervention(oracle, **kwargs), oracle


def test_no_trigger_means_policy_runs():
    interv, oracle = _make()
    for step in range(5):
        assert interv(step, {}) is None
    assert oracle.calls == 0


def test_trigger_hands_control_to_oracle_and_never_gives_it_back():
    interv, oracle = _make()
    interv.trigger()

    for step in range(10):
        action = interv(step, {})
        assert action is not None  # 정책에 돌려주지 않는다
    assert oracle.calls == 10
    assert interv.num_triggers == 1


def test_repeated_trigger_counts_once():
    interv, _ = _make()
    interv._handle_key(ord("s"))
    interv._handle_key(ord("s"))
    assert interv.num_triggers == 1


def test_reset_clears_state_and_resets_oracle():
    interv, oracle = _make()
    interv.trigger()
    interv._handle_key(ord("q"))
    assert interv.should_end()

    interv.reset()
    assert not interv.should_end()
    assert interv.num_triggers == 0
    assert interv(0, {}) is None  # 트리거도 풀린다
    assert oracle.resets == 1


def test_handle_key_ignores_unrelated_keys():
    interv, _ = _make(trigger_key="s", quit_key="q")
    interv._handle_key(ord("x"))
    assert interv(0, {}) is None
    assert not interv.should_end()


def _oracle():
    return SquareAssemblyOracle(env=object())


def _state(R, handle_dir=0.0):
    """핸들 막대가 handle_dir 방향인 상황. _rot()은 handle-nut 방향과 R만 본다."""
    return {"R": np.asarray(R, dtype=float),
            "nut": np.zeros(3),
            "handle": np.array([np.cos(handle_dir), np.sin(handle_dir), 0.0]) * 0.054}


# 열 = [손가락 축, y, 접근 축]. 수직 아래를 보고 손가락 축이 월드 x인 자세.
_DOWN_X = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
# 같은 자세에서 손가락 축만 월드 y로 돌린 것(= 핸들이 x 방향일 때의 정답 자세)
_DOWN_Y = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def test_rot_is_zero_when_already_perpendicular_to_handle():
    action, angle = _oracle()._rot(_state(_DOWN_Y, handle_dir=0.0))
    assert angle < 1e-6
    np.testing.assert_allclose(action, 0.0, atol=1e-9)


def test_rot_turns_90deg_about_z_when_fingers_are_parallel_to_handle():
    action, angle = _oracle()._rot(_state(_DOWN_X, handle_dir=0.0))
    assert abs(angle - np.pi / 2) < 1e-6
    np.testing.assert_allclose(action[:2], 0.0, atol=1e-9)  # z축 회전만
    assert abs(action[2]) == 0.4  # rot_cap으로 잘림


def test_rot_corrects_a_tilted_wrist_not_just_yaw():
    """정렬된 yaw라도 손목이 기울면 x/y 성분으로 세워야 한다 - 이걸 안 해서 잡기에
    실패했던 회귀(2026-09-17)."""
    t = np.deg2rad(30)
    tilt = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    action, angle = _oracle()._rot(_state(tilt @ _DOWN_Y, handle_dir=0.0))
    assert abs(angle - t) < 1e-6
    assert np.abs(action[:2]).max() > 0.05
