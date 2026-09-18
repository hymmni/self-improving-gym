"""마우스 텔레옵 제어 법칙 + 키 상태 머신 검증. cv2/pynput/robosuite 없이 돈다.

실제 조작감(게인·속도)은 서버에서 창을 띄워 손으로 맞추는 것이라 여기서는 "입력이 액션의
어느 성분을 어느 방향으로 움직이는가"와 상한/클램프/초기화만 본다.
"""

import numpy as np
import pytest

from square_assembly.runners.mouse_teleop import (
    MouseTeleopController, MouseTeleopIntervention, TopDownMap,
)
from square_assembly.runners.square_oracle import _POS_SCALE

# 수직 아래를 보는 그리퍼, 손가락 축 = 월드 +x
_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _state(x=0.0, y=0.0, z=1.0, R=_DOWN):
    return {"grip": np.array([x, y, z]), "R": R.copy(),
            "nut": np.array([0.1, 0.0, 0.83]), "handle": np.array([0.154, 0.0, 0.83]),
            "peg": np.array([0.23, 0.1, 0.85])}


def test_xy_is_proportional_and_capped():
    c = MouseTeleopController(kp=1.0, kd=0.0, pos_cap=0.3)
    c.take_over(_state())
    c.set_cursor((0.005, 0.0))
    a = c.action(_state())
    np.testing.assert_allclose(a[:2], [0.005 / _POS_SCALE, 0.0])
    c.set_cursor((-0.2, 0.05))
    a = c.action(_state())
    np.testing.assert_allclose(a[:2], [-0.3, 0.3])  # 상한에 걸림


def test_kd_damps_when_already_moving_toward_target():
    p = MouseTeleopController(kp=1.0, kd=0.0)
    d = MouseTeleopController(kp=1.0, kd=0.5)
    for c in (p, d):
        c.take_over(_state(x=0.0))
        c.set_cursor((0.01, 0.0))
        c.action(_state(x=0.0))
    a_p = p.action(_state(x=0.004))  # 한 스텝에 4mm 전진한 상태
    a_d = d.action(_state(x=0.004))
    assert 0 < a_d[0] < a_p[0]


def test_take_over_holds_position_until_cursor_moves():
    c = MouseTeleopController()
    c.set_cursor((0.3, 0.3))  # 잡기 전에 쌓인 커서 위치는 무시돼야 한다
    c.take_over(_state())
    np.testing.assert_array_equal(c.action(_state())[:2], [0.0, 0.0])


def test_z_keys_and_limits():
    c = MouseTeleopController(z_speed=0.2, z_min=0.83, z_max=1.10)
    c.take_over(_state())
    c.z_up = True
    assert c.action(_state(z=1.0))[2] == pytest.approx(0.2)
    assert c.action(_state(z=1.10))[2] == 0.0  # 상한에서 멈춤
    c.z_up, c.z_down = False, True
    assert c.action(_state(z=1.0))[2] == pytest.approx(-0.2)
    assert c.action(_state(z=0.83))[2] == 0.0
    c.z_up = True  # 둘 다 누르면 정지
    assert c.action(_state(z=1.0))[2] == 0.0


def test_wheel_turns_yaw_target_about_world_z():
    c = MouseTeleopController(yaw_step=np.deg2rad(5.0), rot_cap=0.4)
    c.take_over(_state())
    np.testing.assert_array_equal(c.action(_state())[3:6], [0.0, 0.0, 0.0])  # 목표 = 현재
    c.wheel(+1)
    a = c.action(_state())
    assert a[5] == pytest.approx(np.deg2rad(5.0) / 0.5)  # z축 회전, _ROT_SCALE=0.5
    assert abs(a[3]) < 1e-9 and abs(a[4]) < 1e-9
    c.wheel(-3)
    assert c.action(_state())[5] == pytest.approx(-np.deg2rad(10.0) / 0.5)


def test_gripper_ramps_with_button():
    c = MouseTeleopController(grip_rate=0.5)
    c.take_over(_state())
    assert c.grip_cmd == -1.0
    c.grip_pressed = True
    assert [c.action(_state())[6] for _ in range(3)] == [-0.5, 0.0, 0.5]
    assert c.action(_state())[6] == 1.0 and c.action(_state())[6] == 1.0  # 클램프
    c.grip_pressed = False
    assert c.action(_state())[6] == 0.5


def test_take_over_starts_closed_when_fingers_are_closed():
    c = MouseTeleopController()
    c.take_over(_state(), finger_gap=0.02)
    assert c.grip_cmd == 1.0
    c.take_over(_state(), finger_gap=0.08)
    assert c.grip_cmd == -1.0


def test_map_roundtrip_and_orientation():
    m = TopDownMap((0.0, 0.0), 0.8, 480)
    assert m.to_px((0.0, 0.0)) == (240, 240)
    px, py = m.to_px((0.1, 0.2))
    assert px > 240 and py > 240  # +y는 오른쪽, +x는 아래(로봇 쪽이 위)
    np.testing.assert_allclose(m.to_world(*m.to_px((0.1, 0.2))), [0.1, 0.2], atol=1e-3)


def _make():
    interv = MouseTeleopIntervention(env=object(), state_fn=_state)
    return interv


def test_keys_drive_the_state_machine():
    interv = _make()
    obs = {"robot0_gripper_qpos": np.array([0.04, -0.04])}
    assert interv(0, obs) is None
    interv._handle_key(ord("h"))
    assert interv.num_triggers == 1
    a = interv(1, obs)
    assert a.shape == (7,)
    interv._handle_key(ord("h"))  # 이미 잡은 상태에서 또 눌러도 카운트 안 됨
    assert interv.num_triggers == 1
    interv._handle_key(ord("p"))
    assert interv(2, obs) is None
    interv._handle_key(ord("h"))
    assert interv.num_triggers == 2
    assert not interv._paused
    interv._handle_key(ord("s"))
    assert interv._paused
    interv._handle_key(ord("s"))
    assert not interv._paused
    assert not interv.should_end()
    interv._handle_key(ord("q"))
    assert interv.should_end()


def test_reset_clears_everything():
    interv = _make()
    interv._handle_key(ord("h"))
    interv._handle_key(ord("s"))
    interv._handle_key(ord("q"))
    interv.reset()
    assert interv(0, {}) is None and not interv._paused and not interv.should_end()
    assert interv.num_triggers == 0
