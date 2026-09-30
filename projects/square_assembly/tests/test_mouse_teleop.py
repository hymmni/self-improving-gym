"""마우스 텔레옵 제어 법칙 + 키 상태 머신 검증. 창(Tk)/robosuite 없이 돈다.

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


def test_gripper_follows_the_button_immediately():
    """robosuite가 부호만 보고 스스로 램프한다 — 여기서 램프하면 부호가 바뀔 때까지 0.5초 늦기만 한다."""
    c = MouseTeleopController()
    c.take_over(_state())
    assert c.grip_cmd == -1.0
    c.grip_pressed = True
    assert c.action(_state())[6] == 1.0
    c.grip_pressed = False
    assert c.action(_state())[6] == -1.0


def test_take_over_follows_the_button_not_the_fingers():
    """넘겨받는 순간에도 그리퍼는 버튼만 따른다 — 빈손으로 닫힌 채(헛집기) 넘겨받으면 클릭 없이 바로 연다.
    쥔 너트를 유지하려면 버튼을 누른 채 Tab을 누른다(2026-09-30 사용자 결정)."""
    c = MouseTeleopController()
    c.take_over(_state())
    assert c.grip_cmd == -1.0 and c.action(_state())[6] == -1.0
    c.grip_pressed = True  # 누른 채 Tab
    c.take_over(_state())
    assert c.grip_cmd == 1.0
    for _ in range(30):
        assert c.action(_state())[6] == 1.0
    c.grip_pressed = False
    assert c.action(_state())[6] == -1.0


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
    interv._handle_key("Tab")
    assert interv.num_triggers == 1
    a = interv(1, obs)
    assert a.shape == (7,)
    interv._handle_key("Tab")  # 한 번 더 누르면 정책에 돌려준다
    assert interv(2, obs) is None
    interv._handle_key("Tab")
    assert interv.num_triggers == 2
    assert not interv._paused
    interv._handle_key("s")
    assert interv._paused
    interv._handle_key("s")
    assert not interv._paused
    assert not interv.should_end()
    interv._handle_key("q")
    assert interv.should_end()


def test_reset_clears_everything():
    interv = _make()
    interv._handle_key("Tab")
    interv._handle_key("s")
    interv._handle_key("q")
    interv.reset()
    assert interv(0, {}) is None and not interv._paused and not interv.should_end()
    assert interv.num_triggers == 0


def test_space_and_shift_move_z_for_as_long_as_they_are_held():
    """예전(pynput + 0.6초 유지)엔 꾹 눌러도 처음 0.6초만 움직였다 — 이제 누름~뗌 사이 내내 움직인다."""
    interv = _make()
    interv._on_key("Tab", True)
    interv._on_key("space", True)
    assert all(interv(t, {})[2] > 0 for t in range(40))    # 2초(40스텝) 누르고 있어도 계속
    interv._on_key("space", False)
    interv._on_key("space", True)                         # X11 키 반복 = 뗌+누름 쌍 — 끊기지 않는다
    assert interv(40, {})[2] > 0
    interv._on_key("space", False)
    assert interv(41, {})[2] == 0.0
    interv._on_key("Shift_L", True)
    assert all(interv(t, {})[2] < 0 for t in range(42, 82))  # 수식키는 반복이 없어도 계속
    interv._on_key("B", True)                             # Shift를 누른 채 b(대문자로 온다)도 먹는다
    assert interv.pop_rewind(82) == 62
    interv._on_key("Shift_L", False)
    assert interv(62, {})[2] == 0.0


def test_ctrl_lifts_because_rustdesk_sends_space_as_taps():
    """RustDesk는 꾹 누른 Space를 누름+뗌 쌍으로만 보낸다 — Ctrl은 누름·뗌이 제대로 와서 누르는 동안 올라간다."""
    interv = _make()
    interv._on_key("Tab", True)
    interv._on_key("Control_L", True)
    assert all(interv(t, {})[2] > 0 for t in range(20))
    interv._on_key("Control_L", False)
    assert interv(20, {})[2] == 0.0
    interv._on_key("space", True); interv._on_key("space", False)  # 같은 update()에서 처리되는 탭 쌍
    assert interv(21, {})[2] == 0.0


def test_shift_tab_still_toggles_and_yaw_is_wheel_only():
    interv = _make()
    interv._on_key("Tab", True)
    interv._on_key("a", True); interv._on_key("d", True)  # a/d는 없앴다(2026-09-22) — 휠로만 돈다
    assert interv(0, {})[5] == 0.0
    interv.controller.wheel(+1)
    assert interv(1, {})[5] > 0            # 휠 위 한 칸: +5도 목표 -> z축 양의 회전
    assert interv._on_key("ISO_Left_Tab", True) == "break"  # Shift+Tab도 전환, Tk 포커스 이동은 막는다
    assert interv(2, {}) is None


def test_back_accumulates_pauses_and_clamps_at_zero():
    interv = MouseTeleopIntervention(env=object(), state_fn=_state, back_steps=40)
    assert interv.pop_rewind(100) is None
    interv._handle_key("b")
    interv._handle_key("b")
    assert interv._paused                  # 되감으면 멈춰서 어디로 왔는지 보여준다
    assert interv.pop_rewind(100) == 20    # 두 번 = 80스텝 뒤로
    assert interv._rewound == (100, 20)    # 멈춘 화면에 "어디서 어디로" 띄울 정보
    interv._handle_key("s")
    assert interv._rewound is None         # 재개하면 지운다
    assert interv.pop_rewind(100) is None  # 꺼내면 비워진다
    interv._handle_key("b")
    assert interv.pop_rewind(15) == 0      # 시작보다 앞으로는 못 간다


def test_restart_goes_to_zero_and_beats_back():
    interv = _make()
    interv._handle_key("b")
    interv._handle_key("r")
    interv._handle_key("b")
    assert interv.pop_rewind(300) == 0


def test_rewind_resyncs_the_controller_to_the_restored_pose():
    """되감기 전 목표(커서·야우·그리퍼)를 그대로 들고 가면 재개하자마자 팔이 튄다."""
    interv = _make()
    interv._handle_key("Tab")
    interv.controller.set_cursor((0.2, 0.2))
    assert np.abs(interv(0, {})[:2]).max() > 0
    interv._handle_key("b")
    interv.pop_rewind(50)
    np.testing.assert_allclose(interv(10, {})[:2], 0.0)  # 커서가 다시 움직일 때까지 제자리


def test_back_is_one_second_by_default():
    assert _make().back_steps == 20  # 20Hz 제어


def test_reset_clears_pending_rewind():
    interv = _make()
    interv._handle_key("r")
    interv.reset()
    assert interv.pop_rewind(100) is None


def _modes(interv, pattern):
    """pattern(문자열, h=사람 p=정책)대로 스텝을 밟는다 — 스텝마다 모드가 기록된다."""
    for t, m in enumerate(pattern):
        if (m == "h") != interv._active:
            interv._handle_key("Tab")
        interv(t, {})
    return len(pattern)


def test_left_goes_back_to_where_the_current_mode_started():
    interv = _make()
    step = _modes(interv, "p" * 30 + "h" * 12)       # 30에서 개입, 지금 42
    interv._handle_key("Left")
    assert interv._paused
    assert interv.pop_rewind(step) == 30 and interv._active  # 개입 시작점, 사람 모드 그대로
    interv(30, {})
    interv._handle_key("Left")                         # 전환점 위에서 또 누르면 그 앞 전환점
    assert interv.pop_rewind(31) == 30
    interv._handle_key("Left")
    assert interv.pop_rewind(30) == 0 and not interv._active  # 정책 구간 시작 — 정책 모드로


def test_left_presses_accumulate_across_switches():
    interv = _make()
    step = _modes(interv, "p" * 10 + "h" * 5 + "p" * 5 + "h" * 5)
    interv._handle_key("Left"); interv._handle_key("Left")
    assert interv.pop_rewind(step) == 15 and not interv._active


def test_back_past_the_takeover_hands_control_back_to_the_policy():
    """개입 시작점 앞으로 되감으면 정책이 다시 몬다 — 개입 시점이 되감기 계산으로 앞당겨지지 않는다."""
    interv = _make()
    step = _modes(interv, "p" * 100 + "h" * 10)
    interv._handle_key("b")
    assert interv.pop_rewind(step) == 90
    assert interv(90, {}) is None                      # 90은 원래 정책 스텝
    interv._handle_key("r")
    assert interv.pop_rewind(91) == 0 and not interv._active


def test_rewind_to_jumps_to_that_step_pauses_and_restores_its_mode():
    """렌더 노이즈 복구용 — 수집기가 노이즈 직전 정상 스텝을 직접 지정한다. 키 되감기보다 우선."""
    interv = _make()
    step = _modes(interv, "p" * 20 + "h" * 10)
    interv._handle_key("b")
    interv.rewind_to(12)
    assert interv._paused
    assert interv.pop_rewind(step) == 12 and not interv._active  # 12는 정책 스텝
    assert interv.pop_rewind(12) is None                        # 요청은 한 번만


def test_reset_clears_a_pending_rewind_to():
    interv = _make()
    _modes(interv, "p" * 5)
    interv.rewind_to(2)
    interv.reset()
    assert interv.pop_rewind(5) is None


def _map_state():
    box = [np.array([0.0, 0.0]), np.array([0.02, 0.0]), np.array([0.02, 0.02]), np.array([0.0, 0.02])]
    return {**_state(), "peg_boxes": [box], "nut_boxes": [box, box]}


def test_cameras_stack_on_the_left_and_the_map_takes_the_right():
    pytest.importorskip("cv2")
    interv = MouseTeleopIntervention(env=object(), state_fn=_map_state, map_size=960)
    frames = [np.zeros((480, 480, 3), np.uint8), np.full((480, 480, 3), 255, np.uint8)]
    canvas = interv._compose(frames)
    assert canvas.shape == (960, 480 + 960, 3)
    assert canvas[300, 200].max() == 0 and canvas[700, 200].min() == 255  # agentview 위, wrist 아래


def test_camera_tiles_get_a_border_and_their_name():
    pytest.importorskip("cv2")
    interv = MouseTeleopIntervention(env=object(), state_fn=_map_state, map_size=960)
    gray = np.full((480, 480, 3), 128, np.uint8)
    plain = interv._compose([gray, gray])
    boxed = interv._compose({"agentview": gray, "robot0_eye_in_hand": gray})
    assert (plain[480, 240] == 128).all() and (plain[5:25, 5:60] == 128).all()  # 이름 없으면 안 그림
    assert (boxed[480, 240] != 128).any() and (boxed[240, 240] == 128).all()  # 경계선만, 가운데는 그대로
    assert (boxed[5:25, 5:60] != 128).any()  # 왼쪽 위 이름표


def test_cursor_only_follows_the_mouse_over_the_map():
    pytest.importorskip("cv2")
    interv = MouseTeleopIntervention(env=object(), state_fn=_map_state, map_size=960)
    interv._compose([np.zeros((480, 480, 3), np.uint8)] * 2)
    interv.controller.target_xy = None
    interv._on_motion(100, 480)  # 카메라 열 위
    assert interv.controller.target_xy is None
    interv._on_motion(480 + 480, 480)  # 맵 한가운데 = 테이블 중심
    np.testing.assert_allclose(interv.controller.target_xy, interv.map.center, atol=1e-3)


def _paused_human():
    interv = MouseTeleopIntervention(env=object(), state_fn=_state)  # 그리퍼 (0, 0), 맵 480px = 0.8m
    interv._handle_key("Tab")
    interv._handle_key("s")
    return interv


def _px(interv, xy):
    x, y = interv.map.to_px(xy)
    return x + interv._map_x0, y


def _x_server(interv):
    """X 서버 흉내: 옮긴 포인터 자리마다 그 자리 Motion 이벤트를 보낸다."""
    for w, _ in list(interv._pending):
        interv._on_motion(*w)


def _hold(interv, seconds, dt=0.02):
    for _ in range(int(round(seconds / dt))):
        interv._tick(_state(), dt)
        _x_server(interv)


def _hand(interv, dpx):
    """손이 dpx(px)만큼 움직인 마우스 이벤트 — 지금 포인터 자리에서 상대 이동."""
    _x_server(interv)
    interv._on_motion(*(np.asarray(interv._prev_px, dtype=float) + dpx))
    _x_server(interv)


def _cursor_cm(interv):
    return np.linalg.norm(interv._cur) * 100


def test_paused_pointer_near_the_gripper_snaps_the_target_onto_it():
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.02, 0.01)))          # 2.2cm 옆 — 붙는 반경(2.5cm) 안
    assert interv._snapped
    np.testing.assert_allclose(interv.controller.target_xy, [0.0, 0.0], atol=1e-9)


def test_snapped_cursor_is_pulled_onto_the_gripper_and_the_os_pointer_follows():
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.02, 0.01)))
    _hold(interv, 0.6)
    assert _cursor_cm(interv) < 0.3                          # 커서가 그리퍼 중심으로 끌려왔고
    np.testing.assert_allclose(interv._last_warp, _px(interv, (0.0, 0.0)), atol=1.0)  # 실제 포인터도 옮겼다


def test_a_gentle_tug_stretches_then_springs_back():
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.02, 0.0)))
    _hold(interv, 0.6)
    _hand(interv, (12, 0))                                   # 손이 오른쪽으로 2cm(12px)
    _hold(interv, 0.06)
    assert interv._snapped and 0.3 < _cursor_cm(interv) < 1.0  # 끈적하게 일부만 따라 늘어났다가
    _hold(interv, 0.6)
    assert interv._snapped and _cursor_cm(interv) < 0.3      # 힘을 빼면 돌아온다
    np.testing.assert_allclose(interv.controller.target_xy, [0.0, 0.0], atol=1e-9)


def test_a_hard_yank_breaks_free_and_is_not_pulled_back():
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.02, 0.0)))
    _hold(interv, 0.6)
    _hand(interv, (36, 0))                                   # 한 번에 6cm
    assert not interv._snapped
    assert 2.0 < _cursor_cm(interv) < 2.3                    # 보이던 자리(늘어난 2.1cm)에서 떨어져
    _hold(interv, 0.3)
    popped = _cursor_cm(interv)
    assert 3.1 < popped < 3.5                                # 당긴 쪽으로 1.2cm 미끄러져 나가고
    _hold(interv, 0.5)
    assert _cursor_cm(interv) == pytest.approx(popped, abs=0.05)  # 떨어진 채 — 다시 끌려오지 않는다
    np.testing.assert_allclose(interv.controller.target_xy, interv._cur, atol=1e-9)


def test_nearby_cursor_is_drawn_in_and_captured():
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.04, 0.0)))            # 끌림 반경(4.5cm) 안, 붙는 반경 밖
    assert not interv._snapped
    _hold(interv, 1.0)
    assert interv._snapped


def test_far_pointer_does_not_snap_and_pulling_away_releases():
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.02, 0.0)))
    interv._on_motion(*_px(interv, (0.10, 0.0)))            # 크게 빼면 떨어져 당긴 쪽으로 튀어나간다
    assert not interv._snapped
    np.testing.assert_allclose(interv.controller.target_xy, interv._cur, atol=1e-9)


def test_no_snap_while_running_or_in_policy_mode():
    running = MouseTeleopIntervention(env=object(), state_fn=_state)
    running._handle_key("Tab")
    running._on_motion(*_px(running, (0.02, 0.0)))
    _hold(running, 0.5)
    assert not running._snapped and running._last_warp is None
    np.testing.assert_allclose(running.controller.target_xy, [0.02, 0.0], atol=2e-3)
    policy = MouseTeleopIntervention(env=object(), state_fn=_state)
    policy._handle_key("s")
    policy._on_motion(*_px(policy, (0.02, 0.0)))
    assert not policy._snapped


def test_new_takeover_forgets_the_magnet():
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.02, 0.01)))
    interv._handle_key("s"); interv._handle_key("Tab"); interv._handle_key("Tab")  # 재개 → 정책 → 다시 사람
    interv._on_motion(*_px(interv, (0.05, 0.0)))
    assert not interv._snapped
    np.testing.assert_allclose(interv.controller.target_xy, [0.05, 0.0], atol=2e-3)


def test_capture_also_pulls_the_yaw_onto_the_gripper():
    interv = _paused_human()
    interv.controller.target_yaw = 0.5                      # 멈춘 동안 휠로 돌려둔 목표
    interv._on_motion(*_px(interv, (0.02, 0.0)))
    assert interv.controller.target_yaw == pytest.approx(0.0, abs=1e-9)


def test_wheel_twists_the_snapped_yaw_but_it_springs_back_and_never_breaks_off_alone():
    """회전엔 따로 떨어지는 조건이 없다 — 붙은 채 세게 돌려도 비틀렸다가 원래 각도로 돌아온다."""
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.02, 0.0)))
    for _ in range(8):                                       # 40도를 한 번에
        interv._on_wheel(+1)
    assert interv._snapped and interv._yo > np.deg2rad(5)   # 화면의 목표 그리퍼는 비틀리고
    assert interv.controller.target_yaw == pytest.approx(0.0, abs=1e-9)  # 제어 목표는 그대로
    _hold(interv, 0.8)
    assert abs(interv._yo) < np.deg2rad(1)                  # 휠을 멈추면 원래 각도로
    _hand(interv, (36, 0))                                   # 위치를 당겨 떨어지면
    interv._on_wheel(+1)
    assert interv.controller.target_yaw == pytest.approx(interv.controller.yaw_step, abs=np.deg2rad(1))


def test_wheel_is_plain_yaw_while_running():
    interv = MouseTeleopIntervention(env=object(), state_fn=_state)
    interv._handle_key("Tab")
    interv._on_wheel(+1)
    assert interv.controller.target_yaw == pytest.approx(interv.controller.yaw_step)


def test_events_that_left_before_the_warp_are_measured_from_the_old_spot():
    """옮기기 전에 생겨 늦게 온 이벤트는 옛 기준, 옮긴 자리 이벤트 뒤로는 새 기준 — 손 움직임을 안 잃는다."""
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.02, 0.0)))
    _hold(interv, 0.6)
    stretch0 = interv._stretch.copy()
    old = np.asarray(interv._prev_px, dtype=float)
    interv._on_motion(*(old + (3, 0)))                       # 손 +3px → 끈적하게 따라가며 포인터를 옮긴다
    interv._on_motion(*(old + (6, 0)))                       # 옮기기 전에 생긴 이벤트(옛 기준 +3px)
    _x_server(interv)                                        # 옮긴 자리 이벤트
    interv._on_motion(*(np.asarray(interv._prev_px) + (3, 0)))  # 새 기준 +3px
    moved_px = (interv._stretch - stretch0)[1] * interv.map.scale  # 월드 +y = 화면 오른쪽
    assert moved_px == pytest.approx(9.0, abs=0.5)


def test_far_cursor_is_not_drawn_in():
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.06, 0.0)))            # 끌림 반경(4.5cm) 밖
    _hold(interv, 1.0)
    assert not interv._snapped and _cursor_cm(interv) == pytest.approx(6.0, abs=0.2)


def _break_free(interv, dpx=(30, 0)):
    interv._on_motion(*_px(interv, (0.02, 0.0)))
    _hold(interv, 0.6)
    _hand(interv, dpx)                                       # 5cm 당겨 떨어짐 — 끌림 반경 안에 떨어진다
    assert not interv._snapped
    _hold(interv, 0.3)                                       # 떨어져 미끄러지는 것까지


def test_resting_after_breaking_free_is_not_grabbed_again():
    interv = _paused_human()
    _break_free(interv)
    popped = _cursor_cm(interv)
    _hold(interv, 1.0)
    assert not interv._snapped and _cursor_cm(interv) == pytest.approx(popped, abs=0.05)


def test_moving_back_toward_the_gripper_after_a_short_break_re_attaches():
    interv = _paused_human()
    _break_free(interv)
    _hold(interv, 0.4)                                       # 떨어진 직후의 쉬는 시간이 지나고
    for _ in range(3):
        _hand(interv, (-2, 0))                               # 그리퍼 쪽으로 조금씩
        _hold(interv, 0.1)
    _hold(interv, 0.5)
    assert interv._snapped


def test_capture_turns_the_yaw_the_short_way():
    """350도 돌려둔 목표는 붙을 때 +10도만, 370도는 −10도만 돌아 붙는다."""
    for off, sign in ((350, +1), (370, -1)):
        interv = _paused_human()
        interv.controller.target_yaw = np.deg2rad(off)
        interv._on_motion(*_px(interv, (0.02, 0.0)))
        assert np.degrees(interv._yo) == pytest.approx(-10 * sign, abs=1e-6)  # 보이는 목표는 10도 옆에서
        seen = []
        for _ in range(40):
            interv._tick(_state(), 0.02)
            seen.append(np.degrees(interv._yo))
        assert max(abs(v) for v in seen) <= 10.5 and abs(seen[-1]) < 0.5  # 짧은 쪽으로 돌아 붙는다
        assert interv.controller.target_yaw == pytest.approx(0.0, abs=1e-9)


def test_a_long_wheel_spin_while_snapped_also_returns_the_short_way():
    interv = _paused_human()
    interv._on_motion(*_px(interv, (0.02, 0.0)))
    for _ in range(120):                                     # 600도 — 보이는 비틀림 210도
        interv._on_wheel(+1)
    assert abs(interv._yo) <= np.pi
    start = abs(np.degrees(interv._yo))
    seen = []
    for _ in range(50):
        interv._tick(_state(), 0.02)
        seen.append(abs(np.degrees(interv._yo)))
    assert max(seen) <= start + 1 and seen[-1] < 1
