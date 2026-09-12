"""ScriptedFailureIntervention 상태 머신 + MJPEG 렌더 경로 검증.

키 입력을 상태로 바꾸는 _handle_key()와 트리거 이후 액션 시퀀스를 내는 __call__ 상태
머신, 그리고 render()가 실제로 GUI 없이 프레임을 인코딩해 MJPEG로 내보내는지 확인한다
(2026-09-12, cv2.imshow 창을 없애고 HTTP 스트림으로 바꾼 뒤 추가 — 이 테스트가 실패하면
누군가 다시 cv2.imshow류의 GUI 호출을 넣었다가 X11 없는 환경에서 멈추는 걸 여기서 잡는다).
render()를 호출하지 않는 한 __init__은 포트/터미널에 손대지 않는다 - 여러 테스트가
독립적으로 인스턴스를 만들어도 서로 간섭하지 않는다.
"""

import http.client

import numpy as np

from square_assembly.runners.scripted_intervention import ScriptedFailureIntervention


def _make(**kwargs):
    return ScriptedFailureIntervention(camera_key="agentview_image", action_dim=7, **kwargs)


def test_no_trigger_returns_none_every_step():
    interv = _make()
    for step in range(5):
        assert interv(step, {}) is None


def test_trigger_starts_recovery_for_exact_step_count():
    interv = _make(recovery_steps=3)
    interv.trigger()

    actions = [interv(s, {}) for s in range(3)]
    assert all(a is not None for a in actions)
    assert interv(3, {}) is None  # recovery_steps를 다 쓰면 정책으로 복귀


def test_recovery_action_shape_and_values():
    interv = _make(recovery_steps=1, retract_z=1.0, gripper_open=-1.0)
    interv.trigger()

    action = interv(0, {})
    assert action.shape == (7,)
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action[[0, 1, 3, 4, 5]], 0.0)  # z·그리퍼 외엔 전부 0
    assert action[2] == 1.0   # z축 후퇴
    assert action[-1] == -1.0  # 그리퍼 열기


def test_trigger_while_recovering_is_ignored():
    """회복 도중 다시 트리거해도 recovery_steps가 늘어나거나 재시작하지 않는다(중첩 방지)."""
    interv = _make(recovery_steps=3)
    interv.trigger()
    interv(0, {})
    interv.trigger()  # 회복 중 재트리거 - 무시돼야 함
    interv(1, {})

    assert interv(2, {}) is not None  # 원래 recovery_steps(3)까지는 여전히 액션
    assert interv(3, {}) is None      # 그 다음엔 정책 복귀 (늘어나지 않음)
    assert interv.num_triggers == 1


def test_reset_clears_state():
    interv = _make(recovery_steps=5)
    interv.trigger()
    interv(0, {})
    interv._handle_key(ord("q"))
    assert interv.should_end()

    interv.reset()
    assert not interv.should_end()
    assert interv.num_triggers == 0
    assert interv(0, {}) is None  # 대기 중이던 트리거도 초기화됨


def test_handle_key_trigger_and_quit():
    interv = _make(trigger_key="s", quit_key="q", recovery_steps=2)

    interv._handle_key(ord("x"))  # 무관한 키는 무시
    assert interv(0, {}) is None

    interv._handle_key(ord("s"))
    assert interv(0, {}) is not None
    assert interv.num_triggers == 1

    assert not interv.should_end()
    interv._handle_key(ord("q"))
    assert interv.should_end()


def test_render_streams_jpeg_without_any_gui():
    """render()는 cv2.imshow 창을 안 열고, LiveView(MJPEG)로만 프레임을 내보낸다."""
    interv = _make(http_port=0)  # 0 = OS가 빈 포트 배정(테스트 간 충돌 방지)
    assert interv._view._started is False  # 생성만으로는 서버/터미널에 손대지 않는다

    obs = {"agentview_image": np.zeros((84, 84, 3), dtype=np.uint8)}
    try:
        assert interv.render(obs) is True  # 계속 진행(종료 아님)
        assert interv._view._httpd.latest_jpeg is not None
        assert interv._view._stdin_is_tty is False  # pytest 하의 stdin은 tty가 아님

        port = interv._view.bound_port
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert b"multipart/x-mixed-replace" in resp.getheader("Content-Type").encode()
        chunk = resp.read(64)  # 무한 스트림이라 전체를 다 읽지 않고 앞부분만 확인
        assert chunk.startswith(b"--frame")
        conn.close()
    finally:
        interv.close()
