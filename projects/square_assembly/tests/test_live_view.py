"""LiveView(MJPEG 스트림 + 터미널 키 입력) 단독 검증.

이 파일이 실패한다는 건 어떤 스크립트든 실시간 화면을 재사용할 때 근본이 깨졌다는
뜻이다 - scripted_intervention.py처럼 이 클래스를 감싸는 코드마다 따로 테스트하지
않아도 되게 하는 게 이 모듈을 뽑아낸 이유다.
"""

import http.client

from square_assembly.utils.live_view import LiveView


def test_construction_has_no_side_effects():
    view = LiveView(port=0)
    assert view._started is False
    assert view._httpd is None


def test_poll_key_returns_none_without_tty():
    """pytest 하의 stdin은 tty가 아니므로 키 입력을 조용히 건너뛰어야 한다."""
    view = LiveView(port=0)
    try:
        assert view.poll_key() is None
        assert view._stdin_is_tty is False
    finally:
        view.close()


def test_push_jpeg_is_servable_over_http():
    view = LiveView(port=0)
    try:
        view.push_jpeg(b"fake-jpeg-bytes")
        port = view.bound_port

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert b"multipart/x-mixed-replace" in resp.getheader("Content-Type").encode()
        chunk = resp.read(64)  # 무한 스트림이라 앞부분만 확인
        assert b"fake-jpeg-bytes" in chunk
        conn.close()
    finally:
        view.close()
