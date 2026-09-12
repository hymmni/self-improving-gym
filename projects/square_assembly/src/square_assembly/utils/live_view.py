"""JPEG 프레임을 로컬 MJPEG 스트림으로 내보내고, SSH 터미널에서 키 입력을 읽는 공용 유틸.

X11/RustDesk 없이 원격 GPU 서버의 실시간 화면을 봐야 하는 어떤 스크립트에서든 재사용한다
(2026-09-12, runners/scripted_intervention.py에서 추출 — 원래 이유는 이 서버 sshd가
ssh -X/-Y 모두에서 cv2(Qt/xcb)의 확장 버전 질의에 응답을 안 줘서 GUI 창이 영원히 멈췄고,
RustDesk는 한 번에 한 사람만 접속 가능해 대안이 필요했던 것 — DOCKER.md §4 참고).

사용법:
    view = LiveView(port=8765)          # port=0이면 OS가 빈 포트 배정(테스트용)
    ...
    ok, buf = cv2.imencode(".jpg", bgr)  # GUI 없는 순수 코덱 연산 - X11 불필요
    view.push_jpeg(buf.tobytes())
    key = view.poll_key()                # 논블로킹, 없으면 None
    ...
    view.close()

`docker-compose.yml`의 `network_mode: host`를 전제로 127.0.0.1에만 바인드한다 — 그래야
컨테이너 밖에서도 `ssh -L <port>:localhost:<port> <server>`로 로컬 브라우저까지만 닿고,
서버의 실제 네트워크 인터페이스엔 인증 없이 노출되지 않는다.
"""

import select
import sys
import termios
import threading
import time
import tty
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _MJPEGHandler(BaseHTTPRequestHandler):
    """self.server.latest_jpeg(bytes|None)를 계속 밀어주는 최소 MJPEG 스트림."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                frame = self.server.latest_jpeg
                if frame is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 뷰어가 탭을 닫았을 뿐, 정상 종료

    def log_message(self, format, *args):
        pass  # 매 프레임 요청마다 콘솔에 접속 로그 찍히는 것 방지


class LiveView:
    """MJPEG 스트림(화면 출력) + 터미널 cbreak 키 입력(제어)을 함께 다루는 헬퍼.

    생성만으로는 포트/터미널에 손대지 않는다 - 첫 push_jpeg()/poll_key() 호출 때 지연
    시작한다(단위 테스트가 인스턴스를 만들기만 해도 부작용이 생기지 않게).
    """

    def __init__(self, port=8765):
        self.port = port
        self._httpd = None
        self._thread = None
        self._stdin_is_tty = False
        self._orig_termios = None
        self._started = False

    def _ensure_started(self):
        if self._started:
            return
        self._started = True

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), _MJPEGHandler)
        self._httpd.latest_jpeg = None
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        port = self._httpd.server_port
        print(
            f"[LiveView] 로컬에서 `ssh -L {port}:localhost:{port} <server>` 포트포워딩 후 "
            f"브라우저로 http://localhost:{port} 접속하면 화면이 보인다.",
            flush=True,
        )

        self._stdin_is_tty = sys.stdin.isatty()
        if self._stdin_is_tty:
            self._orig_termios = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())  # ISIG는 유지 -> Ctrl+C 정상 동작

    @property
    def bound_port(self):
        """실제 바인드된 포트(port=0으로 만들었으면 OS가 고른 값)."""
        self._ensure_started()
        return self._httpd.server_port

    def push_jpeg(self, jpeg_bytes):
        self._ensure_started()
        self._httpd.latest_jpeg = jpeg_bytes

    def poll_key(self):
        """stdin에서 논블로킹으로 키 하나 읽기(cbreak라 Enter 불필요). 없으면 None."""
        self._ensure_started()
        if not self._stdin_is_tty:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        return ord(ch) if ch else None

    def close(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._stdin_is_tty and self._orig_termios is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._orig_termios)
