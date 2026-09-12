"""사람이 실패를 "판단"하면 트리거하고, 실제 회복 동작은 고정 스크립트가 실행하는 개입.

KeyboardIntervention(runners/intervention_rollout.py — robosuite Keyboard device로 사람이
로봇을 직접 몬다)과 다르다. 여기선 사람이 조종하지 않는다 — 트리거 시점만 사람이 정하고,
그 뒤 recovery_steps 동안은 고정된 회복 액션(그리퍼 열기 + 위로 후퇴)이 실행된 뒤 정책에
제어를 돌려준다.

collect_episode(runners/intervention_rollout.py)의 두 콜백 계약에 이 클래스 하나로 꽂는다:
  render_fn=interv.render         (매 스텝 프레임 표시 + 트리거/종료 키 감지)
  intervention_fn=interv          (트리거 이후 recovery_steps 동안 고정 액션 반환)
  should_end_fn=interv.should_end (종료 키)

트리거는 render()(env.step 이후, obs_raw_{t+1} 시점)에서 걸리고, 실제 회복 액션은 다음
스텝의 intervention_fn(step+1, ...)부터 나간다 — collect_episode 루프 순서(intervention_fn
-> env.step -> render_fn) 그대로다. recovery_steps를 다 쓰면 intervention_fn이 다시 None을
반환해 정책이 재계획(chunk=None, intervention_rollout.py 기존 로직)하며 이어받는다.

회복 액션은 OSC_POSE 기본 컨트롤러(robosuite Panda 기본값, configs/task/square.yaml
action_dim=7과 일치) 레이아웃을 가정한다: action = [dx, dy, dz, drx, dry, drz, gripper].
z는 월드 상승 방향, gripper는 양수=닫힘/음수=열림(robosuite 표준 관례 — 서버에서
load_composite_controller_config(robot="Panda")로 직접 확인함, 2026-09-09). 다른
로봇/컨트롤러 레이아웃(예: 양팔)엔 그대로 안 맞을 수 있다.

## 화면 표시: cv2.imshow가 아니라 MJPEG 스트림 + 터미널 키 입력 (2026-09-12)
원래는 cv2.imshow(X11/Qt GUI 창)를 썼는데, 이 GPU 서버는 ssh -X(심지어 -Y로 trusted
forwarding을 써도) 상태에서 cv2가 번들한 Qt/xcb가 커넥션 초기화 중 보내는 확장 버전 질의
(xcb_shm_query_version, xcb_xfixes_query_version, ...)에 응답을 영영 못 받고 멈춘다
(DOCKER.md §4 2026-09-11 기록, py-spy 네이티브 스택으로 확인). RustDesk로 서버의 실제 로컬
세션에 붙는 우회도 되지만 그 세션은 한 번에 한 사람만 쓸 수 있어(2026-09-12 실측 필요성
확인) — 그래서 X11/GUI를 아예 안 쓰는 방식으로 바꿨다:
  - 프레임은 cv2.imencode로 JPEG 인코딩만 하고(GUI 없이 순수 코덱 연산이라 X11 불필요),
    내장 HTTP 서버(stdlib http.server)로 MJPEG 스트림을 로컬 포트에 띄운다. 사용자는
    `ssh -L <port>:localhost:<port>`로 평범하게 포트포워딩만 하면 아무 브라우저로 볼 수
    있다 — RustDesk/ssh -X 어느 쪽도 필요 없다.
  - 트리거/종료 키는 cv2 창 대신, 이 스크립트가 실행 중인 SSH 터미널 자체에서 cbreak 모드로
    한 글자씩(엔터 불필요) 논블로킹으로 읽는다(stdlib termios/tty/select).
  - stdin이 tty가 아니면(예: 테스트, 파이프 리다이렉션) 조용히 키 입력을 건너뛴다.
HTTP 서버/터미널 cbreak 모드는 첫 render() 호출 때 지연 시작한다(생성자에서 시작하면 이
클래스를 그냥 만들기만 해도 포트를 점유하고 stdin을 건드려 상태 머신 단위 테스트가 깨진다).
"""

import select
import sys
import termios
import threading
import time
import tty
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_Z_INDEX = 2
_GRIPPER_INDEX = -1


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


class ScriptedFailureIntervention:
    """실패로 보이면 트리거 -> recovery_steps 동안 [그리퍼 열기 + 위로 후퇴] -> 정책 복귀.

    Args:
        camera_key (str): 화면에 띄울 obs rgb 키(예: "agentview_image").
        action_dim (int): 액션 차원(square task=7). 회복 액션은 이 중 z축·그리퍼만 채운다.
        trigger_key (str): 트리거 키 문자(기본 "s").
        quit_key (str): 에피소드 포기 키(기본 "q").
        recovery_steps (int): 트리거 후 스크립트가 제어하는 스텝 수.
        retract_z (float): 회복 중 z축 델타 액션 크기(컨트롤러 input 범위 [-1,1], 양수=상승).
        gripper_open (float): 회복 중 그리퍼 액션 값(음수=열림).
        http_port (int): MJPEG 스트림 포트. 0이면 OS가 빈 포트를 골라준다(테스트용).
    """

    def __init__(self, camera_key, action_dim=7, trigger_key="s", quit_key="q",
                 recovery_steps=20, retract_z=1.0, gripper_open=-1.0, http_port=8765):
        self.camera_key = camera_key
        self.action_dim = action_dim
        self.trigger_key = ord(trigger_key)
        self.quit_key = ord(quit_key)
        self.recovery_steps = recovery_steps
        self.retract_z = retract_z
        self.gripper_open = gripper_open
        self.http_port = http_port
        self._pending_trigger = False
        self._recovery_remaining = 0
        self._quit_requested = False
        self.num_triggers = 0  # 에피소드당 트리거 횟수(진단/로그용)

        self._httpd = None
        self._http_thread = None
        self._stdin_is_tty = False
        self._orig_termios = None
        self._started = False

    def reset(self):
        """에피소드 시작마다 호출한다(collect_episode는 이 객체를 자동 리셋하지 않는다)."""
        self._pending_trigger = False
        self._recovery_remaining = 0
        self._quit_requested = False
        self.num_triggers = 0

    def trigger(self):
        """트리거 요청. 이미 회복 중이면 무시한다(중첩/연장 방지)."""
        if self._recovery_remaining == 0:
            self._pending_trigger = True

    def should_end(self):
        return self._quit_requested

    def _recovery_action(self):
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[_Z_INDEX] = self.retract_z
        action[_GRIPPER_INDEX] = self.gripper_open
        return action

    def __call__(self, step, obs_raw):
        """intervention_fn 계약: action(ndarray) 또는 None(정책이 실행)."""
        if self._recovery_remaining == 0 and self._pending_trigger:
            self._pending_trigger = False
            self._recovery_remaining = self.recovery_steps
            self.num_triggers += 1
        if self._recovery_remaining == 0:
            return None
        self._recovery_remaining -= 1
        return self._recovery_action()

    def _handle_key(self, key):
        """키 코드 -> 상태 갱신. cv2/터미널 의존 없이 테스트 가능하게 render()에서 분리."""
        if key == self.trigger_key:
            self.trigger()
        elif key == self.quit_key:
            self._quit_requested = True

    def _ensure_started(self):
        """MJPEG 서버 + 터미널 cbreak 모드를 첫 render() 호출 때만 지연 시작."""
        if self._started:
            return
        self._started = True

        # 127.0.0.1로만 바인드 - network_mode: host라 0.0.0.0이면 서버 실제 네트워크
        # 인터페이스에 인증 없이 노출된다. 루프백만 열면 서버 자신 또는 `ssh -L` 터널을
        # 통해서만 접근 가능(2026-09-12, 사용자 지적으로 발견).
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.http_port), _MJPEGHandler)
        self._httpd.latest_jpeg = None
        self._http_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._http_thread.start()
        port = self._httpd.server_port
        print(
            f"[ScriptedFailureIntervention] 로컬에서 `ssh -L {port}:localhost:{port} <server>` "
            f"포트포워딩 후 브라우저로 http://localhost:{port} 접속하면 화면이 보인다.",
            flush=True,
        )

        self._stdin_is_tty = sys.stdin.isatty()
        if self._stdin_is_tty:
            self._orig_termios = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())  # ISIG는 유지 -> Ctrl+C 정상 동작

    def _poll_key(self):
        """stdin에서 논블로킹으로 키 하나 읽기(cbreak라 Enter 불필요). 없으면 None."""
        if not self._stdin_is_tty:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        return ord(ch) if ch else None

    def render(self, obs_raw):
        """render_fn 계약: 프레임을 MJPEG로 내보내고 터미널 키 입력을 감지한다.
        False 반환 시 에피소드 종료.

        obs_raw[camera_key]는 postprocess_visual_obs=True(기본)를 거쳐 이미 CHW,float[0,1].
        """
        import cv2  # GUI 없는 순수 코덱 연산(imencode/putText)만 쓴다 - X11 불필요.

        self._ensure_started()

        frame = np.asarray(obs_raw[self.camera_key])
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[0] < frame.shape[-1]:
            frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        bgr = np.ascontiguousarray(frame[:, :, ::-1])
        bgr = cv2.resize(bgr, (bgr.shape[1] * 4, bgr.shape[0] * 4), interpolation=cv2.INTER_NEAREST)

        recovering = self._recovery_remaining > 0
        status = "RECOVERY" if recovering else "policy"
        color = (0, 0, 255) if recovering else (0, 200, 0)
        cv2.putText(bgr, status, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(
            bgr, f"[{chr(self.trigger_key)}]=trigger recovery  [{chr(self.quit_key)}]=quit",
            (6, bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )

        ok, buf = cv2.imencode(".jpg", bgr)
        if ok:
            self._httpd.latest_jpeg = buf.tobytes()

        key = self._poll_key()
        if key is not None:
            self._handle_key(key)
        return not self._quit_requested

    def close(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._stdin_is_tty and self._orig_termios is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._orig_termios)
