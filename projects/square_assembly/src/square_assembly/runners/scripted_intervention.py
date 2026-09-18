"""사람이 실패를 "판단"만 하면, 실제 수습은 룰 기반 오라클이 끝까지 해내는 개입.

KeyboardIntervention(runners/intervention_rollout.py — robosuite Keyboard device로 사람이
로봇을 직접 몬다)과 다르다. 여기선 사람이 조종하지 않는다 — 트리거 시점만 사람이 정하고,
그 뒤로는 SquareAssemblyOracle(runners/square_oracle.py)이 sim의 특권 정보를 읽어 너트를
peg에 꽂을 때까지 제어한다. 그래서 pynput/robosuite Keyboard device가 필요 없다 — 이미
화면에 띄우는 뷰어 창의 키 입력(cv2.waitKey)만으로 충분하다.

한 번 트리거되면 에피소드가 끝날 때까지(성공/포기/max_steps) 오라클이 잡고 있는다.
"실패했다 싶으면 그 에피소드는 스크립트가 성공시켜서 성공 데모로 저장한다"가 목적이라
정책에 제어를 돌려주지 않는다. 포기하려면 종료 키.

collect_episode(runners/intervention_rollout.py)의 세 콜백 계약에 이 클래스 하나로 꽂는다:
  render_fn=lambda ...: interv.render(frame)  (프레임 표시 + 트리거/종료 키 감지)
  intervention_fn=interv                      (트리거 이후 오라클 액션 반환)
  should_end_fn=interv.should_end             (종료 키)

트리거는 render()(env.step 이후, obs_raw_{t+1} 시점)에서 걸리고, 실제 오라클 액션은 다음
스텝의 intervention_fn(step+1, ...)부터 나간다 — collect_episode 루프 순서(intervention_fn
-> env.step -> render_fn) 그대로다.
"""

import numpy as np


def open_window(window_name="rollout"):
    """빈 창을 미리 띄워 cv2(Qt/xcb)의 X 연결을 먼저 확립한다.

    **robosuite/robomimic을 끌어오는 임포트(즉 이 프로젝트의 factory/task_utils/
    intervention_rollout)보다 먼저 호출해야 한다.** 순서가 반대면 첫 `cv2.imshow`가 Qt/xcb의
    확장 질의(`xcb_shm_query_version`)에서 응답을 영영 못 받고 메인 스레드가 멈춘다 —
    NVIDIA EGL/GL 라이브러리가 먼저 로드되면 Xlib 잠금이 꼬이는 것으로 보인다.

    2026-09-12 서버에서 이분 탐색으로 확인(ssh -X/RustDesk 어느 화면이든 동일하게 재현):
    최소 프로세스·torch CUDA·`import robosuite`·`import robomimic`까지는 전부 정상,
    이 프로젝트 모듈 임포트 후 imshow에서 정지, 창을 먼저 열면 그 뒤 전부 정상.
    """
    import cv2

    cv2.imshow(window_name, np.zeros((10, 10, 3), dtype=np.uint8))
    cv2.waitKey(1)


class ScriptedFailureIntervention:
    """트리거 키 -> 오라클이 에피소드를 성공까지 끌고 간다.

    Args:
        oracle: `(step, obs_raw) -> action` 콜러블(SquareAssemblyOracle). `reset()`이 있으면
            에피소드마다 같이 리셋한다.
        trigger_key (str): 오라클에 넘기는 키(기본 "s").
        quit_key (str): 에피소드 포기 키(기본 "q").
        window_name (str): cv2 창 이름.
    """

    def __init__(self, oracle, trigger_key="s", quit_key="q", window_name="rollout"):
        self.oracle = oracle
        self.trigger_key = ord(trigger_key)
        self.quit_key = ord(quit_key)
        self.window_name = window_name
        self._active = False
        self._quit_requested = False
        self.num_triggers = 0  # 진단/로그용

    def reset(self):
        """에피소드 시작마다 호출한다(collect_episode는 이 객체를 자동 리셋하지 않는다)."""
        self._active = False
        self._quit_requested = False
        self.num_triggers = 0
        if hasattr(self.oracle, "reset"):
            self.oracle.reset()

    def trigger(self):
        if not self._active:
            self._active = True
            self.num_triggers += 1

    def should_end(self):
        return self._quit_requested

    def __call__(self, step, obs_raw):
        """intervention_fn 계약: action(ndarray) 또는 None(정책이 실행)."""
        if not self._active:
            return None
        return self.oracle(step, obs_raw)

    def _handle_key(self, key):
        """키 코드 -> 상태 갱신. cv2 의존 없이 테스트 가능하게 render()에서 분리."""
        if key == self.trigger_key:
            self.trigger()
        elif key == self.quit_key:
            self._quit_requested = True

    def render(self, frame):
        """render_fn 계약: 프레임을 화면에 띄우고 키 입력을 감지한다. False 반환 시 에피소드 종료.

        frame은 HWC uint8 RGB(정책 입력과 무관한 표시 전용 고해상도 렌더 — 호출부가
        env.render(mode="rgb_array", height=..., width=...)로 만들어 넘긴다).
        """
        import cv2

        bgr = np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])

        status = f"ORACLE:{getattr(self.oracle, 'phase', '?')}" if self._active else "policy"
        color = (0, 0, 255) if self._active else (0, 200, 0)
        cv2.putText(bgr, status, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(
            bgr, f"[{chr(self.trigger_key)}]=let the script finish it  [{chr(self.quit_key)}]=give up",
            (8, bgr.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        cv2.imshow(self.window_name, bgr)

        self._handle_key(cv2.waitKey(1) & 0xFF)
        return not self._quit_requested

    def close(self):
        import cv2

        cv2.destroyWindow(self.window_name)
