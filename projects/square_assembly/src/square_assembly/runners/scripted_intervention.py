"""사람이 실패를 "판단"하면 트리거하고, 실제 회복 동작은 고정 스크립트가 실행하는 개입.

KeyboardIntervention(runners/intervention_rollout.py — robosuite Keyboard device로 사람이
로봇을 직접 몬다)과 다르다. 여기선 사람이 조종하지 않는다 — 트리거 시점만 사람이 정하고,
그 뒤 recovery_steps 동안은 고정된 회복 액션(그리퍼 열기 + 위로 후퇴)이 실행된 뒤 정책에
제어를 돌려준다. 그래서 pynput/robosuite Keyboard device가 필요 없다 — 이미 화면에
띄우는 뷰어 창의 키 입력(cv2.waitKey)만으로 충분하다.

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
"""

import numpy as np

_Z_INDEX = 2
_GRIPPER_INDEX = -1


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
        window_name (str): cv2 창 이름.
    """

    def __init__(self, camera_key, action_dim=7, trigger_key="s", quit_key="q",
                 recovery_steps=20, retract_z=1.0, gripper_open=-1.0, window_name="rollout"):
        self.camera_key = camera_key
        self.action_dim = action_dim
        self.trigger_key = ord(trigger_key)
        self.quit_key = ord(quit_key)
        self.recovery_steps = recovery_steps
        self.retract_z = retract_z
        self.gripper_open = gripper_open
        self.window_name = window_name
        self._pending_trigger = False
        self._recovery_remaining = 0
        self._quit_requested = False
        self.num_triggers = 0  # 에피소드당 트리거 횟수(진단/로그용)

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
        """키 코드 -> 상태 갱신. cv2 의존 없이 테스트 가능하게 render()에서 분리."""
        if key == self.trigger_key:
            self.trigger()
        elif key == self.quit_key:
            self._quit_requested = True

    def open_window(self):
        """빈 창을 미리 띄워 cv2(Qt/xcb)의 X 연결을 먼저 확립한다.

        **MuJoCo EGL 환경(make_eval_env)을 만들기 전에 반드시 호출해야 한다.** 순서가 반대면
        cv2가 X에 처음 붙는 시점에 첫 확장 질의(xcb_shm_query_version)의 응답을 영영 못 받고
        메인 스레드가 멈춘다 — NVIDIA EGL/GL 라이브러리가 먼저 로드되면 Xlib 잠금이 꼬이는
        것으로 보인다(2026-09-12 서버에서 이분 탐색으로 확인: EGL env 생성 후 imshow = 무한
        정지, 창을 먼저 열면 정상. ssh -X/RustDesk 어느 화면이든 동일하게 재현).
        """
        import cv2

        cv2.imshow(self.window_name, np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.waitKey(1)

    def render(self, obs_raw):
        """render_fn 계약: 프레임을 화면에 띄우고 키 입력을 감지한다. False 반환 시 에피소드 종료.

        obs_raw[camera_key]는 postprocess_visual_obs=True(기본)를 거쳐 이미 CHW,float[0,1]
        - 화면 표시용으로만 HWC,uint8,BGR로 변환한다(cv2 관례, 저장에는 안 씀).
        """
        import cv2

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
        cv2.imshow(self.window_name, bgr)

        self._handle_key(cv2.waitKey(1) & 0xFF)
        return not self._quit_requested

    def close(self):
        import cv2

        cv2.destroyWindow(self.window_name)
