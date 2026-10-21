"""robomimic/robosuite env 생성 — robomimic import는 이 파일(과 datasets 어댑터)에만 존재.

docs/plan.md M0 확정 결과 참고: ObsUtils 초기화가 env 생성 전에 필요, MUJOCO_GL=egl 권장(오프스크린 렌더).
"""

import numpy as np
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_robosuite import EnvRobosuite

# [self-improving-gym 통합 시 축소] 원본 레포는 여기서 커스텀 robosuite env 등록
# 부수효과로 `from square_assembly.envs.custom import door_cabinet_env`를 import했다. DoorCabinet은
# robomimic 표준 task가 아닌 커스텀 task라 이 통합 범위(square)엔 필요 없어 가져오지 않았다.
# square/lift/can 등 robosuite 표준 task는 robosuite 자체가 이미 등록하므로 이 import 없이도
# suite.make(env_name="NutAssemblySquare", ...)가 그대로 동작한다.


def _ensure_obs_utils_initialized(obs_keys, rgb_keys=()):
    if ObsUtils.OBS_KEYS_TO_MODALITIES is not None:
        return
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(obs_keys), "rgb": list(rgb_keys)}})


def make_lowdim_env(env_name, robots, obs_keys, render=False, renderer="mjviewer", gripper_types=None, env_kwargs=None):
    """low_dim obs만 쓰는 robosuite env 생성.

    Args:
        env_name (str): 예: "Lift".
        robots (str | list[str]): 예: "Panda".
        obs_keys (list[str]): SequenceDataset과 동일한 low_dim obs 키 목록.
        render (bool): True면 화면(DISPLAY)에 실시간 뷰어 창을 띄운다.
            이 프로세스의 DISPLAY 환경변수가 가리키는 화면에 창이 뜨므로, 원격 접속 중이고
            X forwarding이 없으면 창이 안 보일 수 있다. MUJOCO_GL은 설정하지 않아야 한다
            (egl로 두면 오프스크린 강제라 사람이 보는 창이 안 뜬다).
        renderer (str): "mjviewer"(MuJoCo 네이티브 뷰어, 마우스 카메라 조작 가능) 또는
            "mujoco"(OpenCV 창으로 프레임만 표시). robomimic의 `EnvRobosuite`는 render=True일 때
            내부적으로 renderer를 "mujoco"(OpenCV)로 강제 덮어쓰므로, mjviewer를 쓰려면 일단
            render=False로 생성한 뒤 내부 raw robosuite env(`env.env`)의 렌더러 속성을 직접
            설정해 우회한다.
        gripper_types (str | None): 예: "RobotiqThreeFingerGripper". None이면 로봇 기본 그리퍼.
        env_kwargs (dict | None): 커스텀 env 클래스별 추가 kwargs 통과용(예: DoorCabinet의
            outside_color). 범용 팩토리에 task별 파라미터를 하나씩 하드코딩하지 않기 위함.
    """
    _ensure_obs_utils_initialized(obs_keys)
    kwargs = dict(env_kwargs) if env_kwargs else {}
    if gripper_types is not None:
        kwargs["gripper_types"] = gripper_types
    if not isinstance(robots, str):
        robots = list(robots)  # hydra ListConfig -> 네이티브 list(2지 이상 로봇, 예: Transport)
    env = EnvRobosuite(
        env_name=env_name,
        robots=robots,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        reward_shaping=False,
        **kwargs,
    )
    if render:
        env.env.has_renderer = True
        env.env.renderer = renderer
    return env


def rot6d_to_matrix(d6):
    """rot6d(회전행렬 앞 두 행, pytorch3d 규약 — DP RotationTransformer) -> 회전행렬."""
    b1 = d6[:3] / np.linalg.norm(d6[:3])
    b2 = d6[3:6] - (b1 @ d6[3:6]) * b1
    b2 = b2 / np.linalg.norm(b2)
    return np.stack([b1, b2, np.cross(b1, b2)])


class AbsRot6dActionEnv:
    """정책 액션(월드 절대 목표 pos3 + rot6d + gripper1, DP 규약)을 지금 손끝 자세 기준 OSC delta 액션으로 바꿔 넘긴다.

    robosuite OSC(delta, goal_update_mode="achieved", input_ref_frame="world")의 목표는 지금 자세 + scale*delta라서,
    |목표 - 지금|이 한 스텝 범위(5cm, 0.5rad) 안이면 absolute 컨트롤러와 같은 목표가 된다 — DP의 절대 액션 자체가
    delta 시연을 이 식으로 바꾼 것이다. 범위 밖이면 잘린다(input 클립과 같다). env를 delta로 두는 이유: 사람
    개입(마우스 텔레옵)·PH 시연·우리 정책이 모두 delta라 한 에피소드에 섞어 기록할 수 있다.
    7차원 액션(사람·0 액션)은 그대로 통과. 실제로 넣은 7차원 액션은 last_action(수집기가 기록한다).
    """

    def __init__(self, env):
        self._env = env
        self.last_action = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        from scipy.spatial.transform import Rotation

        action = np.asarray(action, dtype=np.float64)
        if action.shape[-1] == 10:
            ctrl = self._env.env.robots[0].part_controllers["right"]
            ctrl.update(force=True)
            scale = np.abs(ctrl.output_max - ctrl.output_min) / np.abs(ctrl.input_max - ctrl.input_min)
            drot = Rotation.from_matrix(rot6d_to_matrix(action[3:9]) @ ctrl.ref_ori_mat.T).as_rotvec()
            delta = np.concatenate([action[:3] - ctrl.ref_pos, drot]) / scale
            action = np.concatenate([np.clip(delta, -1.0, 1.0), action[9:]])
        self.last_action = action.astype(np.float32)
        return self._env.step(action)


def make_image_env(env_name, robots, lowdim_keys, rgb_keys, camera_names, image_size=84, gripper_types=None, env_kwargs=None):
    """이미지(픽셀) obs를 쓰는 robosuite env 생성 (오프스크린 렌더).

    Args:
        env_name (str): 예: "NutAssemblySquare".
        robots (str | list[str]): 예: "Panda".
        lowdim_keys (list[str]): proprio 등 low_dim obs 키(예: robot0_eef_pos).
        rgb_keys (list[str]): 이미지 obs 키(예: agentview_image). camera_names와 순서 대응.
        camera_names (list[str]): 렌더할 robosuite 카메라(예: agentview, robot0_eye_in_hand).
            각 카메라 <cam>의 이미지는 obs 키 "<cam>_image"로 나온다.
        image_size (int): 정사각 렌더 해상도(H=W). 학습 데이터셋과 맞춰야 함(기본 84).
        gripper_types (str | None): 예: "RobotiqThreeFingerGripper". None이면 로봇 기본 그리퍼.
        env_kwargs (dict | None): 커스텀 env 클래스별 추가 kwargs 통과용(예: DoorCabinet의
            outside_color).

    오프스크린 렌더이므로 MUJOCO_GL=egl 환경에서 실행한다.
    """
    _ensure_obs_utils_initialized(lowdim_keys, rgb_keys)
    kwargs = dict(env_kwargs) if env_kwargs else {}
    if gripper_types is not None:
        kwargs["gripper_types"] = gripper_types
    if not isinstance(robots, str):
        robots = list(robots)  # hydra ListConfig -> 네이티브 list(2지 이상 로봇, 예: Transport)
    env = EnvRobosuite(
        env_name=env_name,
        robots=robots,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        camera_names=list(camera_names),
        camera_heights=image_size,
        camera_widths=image_size,
        reward_shaping=False,
        **kwargs,
    )
    return env
