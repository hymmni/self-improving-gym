r"""정책이 실패로 가는 걸 사람이 보고 판단해서 트리거하면, 그 뒤는 룰 기반 오라클이
성공까지 끌고 가는 반개입(半介入) 수집. collect_square_rollouts.py(완전 무개입, 헤드리스
배치)의 "성공만 필터해서 추가 학습"용 성공 데모를 더 효율적으로 늘리기 위한 변형이다.

사람은 조종하지 않는다 — 화면을 보고 있다가 실패로 보이면 트리거 키를 누르는 것뿐이고,
그 뒤는 runners/square_oracle.SquareAssemblyOracle이 sim의 특권 정보(너트/핸들/peg 위치)로
너트를 peg에 꽂을 때까지 제어한다. 그래서 KeyboardIntervention(robosuite Keyboard device +
pynput)이 필요 없다. 오라클 구간은 INTV로 라벨되고, 에피소드는 성공으로 끝난다.

## 실행 전제
- 이미지 task라 매 스텝 obs에 카메라 프레임이 이미 들어있다(offscreen render) — 별도
  onscreen 렌더러를 켜지 않고 그 프레임을 cv2 창에 그대로 띄운다. `MUJOCO_GL=egl` +
  `DISPLAY`(X11 화면, 예: 도커 컴포즈가 전달하는 값)가 필요하다(README의 image task
  render=true 패턴과 동일 — mjviewer 온스크린과 동시에 켜면 GL 충돌로 세그폴트한다는
  실측 지뢰가 있으니 mjviewer는 절대 같이 켜지 않는다).
- robosuite/numba가 컨테이너 site-packages 경로에 캐시를 못 써서 import 시점에
  RuntimeError를 낼 수 있다(2026-09-09 서버에서 실측) — `NUMBA_CACHE_DIR=/tmp/numba_cache`
  같은 쓰기 가능한 경로를 지정해 우회한다.

사용(컨테이너 WORKDIR=/workspace, 즉 레포 루트에서 실행 기준):
    MUJOCO_GL=egl NUMBA_CACHE_DIR=/tmp/numba_cache \
    python -m square_assembly.scripts.collect_square_scripted_intervention \
        --base-ckpt projects/square_assembly/checkpoints/square_base_policy/policy_epoch1060.pt \
        --episodes 20 --max-steps 700 \
        --out data/square_scripted_intv_v1.hdf5

창이 뜨면 정책이 자동으로 진행한다. 실패로 보이면 's'를 눌러 오라클에 넘긴다(그 에피소드는
끝까지 오라클이 잡는다 — 정책에 돌려주지 않는다). 포기하려면 'q'. --mode mouse면 Tab으로 사람/정책을
오가고, ←(직전 모드 전환점)·'b'(1초)·'r'(같은 배치로 처음부터)로 되감을 수 있다 — 버린 배치는
다시 못 만나므로 q보다 이쪽을 쓴다. ←를 두 번 연타하고 Enter면 바로 전에 저장한 에피소드를 지우고 그
시작 상태에서 다시 모은다(이번 실행에서 저장한 것만, Esc는 취소).

이미 저장된 에피소드를 다시 모으려면 `--mode mouse --redo demo_113 demo_161 ...`: 에피소드마다 마지막
개입이 시작된 sim 상태를 되돌려 사람 제어로, 멈춘 채 시작한다([s]로 재개). 저장된 앞부분도 이어받으므로
'b'/←로 개입 이전까지 되감아 더 일찍 잡을 수 있고, 'r'이면 같은 배치로 처음부터(정책부터) 다시 한다.
[q]는 그 에피소드를 건너뛴다. 성공으로 끝났을 때만 기존 에피소드를 새 것으로 바꾼다.

정책 없이 사람 시연만 모으려면 `--mode mouse --human-only`: 체크포인트를 안 읽고(--base-ckpt 무시, task는
configs/task/square.yaml) 에피소드마다 사람 제어로 멈춘 채 시작한다([s]로 시작, Tab은 안 먹는다). 모든 스텝이
INTV로 라벨되고 attrs["sampler"]는 "human"이다.

창이 작으면 `--zoom 1.5`(창 모드 배율)나 `--fullscreen`(실행 중 F11로 오간다)으로 키운다 — 다 그린 화면을 늘려 띄운다.

화면은 --display-size 해상도로 따로 렌더해서 보여준다(학습/저장 데이터는 task의
image_size 그대로 84픽셀 — 사람이 보기엔 84픽셀이 너무 작아서 분리). 480을 넘길 순 없다
(_MAX_DISPLAY 참고).

오라클은 한 번에 실패해도 처음부터 반복하므로 스텝만 충분하면 결국 성공한다(고정 시드
20에피소드 실측, 2026-09-17: 트리거 후 성공까지 중앙값 224스텝·최대 539스텝) —
--max-steps는 넉넉히(700 이상) 주는 게 좋다.
"""

import argparse
import time
import os

import h5py
import numpy as np
import torch

# scripted_intervention은 모듈 단계에서 numpy만 쓴다(robosuite/EGL을 안 건드림) — 그래서
# 여기서 임포트해도 안전하다. robosuite/robomimic을 끌어오는 나머지 임포트는 run() 안에서
# open_window() 뒤에 한다(이유는 open_window docstring, DOCKER.md §4).
from square_assembly.datasets.labels import LABEL_INTV, LABEL_PREINTV, LABEL_ROLLOUT
from square_assembly.runners.scripted_intervention import ScriptedFailureIntervention, open_window
from square_assembly.runners.mouse_teleop import MouseTeleopController, MouseTeleopIntervention
from square_assembly.runners.square_oracle import SquareAssemblyOracle


# robosuite 오프스크린 버퍼는 640x480(MJCF 기본)으로 잡히고, 그보다 큰 렌더를 요청하면
# binding_utils.update_offscreen_size가 MjrContext를 다시 만든다. 그런데 cv2(Qt) 창이 이미
# 떠 있는 프로세스에선 그 재생성이 EGL에서 실패한다 — `mujoco.FatalError: Default framebuffer
# is not complete, error 0x0` (2026-09-17 서버 실측: compose run 컨테이너에서 512는 죽고 480은
# 정상, 같은 코드가 창 없이 헤드리스면 512도 정상).
# ponytail: 480 상한. 더 크게 보려면 render context를 직접 큰 max_width/max_height로 만들어
# sim.add_render_context로 꽂아야 하는데, 모니터링 화면에 그만한 값어치는 없다.
_MAX_DISPLAY = 480


def frame_roughness(im):
    """이웃 픽셀 차이의 평균 — 정상 장면은 84픽셀에서 4~10, 노이즈는 30~130, 검은 화면은 ~0."""
    return float(np.abs(np.diff(np.asarray(im).astype(int), axis=1)).mean())


def renderer_is_noisy(env, camera_name, display_size, n_steps=5):
    """reset 뒤 n_steps 스텝을 진행한 프레임까지 보고 렌더 고장을 판정한다.

    2026-09-18 pororo 실측으로 고장의 정확한 서명을 잡았다: 고장난 컨테이너/구간에서는
    **reset 직후 첫 프레임은 정상이고 step 이후 프레임부터 전부 노이즈**(또는 검은 화면)다.
    그래서 첫 프레임만 보던 예전 점검은 고장을 그대로 통과시켰다(v2 수집 첫 에피소드 700스텝이
    노이즈로 저장된 사고). 같은 이미지·GPU·드라이버·코드인데 컨테이너를 만든 시각에 따라
    정상/고장이 갈리고(16:30~16:55에 만든 것 전부 고장, 그 전후는 정상), 한 프로세스 안에서
    시간이 지나면 회복되기도 한다. GPU 부하·DISPLAY·TTY·마운트 디렉터리는 실험으로 배제했다
    (DOCKER.md §4). 원인은 미상 — 그래서 판정과 재시도로 막는다.
    """
    obs = env.reset()
    for _ in range(n_steps):
        obs, _, _, _ = env.step(np.zeros(env.action_dimension if hasattr(env, "action_dimension") else 7))
    small = frame_roughness(np.transpose(obs[camera_name], (1, 2, 0)) * 255.0)
    big = frame_roughness(env.render(mode="rgb_array", height=display_size, width=display_size,
                                     camera_name=camera_name[: -len("_image")]))
    print(f"렌더 점검(step {n_steps} 이후): 이웃 픽셀 차이 84={small:.1f} {display_size}={big:.2f}", flush=True)
    return not (2.0 < small < 15.0) or big < 0.05


def wait_for_renderer(make_env, camera_name, display_size, timeout_s=120.0, retry_s=5.0):
    """렌더가 정상인 env를 돌려준다 — 고장이면 env를 닫고 다시 만들며 timeout_s까지 기다린다."""
    t0 = time.time()
    while True:
        env = make_env()
        if not renderer_is_noisy(env, camera_name, display_size):
            return env
        if hasattr(env, "close"):
            env.close()
        if time.time() - t0 > timeout_s:
            raise RuntimeError(
                f"{timeout_s:.0f}초 동안 렌더가 노이즈/검은 화면만 냈다 — 이대로 수집하면 obs까지 노이즈라 "
                "데이터가 쓸모없다. 컨테이너를 새로 만들어 다시 시도한다(DOCKER.md §4 '렌더가 노이즈로 나올 때')."
            )
        print(f"  렌더 고장 — {retry_s:.0f}초 뒤 env를 다시 만들어 재점검 ({time.time() - t0:.0f}s 경과)", flush=True)
        time.sleep(retry_s)


def wait_for_render_recovery(env, camera_name, display_size, sleep_fn=time.sleep, timeout_s=600.0, retry_s=5.0):
    """에피소드 도중 고장난 렌더가 (같은 env에서) 회복될 때까지 기다린다. 점검이 env를 reset하니
    호출부가 그 뒤 상태를 되돌리거나 새 에피소드를 시작해야 한다."""
    t0 = time.time()
    while renderer_is_noisy(env, camera_name, display_size):
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"{timeout_s / 60:.0f}분 동안 렌더가 복구되지 않았다 — 컨테이너를 새로 만들어 다시 시도한다(DOCKER.md §4).")
        sleep_fn(retry_s)
    print("  렌더 복구됨", flush=True)


def make_noise_tracker(env, interv, camera, display_cameras, display_size, obs_ep, wait_for_recovery):
    """collect_episode의 render_fn — 저장될 obs의 렌더 노이즈를 감시하고 화면을 그린다.

    연속 3프레임 노이즈면, 되감기를 지원하는 개입 장치(마우스)는 에피소드를 살린다: 렌더가 회복될
    때까지 기다린 뒤 노이즈 직전의 정상 스텝으로 되감고 일시정지한다. 그사이 저장된 노이즈 프레임
    (1~2개)은 되감기가 잘라낸다. 돌아갈 정상 스텝이 없거나(첫 프레임부터 노이즈) 되감기가 없는
    장치(오라클)면 False로 에피소드를 끊는다 — 호출부가 버리고 다시 모은다(2026-09-30 이전엔 늘 이쪽).

    Returns: (track, noise) — 끊겼으면 noise["streak"] >= 3.
    """
    noise = {"streak": 0, "last_clean": None}

    def track(obs_raw):
        idx = len(obs_ep)  # 이 obs가 다음 pre_step에서 저장될 자리
        r = frame_roughness(np.transpose(obs_raw[camera], (1, 2, 0)) * 255.0)
        if 2.0 < r < 15.0:
            noise["streak"], noise["last_clean"] = 0, idx
        else:
            noise["streak"] += 1
        if noise["streak"] >= 3:
            if not hasattr(interv, "rewind_to") or noise["last_clean"] is None:
                return False
            print(f"  !! 렌더 노이즈(step {idx}) — 회복을 기다렸다가 step {noise['last_clean']}로 되감는다", flush=True)
            wait_for_recovery()
            interv.rewind_to(noise["last_clean"])
            noise["streak"] = 0
            return True
        # 저장/학습은 obs의 84픽셀 그대로, 화면만 따로 고해상도로 렌더한다.
        frames = {c: env.render(mode="rgb_array", height=display_size, width=display_size, camera_name=c)
                  for c in display_cameras}
        return interv.render(frames if len(frames) > 1 else frames[display_cameras[0]])

    return track, noise


def drop_episode(data_grp, name):
    """저장한 에피소드를 지운다(다시 모으기). Returns: (시작 sim 상태, 길이, 성공 여부)."""
    g = data_grp[name]
    s0, T, ok = np.asarray(g["states"][0]), int(g.attrs["num_samples"]), bool(g.attrs["is_success"])
    del data_grp[name]
    return s0, T, ok


def last_intervention_onset(modes):
    """마지막 사람 개입 구간이 시작한 스텝(없으면 None) — --redo가 되돌아갈 자리."""
    human = np.flatnonzero(np.asarray(modes) == LABEL_INTV)
    if not human.size:
        return None
    gaps = np.flatnonzero(np.diff(human) > 1)
    return int(human[gaps[-1] + 1] if gaps.size else human[0])


def _gripper(env):
    """robosuite 그리퍼 객체(없으면 None — 테스트용 가짜 env). hard_reset이 reset마다 새로 만드니 매번 다시 찾는다."""
    robots = getattr(getattr(env, "env", env), "robots", None)
    return robots[0].gripper[robots[0].arms[0]] if robots else None


def reset_to_state(env, state):
    """새 에피소드로 reset한 뒤 저장된 sim 상태로 되돌린다. Returns: 그 상태의 obs.

    env.reset()은 팔 컨트롤러에 홈 자세를 캐시해 두고, 컨트롤러는 한 스텝을 돌기 전엔 그 캐시를 다시 읽지 않는다
    (robosuite Controller.update의 new_update). 그대로 두면 되돌린 뒤 첫 스텝의 delta 목표가 홈 자세 기준으로
    잡혀 팔이 그쪽으로 튄다(2026-10-01 실측: 기준 위치가 33cm 어긋나 1스텝 뒤 관절 속도 차 0.67rad/s →
    강제로 다시 읽으면 0.008). 에피소드 도중의 되감기는 reset을 안 거쳐서 해당 없다.
    """
    env.reset()
    obs = env.reset_to({"states": state})
    for robot in getattr(getattr(env, "env", env), "robots", None) or []:
        robot.composite_controller.get_controller(robot.arms[0]).update(force=True)
    return obs


def gripper_ramp(grip_cmds, speed):
    """그리퍼 명령(액션 마지막 차원)들을 넣었을 때 그리퍼의 current_action이 지나온 값. ramp[t] = 액션 t 직전 값.

    From: robosuite/models/grippers/panda_gripper.py (PandaGripper.format_action, reset 시 0)
    그리퍼는 열기/닫기 명령을 내부에서 speed씩 적분해 손가락 목표로 쓴다. 이 값은 sim 상태에 없어서,
    상태만 되돌리면 손가락 목표가 엉뚱한 데서 다시 출발해 몇 스텝 동안 쥔 것이 풀리거나 열린 손이 조여진다.
    """
    c, ramp = np.zeros(2), []
    for a in grip_cmds:
        ramp.append(c.copy())
        c = np.clip(c + np.array([-1.0, 1.0]) * speed * np.sign(a), -1.0, 1.0)
    return ramp + [c]


def write_episode(data_grp, name, actions, modes, states, obs, obs_keys, rgb_keys, camera, sampler, model_xml):
    """에피소드 하나를 data_grp[name]에 쓴다. Returns: (그룹, 샘플 프레임들의 거칠기) — 렌더 노이즈 판정용."""
    g = data_grp.create_group(name)
    g.attrs["num_samples"] = len(actions)
    g.attrs["sampler"] = sampler
    g.create_dataset("actions", data=actions)
    g.create_dataset("action_mode", data=modes)
    g.create_dataset("states", data=np.stack(states))
    g.attrs["model_file"] = model_xml
    obs_grp = g.create_group("obs")
    for k in obs_keys:
        obs_grp.create_dataset(k, data=np.stack([o[k] for o in obs], axis=0), compression="gzip" if k in rgb_keys else None)
    frames, T = obs_grp[camera], len(actions)
    return g, [frame_roughness(frames[i]) for i in np.linspace(0, T - 1, min(16, T)).astype(int)]


def make_recorder(env, interv, obs_keys, rgb_keys, stored=None):
    """collect_episode의 pre_step_fn — 매 스텝 액션 직전 obs·sim 상태를 쌓고, 되감기 요청을 처리한다.

    stored=(obs 리스트, states 리스트, 그리퍼 ramp 리스트)면 저장된 에피소드의 앞부분으로 버퍼를 채워
    시작한다(--redo) — 되감기가 그 앞부분으로도 들어갈 수 있다.

    그리퍼의 명령 적분값(gripper_ramp 참고)도 스텝마다 기억했다가 되감을 때 함께 되돌린다.

    obs[t]·states[t]는 actions[t]를 넣기 직전 시점이다(render_fn은 스텝 뒤라 거기서 쌓으면 한 칸
    밀린다 — 2026-09-22 이전 수집분이 그렇다). 상태는 되감기에 쓰고, 저장해 두면 나중에 아무
    해상도로나 다시 렌더할 수 있다(프레임당 45 float).

    sim은 **매번** env.env.sim에서 새로 읽는다 — robosuite hard_reset은 reset마다 MjSim을 새로
    만들어서, 에피소드 시작 전에 잡아둔 sim은 멈춘 옛 객체다. 그걸 읽으면 모든 상태가 이전
    에피소드 끝 값으로 저장되고 되감기가 거기로 순간이동한다(2026-09-22 r0v2 수집에서 실제로 났다).

    Returns: (pre_step, obs_ep, states_ep) — 두 리스트는 pre_step이 채우고 자른다.
    """
    obs_ep, states_ep, ramp_ep = (list(stored[0]), list(stored[1]), list(stored[2])) if stored else ([], [], [])

    def pre_step(step, obs_raw):
        t = interv.pop_rewind(step) if hasattr(interv, "pop_rewind") else None
        grip = _gripper(env)
        if t is not None and t < len(states_ep):
            state = states_ep[t]
            if grip is not None:
                grip.current_action = ramp_ep[t]
            del obs_ep[t:], states_ep[t:], ramp_ep[t:]
            print(f"  << 되감기 step {step} -> {t}", flush=True)
            env.reset_to({"states": state})
            return t
        obs_ep.append({k: _to_storage(k, obs_raw[k], rgb_keys) for k in obs_keys})
        states_ep.append(np.asarray(env.env.sim.get_state().flatten()))
        if grip is not None:
            ramp_ep.append(np.array(grip.current_action, dtype=np.float64))
        return None

    return pre_step, obs_ep, states_ep


def _from_storage(key, val, rgb_keys):
    """_to_storage의 역 — 저장된 HWC uint8을 env가 주는 CHW float[0,1]로(값은 왕복해도 그대로다)."""
    return np.transpose(val, (2, 0, 1)).astype(np.float32) / 255.0 if key in rgb_keys else np.asarray(val)


def _to_storage(key, val, rgb_keys):
    """collect_square_rollouts._to_storage와 동일 — CHW,float[0,1] -> HWC,uint8."""
    val = np.asarray(val)
    if key in rgb_keys:
        if val.ndim == 3 and val.shape[0] in (1, 3, 4) and val.shape[0] < val.shape[-1]:
            val = np.transpose(val, (1, 2, 0))
        val = np.clip(val * 255.0, 0, 255).astype(np.uint8)
    return val


def sampler_label(policy):
    """정책 추론 샘플러 이름(에피소드 attrs["sampler"]) — 예: "DDPM100", "DDIM10 eta=1"."""
    name = type(policy.inference_scheduler).__name__.removesuffix("Scheduler")
    eta = getattr(policy, "inference_step_kwargs", {}).get("eta")
    return f"{name}{policy.num_inference_steps}" + ("" if eta is None else f" eta={eta:g}")


def run(base_ckpt, episodes, max_steps, out, camera, trigger_key, quit_key,
        control_fps, display_size, window_name="rollout", mode="oracle", teleop=None, overwrite=False,
        ddim_steps=10, ddim_eta=1.0, redo=None, human_only=False, zoom=1.0, fullscreen=False):
    # 반드시 아래 임포트들(robosuite/robomimic → EGL 초기화)보다 먼저 — 순서가 바뀌면 첫
    # cv2.imshow가 영영 멈춘다. mouse 모드는 cv2 창이 아니라 Tk 창이라 필요 없다.
    if mode != "mouse":
        open_window(window_name)

    if redo and (mode != "mouse" or overwrite):
        raise ValueError("--redo는 --mode mouse에서만, --overwrite 없이 쓴다")
    if human_only and mode != "mouse":
        raise ValueError("--human-only는 --mode mouse에서만 쓴다")
    if display_size > _MAX_DISPLAY:
        raise ValueError(
            f"--display-size는 {_MAX_DISPLAY} 이하여야 한다(요청 {display_size}) — "
            "그보다 크면 첫 프레임에서 MjrContext 재생성이 EGL에서 실패한다(_MAX_DISPLAY 주석 참고)."
        )

    from square_assembly.datasets.normalization import MinMaxNormalizer, load_stats
    from square_assembly.factory import registry
    from square_assembly.runners.intervention_rollout import _predict_chunk, collect_episode
    from square_assembly.runners.rollout import maybe_speed_up_inference
    from square_assembly.utils.checkpoints import load_epoch_checkpoint, load_run_config
    from square_assembly.utils.task_utils import is_image_task, make_eval_env, task_obs_keys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if human_only:   # 정책이 한 스텝도 안 움직인다 — 체크포인트 없이 task 설정만 읽는다
        from omegaconf import OmegaConf

        task_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "configs", "task", "square.yaml"))
        policy = normalizer = None
        obs_horizon = action_horizon = 1
        sampler = "human"
    else:
        saved = load_run_config(base_ckpt)
        task_cfg, policy_cfg, policy_name = saved.task, saved.policy, saved.policy_name

        policy = registry.create_policy(policy_name, task_cfg, policy_cfg).to(device)
        load_epoch_checkpoint(base_ckpt, policy, device)
        policy.eval()
        maybe_speed_up_inference(policy, ddim_steps or None, eta=ddim_eta)
        sampler = sampler_label(policy)
        obs_horizon, action_horizon = task_cfg.get("obs_horizon", policy_cfg.obs_horizon), policy_cfg.action_horizon

        stats_path = os.path.join(os.path.dirname(base_ckpt), "normalization_stats.json")
        normalizer = MinMaxNormalizer(load_stats(stats_path))
    obs_keys = task_obs_keys(task_cfg)
    rgb_keys = list(task_cfg.rgb_keys) if is_image_task(task_cfg) else []
    if camera not in rgb_keys:
        raise ValueError(f"--camera {camera}는 task.rgb_keys {rgb_keys}에 없다")

    env = wait_for_renderer(lambda: make_eval_env(task_cfg), camera, display_size)
    # 마우스 모드는 모든 카메라(--camera가 맨 위)를 맵 왼쪽에 쌓아 보여주고, 맵은 그 높이 합만큼 키운다.
    shown = [camera] + [k for k in rgb_keys if k != camera] if mode == "mouse" else [camera]
    display_cameras = [k[: -len("_image")] for k in shown]

    if mode == "mouse":
        interv = MouseTeleopIntervention(
            env, controller=MouseTeleopController(**(teleop or {})), map_size=display_size * len(shown),
            window_name=window_name, quit_key=quit_key, human_only=human_only, zoom=zoom, fullscreen=fullscreen,
        )
    else:
        interv = ScriptedFailureIntervention(
            SquareAssemblyOracle(env), trigger_key=trigger_key, quit_key=quit_key, window_name=window_name,
        )

    def predict_fn(history):
        return _predict_chunk(policy, normalizer, history, obs_keys, device, rgb_keys=rgb_keys)

    def recover_render():
        wait_for_render_recovery(env, camera, display_size, sleep_fn=getattr(interv, "idle", time.sleep))

    def collect_once(start_state=None, stored=None, resume=None, grip_at_start=None):
        """에피소드 하나를 돌린다. start_state가 있으면 그 sim 상태에서 시작한다. stored·resume(--redo)이면
        저장된 앞부분을 이어받고, 첫 스텝 전에 되돌린 장면을 먼저 띄운다(멈춘 채 시작했으면 거기서 기다린다)."""
        pre_step, obs_ep, states_ep = make_recorder(env, interv, obs_keys, rgb_keys, stored)
        track, noise = make_noise_tracker(env, interv, camera, display_cameras, display_size, obs_ep, recover_render)

        def reset_fn():
            obs = env.reset() if start_state is None else reset_to_state(env, start_state)
            if resume is not None:
                _gripper(env).current_action = grip_at_start
            if resume is not None or human_only:   # 멈춘 채 시작한다 — 첫 스텝 전에 장면부터 띄운다
                track(obs)
            return obs

        result = collect_episode(
            env, policy, normalizer, obs_keys, obs_horizon, action_horizon, device,
            intervention_fn=interv, max_steps=max_steps, render=False, render_fn=track,
            pre_step_fn=pre_step, should_end_fn=interv.should_end, control_fps=control_fps,
            predict_fn=predict_fn, print_diagnostics=False,
            reset_fn=reset_fn, resume=resume,
        )
        return result, obs_ep, states_ep, noise

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    outcomes = {"success": 0, "fail": 0}

    try:
        # 파일이 이미 있으면 이어서 모은다("a") — 같은 --out으로 다시 실행해도 기존 에피소드가 안 날아간다.
        # --overwrite면 기존 파일을 비우고 처음부터("w").
        if overwrite and os.path.exists(out):
            print(f"--overwrite: 기존 {out}를 비우고 처음부터 모은다", flush=True)
        with h5py.File(out, "w" if overwrite else "a") as f:
            data_grp = f.require_group("data")
            outcomes["success"] = sum(k.startswith("demo_") for k in data_grp.keys())
            outcomes["fail"] = sum(k.startswith("fail_") for k in data_grp.keys())
            total = int(data_grp.attrs.get("total", 0))
            ep = outcomes["success"] + outcomes["fail"]
            if ep:
                print(f"이어서 수집: 기존 성공 {outcomes['success']} / 실패 {outcomes['fail']} -> 목표 성공 {episodes}", flush=True)
                old = {str(g.attrs.get("sampler", "기록 없음(DDPM100 추정)")) for g in data_grp.values()}
                if old != {sampler}:
                    print(f"  !! 기존 에피소드 샘플러 {sorted(old)} ≠ 이번 {sampler} — 섞지 않으려면 --out을 새로", flush=True)
            # episodes = 저장할 *성공* 에피소드 수. 실패도 저장은 되지만(is_success=False, 병합에서 제외)
            # 개수에는 안 센다 — 라운드마다 "성공 N개"를 맞추려는 것이지 시도 횟수가 아니다.
            # ←← + Enter로 다시 모으기: 이번 실행에서 바로 전에 저장한 에피소드와, 지운 뒤 다시 시작할 상태
            # --redo: 저장된 에피소드를 마지막 개입 시작 상태에서 이어받아 다시 모으고, 성공했을 때만 바꾼다.
            replaced = []
            for name in redo or []:
                if "redo_tmp" in data_grp:      # 지난 실행이 중간에 죽으며 남긴 것
                    del data_grp["redo_tmp"]
                old = data_grp[name]
                old_modes = np.asarray(old["action_mode"])
                t0 = last_intervention_onset(old_modes)
                if t0 is None:
                    print(f"{name}: 사람 개입이 없는 에피소드 — 건너뜀", flush=True)
                    continue
                T_old = int(old.attrs["num_samples"])
                pre = {k: np.asarray(old["obs"][k][:t0]) for k in obs_keys}
                obs_st = [{k: pre[k][i] for k in obs_keys} for i in range(t0)]
                states = np.asarray(old["states"][: t0 + 1])
                ramp = gripper_ramp(np.asarray(old["actions"][:t0, -1]), _gripper(env).speed)
                resume = {"obs": [{k: _from_storage(k, o[k], rgb_keys) for k in obs_keys} for o in obs_st],
                          "actions": np.asarray(old["actions"][:t0]),
                          # PREINTV는 개입 직전의 정책 스텝에 붙인 표시라, 되돌려 뒀다가 끝에서 다시 매긴다
                          "modes": np.where(old_modes[:t0] == LABEL_PREINTV, LABEL_ROLLOUT, old_modes[:t0])}
                print(f"{name}: step {t0}/{T_old}(마지막 개입 시작)에서 다시 — 멈춘 채 시작, [s] 재개 / [b]·[←] 더 앞으로 / "
                      f"[r] 처음부터 / [q] 건너뜀", flush=True)
                while True:
                    interv.reset()
                    interv.start_as_human(earlier=old_modes[:t0] == LABEL_INTV)
                    result, obs_ep, states_ep, noise = collect_once(states[t0], (obs_st, states[:t0], ramp[:t0]),
                                                                    resume, ramp[t0])
                    if noise["streak"] < 3:
                        break
                    print("  !! 렌더 노이즈 — 복구를 기다렸다가 같은 상태에서 다시", flush=True)
                    recover_render()
                if not result["success"]:
                    print(f"  {name}: 성공으로 끝나지 않음 — 기존 에피소드를 그대로 둔다", flush=True)
                    continue
                actions = np.asarray(result["actions"], dtype=np.float64)
                assert len(actions) == len(obs_ep) == len(states_ep), "obs/state/action 스텝 수 불일치"
                new, sampled = write_episode(data_grp, "redo_tmp", actions, result["action_modes"], states_ep, obs_ep,
                                             obs_keys, rgb_keys, camera, sampler, env.env.sim.model.get_xml())
                if not all(2.0 < r < 15.0 for r in sampled):
                    del data_grp["redo_tmp"]
                    print(f"  !! {name}: 렌더 노이즈가 섞였다 — 기존 에피소드를 그대로 둔다", flush=True)
                    continue
                new.attrs["render_ok"] = new.attrs["is_success"] = True
                new.attrs["redo_from_step"] = t0
                del data_grp[name]
                data_grp.move("redo_tmp", name)
                total += len(actions) - T_old
                data_grp.attrs["total"] = total
                f.flush()
                replaced.append(name)
                print(f"  {name}: 저장({T_old} -> {len(actions)}스텝)", flush=True)
            if redo:
                print(f"다시 수집: {len(replaced)}/{len(redo)}개 교체 {replaced}", flush=True)
                episodes = 0   # 새 에피소드는 모으지 않는다

            last_saved, redo_state = None, None
            while outcomes["success"] < episodes:
                interv.reset()
                interv.redo_label = last_saved[1] if last_saved else None
                result, obs_ep, states_ep, noise = collect_once(redo_state)
                if noise["streak"] >= 3:
                    print(f"  !! 렌더 노이즈로 에피소드 중단(step {len(obs_ep)}) — 버리고 렌더 복구를 기다린다", flush=True)
                    recover_render()
                    print("  같은 에피소드 번호로 다시 수집", flush=True)
                    continue
                if getattr(interv, "redo_requested", False):
                    redo_state, T_old, was_success = drop_episode(data_grp, last_saved[0])
                    outcomes["success" if was_success else "fail"] -= 1
                    total -= T_old
                    ep -= 1
                    print(f"  << 다시 수집: {last_saved[0]}({T_old}스텝)을 지우고 같은 시작 상태에서 다시 모은다", flush=True)
                    last_saved = None
                    continue

                actions = np.asarray(result["actions"], dtype=np.float64)
                T = len(actions)
                assert T == len(obs_ep) == len(states_ep), f"obs/state/action 스텝 수 불일치: {len(obs_ep)}/{len(states_ep)} vs {T}"

                # 그룹 이름은 성공/실패를 따로 센다: 성공은 demo_0..N(학습용, 번호가 곧 성공 개수),
                # 실패는 fail_0..M(fail-aware STG용으로 보존, 병합은 is_success로 거른다).
                is_success = bool(result["success"])
                demo_grp, sampled = write_episode(
                    data_grp, f"demo_{outcomes['success']}" if is_success else f"fail_{outcomes['fail']}",
                    actions, result["action_modes"], states_ep, obs_ep, obs_keys, rgb_keys, camera, sampler,
                    env.env.sim.model.get_xml())

                # 에피소드 중간에 렌더가 고장나도 데이터에 못 들어가게: 프레임 거칠기가 하나라도
                # 노이즈 범위면 실패로 기록한다(merge_demo_hdf5가 실패분을 버린다).
                render_ok = all(2.0 < r < 15.0 for r in sampled)
                demo_grp.attrs["render_ok"] = render_ok
                if not render_ok:
                    print(f"  !! ep {ep}: 프레임 거칠기 {min(sampled):.1f}~{max(sampled):.1f} — 렌더 노이즈, 실패로 기록", flush=True)
                    if is_success:
                        data_grp.move(demo_grp.name.split("/")[-1], f"fail_{outcomes['fail']}")
                        demo_grp = data_grp[f"fail_{outcomes['fail']}"]
                    is_success = False
                demo_grp.attrs["is_success"] = is_success
                outcomes["success" if is_success else "fail"] += 1
                total += T
                name = demo_grp.name.split("/")[-1]
                last_saved = (name, f"{name} ({'success' if is_success else 'fail'}, {T} steps)")
                redo_state = None
                print(
                    f"ep {ep}: steps={T} success={is_success} triggers={interv.num_triggers}  "
                    f"성공 {outcomes['success']}/{episodes} (시도 {ep + 1})",
                    flush=True,
                )
                ep += 1

            data_grp.attrs["total"] = total
    finally:
        interv.close()
        if hasattr(env, "close"):
            env.close()

    print(f"\n수집 완료: 성공 {outcomes['success']} / 실패 {outcomes['fail']} (총 {outcomes['success'] + outcomes['fail']}에피소드)  saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--base-ckpt",
        default="projects/square_assembly/checkpoints/square_base_policy/policy_epoch1060.pt",
    )
    ap.add_argument("--episodes", type=int, default=20, help="저장할 성공 에피소드 수(실패는 저장되지만 안 셈)")
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="data/square_scripted_intv.hdf5")
    ap.add_argument("--camera", default="agentview_image")
    ap.add_argument("--mode", choices=["oracle", "mouse"], default="oracle",
                    help="oracle: 트리거 키로 스크립트 오라클에 넘김 / mouse: 사람이 2D 맵 위에서 직접 조종"
                         "(Tab=사람/정책 전환 ←=직전 전환점으로 s=일시정지, runners/mouse_teleop.py 참고)")
    ap.add_argument("--trigger-key", default="s", help="[oracle] 실패 판단 시 오라클에 넘기는 키")
    ap.add_argument("--kp", type=float, default=2.0, help="[mouse] xy P 게인")
    ap.add_argument("--kd", type=float, default=2.0, help="[mouse] xy D 게인(0이면 멈출 때 0.4~0.7cm 넘친다)")
    ap.add_argument("--pos-cap", type=float, default=1.0, help="[mouse] xy delta 상한(1.0=5cm/step)")
    ap.add_argument("--z-down", type=float, default=7.0, help="[mouse] Shift 내림 속도(cm/s, PH 시연 중앙값)")
    ap.add_argument("--z-up", type=float, default=18.0, help="[mouse] Ctrl 올림 속도(cm/s, 가속 뒤, PH 시연)")
    ap.add_argument("--yaw-step", type=float, default=5.0, help="[mouse] 휠 한 칸당 야우(도)")
    ap.add_argument("--rot-cap", type=float, default=0.4,
                    help="[mouse] 회전 액션 상한(0.4 = 최고 47도/s, 0.6 = 64도/s. 0.8부터는 위치가 5cm 넘게 밀린다)")
    ap.add_argument("--ddim-steps", type=int, default=10,
                    help="정책 추론을 DDIM N스텝으로(0=학습 그대로 DDPM 100). 2026-09-30 50ep 비교에서 성공률·실패 "
                         "유형 차이 없음, 청크 666→75ms. 에피소드마다 attrs['sampler']에 기록")
    ap.add_argument("--ddim-eta", type=float, default=1.0,
                    help="DDIM 단계별 노이즈(1=DDPO가 단계별 확률을 계산할 수 있음, 0=결정적)")
    ap.add_argument("--quit-key", default="q", help="에피소드를 포기하고 다음으로 넘어가는 키")
    ap.add_argument("--display-size", type=int, default=_MAX_DISPLAY,
                    help=f"화면 표시용 카메라 한 장의 렌더 해상도(저장 데이터와 무관, 최대 {_MAX_DISPLAY}). "
                         "마우스 모드의 맵은 이 값 × 카메라 수")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="[mouse] 창을 이 배율로 늘려 띄운다(그리는 해상도는 그대로 — 조금 흐려진다)")
    ap.add_argument("--fullscreen", action="store_true", help="[mouse] 전체화면으로 시작한다(F11로 오간다)")
    ap.add_argument("--control-fps", type=float, default=20.0, help="사람이 볼 수 있는 속도로 페이싱(0=최대 속도)")
    ap.add_argument("--redo", nargs="+", default=None, metavar="demo_N",
                    help="[mouse] 저장된 에피소드들을 마지막 개입 시작 상태에서 이어받아 다시 수집한다"
                         "(성공했을 때만 교체. 새 에피소드는 모으지 않는다)")
    ap.add_argument("--human-only", action="store_true",
                    help="[mouse] 정책 없이 사람 시연만 모은다 — 체크포인트를 안 읽고(--base-ckpt 무시) 에피소드마다 "
                         "사람 제어로 멈춘 채 시작한다([s]로 시작)")
    ap.add_argument("--overwrite", action="store_true",
                    help="--out 파일이 있으면 지우고 처음부터 모은다(기본: 기존 에피소드 뒤에 이어서)")
    args = ap.parse_args()
    run(args.base_ckpt, args.episodes, args.max_steps, args.out, args.camera,
        args.trigger_key, args.quit_key, args.control_fps, args.display_size,
        mode=args.mode, overwrite=args.overwrite, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta, redo=args.redo,
        human_only=args.human_only, zoom=args.zoom, fullscreen=args.fullscreen,
        teleop=dict(kp=args.kp, kd=args.kd, pos_cap=args.pos_cap, down_speed=args.z_down / 100, up_speed=args.z_up / 100,
                    yaw_step=np.deg2rad(args.yaw_step), rot_cap=args.rot_cap))


if __name__ == "__main__":
    main()
