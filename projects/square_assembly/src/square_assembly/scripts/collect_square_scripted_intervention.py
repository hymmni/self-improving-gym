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
끝까지 오라클이 잡는다 — 정책에 돌려주지 않는다). 포기하려면 'q'.

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


def _to_storage(key, val, rgb_keys):
    """collect_square_rollouts._to_storage와 동일 — CHW,float[0,1] -> HWC,uint8."""
    val = np.asarray(val)
    if key in rgb_keys:
        if val.ndim == 3 and val.shape[0] in (1, 3, 4) and val.shape[0] < val.shape[-1]:
            val = np.transpose(val, (1, 2, 0))
        val = np.clip(val * 255.0, 0, 255).astype(np.uint8)
    return val


def run(base_ckpt, episodes, max_steps, out, camera, trigger_key, quit_key,
        control_fps, display_size, window_name="rollout", mode="oracle", teleop=None):
    # 반드시 아래 임포트들(robosuite/robomimic → EGL 초기화)보다 먼저 — 순서가 바뀌면 첫
    # cv2.imshow가 영영 멈춘다.
    open_window(window_name)

    if display_size > _MAX_DISPLAY:
        raise ValueError(
            f"--display-size는 {_MAX_DISPLAY} 이하여야 한다(요청 {display_size}) — "
            "그보다 크면 첫 프레임에서 MjrContext 재생성이 EGL에서 실패한다(_MAX_DISPLAY 주석 참고)."
        )

    from square_assembly.datasets.normalization import MinMaxNormalizer, load_stats
    from square_assembly.factory import registry
    from square_assembly.runners.intervention_rollout import _predict_chunk, collect_episode
    from square_assembly.utils.checkpoints import load_epoch_checkpoint, load_run_config
    from square_assembly.utils.task_utils import is_image_task, make_eval_env, task_obs_keys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved = load_run_config(base_ckpt)
    task_cfg, policy_cfg, policy_name = saved.task, saved.policy, saved.policy_name

    policy = registry.create_policy(policy_name, task_cfg, policy_cfg).to(device)
    load_epoch_checkpoint(base_ckpt, policy, device)
    policy.eval()

    stats_path = os.path.join(os.path.dirname(base_ckpt), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))
    obs_keys = task_obs_keys(task_cfg)
    rgb_keys = list(task_cfg.rgb_keys) if is_image_task(task_cfg) else []
    if camera not in rgb_keys:
        raise ValueError(f"--camera {camera}는 task.rgb_keys {rgb_keys}에 없다")

    env = wait_for_renderer(lambda: make_eval_env(task_cfg), camera, display_size)
    display_camera = camera[: -len("_image")]

    if mode == "mouse":
        interv = MouseTeleopIntervention(
            env, controller=MouseTeleopController(**(teleop or {})), map_size=display_size,
            window_name=window_name, quit_key=quit_key,
        )
    else:
        interv = ScriptedFailureIntervention(
            SquareAssemblyOracle(env), trigger_key=trigger_key, quit_key=quit_key, window_name=window_name,
        )

    def predict_fn(history):
        return _predict_chunk(policy, normalizer, history, obs_keys, device, rgb_keys=rgb_keys)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    outcomes = {"success": 0, "fail": 0}

    try:
        # 파일이 이미 있으면 이어서 모은다("a") — 같은 --out으로 다시 실행해도 기존 에피소드가 안 날아간다.
        with h5py.File(out, "a") as f:
            data_grp = f.require_group("data")
            outcomes["success"] = sum(k.startswith("demo_") for k in data_grp.keys())
            outcomes["fail"] = sum(k.startswith("fail_") for k in data_grp.keys())
            total = int(data_grp.attrs.get("total", 0))
            ep = outcomes["success"] + outcomes["fail"]
            if ep:
                print(f"이어서 수집: 기존 성공 {outcomes['success']} / 실패 {outcomes['fail']} -> 목표 성공 {episodes}", flush=True)
            # episodes = 저장할 *성공* 에피소드 수. 실패도 저장은 되지만(is_success=False, 병합에서 제외)
            # 개수에는 안 센다 — 라운드마다 "성공 N개"를 맞추려는 것이지 시도 횟수가 아니다.
            while outcomes["success"] < episodes:
                interv.reset()
                obs_ep = []
                noise_streak = [0]

                def track(obs_raw, _store=obs_ep, _streak=noise_streak):
                    _store.append({k: _to_storage(k, obs_raw[k], rgb_keys) for k in obs_keys})
                    # 렌더가 도중에 고장나면(연속 3프레임 노이즈) 에피소드를 즉시 끊는다 —
                    # 사람이 지켜보다 q를 누를 필요 없이 아래서 버리고 복구를 기다린다.
                    r = frame_roughness(np.transpose(obs_raw[camera], (1, 2, 0)) * 255.0)
                    _streak[0] = 0 if 2.0 < r < 15.0 else _streak[0] + 1
                    # 저장/학습은 obs의 84픽셀 그대로, 화면만 따로 고해상도로 렌더한다.
                    keep = interv.render(env.render(
                        mode="rgb_array", height=display_size, width=display_size,
                        camera_name=display_camera,
                    ))
                    return keep and _streak[0] < 3

                result = collect_episode(
                    env, policy, normalizer, obs_keys,
                    task_cfg.get("obs_horizon", policy_cfg.obs_horizon), policy_cfg.action_horizon, device,
                    intervention_fn=interv, max_steps=max_steps, render=False, render_fn=track,
                    should_end_fn=interv.should_end, control_fps=control_fps,
                    predict_fn=predict_fn, print_diagnostics=False,
                )
                if noise_streak[0] >= 3:
                    print(f"  !! 렌더 노이즈로 에피소드 중단(step {len(obs_ep)}) — 버리고 렌더 복구를 기다린다", flush=True)
                    t0 = time.time()
                    while renderer_is_noisy(env, camera, display_size):
                        if time.time() - t0 > 600:
                            raise RuntimeError("10분 동안 렌더가 복구되지 않았다 — 컨테이너를 새로 만들어 다시 시도한다(DOCKER.md §4).")
                        time.sleep(5)
                    print("  렌더 복구됨 — 같은 에피소드 번호로 다시 수집", flush=True)
                    continue

                actions = np.asarray(result["actions"], dtype=np.float64)
                T = len(actions)
                assert T == len(obs_ep), f"obs/action 스텝 수 불일치: {len(obs_ep)} vs {T}"

                # 그룹 이름은 성공/실패를 따로 센다: 성공은 demo_0..N(학습용, 번호가 곧 성공 개수),
                # 실패는 fail_0..M(fail-aware STG용으로 보존, 병합은 is_success로 거른다).
                is_success = bool(result["success"])
                demo_grp = data_grp.create_group(f"demo_{outcomes['success']}" if is_success else f"fail_{outcomes['fail']}")
                demo_grp.attrs["num_samples"] = T
                demo_grp.create_dataset("actions", data=actions)
                demo_grp.create_dataset("action_mode", data=result["action_modes"])
                obs_grp = demo_grp.create_group("obs")
                for k in obs_keys:
                    stacked = np.stack([o[k] for o in obs_ep], axis=0)
                    obs_grp.create_dataset(k, data=stacked, compression="gzip" if k in rgb_keys else None)

                # 에피소드 중간에 렌더가 고장나도 데이터에 못 들어가게: 프레임 거칠기가 하나라도
                # 노이즈 범위면 실패로 기록한다(merge_demo_hdf5가 실패분을 버린다).
                frames = obs_grp[camera]
                sampled = [frame_roughness(frames[i]) for i in np.linspace(0, T - 1, min(16, T)).astype(int)]
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
                print(
                    f"ep {ep}: steps={T} success={is_success} triggers={interv.num_triggers}  "
                    f"누적 성공 {outcomes['success']}/{ep + 1} ({outcomes['success'] / (ep + 1):.1%})",
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
                         "(h=잡기 p=돌려주기 s=일시정지, runners/mouse_teleop.py 참고)")
    ap.add_argument("--trigger-key", default="s", help="[oracle] 실패 판단 시 오라클에 넘기는 키")
    ap.add_argument("--kp", type=float, default=1.0, help="[mouse] xy P 게인")
    ap.add_argument("--kd", type=float, default=0.0, help="[mouse] xy D 게인")
    ap.add_argument("--pos-cap", type=float, default=0.3, help="[mouse] xy delta 상한(1.0=5cm/step)")
    ap.add_argument("--z-speed", type=float, default=0.2, help="[mouse] Space/Shift z 속도(0.2=1cm/step)")
    ap.add_argument("--yaw-step", type=float, default=5.0, help="[mouse] 휠 한 칸당 야우(도)")
    ap.add_argument("--grip-rate", type=float, default=0.1, help="[mouse] 스텝당 그리퍼 명령 변화")
    ap.add_argument("--quit-key", default="q", help="에피소드를 포기하고 다음으로 넘어가는 키")
    ap.add_argument("--display-size", type=int, default=_MAX_DISPLAY,
                    help=f"화면 표시용 렌더 해상도(저장 데이터와 무관, 최대 {_MAX_DISPLAY})")
    ap.add_argument("--control-fps", type=float, default=20.0, help="사람이 볼 수 있는 속도로 페이싱(0=최대 속도)")
    args = ap.parse_args()
    run(args.base_ckpt, args.episodes, args.max_steps, args.out, args.camera,
        args.trigger_key, args.quit_key, args.control_fps, args.display_size,
        mode=args.mode, teleop=dict(kp=args.kp, kd=args.kd, pos_cap=args.pos_cap, z_speed=args.z_speed,
                                    yaw_step=np.deg2rad(args.yaw_step), grip_rate=args.grip_rate))


if __name__ == "__main__":
    main()
