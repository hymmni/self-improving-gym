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
import os

import h5py
import numpy as np
import torch

# scripted_intervention은 모듈 단계에서 numpy만 쓴다(robosuite/EGL을 안 건드림) — 그래서
# 여기서 임포트해도 안전하다. robosuite/robomimic을 끌어오는 나머지 임포트는 run() 안에서
# open_window() 뒤에 한다(이유는 open_window docstring, DOCKER.md §4).
from square_assembly.runners.scripted_intervention import ScriptedFailureIntervention, open_window
from square_assembly.runners.square_oracle import SquareAssemblyOracle


# robosuite 오프스크린 버퍼는 640x480(MJCF 기본)으로 잡히고, 그보다 큰 렌더를 요청하면
# binding_utils.update_offscreen_size가 MjrContext를 다시 만든다. 그런데 cv2(Qt) 창이 이미
# 떠 있는 프로세스에선 그 재생성이 EGL에서 실패한다 — `mujoco.FatalError: Default framebuffer
# is not complete, error 0x0` (2026-09-17 서버 실측: compose run 컨테이너에서 512는 죽고 480은
# 정상, 같은 코드가 창 없이 헤드리스면 512도 정상).
# ponytail: 480 상한. 더 크게 보려면 render context를 직접 큰 max_width/max_height로 만들어
# sim.add_render_context로 꽂아야 하는데, 모니터링 화면에 그만한 값어치는 없다.
_MAX_DISPLAY = 480


def _to_storage(key, val, rgb_keys):
    """collect_square_rollouts._to_storage와 동일 — CHW,float[0,1] -> HWC,uint8."""
    val = np.asarray(val)
    if key in rgb_keys:
        if val.ndim == 3 and val.shape[0] in (1, 3, 4) and val.shape[0] < val.shape[-1]:
            val = np.transpose(val, (1, 2, 0))
        val = np.clip(val * 255.0, 0, 255).astype(np.uint8)
    return val


def run(base_ckpt, episodes, max_steps, out, camera, trigger_key, quit_key,
        control_fps, display_size, window_name="rollout"):
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

    env = make_eval_env(task_cfg)
    interv = ScriptedFailureIntervention(
        SquareAssemblyOracle(env), trigger_key=trigger_key, quit_key=quit_key, window_name=window_name,
    )
    display_camera = camera[: -len("_image")]

    def predict_fn(history):
        return _predict_chunk(policy, normalizer, history, obs_keys, device, rgb_keys=rgb_keys)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    outcomes = {"success": 0, "fail": 0}

    try:
        with h5py.File(out, "w") as f:
            data_grp = f.create_group("data")
            total = 0
            for ep in range(episodes):
                interv.reset()
                obs_ep = []

                def track(obs_raw, _store=obs_ep):
                    _store.append({k: _to_storage(k, obs_raw[k], rgb_keys) for k in obs_keys})
                    # 저장/학습은 obs의 84픽셀 그대로, 화면만 따로 고해상도로 렌더한다.
                    return interv.render(env.render(
                        mode="rgb_array", height=display_size, width=display_size,
                        camera_name=display_camera,
                    ))

                result = collect_episode(
                    env, policy, normalizer, obs_keys,
                    task_cfg.get("obs_horizon", policy_cfg.obs_horizon), policy_cfg.action_horizon, device,
                    intervention_fn=interv, max_steps=max_steps, render=False, render_fn=track,
                    should_end_fn=interv.should_end, control_fps=control_fps,
                    predict_fn=predict_fn, print_diagnostics=False,
                )
                actions = np.asarray(result["actions"], dtype=np.float64)
                T = len(actions)
                assert T == len(obs_ep), f"obs/action 스텝 수 불일치: {len(obs_ep)} vs {T}"

                demo_grp = data_grp.create_group(f"demo_{ep}")
                demo_grp.attrs["num_samples"] = T
                demo_grp.create_dataset("actions", data=actions)
                demo_grp.create_dataset("action_mode", data=result["action_modes"])
                obs_grp = demo_grp.create_group("obs")
                for k in obs_keys:
                    stacked = np.stack([o[k] for o in obs_ep], axis=0)
                    obs_grp.create_dataset(k, data=stacked, compression="gzip" if k in rgb_keys else None)

                is_success = bool(result["success"])
                demo_grp.attrs["is_success"] = is_success
                outcomes["success" if is_success else "fail"] += 1
                total += T
                print(
                    f"ep {ep}: steps={T} success={is_success} triggers={interv.num_triggers}  "
                    f"누적 성공 {outcomes['success']}/{ep + 1} ({outcomes['success'] / (ep + 1):.1%})",
                    flush=True,
                )

            data_grp.attrs["total"] = total
    finally:
        interv.close()
        if hasattr(env, "close"):
            env.close()

    print(f"\n수집 완료: {episodes}에피소드 (성공 {outcomes['success']}, 실패 {outcomes['fail']})  saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--base-ckpt",
        default="projects/square_assembly/checkpoints/square_base_policy/policy_epoch1060.pt",
    )
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="data/square_scripted_intv.hdf5")
    ap.add_argument("--camera", default="agentview_image")
    ap.add_argument("--trigger-key", default="s", help="실패 판단 시 오라클에 넘기는 키")
    ap.add_argument("--quit-key", default="q", help="에피소드를 포기하고 다음으로 넘어가는 키")
    ap.add_argument("--display-size", type=int, default=_MAX_DISPLAY,
                    help=f"화면 표시용 렌더 해상도(저장 데이터와 무관, 최대 {_MAX_DISPLAY})")
    ap.add_argument("--control-fps", type=float, default=20.0, help="사람이 볼 수 있는 속도로 페이싱(0=최대 속도)")
    args = ap.parse_args()
    run(args.base_ckpt, args.episodes, args.max_steps, args.out, args.camera,
        args.trigger_key, args.quit_key, args.control_fps, args.display_size)


if __name__ == "__main__":
    main()
