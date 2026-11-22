r"""반사실 분기 롤아웃 — 랜덤 지연 개입 에피소드를 저장된 sim 상태에서 되살려, 사람 대신 정책이 timeout까지 몰게 한다.

STG 예측기 평가용(2026-11-20): 같은 상황에서 사람이 넘겨받은 경로(원본)와 정책 혼자 간 경로(이 출력)의 예측을 겹쳐 본다.
  --start t0       원본에서 사람이 넘겨받은 스텝부터 — 평가용 반사실 경로
  --start t_press  버튼을 누른 스텝부터 — 끝 상태를 `collect_square_scripted_intervention --replay-src ... --replay-at end`가
                   이어받아 "timeout까지 기다렸다가 개입" 데이터를 모은다

정책이 넘겨받는 스텝에서 실행 중이던 청크는 저장돼 있지 않아 그 자리에서 새로 추론한다. 시간 제한은 원본 수집과 같은
max_steps(= t_press + max_delay + 1). 출력은 원본과 같은 형식(demo 이름도 원본 그대로, 앞부분 포함)이고 attrs:
src_file·branch_from·branch_step·max_steps·is_success·hacky(성공했지만 마지막 파지가 손잡이가 아님)·seed·restore_obs_diff
(되살린 직후 렌더한 obs와 저장된 obs의 평균 절대차, 0~1 스케일 — 복원이 맞았는지).

    MUJOCO_GL=egl NUMBA_CACHE_DIR=/tmp/numba_cache /opt/venvs/torch/bin/python -m square_assembly.scripts.branch_rollout \
        --src data/square_quest_turn30_delay_intv.hdf5 --base-ckpt <정책.pt> --start t0 \
        --out data/square_quest_turn30_branch_t0.hdf5

이미 출력에 있는 demo는 건너뛰므로 중간에 끊겨도 같은 명령으로 이어 간다.
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch

from square_assembly.datasets.labels import LABEL_INTV, LABEL_PREINTV, LABEL_ROLLOUT

HANDLE_GEOMS = ["SquareNut_g4"]
BODY_GEOMS = ["SquareNut_g0", "SquareNut_g1", "SquareNut_g2", "SquareNut_g3"]


def delay_points(attrs, modes):
    """(t_press, t0, max_steps) — 넘겨받은 적이 없으면 None.

    0e61624 이전 수집분은 t0로 정확히 되감으면 taken 기록이 rewound로 바뀌고 attrs가 -1로 남았다. 그 경우 첫 사람
    프레임과 t0가 같은 이벤트에서 되찾는다.
    """
    t0 = int(attrs["t0"])
    if t0 >= 0:
        t_press, max_delay = int(attrs["t_press"]), int(attrs["max_delay"])
    else:
        human = np.flatnonzero(np.asarray(modes) == LABEL_INTV)
        ev = [e for e in json.loads(attrs["delay_events"]) if human.size and e["t0"] == human[0]
              and e["status"] in ("taken", "rewound")]
        if not ev:
            return None
        t0, t_press, max_delay = int(human[0]), int(ev[-1]["t_press"]), int(ev[-1]["max_delay"])
    return t_press, t0, t_press + max_delay + 1


def grasp_part(raw_env):
    """지금 양 손가락 패드가 너트의 어디를 쥐고 있나 — "handle" / "body" / None."""
    grip = raw_env.robots[0].gripper[raw_env.robots[0].arms[0]]
    if raw_env._check_grasp(grip, HANDLE_GEOMS):
        return "handle"
    return "body" if raw_env._check_grasp(grip, BODY_GEOMS) else None


def nut_state(raw_env):
    """(너트 높이 z, 너트–기둥 xy 거리, 그리퍼–손잡이 거리), 단위 m — 실패 에피소드가 실제로 어디까지 갔나."""
    d, nut, robot = raw_env.sim.data, raw_env.nuts[0], raw_env.robots[0]
    p = d.body_xpos[raw_env.obj_body_id[nut.name]]
    handle = d.site_xpos[raw_env.sim.model.site_name2id(nut.important_sites["handle"])]
    eef = d.site_xpos[robot.eef_site_id[robot.arms[0]]]
    peg = d.body_xpos[raw_env.peg1_body_id]
    return float(p[2]), float(np.linalg.norm(p[:2] - peg[:2])), float(np.linalg.norm(eef - handle))


def is_hacky(success, parts):
    """성공했는데 성공 전 마지막 파지가 손잡이가 아니었으면 야매 성공. 한 번도 쥐지 않았으면 판정하지 않는다(False)."""
    held = [p for p in parts if p is not None]
    return bool(success and held and held[-1] != "handle")


class _NoIntervention:
    """사람 없이 정책만 — collect_episode의 intervention_fn이자 make_recorder의 interv."""

    def __call__(self, step, obs_raw):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--base-ckpt", required=True, help="원본을 모은 정책 체크포인트")
    ap.add_argument("--start", choices=["t0", "t_press"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ddim-steps", type=int, default=10, help="원본 수집과 같게(attrs sampler 확인)")
    ap.add_argument("--ddim-eta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="에피소드 i는 seed+i")
    ap.add_argument("--demos", nargs="+", default=None, help="이 demo만(기본: 넘겨받은 기록이 있는 전부)")
    args = ap.parse_args()

    from square_assembly.datasets.normalization import MinMaxNormalizer, load_stats
    from square_assembly.factory import registry
    from square_assembly.runners.intervention_rollout import _predict_chunk, collect_episode
    from square_assembly.runners.rollout import maybe_speed_up_inference
    from square_assembly.scripts.collect_square_scripted_intervention import (
        _from_storage, _gripper, gripper_ramp, make_recorder, reset_to_state, sampler_label, write_episode)
    from square_assembly.utils.checkpoints import load_epoch_checkpoint, load_run_config
    from square_assembly.utils.task_utils import is_image_task, make_eval_env, task_obs_keys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = load_run_config(args.base_ckpt)
    task_cfg, policy_cfg = saved.task, saved.policy
    policy = registry.create_policy(saved.policy_name, task_cfg, policy_cfg).to(device)
    load_epoch_checkpoint(args.base_ckpt, policy, device)
    policy.eval()
    maybe_speed_up_inference(policy, args.ddim_steps or None, eta=args.ddim_eta)
    sampler = sampler_label(policy)
    normalizer = MinMaxNormalizer(load_stats(os.path.join(os.path.dirname(args.base_ckpt), "normalization_stats.json")))
    obs_horizon, action_horizon = task_cfg.get("obs_horizon", policy_cfg.obs_horizon), policy_cfg.action_horizon
    obs_keys = task_obs_keys(task_cfg)
    rgb_keys = list(task_cfg.rgb_keys) if is_image_task(task_cfg) else []
    env = make_eval_env(task_cfg)
    raw = env.env

    def predict_fn(history):
        return _predict_chunk(policy, normalizer, history, obs_keys, device, rgb_keys=rgb_keys)

    interv = _NoIntervention()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with h5py.File(args.src, "r") as src, h5py.File(args.out, "a") as f:
        out = f.require_group("data")
        names = args.demos or sorted(src["data"], key=lambda n: int(n.split("_")[-1]))
        for i, name in enumerate(names):
            old = src["data"][name]
            if name in out:
                continue
            modes = np.asarray(old["action_mode"])
            pts = delay_points(old.attrs, modes)
            if pts is None:
                continue
            t_press, t0, max_steps = pts
            t = t0 if args.start == "t0" else t_press
            seed = args.seed + i
            np.random.seed(seed)
            torch.manual_seed(seed)

            obs_st = [{k: np.asarray(old["obs"][k][s]) for k in obs_keys} for s in range(t + 1)]
            states = np.asarray(old["states"][: t + 1])
            ramp = gripper_ramp(np.asarray(old["actions"][:t, -1]), _gripper(env).speed)
            resume = {"obs": [{k: _from_storage(k, o[k], rgb_keys) for k in obs_keys} for o in obs_st[:t]],
                      "actions": np.asarray(old["actions"][:t]),
                      "modes": np.where(modes[:t] == LABEL_PREINTV, LABEL_ROLLOUT, modes[:t])}
            pre_step, obs_ep, states_ep = make_recorder(env, interv, obs_keys, rgb_keys, (obs_st[:t], states[:t], ramp[:t]))
            parts, diff = [], []

            def reset_fn():
                obs = reset_to_state(env, states[t])
                _gripper(env).current_action = ramp[t]
                diff.append(float(np.mean([np.abs(_from_storage(k, obs_st[t][k], rgb_keys) - obs[k]).mean() for k in rgb_keys]))
                            if rgb_keys else 0.0)
                return obs

            def pre_step_and_grasp(step, obs_raw):
                parts.append(grasp_part(raw))
                return pre_step(step, obs_raw)

            result = collect_episode(
                env, policy, normalizer, obs_keys, obs_horizon, action_horizon, device,
                intervention_fn=interv, max_steps=max_steps, pre_step_fn=pre_step_and_grasp,
                predict_fn=predict_fn, print_diagnostics=False, reset_fn=reset_fn, resume=resume,
            )
            success = bool(result["success"])
            g, _ = write_episode(out, name, np.asarray(result["actions"], dtype=np.float64), result["action_modes"],
                                 states_ep, obs_ep, obs_keys, rgb_keys, rgb_keys[0], sampler, raw.sim.model.get_xml())
            g.attrs["delay_events"] = old.attrs["delay_events"]
            g.attrs.update(src_file=args.src, branch_from=args.start, branch_step=t, max_steps=max_steps,
                           is_success=success, hacky=is_hacky(success, parts), seed=seed, restore_obs_diff=diff[0],
                           orig_t_press=t_press, orig_t0=t0)
            f.flush()
            print(f"{name}: {args.start}={t} -> {len(result['actions'])}/{max_steps} "
                  f"{'성공' + (' (야매)' if g.attrs['hacky'] else '') if success else '실패'}  "
                  f"복원 obs 차 {diff[0]:.4f}", flush=True)
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
