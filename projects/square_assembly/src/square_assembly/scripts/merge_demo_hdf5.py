r"""여러 hdf5(사람 시연 + 롤아웃/개입 수집분)를 학습용 데이터셋 하나로 합친다.

수집 스크립트(collect_square_rollouts.py, collect_square_scripted_intervention.py)는 각
세션을 따로 저장하므로, "기존 데모 + 이번에 늘린 성공분"으로 다시 학습하려면 한 파일로
이어붙여야 한다(예전 레포의 round.py가 하던 일 중 데이터 병합 부분).

라벨(action_mode) 규칙:
- action_mode가 있는 소스(개입 수집분)는 그대로 가져온다 — demo=-1/rollout=0/intv=1/preintv=-10
  (datasets/labels.py). 오라클이 몬 구간이 INTV로 남아 있어야 나중에 weighting=class_based
  (SIRIUS) 실험을 데이터 재수집 없이 할 수 있다.
- action_mode가 없는 소스(순수 시연 hdf5)는 전 프레임을 LABEL_DEMO(-1)로 채운다.

`data.attrs["env_args"]`는 학습 시작 시 task meta를 데이터에서 복원할 때 필요한데
(utils/task_utils.derive_task_meta_from_hdf5), 수집 산출물엔 없다 — robomimic 원본
데이터셋에서 --env-args-from으로 가져와 붙인다.

사용(컨테이너 WORKDIR=/workspace 기준):
    /opt/venvs/torch/bin/python -m square_assembly.scripts.merge_demo_hdf5 \
        --sources data/square_scale3_0_seed1.hdf5 data/square_scripted_intv_v1.hdf5 \
        --env-args-from /home/moai/hymm_ws/square_dataset/square_image_v15.hdf5 \
        --out data/square_demo50_intv20.hdf5
"""

import argparse
import json

import h5py
import numpy as np

from square_assembly.datasets.labels import LABEL_DEMO

_RGB_SUFFIX = "_image"


def _demo_order(grp):
    """demo_0, demo_1, ... 순서로(hdf5 키는 문자열 정렬이라 demo_10이 demo_2보다 앞에 온다)."""
    return sorted(grp.keys(), key=lambda k: int(k.split("_")[-1]))


_NUT_ENV = None


def initial_nut_yaw(state):
    """에피소드 첫 sim 상태(flattened)에서 Square 너트 yaw(도). robosuite env를 한 번 만들어 재사용한다."""
    global _NUT_ENV
    if _NUT_ENV is None:
        import robosuite as suite

        _NUT_ENV = suite.make("NutAssemblySquare", robots="Panda", has_renderer=False,
                              has_offscreen_renderer=False, use_camera_obs=False)
        _NUT_ENV.reset()
    env = _NUT_ENV
    env.sim.set_state_from_flattened(np.asarray(state))
    env.sim.forward()
    from square_assembly.envs.robomimic.factory import nut_yaw_deg

    return nut_yaw_deg(env)


def in_band(yaw_deg, band):
    lo, hi = band
    return (yaw_deg - lo) % 360 < hi - lo


def merge(sources, out, env_args_from=None, only_success=True, first=None, nut_yaw_exclude=None):
    """sources의 에피소드를 순서대로 out 하나에 이어붙인다. first면 소스마다 앞에서 그 개수만(성공 필터 뒤).
    nut_yaw_exclude=(lo, hi)면 첫 상태의 너트 yaw가 그 구간(도)인 에피소드를 빼고(제한 없이 모은 소스도 쓸 수 있게),
    결과를 그 구간으로 모은 데이터로 기록한다. 반환: (에피소드 수, 프레임 수)."""
    excludes = set()
    for src in sources:
        with h5py.File(src, "r") as fin:
            v = fin["data"].attrs.get("nut_yaw_exclude")
            excludes.add(None if v is None else tuple(float(x) for x in v))
    if nut_yaw_exclude is not None:   # 명시하면 구간 기록이 없는 소스는 걸러서 쓰고, 다른 구간으로 모은 소스만 막는다
        exclude = tuple(float(v) for v in nut_yaw_exclude)
        if excludes - {None, exclude}:
            raise ValueError(f"다른 너트 yaw 제외 구간으로 모은 소스가 있다: {excludes - {None, exclude}}")
    elif len(excludes) > 1:
        raise ValueError(f"소스마다 너트 yaw 제외 구간이 다르다 — 초기화 분포가 섞인다: {excludes}")
    else:
        exclude = excludes.pop() if excludes else None

    n_demo, n_frame = 0, 0
    with h5py.File(out, "w") as fout:
        data_out = fout.create_group("data")
        for src in sources:
            with h5py.File(src, "r") as fin:
                kept = 0
                for key in _demo_order(fin["data"]):
                    if first is not None and kept >= first:
                        break
                    demo = fin["data"][key]
                    if only_success and not bool(demo.attrs.get("is_success", True)):
                        continue
                    if nut_yaw_exclude is not None and in_band(initial_nut_yaw(demo["states"][0]), exclude):
                        continue
                    T = int(demo.attrs["num_samples"])
                    grp = data_out.create_group(f"demo_{n_demo}")
                    grp.attrs["num_samples"] = T
                    grp.attrs["is_success"] = bool(demo.attrs.get("is_success", True))
                    grp.attrs["source"] = src
                    grp.create_dataset("actions", data=np.asarray(demo["actions"]))
                    modes = (
                        np.asarray(demo["action_mode"])
                        if "action_mode" in demo
                        else np.full(T, LABEL_DEMO, dtype=np.int64)
                    )
                    grp.create_dataset("action_mode", data=modes)
                    # 시뮬레이터 상태가 있으면 같이 옮긴다 — 병합본만 남겨도 되감아 렌더할 수 있게.
                    if "states" in demo:
                        grp.create_dataset("states", data=np.asarray(demo["states"]))
                    if "model_file" in demo.attrs:
                        grp.attrs["model_file"] = demo.attrs["model_file"]
                    obs_out = grp.create_group("obs")
                    for obs_key in demo["obs"].keys():
                        obs_out.create_dataset(
                            obs_key, data=np.asarray(demo["obs"][obs_key]),
                            compression="gzip" if obs_key.endswith(_RGB_SUFFIX) else None,
                        )
                    n_demo += 1
                    n_frame += T
                    kept += 1
                print(f"  {src}: {kept}/{len(fin['data'].keys())} 에피소드", flush=True)

        data_out.attrs["total"] = n_frame
        if env_args_from is not None:
            with h5py.File(env_args_from, "r") as fref:
                env_args = json.loads(fref["data"].attrs["env_args"])
            if exclude is not None:   # 수집 때의 초기화 범위가 학습 중 평가·RL 롤아웃까지 따라가게
                env_args.setdefault("env_kwargs", {})["nut_yaw_exclude"] = list(exclude)
            data_out.attrs["env_args"] = json.dumps(env_args)
        if exclude is not None:
            data_out.attrs["nut_yaw_exclude"] = list(exclude)
    return n_demo, n_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", nargs="+", required=True, help="합칠 hdf5들(주어진 순서대로)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-args-from", default=None,
                    help="data.attrs['env_args']를 가져올 robomimic 원본 hdf5")
    ap.add_argument("--keep-failures", action="store_true",
                    help="is_success=False 에피소드도 포함(기본은 성공만)")
    ap.add_argument("--first", type=int, default=None, help="소스마다 앞에서 N개만(demo 번호 순)")
    ap.add_argument("--nut-yaw-exclude", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="첫 상태의 너트 yaw가 이 구간(도)인 에피소드를 빼고, 결과를 이 구간으로 모은 데이터로 기록")
    args = ap.parse_args()

    n_demo, n_frame = merge(args.sources, args.out, args.env_args_from, not args.keep_failures, args.first,
                           args.nut_yaw_exclude)
    print(f"병합 완료: {n_demo}에피소드 {n_frame}프레임 -> {args.out}")


if __name__ == "__main__":
    main()
