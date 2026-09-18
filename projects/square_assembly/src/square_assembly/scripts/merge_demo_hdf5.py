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
    python -m square_assembly.scripts.merge_demo_hdf5 \
        --sources data/square_scale3_0_seed1.hdf5 data/square_scripted_intv_v1.hdf5 \
        --env-args-from /home/moai/hymm_ws/square_dataset/square_image_v15.hdf5 \
        --out data/square_demo50_intv20.hdf5
"""

import argparse

import h5py
import numpy as np

from square_assembly.datasets.labels import LABEL_DEMO

_RGB_SUFFIX = "_image"


def _demo_order(grp):
    """demo_0, demo_1, ... 순서로(hdf5 키는 문자열 정렬이라 demo_10이 demo_2보다 앞에 온다)."""
    return sorted(grp.keys(), key=lambda k: int(k.split("_")[-1]))


def merge(sources, out, env_args_from=None, only_success=True):
    """sources의 에피소드를 순서대로 out 하나에 이어붙인다. 반환: (에피소드 수, 프레임 수)."""
    n_demo, n_frame = 0, 0
    with h5py.File(out, "w") as fout:
        data_out = fout.create_group("data")
        for src in sources:
            with h5py.File(src, "r") as fin:
                kept = 0
                for key in _demo_order(fin["data"]):
                    demo = fin["data"][key]
                    if only_success and not bool(demo.attrs.get("is_success", True)):
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
                data_out.attrs["env_args"] = fref["data"].attrs["env_args"]
    return n_demo, n_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", nargs="+", required=True, help="합칠 hdf5들(주어진 순서대로)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-args-from", default=None,
                    help="data.attrs['env_args']를 가져올 robomimic 원본 hdf5")
    ap.add_argument("--keep-failures", action="store_true",
                    help="is_success=False 에피소드도 포함(기본은 성공만)")
    args = ap.parse_args()

    n_demo, n_frame = merge(args.sources, args.out, args.env_args_from, not args.keep_failures)
    print(f"병합 완료: {n_demo}에피소드 {n_frame}프레임 -> {args.out}")


if __name__ == "__main__":
    main()
