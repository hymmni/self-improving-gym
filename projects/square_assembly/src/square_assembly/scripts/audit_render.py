r"""저장된 카메라 이미지가 저장된 sim 상태와 맞는지 프레임마다 다시 그려 확인한다. --repair면 렌더 버그 프레임을 고친다.

robosuite 렌더 버그(envs/robomimic/render_fix.py, 2026-11-20 수정)로 깨진 프레임은 sim은 멀쩡하고 그림만 틀렸으므로,
저장된 상태로 다시 그리면 원래 그림이 나온다. 프레임 종류(카메라별, 다시 그린 t-3..t+3과 비교):
  ok         다시 그린 t와 평균차 <= 3 (0~255)
  other_cam  다른 카메라 그림이 들어가 있음        ┐ 렌더 버그 — 고친다
  garbage    아무것도 안 닮음(노이즈·검은 화면)     ┘ (그 주변 ±5 스텝의 stale도)
  stale      같은 카메라의 다른 스텝 그림
  mismatch   장면은 맞는데 조금 다름(그린 시점과 저장한 상태가 한 substep 어긋나는 등) — 그대로 둔다
sim 상태(`states`)가 없는 파일은 검사하지 못한다.

    MUJOCO_GL=egl /opt/venvs/torch/bin/python -m square_assembly.scripts.audit_render data/a.hdf5 data/b.hdf5 \
        --out audit.json [--repair] [--workers 6]
"""

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

CAMS = ["agentview", "robot0_eye_in_hand"]
# env 설정(카메라·해상도)을 가져올 체크포인트 — 장면 자체는 에피소드의 model_file이 정한다
CKPT = {"square": "outputs/square_quest_20261117/turn30_e500/policy_epoch500.pt",
        "can": "outputs/can_ph10/scratch/policy_epoch1000.pt"}
OK, NEAR, FAR, AROUND = 3.0, 4.0, 40.0, 5
_ENVS = {}


def classify(stored, same, other):
    """stored: 저장 이미지. same/other: {k: 같은/다른 카메라를 t+k 상태로 다시 그린 것}(k=0 포함)."""
    d = lambda a: float(np.abs(stored.astype(np.int16) - a.astype(np.int16)).mean())
    d0 = d(same[0])
    if d0 <= OK:
        return "ok"
    dk = min((d(v) for k, v in same.items() if k), default=np.inf)
    do = min(d(v) for v in other.values())
    if do < NEAR:
        return "other_cam"
    if dk < NEAR and dk < d0 / 2:
        return "stale"
    if min(d0, dk, do) > FAR:
        return "garbage"
    return "mismatch"


def repair_steps(classes):
    """{cam: [class per step]} -> 고칠 스텝 집합: garbage/other_cam과 그 ±AROUND 안의 stale(같은 사건)."""
    bug = {t for cl in classes.values() for t, c in enumerate(cl) if c in ("garbage", "other_cam")}
    near = {t for cl in classes.values() for t, c in enumerate(cl)
            if c == "stale" and any(abs(t - b) <= AROUND for b in bug)}
    return sorted(bug | near)


def _env(kind):
    if kind not in _ENVS:
        from square_assembly.envs.robomimic.render_fix import install
        from square_assembly.utils.checkpoints import load_run_config
        from square_assembly.utils.task_utils import make_eval_env
        install()
        env = make_eval_env(load_run_config(CKPT[kind]).task)
        _ENVS[kind] = getattr(env, "env", env) if type(env).__name__ == "AbsRot6dActionEnv" else env
    return _ENVS[kind]


def audit_episode(path, name):
    """Returns (보고 dict, {step: {cam: 다시 그린 저장 방향 이미지}} — 고칠 프레임)."""
    with h5py.File(path, "r") as f:
        g = f["data"][name]
        xml = g.attrs["model_file"]
        states = np.asarray(g["states"])
        imgs = {c: np.asarray(g["obs"][f"{c}_image"]) for c in CAMS}
    env = _env("square" if "SquareNut" in xml else "can")
    env.reset_to({"model": xml, "states": states[0]})
    sim = env.env.sim
    T = min(len(states), *(len(v) for v in imgs.values()))
    ref = {c: np.empty((T, 84, 84, 3), np.uint8) for c in CAMS}
    for t in range(T):
        sim.set_state_from_flattened(states[t])
        sim.forward()
        for c in CAMS:
            ref[c][t] = sim.render(84, 84, camera_name=c)
    for c in CAMS:   # 저장 방향(robomimic이 상하를 뒤집었는지)은 첫 프레임으로 정한다
        a, b = (np.abs(r.astype(np.int16) - imgs[c][0]).mean() for r in (ref[c][0], ref[c][0][::-1]))
        if b < a:
            ref[c] = ref[c][:, ::-1]
    classes = {}
    for c in CAMS:
        o = [x for x in CAMS if x != c][0]
        cl = []
        for t in range(T):
            if np.abs(imgs[c][t].astype(np.int16) - ref[c][t]).mean() <= OK:
                cl.append("ok")
                continue
            ks = [k for k in range(-3, 4) if 0 <= t + k < T]
            cl.append(classify(imgs[c][t], {k: ref[c][t + k] for k in ks}, {k: ref[o][t + k] for k in ks}))
        classes[c] = cl
    fix = repair_steps(classes)
    counts = {c: {k: v for k, v in zip(*np.unique(classes[c], return_counts=True)) if k != "ok"} for c in CAMS}
    report = {"file": path, "demo": name, "T": int(T), "repair_steps": fix,
              "counts": {c: {k: int(v) for k, v in counts[c].items()} for c in CAMS}}
    return report, {t: {c: ref[c][t] for c in CAMS} for t in fix}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True, help="에피소드별 보고 JSON")
    ap.add_argument("--repair", action="store_true", help="렌더 버그 프레임을 다시 그린 것으로 바꿔 쓴다")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = []
    for p in args.files:
        with h5py.File(p, "r") as f:
            for n in f["data"]:
                g = f["data"][n]
                if isinstance(g, h5py.Group) and "states" in g and "obs" in g and "agentview_image" in g["obs"]:
                    jobs.append((p, n))
    print(f"{len(jobs)} episodes", flush=True)
    reports, patches, t0 = [], [], time.time()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(audit_episode, p, n): (p, n) for p, n in jobs}
        for i, fu in enumerate(as_completed(futs)):
            p, n = futs[fu]
            try:
                rep, patch = fu.result()
            except Exception as e:   # 한 에피소드 실패가 전체를 멈추지 않게
                reports.append({"file": p, "demo": n, "error": repr(e)[:300]})
                print(f"  {p} {n}: ERROR {e!r}"[:300], flush=True)
                continue
            reports.append(rep)
            if patch:
                patches.append((p, n, patch))
                print(f"  {p} {n}: render-bug steps {len(rep['repair_steps'])} {rep['counts']}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"[{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    json.dump(reports, open(args.out, "w"))
    errors = [r for r in reports if "error" in r]
    if args.repair and errors:
        print(f"{len(errors)} episodes failed to audit — not repairing anything", flush=True)
    elif args.repair:   # 작업 프로세스가 모두 끝난 뒤에 쓴다 — 읽는 중인 파일을 열면 HDF5 잠금에 걸린다
        for p, n, patch in patches:
            with h5py.File(p, "r+") as f:
                g = f["data"][n]
                for t, by_cam in patch.items():
                    for c, im in by_cam.items():
                        g["obs"][f"{c}_image"][t] = im
                prev = json.loads(g.attrs.get("render_repaired", "[]"))
                g.attrs["render_repaired"] = json.dumps(sorted(set(prev) | set(patch)))
    bad = [r for r in reports if r.get("repair_steps")]
    print(f"done {len(jobs)} in {time.time() - t0:.0f}s — episodes with render-bug frames: {len(bad)}, "
          f"frames: {sum(len(r['repair_steps']) for r in bad)}{' (repaired)' if args.repair else ''}", flush=True)


if __name__ == "__main__":
    main()
