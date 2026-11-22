r"""STG 롤아웃 규모 실험(2026-11-20) 예측기 하나를 그 정책의 고정 평가 캐시(stg_scaling_data.py)로 채점한다.

주 지표(플랜 "플랜에서 정한 값과 근거" 2, 8~10):
- ① success_mae: 정책 단독 성공 전체 + 개입 성공 구간(개입 시작부터, 지연·즉시·타임아웃 버전 모두)의 프레임을 모은
  |E[d] − 남은 스텝| 평균. 라벨은 학습과 같게 마지막 bin으로 자른 값.
- ② false_progress: 실패 분기 가운데 mean d(분기 시작 8프레임) − mean d(분기 끝 8프레임) > 0인 비율. 8 = action chunk.
  성공한 분기는 빼고 야매(hacky)·진짜 성공(branch_success)으로 센다. W=1 값은 false_progress_w1.
- ② false_success(2026-11-22부터 그림의 ②): 실패 분기 가운데 끝 8프레임 평균 d ≤ FALSE_SUCCESS_D(성공 수준)인 비율.
  조금 내려간 분기(출렁임·그리퍼가 다가간 실제 진전)는 세지 않고, 처음부터 "거의 끝"으로 읽은 분기는 센다(사용자 지적).
  예측기를 다시 돌리지 않고 저장된 pred.npz로 다시 매기려면 --rescore <스윕 root>.
보조: 세트별 MAE·NLL, 실패 분기 프레임 vs 개입 성공 구간 프레임 AUROC, eval_dstg.reward_metrics(stride 1·8).

부트스트랩 단위 = 평가 에피소드(rollout/<원본 이름>: 개입 세 버전 + 분기 둘, solo/<이름>, demo/<이름>), 층 안 복원 추출.
비율 지표는 단위별 합계(unit_stats)에서 다시 계산하므로 여러 예측기의 구간도 같은 함수로 낸다(bootstrap_ci).

    /opt/venvs/torch/bin/python -m square_assembly.scripts.eval_stg_scaling --predictor <predictor.pt> \
        --eval-cache outputs/stg_rollout_scaling_20261120/caches/random30_eval.vip.h5 --out <폴더>
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

SETS = ("policy_success", "intv_delay", "intv_immediate", "intv_timeout", "demo")
INTV = ("intv_delay", "intv_immediate", "intv_timeout")
BRANCHES = ("branch_t0", "branch_t_press")
WINDOW = 8   # action_horizon — 거짓 진전의 시작·끝 평균 창
FALSE_SUCCESS_D = 50   # 성공 수준 — 성공 50스텝 전 예측 d가 34~47(random30·turn30, 2026-11-22 확인)


def _ratio(sets, kind="abs"):
    return [f"{kind}/{s}" for s in sets], [f"n/{s}" for s in sets]


RATIOS = {"success_mae": _ratio(("policy_success",) + INTV),
          "false_progress": (["fp"], ["nb"]), "false_progress_w1": (["fp_w1"], ["nb"]),
          "mae/intv": _ratio(INTV), "nll/success": _ratio(("policy_success",) + INTV, "nll"),
          **{f"mae/{s}": _ratio((s,)) for s in SETS}, **{f"nll/{s}": _ratio((s,), "nll") for s in SETS},
          **{f"false_progress/{b}": ([f"fp_{b}"], [f"nb_{b}"]) for b in BRANCHES},
          "false_success": (["fs"], ["nb"]), **{f"false_success/{b}": ([f"fs_{b}"], [f"nb_{b}"]) for b in BRANCHES}}


def _unit(info):
    prefix = {"demo": "demo", "policy_success": "solo"}.get(info["role"], "rollout")
    return f"{prefix}/{info['source_name']}"


def _episode(d, ep, t, name, *more):
    m = ep == name
    order = np.argsort(t[m], kind="mergesort")
    return (t[m][order], d[m][order]) + tuple(a[m][order] for a in more)


def unit_stats(d, nll, label, ep, t, episodes, window=WINDOW):
    """평가 에피소드 단위의 합계: abs|n|nll/<set>, 실패 분기 fp·fp_w1·nb(+분기 종류별), 성공 분기 hacky·branch_success."""
    stats = defaultdict(lambda: defaultdict(float))
    for name, info in episodes.items():
        tt, dd, ll, nn = _episode(d, ep, t, name, label, nll)
        s, role, keep = stats[_unit(info)], info["role"], tt >= int(info["score_from"])
        if role in SETS:
            s[f"abs/{role}"] += float(np.abs(dd[keep] - ll[keep]).sum())
            s[f"n/{role}"] += int(keep.sum())
            s[f"nll/{role}"] += float(nn[keep].sum())
        elif role in BRANCHES:
            if info["is_success"]:
                s["hacky" if info["hacky"] else "branch_success"] += 1
                continue
            seg = dd[keep]
            fp = {w: float(seg[:min(w, len(seg))].mean() - seg[-min(w, len(seg)):].mean() > 0) for w in (window, 1)}
            s["fp"] += fp[window]
            s[f"fp_{role}"] += fp[window]
            s["fp_w1"] += fp[1]
            fs = float(seg[-min(window, len(seg)):].mean() <= FALSE_SUCCESS_D)
            s["fs"] += fs
            s[f"fs_{role}"] += fs
            s["nb"] += 1
            s[f"nb_{role}"] += 1
        else:
            raise ValueError(f"{name}: unknown role {role}")
    return {u: dict(v) for u, v in stats.items()}


def _value(totals, num, den):
    den_v = sum(totals.get(k, 0.0) for k in den)
    return sum(totals.get(k, 0.0) for k in num) / den_v if den_v else None


def ratio_metrics(stats):
    totals = defaultdict(float)
    for s in stats.values():
        for k, v in s.items():
            totals[k] += v
    return {name: _value(totals, num, den) for name, (num, den) in RATIOS.items()}


def bootstrap_values(stats_list, n_boot=1000, seed=0):
    """예측기마다의 unit_stats 목록 -> {지표: (n_boot, 예측기 수) 배열}(분모 0이면 nan). 층(rollout/solo/demo) 안에서
    단위를 복원 추출하고 같은 재추출을 모든 예측기에 적용한다 — 짝지은 차이도 이 배열을 양쪽에 같이 쓴다."""
    units = sorted(set().union(*stats_list))
    keys = sorted({k for num, den in RATIOS.values() for k in num + den})
    x = np.array([[[st.get(u, {}).get(k, 0.0) for k in keys] for u in units] for st in stats_list])   # (P, U, K)
    strata = defaultdict(list)
    for i, u in enumerate(units):
        strata[u.split("/")[0]].append(i)
    rng = np.random.RandomState(seed)
    weights = np.zeros((n_boot, len(units)))
    for b in range(n_boot):
        for idx in strata.values():
            np.add.at(weights[b], rng.choice(idx, size=len(idx), replace=True), 1)
    sums = np.einsum("bu,puk->bpk", weights, x)
    col = {k: i for i, k in enumerate(keys)}
    out = {}
    for name, (num, den) in RATIOS.items():
        n_ = sums[..., [col[k] for k in num]].sum(-1)
        d_ = sums[..., [col[k] for k in den]].sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(d_ > 0, n_ / np.where(d_ > 0, d_, 1), np.nan)
    return out


def bootstrap_ci(stats_list, n_boot=1000, seed=0):
    """예측기 평균의 [2.5%, 97.5%] (bootstrap_values의 같은 재추출). 분모가 늘 0인 지표는 None."""
    out = {}
    for name, vals in bootstrap_values(stats_list, n_boot, seed).items():
        if np.isnan(vals).all():
            out[name] = None
            continue
        with np.errstate(all="ignore"):
            mean = np.nanmean(vals, axis=1)
        out[name] = [float(v) for v in np.nanpercentile(mean, [2.5, 97.5])]
    return out


def frame_metrics(d, label, ep, t, episodes):
    """점추정만 내는 보조 지표: AUROC(실패 분기 프레임 > 개입 성공 구간 프레임), 세트별 보상 품질(stride 1·8)."""
    from square_assembly.scripts.eval_dstg import auroc, reward_metrics

    scored = np.zeros(len(d), bool)
    role = np.empty(len(d), object)
    for name, info in episodes.items():
        m = ep == name
        scored[m] = t[m] >= int(info["score_from"])
        role[m] = "failed_branch" if info["role"] in BRANCHES and not info["is_success"] else info["role"]
    pos, neg = scored & (role == "failed_branch"), scored & np.isin(role, INTV)
    out = {"auroc_fail_vs_intv": auroc(d[pos], d[neg]), "reward": {}}
    for group, roles in {"policy_success": ("policy_success",), "intv": INTV, "demo": ("demo",)}.items():
        m = scored & np.isin(role, roles)
        if m.any():
            out["reward"][group] = {f"k{k}": {key: v for key, v in reward_metrics(d[m], label[m], ep[m], t[m], stride=k)
                                              .items() if key in ("reward_sign_acc", "spearman", "n_episodes")}
                                    for k in (1, 8)}
    return out


def read_episodes(cache_path):
    import h5py

    with h5py.File(cache_path, "r") as f:
        return {n: {"role": str(g.attrs["role"]), "source_name": str(g.attrs["source_name"]),
                    "score_from": int(g.attrs["score_from"]), "is_success": bool(g.attrs["is_success"]),
                    "hacky": bool(g.attrs.get("hacky", False))} for n, g in f["data"].items()}


def predict(predictor, eval_cache, device, batch_size=4096):
    """-> (d, nll, label, ep, t, ckpt). 라벨은 마지막 bin으로 자른다. 실패 에피소드 라벨(-1)의 nll은 의미 없다."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from square_assembly.datasets.dino_feature_dataset import DinoFeatureWindows
    from square_assembly.policies.diffusion.dino_stg_predictor import load_checkpoint

    head, ckpt = load_checkpoint(predictor, device)
    ds = DinoFeatureWindows(eval_cache, int(ckpt["obs_horizon"]), fail_bin=-1, obs_parts=ckpt.get("obs_parts", "all"))
    ds.apply_frame_stats(np.asarray(ckpt["frame_mean"], np.float32), np.asarray(ckpt["frame_std"], np.float32))
    ds.clamp_labels(int(ckpt["num_bins"]))
    values = ckpt.get("bin_values")
    bins = torch.tensor(values if values is not None else np.arange(int(ckpt["num_bins"])), dtype=torch.float32,
                        device=device)
    d, nll = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False):
            logits = head(x.to(device))
            d.append((logits.softmax(-1) * bins).sum(-1).cpu().numpy())
            nll.append(F.cross_entropy(logits, y.to(device).long().clamp(min=0), reduction="none").cpu().numpy())
    ep = np.array([n for n, _ in ds.samples])
    t = np.array([ti for _, ti in ds.samples])
    return np.concatenate(d), np.concatenate(nll), ds.labels().astype(float), ep, t, ckpt


def _counts(stats):
    counts = defaultdict(int)
    for s in stats.values():
        for k in ("nb", "hacky", "branch_success"):
            counts[k] += int(s.get(k, 0))
    for u in stats:
        counts[f"units/{u.split('/')[0]}"] += 1
    return dict(counts)


def rescore(root, n_boot=1000):
    """<root>/jobs.json의 job마다 저장된 pred.npz로 metrics.json의 units·point·ci·counts를 다시 매긴다.

    지표 정의가 바뀔 때 예측기를 다시 돌리지 않으려고 쓴다. 프레임 순서·라벨은 평가 캐시와 예측기의
    obs_horizon·num_bins로 정해지므로 그 조합마다 한 번만 만든다(predict와 같은 DinoFeatureWindows)."""
    import torch

    from square_assembly.datasets.dino_feature_dataset import DinoFeatureWindows

    jobs = json.load(open(os.path.join(root, "jobs.json")))["jobs"]
    frames, episodes, done = {}, {}, 0
    for job in jobs:
        out = os.path.join(root, "runs", job["id"])
        path = os.path.join(out, "metrics.json")
        if not os.path.exists(path):
            continue
        m = json.load(open(path))
        ckpt = torch.load(m["predictor"], map_location="cpu", weights_only=False)
        key = (m["eval_cache"], int(ckpt["obs_horizon"]), int(ckpt["num_bins"]))
        if key not in frames:
            ds = DinoFeatureWindows(m["eval_cache"], key[1], fail_bin=-1)
            ds.clamp_labels(key[2])
            frames[key] = (ds.labels().astype(float), np.array([n for n, _ in ds.samples]), np.array([t for _, t in ds.samples]))
        if m["eval_cache"] not in episodes:
            episodes[m["eval_cache"]] = read_episodes(m["eval_cache"])
        pred = np.load(os.path.join(out, "pred.npz"))
        label, ep, t = frames[key]
        assert len(pred["d"]) == len(label), (job["id"], len(pred["d"]), len(label))
        stats = unit_stats(pred["d"].astype(np.float64), pred["nll"].astype(np.float64), label, ep, t, episodes[m["eval_cache"]])
        m.update(point=ratio_metrics(stats), ci=bootstrap_ci([stats], n_boot=n_boot), counts=_counts(stats), units=stats)
        tmp = path + ".tmp"
        with open(tmp, "w") as stream:
            json.dump(m, stream, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
        done += 1
    print(f"rescored {done}/{len(jobs)} jobs in {root}", flush=True)


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predictor")
    ap.add_argument("--eval-cache")
    ap.add_argument("--out", help="metrics.json, pred.npz를 쓸 폴더")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rescore", metavar="ROOT", help="스윕 root의 모든 job을 저장된 pred.npz로 다시 매긴다(다른 인자 무시)")
    args = ap.parse_args()
    if args.rescore:
        return rescore(args.rescore, n_boot=args.n_boot)
    if not (args.predictor and args.eval_cache and args.out):
        ap.error("--predictor, --eval-cache, --out이 필요하다(--rescore가 아니면)")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    d, nll, label, ep, t, ckpt = predict(args.predictor, args.eval_cache, device)
    episodes = read_episodes(args.eval_cache)
    stats = unit_stats(d, nll, label, ep, t, episodes)
    counts = _counts(stats)
    metrics = {"predictor": args.predictor, "eval_cache": args.eval_cache, "window": WINDOW,
               "point": ratio_metrics(stats), "ci": bootstrap_ci([stats], n_boot=args.n_boot),
               "frame": frame_metrics(d, label, ep, t, episodes), "counts": dict(counts), "units": stats,
               **{k: ckpt.get(k) for k in ("n_train_samples", "n_train_demos", "num_steps", "obs_parts", "seed",
                                           "num_bins")}}
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "pred.npz"), d=d.astype(np.float16), nll=nll.astype(np.float16))
    tmp = os.path.join(args.out, "metrics.json.tmp")
    with open(tmp, "w") as stream:
        json.dump(metrics, stream, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(args.out, "metrics.json"))   # 다 쓴 뒤에만 생긴다 — 스윕은 이 파일로 완료를 판단
    p = metrics["point"]
    print(f"success_mae={p['success_mae']:.2f} false_progress={p['false_progress']:.3f} false_success={p['false_success']:.3f} "
          f"demo_mae={p['mae/demo']:.2f} n_failed_branches={counts['nb']}", flush=True)


if __name__ == "__main__":
    main()
