r"""학습된 STG 예측기를 강화학습 없이 직접 평가한다.

**왜 MAE/NLL만으론 부족한가**: SI-EFM이 실제로 쓰는 건 d(o)의 절대값이 아니라 (1) 한 스텝
차분 `r_t = d(o_t) - d(o_{t+1})`(식 2)의 부호·일관성과 (2) 성공 판정 임계값(식 3)이다.
평균오차가 작아도 프레임마다 d가 들쭉날쭉하면 차분은 부호가 뒤집히는 잡음이 되고, 그 보상으로
REINFORCE를 돌리면 정책이 망가진다 — MAE는 이 실패 모드를 못 본다. 그래서 여기서는 보상으로서의
품질을 직접 잰다(RL을 돌리는 건 최종 확인이지 측정 도구가 아니다 — 몇 시간이 걸리고, REINFORCE
분산 때문에 "정책이 안 늘었다"가 보상 탓인지 RL 하이퍼파라미터 탓인지 구분이 안 된다).

성공 데모에서 참 steps-to-go는 매 스텝 정확히 1씩 줄기 때문에 **이상적인 보상은 항상 +1**이다.
이 사실이 아래 지표들의 기준선이다.

| 지표 | 의미 | 이상값 |
|---|---|---|
| `mae` / `nll` | 학습 때와 같은 적합도(참고용) | 0 |
| `reward_sign_acc` | r_t > 0인 transition 비율 — 부호가 맞는가 | 1.0 |
| `reward_mae` | \|r_t − (이상적 스텝당 감소)\|의 평균 | 0 |
| `reward_snr` | 에피소드 내 mean(r)/std(r) 중앙값 — 신호 대 잡음 | 클수록 좋음 |
| `spearman` | 에피소드 내 d와 참 라벨의 순위상관 중앙값 | 1.0 |
| `preintv_auroc` | 사람이 곧 개입한 상태(PREINTV)를 평범한 ROLLOUT보다 나쁘게 보는가 | 참라벨 AUROC에 근접 |
| `success_f1` | d ≤ s로 성공 시점을 잡을 때의 최고 F1 | 1.0 |

`preintv_auroc`는 참 라벨로 계산한 값(`preintv_auroc_oracle`)을 같이 낸다 — PREINTV 프레임은
원래 끝까지 남은 스텝이 많으므로 라벨만으로도 어느 정도 갈리기 때문이다. 예측값이 그 상한에
얼마나 근접하는지로 읽어야 하고, 단독 절대값으로 읽으면 안 된다.

사용:
    python -m square_assembly.scripts.eval_dstg \
        --predictor outputs/dstg/square_r0/predictor.pt \
        --hdf5 data/square_demo50_mouse50.hdf5
    python -m square_assembly.scripts.eval_dstg \
        --predictor outputs/dstg_vip/square_r0/predictor.pt \
        --cache data/square_demo50_mouse50.vip.h5
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

_MIN_EPISODE_LEN = 3  # 순위상관·차분을 낼 수 있는 최소 길이


def auroc(pos, neg):
    """pos 값이 neg보다 클 확률(Mann-Whitney U). 한쪽이 비면 nan."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_f1(score, label):
    """score <= s로 양성을 예측할 때 F1이 최대인 s와 그 지표들."""
    label = np.asarray(label, bool)
    if not label.any():
        return {"success_f1": float("nan"), "success_precision": float("nan"),
                "success_recall": float("nan"), "success_threshold": float("nan")}
    best = {"success_f1": -1.0}
    for s in np.linspace(float(score.min()), float(score.max()), 400):
        pred = score <= s
        tp = float((pred & label).sum())
        prec = tp / max(pred.sum(), 1)
        rec = tp / label.sum()
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        if f1 > best["success_f1"]:
            best = {"success_f1": f1, "success_precision": prec, "success_recall": rec,
                    "success_threshold": float(s)}
    return best


_STRIDES = (1, 2, 5, 10, 20)


def reward_metrics(d, label, demo, t, stride=1):
    """에피소드 안에서 시간순으로 이어붙여 r_t = d_t - d_{t+stride}의 품질을 잰다.

    stride>1은 인접 프레임이 거의 같아 보이는 문제를 피한다 — 신호(참 감소량)는 stride에
    비례해 커지는데 d의 추정 잡음은 그대로라, 잡음에 묻힌 보상이 stride를 키우면 살아나는지
    여기서 바로 보인다.
    """
    signs, errs, snrs, rhos = [], [], [], []
    for name in np.unique(demo):
        m = demo == name
        order = np.argsort(t[m], kind="mergesort")
        d_ep, label_ep = d[m][order], label[m][order]
        if len(d_ep) < max(_MIN_EPISODE_LEN, stride + 1):
            continue
        # 이상적인 스텝당 보상은 참 라벨의 차분 그 자체다 — 라벨이 남은 스텝 수면 항상 +1,
        # 남은 비율 x H로 정규화했으면 H/(L-1)이다. 하드코딩하면 정규화 라벨에서 틀린 값을 잰다.
        r = d_ep[:-stride] - d_ep[stride:]
        ideal = label_ep[:-stride] - label_ep[stride:]
        # 참 변화량이 0인 transition(라벨 해상도가 낮아 인접 프레임이 같은 칸에 떨어질 때)은
        # 부호를 물을 대상이 아니다 — 연속값인 d가 정확히 0 차분을 낼 수는 없으므로, 세면
        # 맞히는 게 불가능한 문제를 정확도에 섞는 꼴이 된다. 라벨이 '남은 스텝 수'면 ideal은
        # 항상 1이라 이 마스크는 전부 True이고 예전 계산과 같은 값이 나온다.
        nz = ideal != 0
        signs.append((r[nz] > 0) == (ideal[nz] > 0))
        errs.append(np.abs(r - ideal))
        snrs.append(r.mean() / r.std() if r.std() > 1e-9 else np.inf)
        rhos.append(spearmanr(d_ep, label_ep).statistic)
    signs = [x for x in signs if len(x)]
    if not signs:
        return {}
    return {
        "reward_sign_acc": float(np.concatenate(signs).mean()),
        "reward_mae": float(np.concatenate(errs).mean()),
        "reward_snr": float(np.median(snrs)),
        "spearman": float(np.median(rhos)),
        "n_episodes": len(signs),
    }


def evaluate(d, nll, label, demo, t, mode):
    """수집된 배열 -> 지표 dict. 순수 numpy라 시뮬레이터·GPU 없이 테스트할 수 있다."""
    out = {"n_samples": int(len(d)),
           "mae": float(np.abs(d - label).mean()),
           "nll": float(np.mean(nll))}
    out.update(reward_metrics(d, label, demo, t))
    # 보상을 몇 스텝 간격으로 읽느냐에 따라 품질이 어떻게 변하는지 — stride 1이 잡음이어도
    # stride가 커지며 살아나면 예측기가 아니라 "1스텝 차분"이 문제라는 뜻이다.
    out["by_stride"] = {
        str(k): {n: v for n, v in reward_metrics(d, label, demo, t, stride=k).items()
                 if n in ("reward_sign_acc", "reward_snr")}
        for k in _STRIDES
    }
    out.update(best_f1(d, label == 0))
    if mode is not None:
        from square_assembly.datasets.labels import LABEL_PREINTV, LABEL_ROLLOUT
        pre, roll = mode == LABEL_PREINTV, mode == LABEL_ROLLOUT
        out["preintv_auroc"] = auroc(d[pre], d[roll])
        out["preintv_auroc_oracle"] = auroc(label[pre], label[roll])
        out["n_preintv"] = int(pre.sum())
    return out


def _collect_baseline(predictor_path, hdf5_path, device, batch_size, val_fraction, split_seed):
    """기준선(정책 ResNet 특징) 예측기 — RobomimicSequenceDataset 경로."""
    import os

    from square_assembly.datasets.normalization import MinMaxNormalizer, load_stats
    from square_assembly.datasets.robomimic_dataset import RobomimicSequenceDataset
    from square_assembly.policies.diffusion.dstg_reward import DstgReward
    from square_assembly.scripts.train_dstg import _episode_split
    from square_assembly.utils.checkpoints import load_run_config
    from square_assembly.utils.task_utils import is_image_task
    from torch.utils.data import DataLoader, Subset

    reward = DstgReward(predictor_path, device=device)
    task_cfg = load_run_config(reward.policy_ckpt_path).task
    stats = load_stats(os.path.join(os.path.dirname(reward.policy_ckpt_path),
                                    "normalization_stats.json"))
    dataset = RobomimicSequenceDataset(
        hdf5_path=hdf5_path, obs_keys=reward.obs_keys, obs_horizon=reward.obs_horizon,
        pred_horizon=1, normalizer=MinMaxNormalizer(stats),
        rgb_keys=task_cfg.rgb_keys if is_image_task(task_cfg) else (),
        hdf5_cache_mode="low_dim",
    )
    _, val_idx, _, n_val_demos = _episode_split(dataset, val_fraction, split_seed)
    labels = dataset.get_time_to_success()[val_idx]
    modes = dataset.get_action_mode_first_frame()[val_idx]
    pairs = [dataset._demo_id_and_index_in_demo(i) for i in val_idx]
    print(f"val {n_val_demos} demos / {len(val_idx)} samples", flush=True)

    d_all, nll_all = [], []
    # RobomimicSequenceDataset은 time_to_success를 안 실어준다(train_dstg.py의 _LabeledWindow가
    # 얹는 것) — shuffle=False라 배치 순서가 val_idx 순서와 같으므로 위에서 뽑아둔 labels를 잘라 쓴다.
    label_t, offset = torch.from_numpy(np.asarray(labels)).long(), 0
    loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for b_i, batch in enumerate(loader):
            obs = {k: v.to(device) for k, v in batch["obs"].items()}
            logits = reward.predictor(obs)
            y = label_t[offset:offset + logits.shape[0]].to(device)
            offset += logits.shape[0]
            d_all.append((F.softmax(logits, -1) * reward.bin_vals).sum(-1).cpu().numpy())
            nll_all.append(F.cross_entropy(logits, y, reduction="none").cpu().numpy())
            if b_i % 20 == 0:
                print(f"  배치 {b_i + 1}/{len(loader)}", flush=True)
    return (np.concatenate(d_all), np.concatenate(nll_all), labels,
            np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]), modes)


def _collect_cached(predictor_path, cache_path, source_hdf5, device, batch_size,
                    val_fraction, split_seed):
    """DINO/VIP 예측기 — 미리 구운 특징 캐시 경로(DinoFeatureWindows)."""
    import h5py
    from square_assembly.datasets.dino_feature_dataset import DinoFeatureWindows
    from square_assembly.policies.diffusion.dino_stg_predictor import load_checkpoint
    from torch.utils.data import DataLoader, Subset

    head, ckpt = load_checkpoint(predictor_path, device)
    dataset = DinoFeatureWindows(cache_path, int(ckpt["obs_horizon"]), fail_bin=ckpt.get("fail_bin"),
                                 label_horizon=ckpt.get("label_horizon"))
    dataset.apply_frame_stats(np.asarray(ckpt["frame_mean"], np.float32),
                              np.asarray(ckpt["frame_std"], np.float32))
    train_idx, val_idx, val_demos = dataset.split_indices(val_fraction, split_seed)
    print(f"val {len(val_demos)} demos / {len(val_idx)} samples", flush=True)

    labels = dataset.labels()[val_idx]
    demo = np.array([dataset.samples[i][0] for i in val_idx])
    t = np.array([dataset.samples[i][1] for i in val_idx])
    modes = None
    if source_hdf5:  # 캐시엔 action_mode가 없어서 원본 hdf5에서 프레임별로 읽어온다
        with h5py.File(source_hdf5, "r") as f:
            per_demo = {n: np.asarray(f["data"][n]["action_mode"]) for n in set(demo.tolist())
                        if "action_mode" in f["data"][n]}
        if per_demo:
            modes = np.array([per_demo[n][min(ti, len(per_demo[n]) - 1)] if n in per_demo else 0
                              for n, ti in zip(demo, t)])

    bins = torch.arange(int(ckpt["num_bins"]), dtype=torch.float32, device=device)
    d_all, nll_all = [], []
    loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            logits = head(x.to(device))
            y = y.to(device).long()
            d_all.append((F.softmax(logits, -1) * bins).sum(-1).cpu().numpy())
            nll_all.append(F.cross_entropy(logits, y, reduction="none").cpu().numpy())
    return np.concatenate(d_all), np.concatenate(nll_all), labels, demo, t, modes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predictor", required=True, help="train_dstg*.py가 저장한 predictor.pt")
    ap.add_argument("--hdf5", default=None, help="기준선 예측기용 학습 데이터셋")
    ap.add_argument("--cache", default=None, help="DINO/VIP 예측기용 특징 캐시(.h5)")
    ap.add_argument("--source-hdf5", default=None,
                    help="[--cache일 때] action_mode를 읽어올 원본 hdf5(PREINTV 지표용)")
    ap.add_argument("--val-fraction", type=float, default=0.2, help="학습 때와 같은 값이어야 held-out이다")
    ap.add_argument("--split-seed", type=int, default=0, help="학습 때와 같은 값이어야 held-out이다")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="지표를 JSON으로 저장할 경로")
    args = ap.parse_args()

    if bool(args.hdf5) == bool(args.cache):
        ap.error("--hdf5(기준선) 또는 --cache(DINO/VIP) 중 정확히 하나를 줘야 한다")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.hdf5:
        arrays = _collect_baseline(args.predictor, args.hdf5, device, args.batch_size,
                                   args.val_fraction, args.split_seed)
    else:
        arrays = _collect_cached(args.predictor, args.cache, args.source_hdf5, device,
                                 args.batch_size, args.val_fraction, args.split_seed)

    metrics = evaluate(*arrays)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
