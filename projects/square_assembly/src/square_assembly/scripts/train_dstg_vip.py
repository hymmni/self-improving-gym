"""VIP 특징 캐시(cache_vip_feats.py) 위의 STG 예측기 학습 진입점.

train_dstg_dino.py와 학습 루프·평가 지표·분할 규칙이 완전히 같다 — 데이터셋
(DinoFeatureWindows)과 헤드(DinoStgHead)가 인코더를 모르고 캐시의 feat/lowdim
포맷에만 의존하므로 그대로 재사용한다(이름은 DINO 실험 때 지어졌지만 인코더에
종속된 코드가 아니다). 이 파일이 새로 필요한 건 캐시 경로와 하이퍼파라미터 기본값뿐.

비교가 목적이므로 하이퍼파라미터(bin 공간, 분할 시드, lr, epochs, head 크기)는
DINO/ResNet 기준선과 같은 값을 기본값으로 둔다 — configs/train_dstg_vip.yaml 참고.

사용:
    python -m square_assembly.scripts.train_dstg_vip \
        cache_path=data/square_scale3_0_seed1.vip.h5 \
        out=outputs/dstg_vip/square_scale3_0_seed1/predictor.pt
"""

import json
import logging
import math
import os

import h5py
import numpy as np
import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from square_assembly.datasets.dino_feature_dataset import DinoFeatureWindows
from square_assembly.policies.diffusion.dino_stg_predictor import DinoStgHead, save_checkpoint

logger = logging.getLogger(__name__)


def _gaussian_targets(labels, num_bins, sigma):
    """HL-Gauss: integrate a truncated Gaussian over integer-centered bins.

    Formula: Farebrother et al. (2024), arxiv.org/abs/2403.03950, Appendix A.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("gauss_sigma must be finite and positive")
    edges = torch.arange(num_bins + 1, device=labels.device, dtype=torch.float32) - 0.5
    cdf = torch.erf((edges - labels.float().unsqueeze(-1)) / (math.sqrt(2) * sigma))
    mass = cdf[:, 1:] - cdf[:, :-1]
    return mass / mass.sum(-1, keepdim=True)


class _PairedWindows(torch.utils.data.Dataset):
    """Keep every classification sample; supervise pairs only when both endpoints are kept."""

    def __init__(self, dataset, indices, stride):
        if stride < 1:
            raise ValueError("pair_stride must be positive")
        self.dataset, self.indices = dataset, indices
        lookup = {dataset.samples[i]: i for i in indices}
        self.pairs = []
        for i in indices:
            name, t = dataset.samples[i]
            self.pairs.append(lookup.get((name, t + stride)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.dataset[self.indices[i]]
        j = self.pairs[i]
        xp, yp = self.dataset[j] if j is not None else (x, y)
        return x, y, xp, yp, j is not None


def _pair_loss(logits, paired_logits, labels, pair_labels, valid, stride):
    """Match the signed anchor-label difference per frame, including backward progress."""
    bins = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    d = (logits.softmax(-1) * bins).sum(-1)
    dp = (paired_logits.softmax(-1) * bins).sum(-1)
    error = ((d - dp) - (labels.float() - pair_labels.float())) / stride
    if not valid.any():
        return logits.sum() * 0.0
    return F.smooth_l1_loss(error[valid], torch.zeros_like(error[valid]))


def _mask_modalities(x, obs_horizon, feat_dim, probability):
    """Drop vision or proprioception per sample; zero is the standardized train mean."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("modality_dropout must be between 0 and 1")
    if not probability:
        return x
    frames = x.clone().reshape(len(x), obs_horizon, -1)
    if not 0 < feat_dim < frames.shape[-1]:
        raise ValueError("modality dropout requires both feature and lowdim inputs")
    draw = torch.rand(len(x), device=x.device)
    frames[draw < probability / 2, :, :feat_dim] = 0
    frames[(draw >= probability / 2) & (draw < probability), :, feat_dim:] = 0
    return frames.reshape_as(x)


def _run_epoch(head, loader, device, num_bins, optimizer=None,
               gauss_sigma=0.0, pair_weight=0.0, pair_stride=8,
               modality_dropout=0.0, obs_horizon=2, feat_dim=0):
    """train_dstg._run_epoch와 같은 지표를 낸다: NLL(cross entropy)과 기댓값 MAE."""
    head.train(optimizer is not None)
    bin_idx = torch.arange(num_bins, device=device, dtype=torch.float32)
    total_nll, total_abs_err, n = 0.0, 0.0, 0

    for batch in loader:
        x, labels = batch[:2]
        x, labels = x.to(device), labels.to(device).long()
        if optimizer is not None and modality_dropout:
            if pair_weight:
                # Concatenate histories so both pair endpoints receive the same modality mask.
                both = _mask_modalities(torch.cat([x, batch[2].to(device)], dim=-1),
                                        obs_horizon * 2, feat_dim, modality_dropout)
                x, xp = both.chunk(2, dim=-1)
            else:
                x = _mask_modalities(x, obs_horizon, feat_dim, modality_dropout)
        logits = head(x)
        nll = F.cross_entropy(logits, labels)
        loss = (F.cross_entropy(logits, _gaussian_targets(labels, num_bins, gauss_sigma))
                if gauss_sigma else nll)
        if optimizer is not None and pair_weight:
            if not modality_dropout:
                xp = batch[2].to(device)
            yp, valid = (v.to(device) for v in batch[3:])
            loss = loss + pair_weight * _pair_loss(
                logits, head(xp), labels, yp, valid, pair_stride)
        if not torch.isfinite(loss):
            raise ValueError("non-finite STG loss")

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            expected = (F.softmax(logits, dim=-1) * bin_idx).sum(dim=-1)
            total_abs_err += (expected - labels.float()).abs().sum().item()
            total_nll += nll.item() * labels.shape[0]
            n += labels.shape[0]

    return total_nll / n, total_abs_err / n


def _collect_curve(head, dataset, indices, device, num_bins, batch_size):
    """Evaluate epoch-end weights without consuming the training shuffle RNG."""
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    was_training = head.training
    head.eval()
    distances, nlls, labels = [], [], []
    bins = torch.arange(num_bins, device=device, dtype=torch.float32)
    try:
        with torch.no_grad():
            for x, y in loader:
                logits, y = head(x.to(device)), y.to(device).long()
                distances.append((logits.softmax(-1) * bins).sum(-1).cpu().numpy())
                nlls.append(F.cross_entropy(logits, y, reduction="none").cpu().numpy())
                labels.append(y.cpu().numpy())
    finally:
        head.train(was_training)
    return (np.concatenate(distances), np.concatenate(nlls), np.concatenate(labels),
            np.array([dataset.samples[i][0] for i in indices]),
            np.array([dataset.samples[i][1] for i in indices]))


def _curve_metrics(arrays, mask=None):
    from square_assembly.scripts.eval_dstg import reward_metrics

    d, nll, label, demo, t = arrays
    if mask is not None:
        d, nll, label, demo, t = (a[mask] for a in arrays)
    return {"n": len(d), "mae": float(np.abs(d - label).mean()) if len(d) else None,
            "nll": float(nll.mean()) if len(d) else None,
            "k8": reward_metrics(d, label, demo, t, stride=8).get("reward_sign_acc")}


def _log_curve(run, row):
    payload = {"epoch": row["epoch"]}
    payload.update({f"{region}/{metric}": value for region, values in row.items()
                    if isinstance(values, dict) for metric, value in values.items() if value is not None})
    run.log(payload, step=row["epoch"])


@hydra.main(config_path="../configs", config_name="train_dstg_vip", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    gauss_sigma = float(cfg.get("gauss_sigma", 0.0))
    pair_weight = float(cfg.get("pair_weight", 0.0))
    pair_stride = int(cfg.get("pair_stride", 8))
    modality_dropout = float(cfg.get("modality_dropout", 0.0))
    success_tail_only = bool(cfg.get("success_tail_only", False))
    record_curves = bool(cfg.get("record_curves", False))
    if cfg.get("use_wandb", False) and not record_curves:
        raise ValueError("use_wandb requires record_curves=true")
    if record_curves and not success_tail_only:
        raise ValueError("record_curves currently requires success_tail_only for comparable train/val regions")
    if success_tail_only and (cfg.get("label_horizon") is not None
                              or cfg.get("preintv", "none") != "none"
                              or cfg.get("train_modes") or int(cfg.get("preintv_weight") or 1) != 1):
        raise ValueError("success_tail_only requires raw countdown, preintv=none, all train_modes, weight=1")
    if not math.isfinite(modality_dropout) or not 0 <= modality_dropout <= 1:
        raise ValueError("modality_dropout must be between 0 and 1")
    if modality_dropout and cfg.get("obs_parts", "all") != "all":
        raise ValueError("modality dropout requires obs_parts=all")
    if not math.isfinite(gauss_sigma) or gauss_sigma < 0:
        raise ValueError("gauss_sigma must be finite and nonnegative")
    if not math.isfinite(pair_weight) or pair_weight < 0 or pair_stride < 1:
        raise ValueError("pair_weight must be nonnegative; pair_stride must be positive")
    if (gauss_sigma or pair_weight) and cfg.get("fail_bin") is not None:
        raise ValueError("ordinal losses cannot smooth a separate failure class")

    dataset = DinoFeatureWindows(cfg.cache_path, cfg.obs_horizon, fail_bin=cfg.get("fail_bin"),
                             label_horizon=cfg.get("label_horizon"),
                             mode_hdf5=cfg.get("mode_hdf5"),
                             preintv=cfg.get("preintv", "none"),
                             preintv_len=cfg.get("preintv_len"),
                             obs_parts=cfg.get("obs_parts", "all"), strict_modes=success_tail_only)
    with h5py.File(cfg.cache_path, "r") as f:
        cache_meta = json.loads(f.attrs["meta"])
        feat_dim = next(iter(f["data"].values()))["feat"].shape[-1]
    logger.info(f"cache={cfg.cache_path} meta={cache_meta}")

    train_idx, val_idx, val_demos = dataset.split_indices(cfg.val_fraction, cfg.split_seed)
    n_val_demos = len(val_demos)
    n_train_demos = len(dataset.frames) - n_val_demos

    if success_tail_only:
        before = len(train_idx)
        train_idx = dataset.retain_success_tails(train_idx)
        n_train_demos = len({dataset.samples[i][0] for i in train_idx})
        logger.info(f"success_tail_only: train samples {before} -> {len(train_idx)}")

    if cfg.get("train_modes"):
        from square_assembly.datasets.labels import LABEL_DEMO, LABEL_INTV, LABEL_PREINTV, LABEL_ROLLOUT
        keep = {"demo": LABEL_DEMO, "rollout": LABEL_ROLLOUT, "intv": LABEL_INTV, "preintv": LABEL_PREINTV}
        keep_ids = {keep[m] for m in cfg.train_modes}
        sample_modes = dataset.sample_modes()
        before = len(train_idx)
        train_idx = [i for i in train_idx if sample_modes[i] in keep_ids]
        logger.info(f"train_modes={list(cfg.train_modes)}: train 샘플 {before} -> {len(train_idx)}")

    if cfg.get("preintv") == "drop":
        drop = set(dataset.preintv_indices())
        before = len(train_idx)
        train_idx = [i for i in train_idx if i not in drop]
        logger.info(f"preintv=drop: train 샘플 {before} -> {len(train_idx)}")

    # PREINTV 오버샘플링 — train_dstg.py와 같다(인덱스 복제, val은 그대로).
    w = int(cfg.get("preintv_weight") or 1)
    if w > 1:
        pre = set(dataset.preintv_indices())
        extra = [i for i in train_idx if i in pre] * (w - 1)
        train_idx = train_idx + extra
        logger.info(f"preintv_weight={w}: train 샘플 +{len(extra)} -> {len(train_idx)}")

    # train_demos_limit: val 데모는 그대로 두고 train 데모만 줄인다 — 같은 held-out 위에서
    # "데이터가 더 있으면 나아지는가"를 재는 규모 곡선용.
    limit = cfg.get("train_demos_limit")
    if limit and int(limit) < n_train_demos:
        pool = sorted(set(dataset.frames) - set(val_demos))
        rng = np.random.RandomState(cfg.split_seed)
        keep = {pool[i] for i in rng.permutation(len(pool))[: int(limit)]}
        train_idx = [i for i in train_idx if dataset.samples[i][0] in keep]
        n_train_demos = len(keep)
    logger.info(
        f"episode split: train={n_train_demos} demos/{len(train_idx)} samples, "
        f"val={n_val_demos} demos/{len(val_idx)} samples"
    )

    # 표준화 통계는 train 데모에서만 — val 누출 방지.
    frame_mean, frame_std = dataset.compute_frame_stats(train_idx, sample_only=success_tail_only)
    dataset.apply_frame_stats(frame_mean, frame_std)

    labels = dataset.labels()
    max_label = int(labels.max())
    if cfg.get("num_bins_override"):
        num_bins = int(cfg.num_bins_override)
        if num_bins <= max_label:
            raise ValueError(
                f"num_bins_override({num_bins})가 실제 관측된 최대 라벨({max_label})보다 "
                f"작거나 같다 — 라벨이 잘린다."
            )
    else:
        num_bins = max_label + 1
    in_dim = cfg.obs_horizon * dataset.frame_dim
    logger.info(
        f"dataset len={len(dataset)} in_dim={in_dim} num_bins={num_bins} (max label={max_label})"
    )

    train_data = (_PairedWindows(dataset, train_idx, pair_stride) if pair_weight
                  else Subset(dataset, train_idx))
    if pair_weight:
        logger.info(f"valid training pairs={sum(j is not None for j in train_data.pairs)}")
    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )

    wandb_run = None
    if cfg.get("use_wandb", False):
        import wandb

        os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
        wandb_config = {k: cfg.get(k) for k in ("seed", "split_seed", "num_epochs", "batch_size", "lr",
                       "weight_decay", "obs_horizon", "obs_parts", "label_horizon",
                       "preintv", "success_tail_only", "gauss_sigma", "pair_weight", "modality_dropout")}
        wandb_config["head_hidden"] = list(cfg.head_hidden)
        wandb_run = wandb.init(
            project=cfg.wandb_project, entity=cfg.get("wandb_entity"),
            name=cfg.get("wandb_run_name") or f"stg-success-tail-split{cfg.split_seed}",
            group=cfg.get("wandb_group"), mode=cfg.get("wandb_mode", "online"),
            dir=os.path.dirname(cfg.out), save_code=False,
            settings=wandb.Settings(disable_git=True),
            config=wandb_config,
        )
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("*", step_metric="epoch")
        wandb_run.config.update({"num_bins": num_bins, "n_train_samples": len(train_idx),
                                "n_val_samples": len(val_idx), "n_train_demos": n_train_demos,
                                "n_val_demos": n_val_demos})
        with open(os.path.join(os.path.dirname(cfg.out), "wandb_run.json"), "w") as stream:
            json.dump({"id": wandb_run.id, "url": wandb_run.url, "entity": wandb_run.entity,
                       "project": wandb_run.project, "group": wandb_run.group,
                       "mode": wandb_run.settings.mode, "dir": wandb_run.dir}, stream, indent=2)
    # SDK initialization must not alter model initialization or the training shuffle stream.
    torch.manual_seed(cfg.seed)
    head = DinoStgHead(in_dim, num_bins, head_hidden=tuple(cfg.head_hidden)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    curves, best_state, best_row = [], None, None
    if record_curves:
        os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
        val_starts, cohorts = {}, {}
        for name in val_demos:
            mode = dataset.modes[name]
            human = mode == 1
            onsets = np.flatnonzero(human & ~np.r_[False, human[:-1]])
            val_starts[name] = int(onsets[-1]) if len(onsets) else 0
            cohorts[name] = "intervention_tail" if human.any() else "expert" if (mode == -1).all() else "policy_success"

    for epoch in range(cfg.num_epochs):
        train_nll, train_mae = _run_epoch(
            head, train_loader, device, num_bins, optimizer=optimizer,
            gauss_sigma=gauss_sigma, pair_weight=pair_weight, pair_stride=pair_stride,
            modality_dropout=modality_dropout, obs_horizon=cfg.obs_horizon, feat_dim=feat_dim)
        if epoch % cfg.log_every == 0 or epoch == cfg.num_epochs - 1:
            msg = f"epoch {epoch} train_nll={train_nll:.4f} train_mae={train_mae:.3f}"
            logger.info(msg)
            print(msg, flush=True)
        if record_curves:
            train_arrays = _collect_curve(head, dataset, sorted(set(train_idx)), device, num_bins, cfg.batch_size)
            val_arrays = _collect_curve(head, dataset, val_idx, device, num_bins, cfg.batch_size)
            demo, t = val_arrays[3:]
            eligible = np.array([ti >= val_starts[n] for n, ti in zip(demo, t)])
            row = {"epoch": epoch + 1, "train": _curve_metrics(train_arrays),
                   "val_all": _curve_metrics(val_arrays),
                   "val_retained": _curve_metrics(val_arrays, eligible),
                   "val_excluded_prefix": _curve_metrics(val_arrays, ~eligible)}
            for cohort in ("expert", "policy_success", "intervention_tail"):
                mask = eligible & np.array([cohorts[n] == cohort for n in demo])
                row[f"val_{cohort}"] = _curve_metrics(val_arrays, mask)
            curves.append(row)
            if wandb_run is not None:
                _log_curve(wandb_run, row)
            if best_row is None or row["val_retained"]["mae"] < best_row["val_retained"]["mae"]:
                best_row = row
                best_state = {k: v.detach().cpu().clone() for k, v in head.head.state_dict().items()}
            curve_path = os.path.join(os.path.dirname(cfg.out), "learning_curve.json")
            with open(curve_path, "w") as stream:
                json.dump({"rows": curves, "best_epoch": best_row["epoch"],
                           "selection_metric": "val_retained.mae"}, stream, indent=2, allow_nan=False)
            print(f"curve epoch={epoch + 1} train_mae={row['train']['mae']:.3f} "
                  f"val_retained_mae={row['val_retained']['mae']:.3f} "
                  f"val_retained_k8={row['val_retained']['k8']}", flush=True)

    val_nll, val_mae = _run_epoch(head, val_loader, device, num_bins, optimizer=None)
    logger.info(
        f"[val] nll={val_nll:.4f} mae={val_mae:.3f} (n={len(val_idx)} samples, {n_val_demos} demos)"
    )
    print({"val_nll": val_nll, "val_mae": val_mae, "num_bins": num_bins,
           "n_train_demos": n_train_demos, "n_val_demos": n_val_demos}, flush=True)

    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    meta = {
        "in_dim": in_dim,
        "num_bins": num_bins,
        "head_hidden": list(cfg.head_hidden),
        "obs_horizon": cfg.obs_horizon,
        "fail_bin": cfg.get("fail_bin"),
        "label_horizon": cfg.get("label_horizon"),
        "gauss_sigma": gauss_sigma,
        "pair_weight": pair_weight,
        "pair_stride": pair_stride,
        "modality_dropout": modality_dropout,
        "seed": cfg.seed,
        "split_seed": cfg.split_seed,
        "train_modes": list(cfg.train_modes) if cfg.get("train_modes") else None,
        "success_tail_only": success_tail_only,
        "train_history_starts": {n: s for n, s in dataset.history_starts.items()
                                 if n in {dataset.samples[i][0] for i in train_idx}},
        "n_train_samples": len(train_idx),
        "n_val_samples": len(val_idx),
        "preintv": cfg.get("preintv", "none"),
        "preintv_weight": cfg.get("preintv_weight"),
        "preintv_len": cfg.get("preintv_len"),
        "obs_parts": cfg.get("obs_parts", "all"),
        "frame_mean": frame_mean,
        "frame_std": frame_std,
        "cache_path": os.path.abspath(cfg.cache_path),
        "cache_meta": cache_meta,
        "epoch": cfg.num_epochs,
        "val_nll": val_nll,
        "val_mae": val_mae,
        "n_train_demos": n_train_demos,
        "n_val_demos": n_val_demos,
        "record_curves": record_curves,
    }
    save_checkpoint(cfg.out, head, meta)
    if best_state is not None:
        best_meta = {**meta, "epoch": best_row["epoch"],
                     "val_nll": best_row["val_all"]["nll"], "val_mae": best_row["val_all"]["mae"],
                     "selection_metric": "val_retained.mae", "selection_value": best_row["val_retained"]["mae"]}
        head.head.load_state_dict(best_state)
        save_checkpoint(os.path.join(os.path.dirname(cfg.out), "best_predictor.pt"), head, best_meta)
    if wandb_run is not None:
        wandb_run.summary.update({"best_epoch": best_row["epoch"],
                                  "best_val_retained_mae": best_row["val_retained"]["mae"],
                                  "best_epoch_val_retained_k8": best_row["val_retained"]["k8"],
                                  "final_epoch": cfg.num_epochs})
        wandb_run.finish()
    logger.info(f"saved: {cfg.out}")


if __name__ == "__main__":
    main()
