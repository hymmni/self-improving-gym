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


def _run_epoch(head, loader, device, num_bins, optimizer=None):
    """train_dstg._run_epoch와 같은 지표를 낸다: NLL(cross entropy)과 기댓값 MAE."""
    head.train(optimizer is not None)
    bin_idx = torch.arange(num_bins, device=device, dtype=torch.float32)
    total_nll, total_abs_err, n = 0.0, 0.0, 0

    for x, labels in loader:
        x, labels = x.to(device), labels.to(device).long()
        logits = head(x)
        loss = F.cross_entropy(logits, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            expected = (F.softmax(logits, dim=-1) * bin_idx).sum(dim=-1)
            total_abs_err += (expected - labels.float()).abs().sum().item()
            total_nll += loss.item() * labels.shape[0]
            n += labels.shape[0]

    return total_nll / n, total_abs_err / n


@hydra.main(config_path="../configs", config_name="train_dstg_vip", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    dataset = DinoFeatureWindows(cfg.cache_path, cfg.obs_horizon, fail_bin=cfg.get("fail_bin"),
                             label_horizon=cfg.get("label_horizon"))
    with h5py.File(cfg.cache_path, "r") as f:
        cache_meta = json.loads(f.attrs["meta"])
    logger.info(f"cache={cfg.cache_path} meta={cache_meta}")

    train_idx, val_idx, val_demos = dataset.split_indices(cfg.val_fraction, cfg.split_seed)
    n_val_demos = len(val_demos)
    n_train_demos = len(dataset.frames) - n_val_demos

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
    frame_mean, frame_std = dataset.compute_frame_stats(train_idx)
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

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )

    head = DinoStgHead(in_dim, num_bins, head_hidden=tuple(cfg.head_hidden)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.num_epochs):
        train_nll, train_mae = _run_epoch(head, train_loader, device, num_bins, optimizer=optimizer)
        if epoch % cfg.log_every == 0 or epoch == cfg.num_epochs - 1:
            msg = f"epoch {epoch} train_nll={train_nll:.4f} train_mae={train_mae:.3f}"
            logger.info(msg)
            print(msg, flush=True)

    val_nll, val_mae = _run_epoch(head, val_loader, device, num_bins, optimizer=None)
    logger.info(
        f"[val] nll={val_nll:.4f} mae={val_mae:.3f} (n={len(val_idx)} samples, {n_val_demos} demos)"
    )
    print({"val_nll": val_nll, "val_mae": val_mae, "num_bins": num_bins,
           "n_train_demos": n_train_demos, "n_val_demos": n_val_demos}, flush=True)

    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    save_checkpoint(cfg.out, head, {
        "in_dim": in_dim,
        "num_bins": num_bins,
        "head_hidden": list(cfg.head_hidden),
        "obs_horizon": cfg.obs_horizon,
        "fail_bin": cfg.get("fail_bin"),
        "label_horizon": cfg.get("label_horizon"),
        "frame_mean": frame_mean,
        "frame_std": frame_std,
        "cache_path": os.path.abspath(cfg.cache_path),
        "cache_meta": cache_meta,
        "epoch": cfg.num_epochs,
        "val_nll": val_nll,
        "val_mae": val_mae,
        "n_train_demos": n_train_demos,
        "n_val_demos": n_val_demos,
    })
    logger.info(f"saved: {cfg.out}")


if __name__ == "__main__":
    main()
