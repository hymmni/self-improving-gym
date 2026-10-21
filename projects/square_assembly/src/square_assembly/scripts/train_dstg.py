"""DstgPredictor(d(o,g) := E[steps-to-go | o]) 학습 진입점 — square task, PH(성공만) 데모.

scripts/train.py 패턴을 따르되(hydra 진입점), task/policy는 hydra defaults가 아니라
policy_ckpt 옆 run_config.yaml에서 직접 복원한다(그 체크포인트가 실제로 학습된 설정이
유일한 출처 — scripts/eval.py의 _apply_run_config와 동일한 이유).

hdf5_path는 명시적으로 오버라이드해야 한다 — run_config.yaml에 박힌 학습 당시 상대경로는
이 머신에서 그대로 안 열린다(phases/5-mani-sim-ddpo/step1.md 참고).

robomimic PH 데모는 전부 성공 시연이라 이 스크립트는 succ 버전만 만든다 — fail-aware
버전(실패 bin 포함)은 실패 롤아웃 데이터가 쌓인 뒤의 다음 phase로 미룬다.

사용:
    python -m square_assembly.scripts.train_dstg \
        policy_ckpt=/path/to/policy_epoch1060.pt \
        hdf5_path=/path/to/square_image_v15.hdf5 \
        out=outputs/dstg/square_demo50_succ/predictor.pt
"""

import json
import logging
import os
import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, TensorDataset

from square_assembly.datasets.labels import LABEL_DEMO, LABEL_INTV, LABEL_PREINTV, LABEL_ROLLOUT
from square_assembly.datasets.normalization import MinMaxNormalizer, load_stats
from square_assembly.datasets.robomimic_dataset import RobomimicSequenceDataset
from square_assembly.datasets.stg_labels import build_labels, success_tail_start
from square_assembly.factory import registry
from square_assembly.policies.diffusion.dstg_predictor import DstgPredictor
from square_assembly.utils.checkpoints import load_epoch_checkpoint, load_run_config
from square_assembly.utils.task_utils import task_obs_keys

logger = logging.getLogger(__name__)

MODE_IDS = {"demo": LABEL_DEMO, "rollout": LABEL_ROLLOUT, "intv": LABEL_INTV, "preintv": LABEL_PREINTV}



class _LabeledWindow(Dataset):
    """RobomimicSequenceDataset의 한 샘플에 time_to_success 라벨을 붙이는 얇은 어댑터
    (robomimic_dataset.py는 안 건드림 — get_time_to_success()는 전체 라벨을 한 번에 반환하는
    별도 메서드이지, __getitem__에 라벨을 얹는 게 아니다, ADR-005)."""

    def __init__(self, base, labels, indices, tail_starts=None):
        self.base = base
        self.labels = labels
        self.indices = list(indices)
        self.tail_starts = tail_starts or {}  # {데모: 성공-tail 시작 프레임} — train 데모만

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        item = self.base[idx]
        item["time_to_success"] = int(self.labels[idx])
        # 성공-tail: 잘라낸 앞쪽 프레임이 이력 창에 섞이지 않게 tail 첫 프레임으로 덮는다
        # (DinoFeatureWindows.__getitem__의 lower clamp와 같은 처리).
        start, t = self.tail_starts.get(item["demo_id"], 0), item["index_in_demo"]
        if start and t >= start:
            for k, v in item["obs"].items():
                pad = start - (t - len(v) + 1)
                if pad > 0:
                    v = v.clone()
                    v[:pad] = v[pad]
                    item["obs"][k] = v
        return item


def _stg_labels(dataset, cfg):
    """RobomimicSequenceDataset 샘플에 stg_labels.build_labels를 적용한다.

    데모 이름·프레임 인덱스는 get_time_to_success와 같은 경로(_demo_id_and_index_in_demo)로,
    길이와 action_mode는 hdf5에서 직접 읽는다(robomimic_dataset.py는 안 건드림, ADR-005).
    """
    seq = dataset._seq_dataset
    pairs = [dataset._demo_id_and_index_in_demo(i) for i in range(len(dataset))]
    names = [n for n, _ in pairs]
    ts = [t for _, t in pairs]
    demos = sorted(set(names))
    lengths = {n: seq.hdf5_file[f"data/{n}/actions"].shape[0] for n in demos}
    success = {n: True for n in demos}   # get_time_to_success와 같은 전제(에피소드=성공 경로)
    modes = {}
    if cfg.get("preintv", "none") != "none" or int(cfg.get("preintv_weight") or 1) > 1:
        for n in demos:
            g = seq.hdf5_file[f"data/{n}"]
            if "action_mode" in g:
                m = np.asarray(g["action_mode"])
                if len(m) < lengths[n]:
                    m = np.concatenate([m, np.repeat(m[-1:], lengths[n] - len(m))])
                modes[n] = m[: lengths[n]]
    return build_labels(names, ts, lengths, success, modes=modes,
                        fail_bin=cfg.get("fail_bin"),
                        label_horizon=cfg.get("label_horizon"),
                        preintv=cfg.get("preintv", "none"),
                        preintv_len=cfg.get("preintv_len"))


def _episode_split(dataset, val_fraction, seed):
    """무작위 에피소드(데모) 단위 분할 — 같은 데모의 프레임이 train/val에 걸쳐 섞이지
    않게 한다(무작위 transition 분할이면 인접 프레임 유출로 val이 낙관적으로 나온다)."""
    # _index_to_demo_id는 {sample_index: demo_id} dict — list()로 감싸면 키(=0..N-1)만
    # 나오는 함정이 있어(demo_id별 그룹핑이 아니라 사실상 무작위 transition 분할이 되어버림),
    # 반드시 .values()로 값(demo_id)을 순서대로 꺼내야 한다.
    demo_ids = [dataset._seq_dataset._index_to_demo_id[i] for i in range(len(dataset))]
    unique_demos = sorted(set(demo_ids))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_demos))
    n_val = max(1, int(round(len(unique_demos) * val_fraction)))
    val_demos = {unique_demos[i] for i in perm[:n_val]}
    train_idx = [i for i, d in enumerate(demo_ids) if d not in val_demos]
    val_idx = [i for i, d in enumerate(demo_ids) if d in val_demos]
    return train_idx, val_idx, len(unique_demos) - len(val_demos), len(val_demos)


def _run_epoch(predictor, loader, device, num_bins, optimizer=None, epoch_label=""):
    train = optimizer is not None
    predictor.train(train)  # frozen_policy도 같이 토글됨 — VisionEncoder crop_hw가
    # self.training으로 RandomCrop/CenterCrop을 고르므로 train=RandomCrop, eval=CenterCrop
    # (원본 diffusion policy 학습과 동일한 augmentation 관례, encoders 자체 가중치는
    # requires_grad_(False)로 이미 얼어 있어 이 토글이 학습에 영향 없음).

    total_nll, total_abs_err, n = 0.0, 0.0, 0
    bin_idx = torch.arange(num_bins, device=device, dtype=torch.float32)
    n_batches = len(loader)
    for b_i, raw_batch in enumerate(loader):
        if n_batches > 20 and (b_i % max(1, n_batches // 10) == 0 or b_i == n_batches - 1):
            print(f"  {epoch_label} 배치 {b_i + 1}/{n_batches}", flush=True)
        if isinstance(raw_batch, dict):
            obs = {k: v.to(device) for k, v in raw_batch["obs"].items()}
            labels = raw_batch["time_to_success"].to(device).long()
            logits = predictor(obs)
        else:  # random_crop=false: 미리 뽑아 둔 CenterCrop 특징 (feat, label)
            labels = raw_batch[1].to(device).long()
            logits = predictor.head(raw_batch[0].to(device))
        loss = F.cross_entropy(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            expected = (F.softmax(logits, dim=-1) * bin_idx).sum(dim=-1)
            total_abs_err += (expected - labels.float()).abs().sum().item()
            total_nll += loss.item() * labels.shape[0]
            n += labels.shape[0]

    return total_nll / n, total_abs_err / n


@torch.no_grad()
def _encode(predictor, data, device, num_workers):
    """얼린 인코더의 eval(CenterCrop) 특징 (N, global_cond_dim). 인코더가 안 변하므로 한 번만
    뽑아 epoch별 곡선과 random_crop=false 학습이 같이 쓴다."""
    predictor.eval()
    loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=num_workers)
    return torch.cat([predictor.frozen_policy.get_global_cond({k: v.to(device) for k, v in b["obs"].items()})
                      for b in loader])


@torch.no_grad()
def _curve_arrays(head, feats, labels, demo, t):
    """train_dstg_vip._curve_metrics가 받는 (d, nll, label, demo, t) — 지표 코드를 VIP 경로와 공유한다."""
    bins = torch.arange(head[-1].out_features, device=feats.device, dtype=torch.float32)
    y = torch.as_tensor(labels, device=feats.device)
    d, nll = [], []
    for i in range(0, len(feats), 4096):
        logits = head(feats[i:i + 4096])
        d.append((logits.softmax(-1) * bins).sum(-1))
        nll.append(F.cross_entropy(logits, y[i:i + 4096], reduction="none"))
    return torch.cat(d).cpu().numpy(), torch.cat(nll).cpu().numpy(), labels, demo, t


@hydra.main(config_path="../configs", config_name="train_dstg", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    # 아래 세 옵션은 VIP 경로(train_dstg_vip.py)의 같은 이름 옵션과 같은 뜻이다. 전부 끄면(기본) 예전 동작 그대로.
    success_tail_only = bool(cfg.get("success_tail_only", False))
    record_curves = bool(cfg.get("record_curves", False))
    random_crop = bool(cfg.get("random_crop", True))
    use_feats = record_curves or not random_crop
    if cfg.get("use_wandb", False) and not record_curves:
        raise ValueError("use_wandb requires record_curves=true")
    if record_curves and not success_tail_only:
        raise ValueError("record_curves currently requires success_tail_only for comparable train/val regions")
    if success_tail_only and (cfg.get("label_horizon") is not None or cfg.get("preintv", "none") != "none"
                              or cfg.get("train_modes") or int(cfg.get("preintv_weight") or 1) != 1):
        raise ValueError("success_tail_only requires raw countdown, preintv=none, all train_modes, weight=1")
    if use_feats and cfg.get("encoder_lr"):
        raise ValueError("record_curves/random_crop=false는 얼린 인코더 특징을 한 번만 뽑아 쓴다 — encoder_lr과 같이 못 쓴다")

    saved = load_run_config(cfg.policy_ckpt)
    if saved is None:
        raise ValueError(f"run_config.yaml을 {cfg.policy_ckpt} 옆에서 못 찾음 — 학습 당시 설정 없이는 정책 아키텍처를 복원할 수 없다.")
    task_cfg, policy_cfg, policy_name = saved.task, saved.policy, saved.policy_name

    OmegaConf.set_struct(task_cfg, False)
    task_cfg.hdf5_path = cfg.hdf5_path  # run_config.yaml의 상대경로는 이 머신에서 안 열림
    OmegaConf.set_struct(task_cfg, True)

    frozen_policy = registry.create_policy(policy_name, task_cfg, policy_cfg).to(device)
    load_epoch_checkpoint(cfg.policy_ckpt, frozen_policy, device)
    frozen_policy.eval()
    logger.info(f"frozen policy 복원: {cfg.policy_ckpt} (task={task_cfg.name} policy_name={policy_name})")

    stats_path = os.path.join(os.path.dirname(cfg.policy_ckpt), "normalization_stats.json")
    normalizer = MinMaxNormalizer(load_stats(stats_path))

    obs_keys = task_obs_keys(task_cfg)
    cache_mode = "all" if cfg.num_workers >= 1 else "low_dim"  # h5py fork 크래시 회피(diffusion_trainer.py와 동일)
    dataset = RobomimicSequenceDataset(
        hdf5_path=cfg.hdf5_path, obs_keys=obs_keys, obs_horizon=policy_cfg.obs_horizon,
        pred_horizon=1,  # STG는 이 시점 관측 하나에 대한 라벨이지 행동 청크가 아니다
        normalizer=normalizer, rgb_keys=task_cfg.rgb_keys, hdf5_cache_mode=cache_mode,
    )
    # 라벨 규칙은 DINO/VIP 경로와 같은 곳(stg_labels.build_labels)을 쓴다 — 인코더를
    # 비교한다면서 라벨 처리를 비교하게 되는 걸 막는다.
    labels, preintv_mask = _stg_labels(dataset, cfg)
    # 기본은 데이터에서 관측된 최대 time-to-success로 정한다(고정 상수 금지, 원래 방침).
    # cfg.num_bins_override(기본 None)가 있으면 그 값을 강제로 쓴다 — 2026-08-11: 데이터
    # 스케일링 비교 실험처럼 "여러 체크포인트가 같은 bin 공간을 써야 mu/sigma를 그대로
    # 비교할 수 있는" 상황 전용. 그 외엔 기본값(None)을 유지해 기존 동작 그대로 둔다.
    if cfg.get("num_bins_override"):
      num_bins = int(cfg.num_bins_override)
      if num_bins <= int(labels.max()):
        raise ValueError(f"num_bins_override({num_bins})가 실제 관측된 최대 time-to-success"
                         f"({int(labels.max())})보다 작거나 같다 — 라벨이 잘린다.")
    else:
      num_bins = int(labels.max()) + 1
    logger.info(f"dataset len={len(dataset)} num_bins={num_bins} (max observed time-to-success={int(labels.max())})")

    train_idx, val_idx, n_train_demos, n_val_demos = _episode_split(dataset, cfg.val_fraction, cfg.split_seed)
    pairs = [dataset._demo_id_and_index_in_demo(i) for i in range(len(dataset))]
    demo_of, t_of = np.array([n for n, _ in pairs]), np.array([t for _, t in pairs])
    tail_starts = None
    if success_tail_only:   # train만 거른다 — val은 전체 프레임 그대로(held-out 비교가 깨지지 않게)
        seq = dataset._seq_dataset
        all_modes = {n: np.asarray(seq.hdf5_file[f"data/{n}/action_mode"]) for n in sorted(set(demo_of))}
        for n, m in all_modes.items():
            if m.ndim != 1 or len(m) != seq.hdf5_file[f"data/{n}/actions"].shape[0] \
                    or not np.isin(m, list(MODE_IDS.values())).all():
                raise ValueError(f"{n}: complete valid action_mode is required for success tails")
        all_starts = {n: success_tail_start(m) for n, m in all_modes.items()}
        before = len(train_idx)
        train_idx = [i for i in train_idx if t_of[i] >= all_starts[demo_of[i]]]
        tail_starts = {n: all_starts[n] for n in set(demo_of[train_idx])}
        n_train_demos = len(tail_starts)
        logger.info(f"success_tail_only: train samples {before} -> {len(train_idx)}")
    if cfg.get("train_modes"):
        # 프레임 종류로 train을 거른다(val은 그대로) — 예: 정책 롤아웃 프레임을 빼고
        # 사람 시연·개입과 개입 직전만 남기기.
        keep_ids = {MODE_IDS[m] for m in cfg.train_modes}
        sample_modes = dataset.get_action_mode_first_frame()
        before = len(train_idx)
        train_idx = [i for i in train_idx if sample_modes[i] in keep_ids]
        logger.info(f"train_modes={list(cfg.train_modes)}: train 샘플 {before} -> {len(train_idx)}")
    if cfg.get("train_sources"):
        # 병합본의 demo attrs["source"](merge_demo_hdf5)로 train만 거른다 — val은 그대로 둬야
        # 서로 다른 데이터로 학습한 arm들이 같은 held-out에서 비교된다.
        seq = dataset._seq_dataset
        src = {d: str(seq.hdf5_file[f"data/{d}"].attrs.get("source", "")) for d in set(seq._index_to_demo_id.values())}
        keep = lambda i: any(k in src[seq._index_to_demo_id[i]] for k in cfg.train_sources)
        before = len(train_idx)
        train_idx = [i for i in train_idx if keep(i)]
        n_train_demos = len({seq._index_to_demo_id[i] for i in train_idx})
        logger.info(f"train_sources={list(cfg.train_sources)}: train 샘플 {before} -> {len(train_idx)}")
    if cfg.get("preintv") == "drop":   # val은 건드리지 않는다 — held-out 비교가 깨진다
        before = len(train_idx)
        train_idx = [i for i in train_idx if not preintv_mask[i]]
        logger.info(f"preintv=drop: train 샘플 {before} -> {len(train_idx)}")

    # PREINTV 오버샘플링 — 학습 프레임의 2.7%뿐이라 그래디언트에 거의 안 잡힌다. 인덱스를
    # 그대로 복제해 더 자주 보게 한다(val은 건드리지 않으므로 held-out은 그대로다).
    w = int(cfg.get("preintv_weight") or 1)
    if w > 1:
        extra = [i for i in train_idx if preintv_mask[i]] * (w - 1)
        train_idx = train_idx + extra
        logger.info(f"preintv_weight={w}: train 샘플 +{len(extra)} -> {len(train_idx)}")
    logger.info(
        f"episode split: train={n_train_demos} demos/{len(train_idx)} samples, "
        f"val={n_val_demos} demos/{len(val_idx)} samples"
    )

    train_loader = DataLoader(
        _LabeledWindow(dataset, labels, train_idx, tail_starts), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, persistent_workers=cfg.num_workers >= 1,
    )
    val_loader = DataLoader(
        _LabeledWindow(dataset, labels, val_idx), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers >= 1,
    )

    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    wandb_run = None
    if cfg.get("use_wandb", False):   # train_dstg_vip.py와 같은 설정·같은 지표 이름 — 두 경로를 한 화면에 겹쳐 본다
        import wandb

        wandb_config = {k: cfg.get(k) for k in ("seed", "split_seed", "num_epochs", "batch_size", "lr", "weight_decay",
                                                 "label_horizon", "preintv", "success_tail_only", "random_crop")}
        wandb_config.update(encoder="policy_resnet18", head_hidden=list(cfg.head_hidden), num_bins=num_bins,
                            obs_horizon=policy_cfg.obs_horizon, n_train_samples=len(train_idx),
                            n_val_samples=len(val_idx), n_train_demos=n_train_demos, n_val_demos=n_val_demos)
        wandb_run = wandb.init(
            project=cfg.wandb_project, entity=cfg.get("wandb_entity"),
            name=cfg.get("wandb_run_name") or f"dstg-resnet-split{cfg.split_seed}",
            group=cfg.get("wandb_group"), mode=cfg.get("wandb_mode", "online"),
            dir=os.path.dirname(cfg.out), save_code=False,
            settings=wandb.Settings(disable_git=True), config=wandb_config,
        )
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("*", step_metric="epoch")
        with open(os.path.join(os.path.dirname(cfg.out), "wandb_run.json"), "w") as stream:
            json.dump({"id": wandb_run.id, "url": wandb_run.url, "entity": wandb_run.entity,
                       "project": wandb_run.project, "group": wandb_run.group}, stream, indent=2)
    if use_feats:  # wandb 초기화가 헤드 초기값·셔플 순서를 바꾸지 않게 한다(VIP 경로와 같은 처리)
        torch.manual_seed(cfg.seed)

    encoder_lr = cfg.get("encoder_lr")
    predictor = DstgPredictor(frozen_policy, num_bins, head_hidden=tuple(cfg.head_hidden),
                              train_encoder=bool(encoder_lr)).to(device)
    if cfg.get("init_ckpt"):  # 파인튜닝 — 헤드만 이어받는다(인코더는 원래 정책 것 그대로)
        init = torch.load(cfg.init_ckpt, map_location=device, weights_only=False)
        if int(init["num_bins"]) != num_bins:
            raise ValueError(f"init_ckpt num_bins {init['num_bins']} != {num_bins} — num_bins_override를 맞춰라")
        predictor.head.load_state_dict(init["model"])
        logger.info(f"init_ckpt에서 헤드를 이어받음: {cfg.init_ckpt}")
    # 이중 안전판: requires_grad_(False)(DstgPredictor 생성자) + 옵티마이저 파라미터 그룹을
    # head로만 좁힘(frozen_policy 전체를 넘기지 않음).
    groups = [{"params": predictor.head.parameters(), "lr": cfg.lr}]
    if encoder_lr:  # 풀 파인튜닝 — 인코더만 더하고 unet은 여전히 뺀다
        groups.append({"params": frozen_policy.encoders.parameters(), "lr": float(encoder_lr)})
        logger.info(f"encoder_lr={encoder_lr}: 비전 인코더도 학습")
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    curves, best_state, best_row, stats = [], None, None, {}
    if use_feats:
        started = time.time()
        curve_idx = sorted(set(train_idx))
        train_feats = _encode(predictor, _LabeledWindow(dataset, labels, curve_idx, tail_starts), device, cfg.num_workers)
        val_feats = _encode(predictor, _LabeledWindow(dataset, labels, val_idx), device, cfg.num_workers)
        stats["feature_cache_seconds"] = time.time() - started
        logger.info(f"CenterCrop 특징 캐시: train {tuple(train_feats.shape)} val {tuple(val_feats.shape)} "
                    f"{stats['feature_cache_seconds']:.1f}s")
        if not random_crop:   # 증강 없이 학습 — 같은 특징을 그대로 쓴다(복제된 인덱스도 그대로 복제)
            pos = {i: k for k, i in enumerate(curve_idx)}
            rows = torch.tensor([pos[i] for i in train_idx])
            train_loader = DataLoader(
                TensorDataset(train_feats.cpu()[rows], torch.as_tensor(labels[train_idx])),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    if record_curves:
        from square_assembly.scripts.train_dstg_vip import _curve_metrics, _log_curve

        val_demo, val_t = demo_of[val_idx], t_of[val_idx]
        eligible = val_t >= np.array([all_starts[n] for n in val_demo])
        cohort_of = {n: "intervention_tail" if (m == LABEL_INTV).any() else "expert" if (m == LABEL_DEMO).all()
                     else "policy_success" for n, m in all_modes.items()}
        val_cohort = np.array([cohort_of[n] for n in val_demo])
        curve_path = os.path.join(os.path.dirname(cfg.out), "learning_curve.json")

    def save(path, epochs_done, **extra):
        torch.save({
            "model": predictor.head.state_dict(),  # frozen_policy는 policy_ckpt에서 다시 로드하는 게 정책
            "epoch": epochs_done,
            "num_bins": num_bins,
            "label_horizon": cfg.get("label_horizon"),
            "preintv": cfg.get("preintv", "none"),
            "preintv_weight": cfg.get("preintv_weight"),
            "preintv_len": cfg.get("preintv_len"),
            "init_ckpt": cfg.get("init_ckpt"),
            "train_sources": list(cfg.train_sources) if cfg.get("train_sources") else None,
            "train_modes": list(cfg.train_modes) if cfg.get("train_modes") else None,
            "policy_ckpt": os.path.abspath(cfg.policy_ckpt),
            "obs_keys": obs_keys,
            "head_hidden": list(cfg.head_hidden),
            "success_tail_only": success_tail_only,
            "random_crop": random_crop,
            "seed": cfg.seed,
            "split_seed": cfg.split_seed,
            "encoder_lr": encoder_lr,
            **({"encoder": frozen_policy.encoders.state_dict()} if encoder_lr else {}),
            **extra,
        }, path)
        logger.info(f"saved: {path}")

    # 중간 저장(학습 곡선) — "몇 epoch 파인튜닝이 적당한가"를 값 하나 정하지 않고 곡선으로 본다.
    save_epochs = set(cfg.get("save_epochs") or [])
    stats["train_seconds"] = 0.0
    for epoch in range(cfg.num_epochs):
        started = time.time()
        train_nll, train_mae = _run_epoch(predictor, train_loader, device, num_bins, optimizer=optimizer,
                                          epoch_label=f"epoch {epoch}/{cfg.num_epochs}")
        stats["train_seconds"] += time.time() - started
        if epoch % cfg.log_every == 0 or epoch == cfg.num_epochs - 1:
            msg = f"epoch {epoch} train_nll={train_nll:.4f} train_mae={train_mae:.3f}"
            logger.info(msg)
            print(msg, flush=True)
        if epoch + 1 in save_epochs and epoch + 1 < cfg.num_epochs:
            save(cfg.out.replace(".pt", f"_ep{epoch + 1}.pt"), epoch + 1)
        if record_curves:
            train_arrays = _curve_arrays(predictor.head, train_feats, labels[curve_idx], demo_of[curve_idx], t_of[curve_idx])
            val_arrays = _curve_arrays(predictor.head, val_feats, labels[val_idx], val_demo, val_t)
            row = {"epoch": epoch + 1, "train": _curve_metrics(train_arrays),
                   "val_all": _curve_metrics(val_arrays),
                   "val_retained": _curve_metrics(val_arrays, eligible),
                   "val_excluded_prefix": _curve_metrics(val_arrays, ~eligible)}
            for cohort in ("expert", "policy_success", "intervention_tail"):
                row[f"val_{cohort}"] = _curve_metrics(val_arrays, eligible & (val_cohort == cohort))
            curves.append(row)
            if wandb_run is not None:
                _log_curve(wandb_run, row)
            if best_row is None or row["val_retained"]["mae"] < best_row["val_retained"]["mae"]:
                best_row = row
                best_state = {k: v.detach().cpu().clone() for k, v in predictor.head.state_dict().items()}
            if device.type == "cuda":
                stats["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
                stats["peak_gpu_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
            with open(curve_path, "w") as stream:
                json.dump({"rows": curves, "best_epoch": best_row["epoch"], "selection_metric": "val_retained.mae",
                           "val_demos": sorted(set(val_demo)), "tail_starts": all_starts, **stats},
                          stream, indent=2, allow_nan=False)
            print(f"curve epoch={epoch + 1} train_mae={row['train']['mae']:.3f} "
                  f"val_retained_mae={row['val_retained']['mae']:.3f} "
                  f"val_retained_k8={row['val_retained']['k8']}", flush=True)

    val_nll, val_mae = _run_epoch(predictor, val_loader, device, num_bins, optimizer=None, epoch_label="[val]")
    logger.info(f"[val] nll={val_nll:.4f} mae={val_mae:.3f} (n={len(val_idx)} samples, {n_val_demos} demos)")
    print({"val_nll": val_nll, "val_mae": val_mae, "num_bins": num_bins,
           "n_train_demos": n_train_demos, "n_val_demos": n_val_demos})
    save(cfg.out, cfg.num_epochs, val_nll=val_nll, val_mae=val_mae)
    if best_state is not None:   # val 유지 구간 MAE가 가장 낮았던 epoch의 헤드 — DstgReward가 그대로 읽는 형식
        predictor.head.load_state_dict(best_state)
        save(os.path.join(os.path.dirname(cfg.out), "best_predictor.pt"), best_row["epoch"],
             val_nll=best_row["val_all"]["nll"], val_mae=best_row["val_all"]["mae"],
             selection_metric="val_retained.mae", selection_value=best_row["val_retained"]["mae"])
    if wandb_run is not None:
        wandb_run.summary.update({"best_epoch": best_row["epoch"],
                                  "best_val_retained_mae": best_row["val_retained"]["mae"],
                                  "best_epoch_val_retained_k8": best_row["val_retained"]["k8"],
                                  "final_epoch": cfg.num_epochs, **stats})
        wandb_run.finish()


if __name__ == "__main__":
    main()
