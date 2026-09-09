# DINO STG Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** square task의 steps-to-go 예측기 `d(o,g)`의 비전 인코더를 diffusion policy의 ResNet18에서 얼린 DINOv2로 교체하되, 특징을 1회 사전추출해 캐시하고 그 위의 MLP 헤드만 학습한다.

**Architecture:** 인코더가 얼려 있으므로 "같은 프레임 → 항상 같은 특징"이 성립한다. 따라서 (1) `cache_dino_feats.py`가 robomimic hdf5의 모든 프레임을 DINOv2로 인코딩해 특징 캐시 hdf5로 굽고, (2) `dino_feature_dataset.py`가 그 캐시를 기준선과 **동일한 인덱싱 규칙**으로 (obs 윈도우, steps-to-go 라벨) 샘플로 자르고, (3) `train_dstg_dino.py`가 기준선과 동일한 하이퍼파라미터로 MLP 헤드를 학습한다. 기존 `dstg_predictor.py` / `train_dstg.py` 경로는 비교 기준선이므로 전혀 건드리지 않는다.

**Tech Stack:** PyTorch 2.x, timm (DINOv2 `vit_small_patch14_reg4_dinov2.lvd142m`), h5py, hydra, pytest. robomimic·robosuite·mujoco는 **쓰지 않는다**.

**Spec:** `docs/superpowers/specs/2026-09-09-dino-stg-encoder-design.md`

## Global Constraints

- 작업 디렉토리는 `projects/square_assembly/`다. 이 문서의 모든 상대경로는 거기 기준.
- 브랜치: `feat/dino-stg-encoder` (main에서 분기). main에 직접 커밋하지 않는다.
- **기존 파일 수정 금지 (ADR-005, 비교 기준선 보존)**: `src/square_assembly/policies/diffusion/dstg_predictor.py`, `src/square_assembly/scripts/train_dstg.py`, `src/square_assembly/scripts/train_dstg_failaware.py`, `src/square_assembly/datasets/robomimic_dataset.py`. 전부 **읽기만** 한다.
- **새 의존성 추가 금지.** `timm`은 `requirements.txt:57`에 `timm==0.9.16`으로 이미 pin돼 있다. `requirements.txt`를 수정하지 않는다.
- **robomimic import 금지.** 로컬 PC에 robomimic이 설치돼 있지 않다. 모든 새 코드와 모든 테스트는 robomimic 없이 동작해야 한다. (예외: `train_dstg._episode_split`는 import 가능하다 — `robomimic_dataset.py`가 robomimic import를 생성자로 지연시켜 뒀기 때문. 실측 확인함.)
- **로컬 실행 인터프리터**: `/home/hymm/miniconda3/bin/python` (base env — torch 2.10.0+cu128, timm 1.0.20, h5py 3.16.0, hydra 1.3.2). 로컬엔 `square_assembly` conda env가 없고 GPU도 없다. 모든 명령은 `PYTHONPATH=src`를 붙여 실행한다.
- **device 하드코딩 금지.** 기본값은 `"cuda" if torch.cuda.is_available() else "cpu"`.
- **비교 조건 고정 (spec §4)**: 데이터 `data/square_scale3_0_seed1.hdf5`, `num_bins_override=501`, `split_seed=0`, `val_fraction=0.2`, `batch_size=64`, `lr=1e-4`, `weight_decay=1e-6`, `num_epochs=30`, `head_hidden=[256,256]`, `obs_horizon=2`, `seed=1`.
  기준선 실측값: **val_mae 15.85 / val_nll 4.67** (`outputs/dstg/square_scale3_0_seed1/predictor.pt`).
- 커밋 메시지는 **영어**, Scoped Conventional Commits, 제목은 서술형(동사 중심), 본문은 `-` 글머리표 멀티라인 (CLAUDE.md).
- 커밋 메시지 말미에 다음 두 줄을 붙인다:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_019LW9kqFwrDYHnMCU8jgS8X
  ```

---

## Task 0: 브랜치 생성

**Files:** 없음 (git 상태 변경만)

- [ ] **Step 1: main에서 feature 브랜치를 만든다**

```bash
cd /home/hymm/Projects/self-improving-gym
git checkout -b feat/dino-stg-encoder
git status
```

Expected: `On branch feat/dino-stg-encoder`, working tree에 spec/plan 두 개의 untracked 파일.

- [ ] **Step 2: spec과 plan을 커밋**

```bash
git add docs/superpowers/specs/2026-09-09-dino-stg-encoder-design.md docs/superpowers/plans/2026-09-09-dino-stg-encoder.md
git commit -m "$(cat <<'EOF'
docs(dstg): specify DINO-feature steps-to-go predictor for square

- add design spec: frozen DINOv2 ViT-S/14 encoder, one-shot feature cache, MLP head
- pin the comparison protocol against the ResNet baseline (val_mae 15.85 / val_nll 4.67 on square_scale3_0_seed1)
- add task-by-task implementation plan

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019LW9kqFwrDYHnMCU8jgS8X
EOF
)"
```

---

## Task 1: DINO 특징 캐시 스크립트

**Files:**
- Create: `projects/square_assembly/src/square_assembly/scripts/cache_dino_feats.py`
- Test: `projects/square_assembly/tests/test_dino_stg.py`

**Interfaces:**
- Produces:
  - `build_encoder(model_name=DEFAULT_MODEL, pretrained=True, device="cpu") -> torch.nn.Module`
  - `encode_frames(model, imgs_u8, crop=76, size=224, device="cpu", batch_size=128) -> np.ndarray` — `(T,H,W,3) uint8` → `(T, 2*D) float16`
  - 모듈 상수 `DEFAULT_MODEL`, `DEFAULT_RGB_KEYS`, `DEFAULT_LOWDIM_KEYS`, `IMAGENET_MEAN`, `IMAGENET_STD`
  - CLI: `python -m square_assembly.scripts.cache_dino_feats --hdf5 <in> --out <out>` → 캐시 hdf5 (`data/<demo>/feat` `(T,1536) float16`, `data/<demo>/lowdim` `(T,9) float32`, attrs `is_success`/`length`, 파일 attrs `meta` JSON)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`projects/square_assembly/tests/test_dino_stg.py` 생성:

```python
"""DINO 특징 캐시·데이터셋·STG 헤드 검증.

이 파일은 robomimic 없이 돌아야 한다(로컬 PC에 robomimic이 없다). 사전학습
가중치 다운로드도 하지 않는다(pretrained=False) — 네트워크 없이 CI/로컬 어디서나
돈다. 실제 가중치 다운로드는 Task 1 Step 5의 스모크가 확인한다.
"""

import numpy as np
import torch

from square_assembly.scripts.cache_dino_feats import DEFAULT_MODEL, build_encoder, encode_frames


def test_encode_frames_shape_and_dtype():
    model = build_encoder(DEFAULT_MODEL, pretrained=False, device="cpu")
    imgs = np.random.randint(0, 256, size=(3, 84, 84, 3), dtype=np.uint8)

    feat = encode_frames(model, imgs, crop=76, size=224, device="cpu", batch_size=2)

    assert feat.shape == (3, 2 * model.embed_dim)  # [CLS ; mean(patch)]
    assert feat.dtype == np.float16
    assert np.isfinite(feat).all()


def test_encode_frames_is_deterministic():
    """얼린 인코더이므로 같은 프레임은 항상 같은 특징이어야 한다 — 캐싱 방식 자체의 전제."""
    model = build_encoder(DEFAULT_MODEL, pretrained=False, device="cpu")
    imgs = np.random.randint(0, 256, size=(2, 84, 84, 3), dtype=np.uint8)

    a = encode_frames(model, imgs, crop=76, size=224, device="cpu", batch_size=2)
    b = encode_frames(model, imgs, crop=76, size=224, device="cpu", batch_size=1)

    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
PYTHONPATH=src /home/hymm/miniconda3/bin/python -m pytest tests/test_dino_stg.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'square_assembly.scripts.cache_dino_feats'`

- [ ] **Step 3: 캐시 스크립트를 구현한다**

`projects/square_assembly/src/square_assembly/scripts/cache_dino_feats.py` 생성:

```python
"""robomimic 형식 hdf5의 모든 프레임을 얼린 DINOv2로 인코딩해 특징 캐시(hdf5)로 굽는다.

STG 예측기 d(o,g)의 비전 인코더를, 학습된 diffusion policy의 ResNet18+SpatialSoftmax
(policies/diffusion/dstg_predictor.py) 대신 자기지도 사전학습 ViT로 갈아끼우기 위한
1회성 전처리다. 인코더가 얼어 있으니 "같은 프레임 → 항상 같은 특징"이고, 매 에폭
이미지를 다시 로드해 인코더를 다시 통과시킬 이유가 없다
(docs/superpowers/specs/2026-09-09-dino-stg-encoder-design.md).

전처리는 기준선의 **eval 시 경로**와 맞춘다: 84x84 -> CenterCrop 76 -> resize 224 ->
ImageNet 정규화. crop 76은 configs/policy/diffusion_unet.yaml의 crop_hw와 같은 값이다 —
시야각이 다르면 val 지표 비교 자체가 성립하지 않는다. (기준선은 학습 시엔 RandomCrop을
쓰지만 캐시 방식은 증강을 잃는다 — spec §3의 알려진 트레이드오프.)

robomimic을 쓰지 않는다 — h5py로 hdf5를 직접 읽는다.

사용:
    python -m square_assembly.scripts.cache_dino_feats \
        --hdf5 data/square_scale3_0_seed1.hdf5 \
        --out data/square_scale3_0_seed1.dino.h5
"""

import argparse
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# timm 0.9.16(서버 canonical env)·1.0.20(로컬 base env) 양쪽에 있는 이름.
# 없다면 --model vit_small_patch14_dinov2.lvd142m 로 폴백한다(레지스터 없는 버전).
DEFAULT_MODEL = "vit_small_patch14_reg4_dinov2.lvd142m"
DEFAULT_RGB_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
DEFAULT_LOWDIM_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def build_encoder(model_name=DEFAULT_MODEL, pretrained=True, device="cpu"):
    """num_classes=0으로 분류 헤드를 떼고 특징 추출기로만 쓴다."""
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    return model.eval().to(device)


@torch.no_grad()
def encode_frames(model, imgs_u8, crop=76, size=224, device="cpu", batch_size=128):
    """(T,H,W,3) uint8 -> (T, 2*D) float16.  프레임당 [CLS ; mean(patch tokens)].

    캐싱과 (후속 작업의) 온라인 추론이 반드시 같은 전처리를 타도록, 전처리는 이 함수
    하나에만 존재한다.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    n_prefix = model.num_prefix_tokens  # CLS(+register) 토큰 수 — 패치 토큰의 시작 위치

    out = []
    for i in range(0, len(imgs_u8), batch_size):
        chunk = np.ascontiguousarray(imgs_u8[i : i + batch_size])
        x = torch.from_numpy(chunk).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        if crop:
            x = TF.center_crop(x, [crop, crop])
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        tokens = model.forward_features(x)  # (B, n_prefix + N, D)
        feat = torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(dim=1)], dim=-1)
        out.append(feat.half().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True, help="robomimic 형식 입력 hdf5")
    p.add_argument("--out", required=True, help="출력 특징 캐시 hdf5")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--rgb-keys", nargs="+", default=list(DEFAULT_RGB_KEYS))
    p.add_argument("--lowdim-keys", nargs="+", default=list(DEFAULT_LOWDIM_KEYS))
    p.add_argument("--crop", type=int, default=76, help="0이면 crop 없이 원본 해상도 사용")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--limit-demos", type=int, default=0, help=">0이면 앞 N개 데모만 (스모크용)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    import timm

    model = build_encoder(args.model, pretrained=True, device=args.device)
    print(f"encoder={args.model} timm={timm.__version__} device={args.device}", flush=True)

    with h5py.File(args.hdf5, "r") as src, h5py.File(args.out, "w") as dst:
        demos = sorted(src["data"].keys())
        if args.limit_demos:
            demos = demos[: args.limit_demos]
        for i, name in enumerate(demos):
            g = src["data"][name]
            length = int(g["actions"].shape[0])
            feat = np.concatenate(
                [
                    encode_frames(model, g["obs"][k][:], args.crop, args.size, args.device, args.batch_size)
                    for k in args.rgb_keys
                ],
                axis=-1,
            )
            low = np.concatenate(
                [np.asarray(g["obs"][k][:], dtype=np.float32) for k in args.lowdim_keys], axis=-1
            )
            if feat.shape[0] != length or low.shape[0] != length:
                raise ValueError(
                    f"{name}: 프레임 수 불일치 feat={feat.shape[0]} lowdim={low.shape[0]} actions={length}"
                )

            og = dst.create_group(f"data/{name}")
            og.create_dataset("feat", data=feat)
            og.create_dataset("lowdim", data=low)
            og.attrs["is_success"] = bool(np.asarray(g.attrs.get("is_success", True)))
            og.attrs["length"] = length
            print(f"[{i + 1}/{len(demos)}] {name} T={length} feat={feat.shape}", flush=True)

        dst.attrs["meta"] = json.dumps(
            {
                "source_hdf5": args.hdf5,
                "model": args.model,
                "timm": timm.__version__,
                "crop": args.crop,
                "size": args.size,
                "rgb_keys": list(args.rgb_keys),
                "lowdim_keys": list(args.lowdim_keys),
            }
        )
    print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
PYTHONPATH=src /home/hymm/miniconda3/bin/python -m pytest tests/test_dino_stg.py -v
```

Expected: 2 passed. (ViT-S를 CPU에서 랜덤 초기화하므로 30초 정도 걸릴 수 있다.)

- [ ] **Step 5: 실데이터 2데모로 스모크 — 사전학습 가중치 다운로드까지 확인**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
PYTHONPATH=src /home/hymm/miniconda3/bin/python -m square_assembly.scripts.cache_dino_feats \
    --hdf5 data/square_scale3_0_seed1.hdf5 \
    --out /tmp/claude-1000/-home-hymm-Projects-self-improving-gym/f01ae550-9c56-4f03-9cdf-493e0bef2519/scratchpad/smoke.dino.h5 \
    --limit-demos 2
```

Expected: `encoder=vit_small_patch14_reg4_dinov2.lvd142m timm=1.0.20 device=cpu` 뒤에 데모 2개 라인, `feat=(T, 1536)`.

**실패 시 대응:**
- `RuntimeError: Unknown model` → 이 timm 버전에 reg4 변종이 없다. `--model vit_small_patch14_dinov2.lvd142m`로 재실행하고, 그 이름을 `DEFAULT_MODEL`로 바꾼 뒤 spec §3 표의 백본 칸도 함께 고친다.
- HF hub 다운로드 실패(네트워크) → 이 시점에서 사용자에게 보고하고 멈춘다. 이후 Task는 전부 이 가중치에 의존한다.

- [ ] **Step 6: 캐시 산출물을 눈으로 확인한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
/home/hymm/miniconda3/bin/python - <<'EOF'
import h5py, json
f = h5py.File("/tmp/claude-1000/-home-hymm-Projects-self-improving-gym/f01ae550-9c56-4f03-9cdf-493e0bef2519/scratchpad/smoke.dino.h5", "r")
print(json.loads(f.attrs["meta"]))
for n in f["data"]:
    g = f["data"][n]
    print(n, g["feat"].shape, g["feat"].dtype, g["lowdim"].shape, dict(g.attrs))
EOF
```

Expected: `feat (T,1536) float16`, `lowdim (T,9) float32`, `{'is_success': True, 'length': T}`.

- [ ] **Step 7: 커밋**

```bash
cd /home/hymm/Projects/self-improving-gym
git add projects/square_assembly/src/square_assembly/scripts/cache_dino_feats.py projects/square_assembly/tests/test_dino_stg.py
git commit -m "$(cat <<'EOF'
feat(dstg): precompute frozen DINOv2 features into an hdf5 cache

- add cache_dino_feats.py: encode every robomimic frame with DINOv2 ViT-S/14 (reg4), store [CLS ; mean(patch)] per camera as float16
- match the baseline eval-time preprocessing (84 -> CenterCrop 76 -> resize 224 -> ImageNet norm) so val metrics stay comparable
- keep preprocessing in a single encode_frames() so a later online path cannot drift from the cache
- read hdf5 directly with h5py; the cache path never imports robomimic

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019LW9kqFwrDYHnMCU8jgS8X
EOF
)"
```

---

## Task 2: 특징 캐시 데이터셋 + 기준선과 동일한 분할

**Files:**
- Create: `projects/square_assembly/src/square_assembly/datasets/dino_feature_dataset.py`
- Modify: `projects/square_assembly/tests/test_dino_stg.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 1의 캐시 hdf5 레이아웃 (`data/<demo>/feat`, `data/<demo>/lowdim`, attrs `is_success`/`length`)
- Produces:
  - `episode_split_by_name(demo_names, val_fraction, seed) -> set[str]` (val 데모 이름 집합)
  - `DinoFeatureWindows(cache_path, obs_horizon, fail_bin=None)` — `torch.utils.data.Dataset`, `__getitem__` 반환 `(torch.FloatTensor (obs_horizon*frame_dim,), int label)`
  - 속성: `.samples: list[(demo_name, t)]`, `.frame_dim: int`, `.lengths: dict`, `.success: dict`
  - 메서드: `.labels() -> np.ndarray`, `.split_indices(val_fraction, seed) -> (train_idx, val_idx, val_demos)`, `.compute_frame_stats(indices) -> (mean, std)`, `.apply_frame_stats(mean, std)`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`projects/square_assembly/tests/test_dino_stg.py` 맨 아래에 추가 (import 줄도 함께 추가):

```python
import types

import h5py
import pytest

from square_assembly.datasets.dino_feature_dataset import DinoFeatureWindows, episode_split_by_name


def _write_fake_cache(path, lengths, feat_dim=4, low_dim=2, success=None):
    """가짜 특징 캐시를 만든다 — 값은 (demo, t)로부터 결정되게 넣어 윈도우 검증이 가능하게."""
    with h5py.File(path, "w") as f:
        for i, L in enumerate(lengths):
            name = f"demo_{i}"
            feat = np.stack([np.full(feat_dim, i * 1000 + t, dtype=np.float16) for t in range(L)])
            low = np.stack([np.full(low_dim, -(i * 1000 + t), dtype=np.float32) for t in range(L)])
            g = f.create_group(f"data/{name}")
            g.create_dataset("feat", data=feat)
            g.create_dataset("lowdim", data=low)
            g.attrs["length"] = L
            g.attrs["is_success"] = True if success is None else bool(success[i])
        f.attrs["meta"] = "{}"


def test_sample_count_and_labels_match_baseline_rule(tmp_path):
    """샘플 수와 라벨은 robomimic(frame_stack=To, pad_frame_stack=True) + get_time_to_success와
    같아야 한다: 데모당 L개 샘플, t=0..L-1, 라벨 = L-1-t."""
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[5, 3])
    ds = DinoFeatureWindows(str(cache), obs_horizon=2)

    assert len(ds) == 8  # 5 + 3
    assert ds.samples[:3] == [("demo_0", 0), ("demo_0", 1), ("demo_0", 2)]
    np.testing.assert_array_equal(ds.labels(), [4, 3, 2, 1, 0, 2, 1, 0])


def test_window_pads_by_repeating_the_first_frame(tmp_path):
    """t=0에서는 앞쪽 이력이 없으므로 첫 프레임이 복제돼야 한다(pad_frame_stack=True와 동일)."""
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[5], feat_dim=4, low_dim=2)
    ds = DinoFeatureWindows(str(cache), obs_horizon=2)

    x0, y0 = ds[0]  # t=0 -> 윈도우 [frame 0, frame 0]
    assert x0.shape == (2 * 6,)
    np.testing.assert_array_equal(x0[:4].numpy(), [0, 0, 0, 0])      # frame 0 feat
    np.testing.assert_array_equal(x0[6:10].numpy(), [0, 0, 0, 0])    # frame 0 feat (복제)
    assert y0 == 4

    x2, y2 = ds[2]  # t=2 -> 윈도우 [frame 1, frame 2]
    np.testing.assert_array_equal(x2[:4].numpy(), [1, 1, 1, 1])
    np.testing.assert_array_equal(x2[6:10].numpy(), [2, 2, 2, 2])
    np.testing.assert_array_equal(x2[4:6].numpy(), [-1, -1])         # frame 1 lowdim
    assert y2 == 2


def test_split_matches_train_dstg_episode_split(tmp_path):
    """val 분할이 기준선과 글자 그대로 같아야 val_mae 비교가 성립한다.
    train_dstg._episode_split을 가짜 dataset으로 직접 호출해 대조한다(robomimic 불필요)."""
    from square_assembly.scripts.train_dstg import _episode_split

    lengths = [7, 5, 9, 4, 6, 8, 3, 5, 6, 7, 4, 5]
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=lengths)
    ds = DinoFeatureWindows(str(cache), obs_horizon=2)
    train_idx, val_idx, _ = ds.split_indices(val_fraction=0.2, seed=0)

    demo_ids = [name for name, _ in ds.samples]

    class _FakeDataset:
        def __init__(self, ids):
            self._seq_dataset = types.SimpleNamespace(
                _index_to_demo_id={i: d for i, d in enumerate(ids)}
            )

        def __len__(self):
            return len(self._seq_dataset._index_to_demo_id)

    ref_train, ref_val, _, _ = _episode_split(_FakeDataset(demo_ids), 0.2, 0)

    assert train_idx == ref_train
    assert val_idx == ref_val


def test_failure_demo_needs_a_fail_bin(tmp_path):
    """실패 데모가 섞였는데 fail_bin이 없으면 조용히 잘못된 라벨을 붙이는 대신 즉시 죽어야 한다."""
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[4, 4], success=[True, False])

    with pytest.raises(ValueError, match="fail_bin"):
        DinoFeatureWindows(str(cache), obs_horizon=2)

    ds = DinoFeatureWindows(str(cache), obs_horizon=2, fail_bin=99)
    np.testing.assert_array_equal(ds.labels(), [3, 2, 1, 0, 99, 99, 99, 99])


def test_frame_stats_use_training_demos_only(tmp_path):
    """표준화 통계가 val 데모를 보면 누출이다 — train 인덱스만 봐야 한다."""
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[4] * 10)
    ds = DinoFeatureWindows(str(cache), obs_horizon=2)
    train_idx, val_idx, val_demos = ds.split_indices(val_fraction=0.2, seed=0)

    mean, std = ds.compute_frame_stats(train_idx)
    train_frames = np.concatenate(
        [ds.frames[n] for n in sorted(ds.frames) if n not in val_demos], axis=0
    )
    np.testing.assert_allclose(mean, train_frames.mean(0), rtol=1e-5)
    assert (std > 0).all()

    ds.apply_frame_stats(mean, std)
    x, _ = ds[train_idx[0]]
    assert torch.isfinite(x).all()
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
PYTHONPATH=src /home/hymm/miniconda3/bin/python -m pytest tests/test_dino_stg.py -v
```

Expected: 앞선 2개는 PASS, 새 5개는 FAIL — `ModuleNotFoundError: No module named 'square_assembly.datasets.dino_feature_dataset'`

- [ ] **Step 3: 데이터셋을 구현한다**

`projects/square_assembly/src/square_assembly/datasets/dino_feature_dataset.py` 생성:

```python
"""DINO 특징 캐시(scripts/cache_dino_feats.py 산출물)를 STG 학습 샘플로 자른다.

인덱싱은 기준선(scripts/train_dstg.py)이 쓰는 robomimic SequenceDataset
(frame_stack=To, seq_length=1, pad_frame_stack=True, pad_seq_length=True) +
RobomimicSequenceDataset.get_time_to_success와 **정확히 같게** 맞춘다:

  - 데모 d의 샘플 수 = L_d, index_in_demo t = 0..L_d-1
  - 관측 윈도우 = 프레임 clip(t-To+1 .. t, 0, L_d-1)   (앞쪽 패딩 = 첫 프레임 복제)
  - 라벨 = L_d - 1 - t (성공 데모) / fail_bin (실패 데모)

여기가 반 칸만 어긋나도 학습은 멀쩡히 돌고 val 지표만 조용히 틀리므로, 이 동일성은
tests/test_dino_stg.py가 손으로 계산한 기대값과 train_dstg._episode_split 대조로 못 박는다.

robomimic을 import하지 않는다 — 캐시에 필요한 정보(데모 이름, 길이, 성공 여부)가 전부 들어 있다.
"""

import h5py
import numpy as np
import torch


def episode_split_by_name(demo_names, val_fraction, seed):
    """train_dstg._episode_split과 동일한 val 데모 집합을, 데모 이름만으로 재현한다.

    원본은 robomimic의 _index_to_demo_id에서 데모 이름을 뽑아
    sorted(set(...)) (사전순) -> RandomState(seed).permutation -> 앞 n_val개를 val로 쓴다.
    캐시엔 데모 이름이 그대로 남아 있으므로 robomimic 없이 같은 결과가 나온다.
    """
    unique = sorted(set(demo_names))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique))
    n_val = max(1, int(round(len(unique) * val_fraction)))
    return {unique[i] for i in perm[:n_val]}


class DinoFeatureWindows(torch.utils.data.Dataset):
    """캐시를 통째로 메모리에 올려 (obs 윈도우, steps-to-go 라벨) 샘플을 낸다.

    Args:
        cache_path (str): cache_dino_feats.py가 만든 hdf5.
        obs_horizon (int): To — 기준선 정책과 같은 값을 써야 한다(configs/policy/diffusion_unet.yaml: 2).
        fail_bin (int | None): 실패 데모에 붙일 별도 클래스. 실패 데모가 있는데 None이면 죽는다.
    """

    def __init__(self, cache_path, obs_horizon, fail_bin=None):
        self.obs_horizon = obs_horizon
        self.fail_bin = fail_bin
        self.frames, self.lengths, self.success = {}, {}, {}
        self.samples = []

        with h5py.File(cache_path, "r") as f:
            for name in sorted(f["data"].keys()):
                g = f["data"][name]
                length = int(g.attrs["length"])
                # ponytail: 캐시 전체를 float32로 메모리에 올린다. 7.5k 프레임(=46MB)엔 과하지 않지만
                # 19만 프레임(square_scale3_1000)이면 ~1.2GB — 그때 float16 유지 + 배치 단위 캐스팅으로 바꾼다.
                feat = np.asarray(g["feat"][:], dtype=np.float32)
                low = np.asarray(g["lowdim"][:], dtype=np.float32)
                self.frames[name] = np.concatenate([feat, low], axis=-1)
                self.lengths[name] = length
                self.success[name] = bool(g.attrs["is_success"])
                self.samples += [(name, t) for t in range(length)]

        if not self.samples:
            raise ValueError(f"{cache_path}에 데모가 없다")
        if fail_bin is None and not all(self.success.values()):
            raise ValueError(
                "실패 데모가 섞여 있는데 fail_bin이 없다 — 실패 transition에 성공 기준 "
                "steps-to-go 라벨을 붙이는 건 범주 오류다(train_dstg_failaware.py 참고)"
            )
        self.frame_dim = next(iter(self.frames.values())).shape[1]

    def __len__(self):
        return len(self.samples)

    def labels(self):
        """(N,) int64 — num_bins 결정과 로깅용. __getitem__을 N번 부르지 않고 한 번에 계산한다."""
        return np.array(
            [
                (self.lengths[n] - 1 - t) if self.success[n] else self.fail_bin
                for n, t in self.samples
            ],
            dtype=np.int64,
        )

    def split_indices(self, val_fraction, seed):
        val_demos = episode_split_by_name(list(self.frames), val_fraction, seed)
        train_idx = [i for i, (n, _) in enumerate(self.samples) if n not in val_demos]
        val_idx = [i for i, (n, _) in enumerate(self.samples) if n in val_demos]
        return train_idx, val_idx, val_demos

    def compute_frame_stats(self, indices):
        """주어진 샘플 인덱스가 속한 데모들의 프레임으로 (mean, std)를 낸다 — val 누출 방지."""
        names = sorted({self.samples[i][0] for i in indices})
        stacked = np.concatenate([self.frames[n] for n in names], axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std[std < 1e-6] = 1.0  # 상수 차원(예: 항상 같은 값인 lowdim)에서 0으로 나누지 않게
        return mean.astype(np.float32), std.astype(np.float32)

    def apply_frame_stats(self, mean, std):
        for name in self.frames:
            self.frames[name] = (self.frames[name] - mean) / std

    def __getitem__(self, i):
        name, t = self.samples[i]
        length = self.lengths[name]
        idx = np.clip(np.arange(t - self.obs_horizon + 1, t + 1), 0, length - 1)
        x = self.frames[name][idx].reshape(-1)
        label = (length - 1 - t) if self.success[name] else self.fail_bin
        return torch.from_numpy(x.copy()), int(label)
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
PYTHONPATH=src /home/hymm/miniconda3/bin/python -m pytest tests/test_dino_stg.py -v
```

Expected: 7 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/hymm/Projects/self-improving-gym
git add projects/square_assembly/src/square_assembly/datasets/dino_feature_dataset.py projects/square_assembly/tests/test_dino_stg.py
git commit -m "$(cat <<'EOF'
feat(data): window cached DINO features exactly like the robomimic baseline

- add DinoFeatureWindows: L samples per demo, first-frame padding, label = L-1-t, optional fail_bin
- add episode_split_by_name reproducing train_dstg._episode_split from demo names alone, so val demos are identical without robomimic
- compute standardization stats from training demos only
- pin both the indexing rule and the split equality with tests

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019LW9kqFwrDYHnMCU8jgS8X
EOF
)"
```

---

## Task 3: STG 헤드 + 학습 스크립트

**Files:**
- Create: `projects/square_assembly/src/square_assembly/policies/diffusion/dino_stg_predictor.py`
- Create: `projects/square_assembly/src/square_assembly/scripts/train_dstg_dino.py`
- Create: `projects/square_assembly/src/square_assembly/configs/train_dstg_dino.yaml`
- Modify: `projects/square_assembly/tests/test_dino_stg.py` (테스트 추가)

**Interfaces:**
- Consumes: `DinoFeatureWindows`, `episode_split_by_name` (Task 2)
- Produces:
  - `DinoStgHead(in_dim, num_bins, head_hidden=(256,256))` — `forward(x: (B,in_dim)) -> (B,num_bins)` logits
  - `save_checkpoint(path, head, meta)` / `load_checkpoint(path, device="cpu") -> (DinoStgHead, dict)`
  - 체크포인트 키: `model`, `in_dim`, `num_bins`, `head_hidden`, `obs_horizon`, `frame_mean`, `frame_std`, `cache_path`, `cache_meta`, `val_nll`, `val_mae`, `n_train_demos`, `n_val_demos`
  - CLI: `python -m square_assembly.scripts.train_dstg_dino cache_path=... out=...`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`projects/square_assembly/tests/test_dino_stg.py` 맨 아래에 추가:

```python
from square_assembly.policies.diffusion.dino_stg_predictor import (
    DinoStgHead,
    load_checkpoint,
    save_checkpoint,
)


def test_head_shape_and_gradients():
    head = DinoStgHead(in_dim=12, num_bins=7, head_hidden=(8, 8))
    x = torch.randn(3, 12)

    logits = head(x)
    assert logits.shape == (3, 7)

    logits.sum().backward()
    grads = [p.grad for p in head.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    """온라인 추론이 학습과 같은 전처리를 재현할 수 있어야 하므로, 전처리 메타가
    체크포인트에 남고 헤드가 그대로 복원돼야 한다."""
    head = DinoStgHead(in_dim=12, num_bins=7, head_hidden=(8, 8))
    x = torch.randn(3, 12)
    expected = head(x)

    path = tmp_path / "predictor.pt"
    save_checkpoint(
        str(path),
        head,
        {
            "in_dim": 12,
            "num_bins": 7,
            "head_hidden": [8, 8],
            "obs_horizon": 2,
            "frame_mean": np.zeros(6, dtype=np.float32),
            "frame_std": np.ones(6, dtype=np.float32),
            "cache_meta": {"model": "vit_small_patch14_reg4_dinov2.lvd142m", "crop": 76, "size": 224},
        },
    )

    loaded, ckpt = load_checkpoint(str(path))
    torch.testing.assert_close(loaded(x), expected)
    assert ckpt["cache_meta"]["crop"] == 76
    assert ckpt["num_bins"] == 7
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
PYTHONPATH=src /home/hymm/miniconda3/bin/python -m pytest tests/test_dino_stg.py -v
```

Expected: 앞선 7개 PASS, 새 2개 FAIL — `ModuleNotFoundError: ... dino_stg_predictor`

- [ ] **Step 3: 헤드 모듈을 구현한다**

`projects/square_assembly/src/square_assembly/policies/diffusion/dino_stg_predictor.py` 생성:

```python
"""사전추출된 DINO 특징 위의 STG(steps-to-go) 카테고리컬 헤드.

역할은 기존 DstgPredictor(dstg_predictor.py)와 같다 — 논문 식(1)
d(o,g) := E[steps-to-go | o]를 bin categorical로 예측한다. 다른 건 입력뿐이다:
얼린 diffusion policy 인코더의 global_cond 대신, scripts/cache_dino_feats.py가
미리 구운 DINO 특징 윈도우를 받는다. 원본은 비교 기준선이라 건드리지 않고 새 파일로 둔다
(ADR-005).

헤드 구조는 기준선과 동일하게 유지한다: MLP(head_hidden) -> Linear(num_bins, bias=False).
인코더 말고 다른 변인이 끼면 비교가 성립하지 않는다.

체크포인트엔 전처리 메타(cache_meta: model/crop/size)와 표준화 통계(frame_mean/frame_std)를
같이 저장한다 — 후속 작업(SI 루프 온라인 보상)이 encode_frames()에 같은 설정을 그대로
먹여 train/serve 불일치 없이 재현할 수 있게 하기 위함이다.
"""

import numpy as np
import torch
import torch.nn as nn


class DinoStgHead(nn.Module):
    def __init__(self, in_dim, num_bins, head_hidden=(256, 256)):
        super().__init__()
        layers, d = [], in_dim
        for h in head_hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, num_bins, bias=False))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, obs_horizon * frame_dim) 표준화된 특징 윈도우 -> (B, num_bins) logits."""
        return self.head(x)


def save_checkpoint(path, head, meta):
    payload = {"model": head.head.state_dict()}
    payload.update(meta)
    for key in ("frame_mean", "frame_std"):
        if key in payload and isinstance(payload[key], np.ndarray):
            payload[key] = payload[key].tolist()
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    """반환: (head, ckpt). ckpt의 cache_meta/frame_mean/frame_std로 전처리를 그대로 재현할 수 있다."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head = DinoStgHead(ckpt["in_dim"], ckpt["num_bins"], tuple(ckpt["head_hidden"]))
    head.head.load_state_dict(ckpt["model"])
    return head.to(device).eval(), ckpt
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
PYTHONPATH=src /home/hymm/miniconda3/bin/python -m pytest tests/test_dino_stg.py -v
```

Expected: 9 passed.

- [ ] **Step 5: hydra 설정을 만든다**

`projects/square_assembly/src/square_assembly/configs/train_dstg_dino.yaml` 생성:

```yaml
# DINO 특징 캐시 위의 STG 예측기 학습 — scripts/train_dstg_dino.py.
# 하이퍼파라미터는 ResNet 기준선(train_dstg.yaml + outputs/dstg/hydra/*/.hydra/config.yaml
# 실측값)과 동일하게 고정한다. 인코더 말고 다른 변인이 끼면 val_mae 비교가 성립하지 않는다.

cache_path: ???   # scripts/cache_dino_feats.py 산출물
out: outputs/dstg_dino/predictor.pt
num_bins_override: 501   # 기준선과 같은 bin 공간을 써야 mae/nll을 그대로 비교할 수 있다
fail_bin: null           # 실패 롤아웃 데이터를 쓸 때만 설정(이번 실험은 PH 성공 데모만)

obs_horizon: 2           # configs/policy/diffusion_unet.yaml과 동일
seed: 1
device: cuda
num_workers: 0           # 특징이 이미 메모리에 있어 워커가 이득이 없다

batch_size: 64
lr: 1.0e-4
weight_decay: 1.0e-6
num_epochs: 30
log_every: 5
head_hidden: [256, 256]

val_fraction: 0.2
split_seed: 0

hydra:
  run:
    dir: outputs/dstg_dino/hydra/${now:%Y%m%d-%H%M%S}
```

- [ ] **Step 6: 학습 스크립트를 구현한다**

`projects/square_assembly/src/square_assembly/scripts/train_dstg_dino.py` 생성:

```python
"""DINO 특징 캐시 위의 STG 예측기 학습 진입점.

scripts/train_dstg.py(ResNet 기준선)의 학습 루프·평가 지표·분할 규칙을 그대로 따르되,
데이터 경로만 바꾼다 — 얼린 diffusion policy 인코더 대신 미리 구운 DINO 특징을 읽는다.
그래서 policy_ckpt / run_config.yaml / normalization_stats.json 의존이 전부 사라지고,
robomimic도 필요 없다.

비교가 목적이므로 하이퍼파라미터(bin 공간, 분할 시드, lr, epochs, head 크기)는
기준선과 같은 값을 기본값으로 둔다 — configs/train_dstg_dino.yaml 참고.

사용:
    python -m square_assembly.scripts.train_dstg_dino \
        cache_path=data/square_scale3_0_seed1.dino.h5 \
        out=outputs/dstg_dino/square_scale3_0_seed1/predictor.pt
"""

import json
import logging
import os

import h5py
import hydra
import numpy as np
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


@hydra.main(config_path="../configs", config_name="train_dstg_dino", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    dataset = DinoFeatureWindows(cfg.cache_path, cfg.obs_horizon, fail_bin=cfg.get("fail_bin"))
    with h5py.File(cfg.cache_path, "r") as f:
        cache_meta = json.loads(f.attrs["meta"])
    logger.info(f"cache={cfg.cache_path} meta={cache_meta}")

    train_idx, val_idx, val_demos = dataset.split_indices(cfg.val_fraction, cfg.split_seed)
    n_val_demos = len(val_demos)
    n_train_demos = len(dataset.frames) - n_val_demos
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
                f"num_bins_override({num_bins})가 실제 관측된 최대 라벨({max_label})보다 작거나 같다 — 라벨이 잘린다."
            )
    else:
        num_bins = max_label + 1
    in_dim = cfg.obs_horizon * dataset.frame_dim
    logger.info(f"dataset len={len(dataset)} in_dim={in_dim} num_bins={num_bins} (max label={max_label})")

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
    logger.info(f"[val] nll={val_nll:.4f} mae={val_mae:.3f} (n={len(val_idx)} samples, {n_val_demos} demos)")
    print({"val_nll": val_nll, "val_mae": val_mae, "num_bins": num_bins,
           "n_train_demos": n_train_demos, "n_val_demos": n_val_demos}, flush=True)

    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    save_checkpoint(cfg.out, head, {
        "in_dim": in_dim,
        "num_bins": num_bins,
        "head_hidden": list(cfg.head_hidden),
        "obs_horizon": cfg.obs_horizon,
        "fail_bin": cfg.get("fail_bin"),
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
```

- [ ] **Step 7: Task 1의 2데모 스모크 캐시로 end-to-end 실행**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
SCRATCH=/tmp/claude-1000/-home-hymm-Projects-self-improving-gym/f01ae550-9c56-4f03-9cdf-493e0bef2519/scratchpad
PYTHONPATH=src /home/hymm/miniconda3/bin/python -m square_assembly.scripts.train_dstg_dino \
    cache_path=$SCRATCH/smoke.dino.h5 \
    out=$SCRATCH/smoke_predictor.pt \
    device=cpu num_epochs=2 log_every=1
```

Expected: 2 에폭 로그 + `{'val_nll': ..., 'val_mae': ..., 'num_bins': 501, 'n_train_demos': 1, 'n_val_demos': 1}`. (데모 2개짜리 스모크라 지표 자체엔 의미가 없다 — 파이프라인이 끝까지 도는지만 본다.)

- [ ] **Step 8: 전체 테스트를 돌려 회귀가 없는지 확인한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
PYTHONPATH=src /home/hymm/miniconda3/bin/python -m pytest tests/test_dino_stg.py -v
```

Expected: 9 passed.

> 나머지 `tests/`(test_dstg_predictor.py 등)는 robomimic·실서버 데이터 경로를 요구해 이 PC에서 수집조차 안 된다 — 이번 작업은 그 파일들을 건드리지 않으므로 회귀 대상이 아니다.

- [ ] **Step 9: 커밋**

```bash
cd /home/hymm/Projects/self-improving-gym
git add projects/square_assembly/src/square_assembly/policies/diffusion/dino_stg_predictor.py \
        projects/square_assembly/src/square_assembly/scripts/train_dstg_dino.py \
        projects/square_assembly/src/square_assembly/configs/train_dstg_dino.yaml \
        projects/square_assembly/tests/test_dino_stg.py
git commit -m "$(cat <<'EOF'
feat(dstg): train the steps-to-go head on cached DINO features

- add DinoStgHead with the baseline head shape (MLP -> Linear(num_bins, bias=False)) so the encoder stays the only changed variable
- add train_dstg_dino.py: same loss, metrics, split seed and bin space as train_dstg.py, but reading the feature cache
- drop the policy_ckpt / run_config / normalization_stats dependency chain entirely
- store preprocessing meta and standardization stats in the checkpoint for a later online path

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019LW9kqFwrDYHnMCU8jgS8X
EOF
)"
```

---

## Task 4: 본 실험 — 전체 캐시 + 학습 + 기준선 비교

**Files:**
- Create: `experiments/2026-09-09_dino-stg-encoder.md`
- 산출물(git 미추적): `projects/square_assembly/data/square_scale3_0_seed1.dino.h5`, `projects/square_assembly/outputs/dstg_dino/square_scale3_0_seed1/predictor.pt`

**Interfaces:**
- Consumes: Task 1~3의 CLI 전부

- [ ] **Step 1: 전체 데이터셋 특징 캐시를 굽는다**

50 데모 / 7,561 프레임 × 카메라 2대 = 15,122 이미지. 16코어 CPU에서 10~25분 예상.

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
time PYTHONPATH=src /home/hymm/miniconda3/bin/python -m square_assembly.scripts.cache_dino_feats \
    --hdf5 data/square_scale3_0_seed1.hdf5 \
    --out data/square_scale3_0_seed1.dino.h5 2>&1 | tail -20
```

Expected: 데모 50개 라인 후 `saved: ...`. 파일 크기 ~23MB (7561 × 1536 × 2B).

- [ ] **Step 2: 기준선과 동일한 조건으로 학습한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
PYTHONPATH=src /home/hymm/miniconda3/bin/python -m square_assembly.scripts.train_dstg_dino \
    cache_path=data/square_scale3_0_seed1.dino.h5 \
    out=outputs/dstg_dino/square_scale3_0_seed1/predictor.pt \
    device=cpu 2>&1 | tail -20
```

Expected: 30 에폭 로그 + 최종 `{'val_nll': ..., 'val_mae': ..., 'num_bins': 501, 'n_train_demos': 40, 'n_val_demos': 10}`.

- [ ] **Step 3: 분할이 실제로 기준선과 같은지 확인한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
PYTHONPATH=src /home/hymm/miniconda3/bin/python - <<'EOF'
from square_assembly.datasets.dino_feature_dataset import DinoFeatureWindows
ds = DinoFeatureWindows("data/square_scale3_0_seed1.dino.h5", obs_horizon=2)
tr, va, val_demos = ds.split_indices(0.2, 0)
print("val demos:", sorted(val_demos))
print("n_train_samples", len(tr), "n_val_samples", len(va), "total", len(ds))
EOF
```

Expected: val 데모 10개, `total 7561`. (샘플 총수가 7561이면 인덱싱이 기준선과 같은 샘플 공간을 쓴다는 뜻이다.)

- [ ] **Step 4: 기준선과 나란히 비교한다**

```bash
cd /home/hymm/Projects/self-improving-gym/projects/square_assembly
/home/hymm/miniconda3/bin/python - <<'EOF'
import torch
base = torch.load("outputs/dstg/square_scale3_0_seed1/predictor.pt", map_location="cpu", weights_only=False)
dino = torch.load("outputs/dstg_dino/square_scale3_0_seed1/predictor.pt", map_location="cpu", weights_only=False)
print(f"{'':10} {'val_mae':>9} {'val_nll':>9} {'num_bins':>9}")
for tag, c in [("resnet", base), ("dino", dino)]:
    print(f"{tag:10} {c['val_mae']:9.3f} {c['val_nll']:9.3f} {c['num_bins']:9d}")
print(f"ratio(mae) = {dino['val_mae'] / base['val_mae']:.3f}")
EOF
```

판정 (spec §4):
- ratio ≤ 1.10 → **동등 이상**. 성공으로 보고 다음 단계(더 큰 데이터 / SI 루프 연결)를 사용자와 논의.
- ratio > 1.20 → **실패**. spec §3의 승급 경로(카메라당 4×4 pooled patch grid = 6144차원)를 사용자에게 제안하고 **승인 전에는 진행하지 않는다**.
- 1.10 < ratio ≤ 1.20 → 애매. 판단을 사용자에게 넘긴다.

- [ ] **Step 5: 실험 기록을 남긴다**

`experiments/2026-09-09_dino-stg-encoder.md`를 `experiments/LOG_TEMPLATE.md` 형식으로 작성한다. 반드시 포함할 것:
- 수정/신규 파일 목록과 각각의 역할
- 실제로 로드된 모델명·timm 버전·전처리(crop/size)·특징 차원 (체크포인트 `cache_meta`에서 그대로 인용)
- 캐시 소요 시간, 캐시 파일 크기, 학습 소요 시간
- 기준선 대비 표: `val_mae`, `val_nll`, ratio (Step 4 출력 그대로)
- 알려진 트레이드오프: 학습 시 RandomCrop 증강 상실 (spec §3)
- 다음 후보: 4×4 pooled patch grid, DINOv3(timm 업그레이드 필요), `square_scale3_1000_*`로 데이터 확장, `dstg_reward.py` 온라인 경로 연결

- [ ] **Step 6: 커밋**

```bash
cd /home/hymm/Projects/self-improving-gym
git add experiments/2026-09-09_dino-stg-encoder.md
git commit -m "$(cat <<'EOF'
docs(experiment): record the DINO-feature steps-to-go result vs the ResNet baseline

- report val_mae / val_nll against outputs/dstg/square_scale3_0_seed1 under identical bins, split seed and hyperparameters
- record the resolved backbone, timm version, preprocessing and feature dimensions
- note the lost train-time RandomCrop augmentation and the pooled-patch-grid upgrade path

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019LW9kqFwrDYHnMCU8jgS8X
EOF
)"
```

> `experiments/`는 git 미추적일 수 있다(ARCHITECTURE.md §3). `git add`가 무시되면 파일만 남기고 커밋은 건너뛴 뒤 그 사실을 보고한다.

---

## Self-Review

**Spec coverage:**
- §2 캐시 파이프라인 → Task 1
- §3 백본/전처리/특징 형태/캐시 포맷 → Task 1 (모델·crop·size·CLS+mean patch·hdf5 레이아웃)
- §3 lowdim 정규화(train 데모 mean/std) → Task 2 `compute_frame_stats` + Task 3 저장
- §3 온라인 경로 = 범위 밖, 전처리 단일화 + 메타 저장으로 대비 → Task 1 `encode_frames`, Task 3 체크포인트 메타
- §4 비교 조건 고정·분할 동일성 → Task 2 `episode_split_by_name` + 대조 테스트, Task 3 config 기본값, Task 4 Step 3~4
- §5 로컬 CPU 실행 → Global Constraints의 인터프리터 지정, Task 4 Step 1~2
- §6 fail-aware는 범위 밖이되 구조는 지원 → Task 2 `fail_bin` + 테스트, Task 3 config `fail_bin: null`
- §7 리스크 4개 → 인덱싱(Task 2 테스트), 모델명 부재(Task 1 Step 5 대응), 다운로드 실패(Task 1 Step 5), 저해상도(Task 4 Step 4 판정 기준)

**Placeholder scan:** 없음 — 모든 코드 스텝에 실제 코드가 들어 있고, 모든 실행 스텝에 실제 명령과 기대 출력이 있다.

**Type consistency:** `encode_frames` 반환 `(T, 2*D) float16` → 캐시 `feat` → `DinoFeatureWindows.frames`(feat+lowdim concat, float32) → `__getitem__` `(obs_horizon*frame_dim,)` → `DinoStgHead(in_dim=obs_horizon*frame_dim)`. `save_checkpoint`가 쓰는 키와 `load_checkpoint`가 읽는 키(`in_dim`/`num_bins`/`head_hidden`/`model`)가 일치하고, Task 3 Step 1의 테스트가 그 계약을 그대로 호출한다.
