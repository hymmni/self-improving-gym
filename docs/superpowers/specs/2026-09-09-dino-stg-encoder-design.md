# DINO 인코더 기반 STG 예측기 (square task) — 설계

## 1. 배경

`square_assembly`의 steps-to-go 예측기 `d(o,g) := E[steps-to-go | o]`는 현재
**이미 학습된 diffusion policy의 비전 인코더(ResNet18 + SpatialSoftmax)를 얼려서
재사용**하고, 그 위에 STG bin을 예측하는 MLP 헤드만 학습한다
(`policies/diffusion/dstg_predictor.py`, `scripts/train_dstg.py`).

이 방식의 제약:

- 인코더가 **그 diffusion policy 체크포인트에 종속**된다 — 정책이 바뀌면 STG 예측기의
  특징 공간도 같이 바뀌고, `policy_ckpt` + `run_config.yaml` + `normalization_stats.json`
  세 파일이 항상 붙어다녀야 한다.
- 인코더가 얼어 있는데도 **매 에폭 이미지를 다시 로드하고 ResNet을 다시 통과**시킨다.
  현재 학습 시간의 대부분이 여기서 나온다.
- 인코더가 robomimic square 50 데모로만 학습된 특징이라 **일반적인 시각 사전지식이 없다**.

## 2. 제안

인코더를 **DINOv2(자기지도 사전학습 ViT)** 로 교체한다. DINO도 얼려 쓰므로,
"같은 프레임 → 항상 같은 특징"이 성립한다 → **특징을 1회 사전추출해 캐시**하고,
학습은 캐시된 특징 벡터 위의 MLP만 돌린다.

```
robomimic hdf5 ──[cache_dino_feats.py, 1회]──> <name>.dino.h5
                                                     │
                                       [train_dstg_dino.py] ──> MLP head (num_bins CE)
```

기대 효과:
1. 학습 루프에서 이미지 로딩·인코더 forward가 사라짐 → 에폭이 초 단위.
2. STG 예측기가 diffusion policy 체크포인트로부터 **독립**.
3. 백본 교체(DINOv2 ↔ DINOv3 ↔ 사이즈 변경) 비용 = 캐시 스크립트 재실행 1회.

## 3. 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 백본 | **DINOv2 ViT-S/14 (reg4)** — `vit_small_patch14_reg4_dinov2.lvd142m` | `requirements.txt`에 이미 `timm==0.9.16`이 pin돼 있어 **의존성 추가 0**. reg(레지스터) 버전이 동일 비용에 특징이 더 깔끔. |
| DINOv3 | 이번엔 안 씀. `--model` 인자로 교체 가능하게만 열어둠 | timm 1.0.20+ 필요(서버 env는 0.9.16). timm pin은 원래 OpenVLA 때문인데 OpenVLA는 이 서브셋에 없고 코드 어디서도 timm을 import하지 않으므로, 나중에 올리는 건 안전하다. v2가 기준선에 못 미치면 그때 시도. |
| 입력 전처리 | `84×84 → CenterCrop 76 → resize 224` | crop 76은 기준선(`configs/policy/diffusion_unet.yaml: crop_hw=[76,76]`)의 **eval 시 전처리와 동일** — 시야각을 맞춰야 비교가 성립. 224는 patch14로 나눠떨어지는 표준 해상도(16×16 패치). |
| 특징 형태 | 카메라당 `[CLS ; mean(patch tokens)]` = 384×2 = **768**, 카메라 2대 → 프레임당 1536 | CLS만 쓰면 공간 정보가 약하고, 전체 패치 토큰(256×384)을 캐싱하면 ~75GB로 논외. 중간값부터 시작. **기준선에 못 미치면 4×4 pooled patch grid(카메라당 6144)로 승급**한다. |
| 캐시 포맷 | hdf5, 데모별 `feat (T,1536) fp16` + `lowdim (T,9) fp32` + attrs `is_success`,`length` | 학습 시 원본 hdf5·robomimic이 아예 필요 없어짐. 19만 프레임(최대 데이터)도 ~580MB. |
| lowdim 정규화 | 학습 스크립트가 **train 데모에서 프레임 단위 mean/std**를 계산해 feat+lowdim 전체에 적용, 체크포인트에 저장 | 기준선의 MinMax 통계는 policy 디렉토리에 종속. 표준화 1개로 두 종류 특징을 함께 처리하는 게 더 짧다. |
| 증강 | 없음 (CenterCrop 고정) | 기준선은 학습 시 RandomCrop(76/84)을 쓴다. 캐시 방식은 이걸 잃는다 — **알려진 트레이드오프**. 필요해지면 프레임당 K개 랜덤크롭 뷰를 캐싱하는 게 업그레이드 경로. |
| 온라인(SI 루프) 경로 | **이번 범위 밖.** 단, 전처리는 `encode_frames()` 하나로 통일하고 체크포인트에 `model/crop/size/frame_mean/frame_std`를 저장해 나중에 그대로 재현 가능하게 한다 | `dstg_reward.py` 연결은 별도 작업. train/serve 전처리 불일치만 구조적으로 막아둔다. |

## 4. 비교 조건 (고정)

DINO 버전의 성패는 **기존 ResNet 기준선과 같은 조건에서의 val 지표**로 판단한다.

- 데이터: `data/square_scale3_0_seed1.hdf5` — robomimic square PH 데모 50개, 7,561 프레임, 전부 성공, 최장 219스텝.
- 기준선: `outputs/dstg/square_scale3_0_seed1/predictor.pt` → **val_mae 15.85, val_nll 4.67**
  (같은 데이터의 seed2: 15.89 / 4.68 — 시드 간 편차는 0.05 수준이라 비교 기준으로 안정적)
- 동일하게 유지할 하이퍼파라미터 (`outputs/dstg/hydra/*/.hydra/config.yaml` 실측):
  `num_bins_override=501`, `split_seed=0`, `val_fraction=0.2`, `batch_size=64`,
  `lr=1e-4`, `weight_decay=1e-6`, `num_epochs=30`, `head_hidden=[256,256]`, `obs_horizon=2`
- **val 분할이 완전히 같아야 한다**: `train_dstg._episode_split`은
  `sorted(set(demo_ids))`(사전순) → `np.random.RandomState(split_seed).permutation` →
  앞 `n_val`개를 val로 쓴다. robomimic 없이도 데모 이름 목록만으로 재현 가능하며,
  이 동일성은 테스트로 못 박는다.

판정: val_mae가 기준선의 ±10% 이내면 "동등", 그보다 낮으면 "개선", 20% 이상 높으면
"실패 → 특징 형태를 4×4 pooled patch grid로 승급 후 재시도".

## 5. 실행 환경

- GPU 서버 `rupy`는 현재 이 PC에서 접속 불가(DNS), 로컬 PC엔 GPU도 robomimic도 없다.
- 하지만 **이 작업은 robomimic이 필요 없다** — 캐시 스크립트는 h5py로 hdf5를 직접 읽고,
  학습은 캐시만 읽는다. 필요한 패키지(torch, timm, h5py, hydra, numpy)는 로컬
  base env(`/home/hymm/miniconda3/bin/python`, torch 2.10 / timm 1.0.20)에 이미 다 있다.
- 규모 추정: 7,561프레임 × 카메라 2대 = 15,122 이미지. ViT-S/14@224는 이미지당 ~4.6 GFLOPs
  → 16코어 CPU에서 10~25분. 학습은 7,561샘플 MLP 30에폭이라 CPU로 수 분.
- **결론: 첫 실험은 로컬 CPU로 완결한다.** 서버가 필요해지는 건 더 큰 데이터
  (`square_scale3_1000_*`, 19만 프레임)로 확장할 때다.
- 로컬 base env는 timm 1.0.20, 서버 canonical env는 0.9.16 → 실제로 로드된
  모델명·timm 버전을 체크포인트 meta에 기록해 추적 가능하게 한다.

## 6. 비목표 (out of scope)

- fail-aware 라벨링(실패 롤아웃 포함) — 코드 구조는 `is_success`/`fail_bin`을 지원하되,
  이번 실험은 PH 성공 데모만 쓴다.
- `dstg_reward.py` / SI 루프 연결.
- 기존 `dstg_predictor.py`·`train_dstg.py`·`train_dstg_failaware.py`·
  `robomimic_dataset.py` 수정 — 기준선이므로 보존한다 (ADR-005).
- DINOv3, 더 큰 백본(ViT-B/L), 랜덤크롭 뷰 캐싱 — 전부 후속 실험 경로로만 열어둔다.

## 7. 리스크

| 리스크 | 대응 |
|---|---|
| 윈도우/라벨 인덱싱이 기준선과 반 칸 어긋남 → 학습은 멀쩡히 돌고 결과만 틀림 | 가장 큰 위험. 손으로 계산한 기대값 테스트 + `train_dstg._episode_split`와의 분할 일치 테스트로 못 박는다. |
| `timm==0.9.16`에 `..._reg4_...` 모델명이 없음 | `--model`로 교체 가능. 폴백은 `vit_small_patch14_dinov2.lvd142m`. |
| HF hub에서 사전학습 가중치 다운로드 실패(오프라인) | Task 1 스모크에서 즉시 드러난다. |
| 84×84 저해상도 시뮬 이미지가 DINO 학습 분포와 멀어 특징이 약할 수 있음 | 이 실험이 확인하려는 것 자체. 실패 시 §3의 승급 경로(패치 그리드)로 진행. |
