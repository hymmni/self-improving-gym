# STG 롤아웃 규모 실험 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 정책 롤아웃 수 n을 10 단위로 늘리며 VIP STG 예측기를 학습·평가하는 스윕(정책 2개, 약 2,200개 예측기)에 필요한 데이터·학습 옵션·평가·실행 묶음을 만들고, 로컬에서 시험 실행해 전체 소요 시간을 잰다. 전체 스윕은 이 플랜 범위 밖이다(사용자 확인 뒤 띄운다).

**Architecture:** VIP 특징은 원본 hdf5마다 한 번만 뽑는다(`cache_vip_feats.py` 그대로). 그 캐시들을 이어 붙여 정책·개입 조건별 학습 캐시와 정책별 평가 캐시를 만든다(`stg_scaling_data.py`, 원본 이름·역할·채점 시작점을 attrs에, action_mode를 그룹에 넣어 캐시 하나가 `mode_hdf5`까지 겸한다). 스윕 모듈(`stg_scaling_sweep.py`)이 원본 이름 순열로 학습 에피소드 목록을 정해 `train_dstg_vip.py`(새 옵션 `num_steps`, `train_episodes_file`, `val_fraction=0`)와 평가 스크립트(`eval_stg_scaling.py`)를 하위 프로세스로 돌리고, 결과 폴더를 모아 표·그림을 만든다.

**Tech Stack:** PyTorch 2.7, h5py, hydra/omegaconf, numpy, scipy, matplotlib, pytest. robomimic·robosuite·mujoco는 학습·평가 경로에서 쓰지 않는다(Eddy는 도커 없이 돌릴 수 있어야 한다).

**Spec:** `docs/superpowers/specs/2026-11-20-stg-rollout-scaling-design.md`

## Global Constraints

- 브랜치 `feat/stg-rollout-scaling`. main 커밋·병합·push 금지. hook 우회 금지.
- 실행은 컨테이너 안: `docker compose exec -T dev /opt/venvs/torch/bin/python ...`, VIP 가중치가 필요하면 `-e HOME=/workspace/outputs/.model_cache`.
- 테스트: `docker compose exec -T dev bash -lc "cd /workspace && /opt/venvs/torch/bin/python -m pytest projects/square_assembly/tests -q"`. 기존 실패 `test_wandb_curve_logging_uses_epoch_and_skips_missing_cohort`는 무시한다.
- `data/`의 기존 파일은 읽기만 한다. 파생 산출물은 `outputs/stg_rollout_scaling_20261120/` 아래(캐시 `caches/`, 실행 결과 `runs/`).
- 기존 스크립트의 기본 동작은 그대로 둔다(새 옵션의 기본값 = 지금 동작).
- bin 수: random30 **522**, turn30 **888** (`num_bins_override`로 반드시 덮어쓴다). 범위 밖 라벨은 마지막 bin.
- 학습 프레임: 데모 전체 + 개입 에피소드의 intv 구간 + (넣는 조건이면) 정책 단독 성공 전체 = `success_tail_only=true`.
- 평가용 롤아웃: 정책마다 개입 15 + 정책 단독 성공 6, 세 조건 공통 에피소드에서. 모든 조건·n에서 같은 목록.
- 부분집합 순서 5개 × 학습 시드 2개 = 조건·n마다 예측기 10개. n = 0, 10, 20, …, 풀 크기(마지막은 풀 전체).
- 입력 7조합: `all`, `feat`, `lowdim`, `agentview`, `wrist`, `wrist_lowdim`(신규), `agentview_lowdim`(신규). 입력 비교는 지연 + 성공 포함 + 같은 에피소드 한 조건.
- 야매 성공(`hacky`)은 학습·평가 지표에서 빼고 개수만 보고한다.
- W&B는 끈다(예측기가 많아 폴더별 결과 파일로 남긴다, 2026-10-19 bin 비교와 같은 이유).
- 인프라 정보(서버 주소)는 커밋하지 않는다.
- 커밋: 영어, `type(scope): 동사 문장`, 본문 `-` 목록, 끝에 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_015opkswgWqcquERQKJmywUc`.

## 플랜에서 정한 값과 근거

스펙이 플랜에 맡긴 두 값과, 스펙이 정하지 않았지만 구현에 필요한 선택을 여기 적는다.

1. **gradient step 수(모든 n 공통):** Task 7에서 가장 큰 학습 세트(학습 프레임이 가장 많은 조건의 풀 전체)로 학습 곡선을 잰 뒤 정한다.
   - 방법: `record_curves=true`, `val_fraction=0.2`(학습 풀 안의 에피소드 분할 — 평가 세트는 건드리지 않아 고정 평가 세트가 길이 선택에 새지 않는다), 충분히 긴 epoch.
   - 규칙: 검증 성공-구간 MAE(`val_retained.mae`)가 최솟값의 2% 안으로 들어온 첫 step을 S80으로 두고, 학습 풀이 80%라 ×1.25 한 뒤 1,000 단위로 올린다.
   - 결과: (Task 7에서 채움)
2. **거짓 진전의 시작·끝:** 시작 = 분기 시작 프레임(`branch_step`)부터 8프레임의 d 평균, 끝 = 분기 마지막 8프레임의 d 평균. 거짓 진전 = 시작 − 끝 > 0.
   - 8 = 정책의 action chunk 길이(`action_horizon: 8`). RL에서 보상은 청크 경계마다 매겨지고 합치면 d(첫 관측) − d(마지막 관측)으로 접힌다. 한 프레임 값은 그 리턴과 정확히 같지만, 프레임마다 떠는 잡음(예전 `residual.step_noise_std`)이 0 근처 분기의 부호를 무작위로 뒤집는다. 한 청크 평균은 RL이 실제로 보는 시간 해상도 안에서 그 잡음만 줄인다.
   - 한 프레임 값(W=1)도 보조로 같이 낸다.
3. **평가용 롤아웃 고르기:** 수집 순서(`demo_k`의 k)로 정렬해 15(또는 6)개 연속 구간으로 나누고, 구간마다 `RandomState(0)`로 하나씩 뽑는다. 결과는 `projects/square_assembly/src/square_assembly/configs/stg_scaling/heldout_{policy}.json`에 커밋한다.
4. **순열:** 정책마다 (나) 학습 풀(개입 + 정책 단독 성공, 이름을 k 순으로 정렬)을 `RandomState(order)`로 섞는다(order = 0..4). (가)는 같은 순열에서 공통 에피소드가 아닌 이름만 걸러 낸다. 그래서 (가)와 (나)의 같은 n은 가능한 한 같은 원본 에피소드를 쓴다.
5. **학습 시드:** `seed = 2·order + s` (s = 0, 1). n = 0은 순서가 무의미하므로 이 규칙으로 서로 다른 시드 10개가 된다. n = 0은 입력 조합마다 모든 조건이 공유한다.
6. **같은 학습이 겹치면 한 번만:** (정책, 캐시, 입력, 시드, 학습 에피소드 목록)이 같은 실행은 하나의 job으로 합친다. 예: (나)에 추가 에피소드가 아직 안 들어온 n, 성공 제외 조건에서 뽑힌 n개에 성공이 없는 경우. 결과는 두 조건 모두에 그대로 쓴다.
7. **합친 hdf5 대신 출처별 캐시를 이어 붙인다.** 학습·평가에 필요한 건 특징·lowdim·action_mode·이름뿐이라 이미지까지 다시 쓰는 `merge_demo_hdf5`는 필요 없다. 데모와 정책 단독 성공을 세 번 인코딩하지 않는다(스펙 설계 11 "특징은 한 번만").
8. **평가 채점 규칙:**
   - 개입 성공 구간은 개입 시작 프레임부터 채점하고, 관측 창은 실제 직전 프레임을 쓴다(기존 `record_curves`의 검증 구간과 같다).
   - 라벨은 학습과 같게 마지막 bin으로 자른 값과 비교한다(기존 `eval_dstg`와 같다).
   - 실패 분기 = `is_success=False`인 분기. 성공한 분기는 야매면 "야매 제외", 아니면 "진짜 성공 제외"로 세기만 한다.
9. **부트스트랩:** 단위는 평가 에피소드. 층은 셋이다: 평가용 개입 원본 에피소드 15개(그 지연·즉시·타임아웃 버전과 분기 둘이 한 단위), 정책 단독 성공 6개, 평가용 데모 20개. 층 안에서 복원 추출 1,000번, 백분위 95% 구간.
   - 비율 지표(MAE, NLL, 거짓 진전 비율)는 단위별 합계로 정확히 다시 계산한다. 조건·n 점의 구간은 같은 재추출을 예측기 10개 모두에 적용해 평균을 낸다. 시드 편차는 10개의 표준편차로 따로 낸다.
   - AUROC와 보상 품질 지표는 점추정 + 시드 편차만 낸다.
10. **AUROC(보조):** 양성 = 실패 분기의 분기 시작 이후 프레임, 음성 = 같은 평가 에피소드들의 개입 성공 구간 프레임, 점수 = d. 둘 다 사람이 위험하다고 본 시점(t_press) 이후의 상태라 "이미 위험해 보이는 상태 안에서의 구분"이다.

## Review Focus

- 학습 목록에 캐시에 없는 이름이 있으면 조용히 빼지 말고 죽는다 — Task 2 테스트.
- 출처별 캐시의 메타(카메라 순서, crop, size)가 다르면 이어 붙이지 않는다 — Task 3 테스트.
- 평가용 원본 이름이 어느 학습 캐시에도(어느 버전으로도) 들어가지 않는다 — Task 3 테스트 + 빌드 때 assert.
- 야매·진짜 성공 분기는 거짓 진전 분모에서 빠지고 개수로 남는다 — Task 5 테스트.
- 스윕을 다시 실행하면 끝난 job은 건너뛰고 실패한 job은 다시 돈다 — Task 6 테스트.

---

### Task 1: `obs_parts`에 손목+고유수용, agentview+고유수용 추가

**Files:**
- Modify: `projects/square_assembly/src/square_assembly/datasets/dino_feature_dataset.py` (`OBS_PARTS`, `CAMERA_PARTS`, 열 자르기 루프)
- Modify: `projects/square_assembly/src/square_assembly/configs/train_dstg_vip.yaml` (`obs_parts` 주석)
- Test: `projects/square_assembly/tests/test_dino_stg.py`

**Interfaces:**
- Produces: `obs_parts` 값 `"wrist_lowdim"`, `"agentview_lowdim"` — 그 카메라 특징 열 + lowdim 전부.

- [ ] **Step 1: 실패하는 테스트** `test_camera_plus_lowdim_parts_keep_one_camera_and_all_proprioception` — 가짜 캐시(`rgb_keys=["agentview_image","robot0_eye_in_hand_image"]`, feat 4열, lowdim 2열)에서 `wrist_lowdim`은 `frame_dim == 4`이고 창이 `[손목 2열, lowdim 2열] × 2프레임`, `agentview_lowdim`은 agentview 2열 + lowdim. lowdim 열은 카메라 폭으로 잘리지 않는다.
- [ ] **Step 2: 실행해 실패 확인** (`ValueError: obs_parts는 ...`)
- [ ] **Step 3: 구현** — 두 키를 `OBS_PARTS`(`("feat","lowdim")`)와 `CAMERA_PARTS`에 넣고, 카메라 열 자르기를 `part == "feat"`일 때만 한다.
- [ ] **Step 4: 전체 테스트 통과 확인**
- [ ] **Step 5: 커밋** `feat(stg): add wrist and agentview camera plus proprioception input combos`

### Task 2: `train_dstg_vip.py`에 고정 step 수, 학습 에피소드 목록, 검증 분할 끄기

**Files:**
- Modify: `projects/square_assembly/src/square_assembly/scripts/train_dstg_vip.py`
- Modify: `projects/square_assembly/src/square_assembly/configs/train_dstg_vip.yaml` (`num_steps: null`, `train_episodes_file: null`)
- Test: `projects/square_assembly/tests/test_dino_stg.py`

**Interfaces:**
- Produces (hydra 키):
  - `num_steps: int | null` — 설정하면 epoch 대신 정확히 이만큼 gradient step(배치 64, 에포크식 비복원 순열을 이어 붙인 `RandomSampler(num_samples=num_steps·batch_size)`, `drop_last=True`). `record_curves`와 같이 못 쓴다.
  - `train_episodes_file: str | null` — 한 줄에 캐시 에피소드 이름 하나. 학습 인덱스를 이 이름들로 거른다. 캐시에 없는 이름이 있으면 `ValueError`.
  - `val_fraction: 0` — 검증 분할 없이 전부 학습. 마지막 val 평가는 건너뛰고 meta의 `val_*`는 `None`.
  - 체크포인트 meta에 `train_episodes`(실제 학습한 이름 정렬 목록)와 `num_steps` 추가.

- [ ] **Step 1: 실패하는 테스트**
  - `test_fixed_steps_train_exactly_that_many_batches` — `num_steps=7, batch_size=2, val_fraction=0`으로 학습하면 optimizer step이 7번(AdamW `state[p]["step"] == 7`), meta `num_steps == 7`, `val_mae is None`.
  - `test_train_episodes_file_keeps_only_listed_episodes` — 목록 `demo_1, demo_3` → meta `train_episodes == ["demo_1","demo_3"]`, `n_train_samples == 12`(길이 6 × 2).
  - `test_train_episodes_file_rejects_unknown_names` — 목록에 `demo_9` → `ValueError`.
- [ ] **Step 2: 실행해 실패 확인**
- [ ] **Step 3: 구현** (기본값이면 기존 경로 그대로)
- [ ] **Step 4: 전체 테스트 통과 확인**
- [ ] **Step 5: 커밋** `feat(stg): train cached STG heads for a fixed step count on a listed episode subset`

### Task 3: 평가용 목록, 조건별 학습 캐시, 평가 캐시

**Files:**
- Create: `projects/square_assembly/src/square_assembly/scripts/stg_scaling_data.py`
- Create: `projects/square_assembly/src/square_assembly/configs/stg_scaling/heldout_random30.json`, `heldout_turn30.json` (빌드 산출물, 커밋)
- Test: `projects/square_assembly/tests/test_stg_scaling.py`

**Interfaces:**
- Produces:
  - `SOURCES: dict[policy, dict]` — 원본 hdf5 경로(demos, delay, immediate, timeout, branch_t0, branch_t_press, eval_demos + 이름 목록).
  - `pick_heldout(names: list[str], k: int, seed: int = 0) -> list[str]` — k 순 정렬 → `np.array_split(k)` → 구간마다 하나.
  - `rollout_pools(policy) -> dict` — `intv`(지연 파일의 개입), `solo`(정책 단독 성공), `immediate`, `timeout`, `common`(세 조건 공통) 이름 목록.
  - `assemble_cache(out: str, entries: list[dict]) -> None` — entry = `{"cache", "hdf5", "name", "role", **attrs}`. 그룹 `demo_{i}`에 `feat`, `lowdim`, `action_mode`와 attrs `length, is_success, source, source_name, role, score_from` (+ 분기면 `hacky`). 메타가 다른 캐시를 섞으면 `ValueError`.
  - 캐시 경로 규칙: `{root}/caches/src/{hdf5 basename}.vip.h5`(출처별), `{root}/caches/{policy}_train_{kind}.vip.h5`, `{root}/caches/{policy}_eval.vip.h5`.
  - 역할(role): 학습 캐시 `demo` | `intv` | `solo`; 평가 캐시 `demo` | `policy_success` | `intv_delay` | `intv_immediate` | `intv_timeout` | `branch_t0` | `branch_t_press`. `score_from` = 개입 시작(개입), `branch_step`(분기), 0(나머지).
  - CLI: `python -m square_assembly.scripts.stg_scaling_data heldout --policy P`, `... caches --policy P --root R`.

- [ ] **Step 1: 실패하는 테스트**
  - `test_pick_heldout_spreads_over_collection_order` — 이름 `demo_0..demo_29`, k=6 → 6개, 구간 `[0..4], [5..9], …`마다 정확히 하나, 같은 seed면 같은 결과, `demo_10`이 `demo_2` 뒤로 정렬된다(숫자 순).
  - `test_assemble_cache_keeps_source_names_modes_and_rejects_mixed_meta` — 가짜 출처 캐시 두 개 + 가짜 hdf5에서 이어 붙인 결과의 attrs `source_name`, `role`, `score_from`, `action_mode`가 맞고, `DinoFeatureWindows(out, 2, mode_hdf5=out, strict_modes=True)`가 열린다. `rgb_keys` 순서가 다른 캐시를 섞으면 `ValueError`.
- [ ] **Step 2: 실행해 실패 확인**
- [ ] **Step 3: 구현** — `caches` 명령은 평가용 이름이 학습 캐시 어디에도 없음을 assert하고, 풀 크기를 출력한다.
- [ ] **Step 4: 전체 테스트 통과 확인**
- [ ] **Step 5: 출처별 VIP 캐시 굽기** (정책당 7개 파일, `cache_vip_feats` 그대로, `HOME=/workspace/outputs/.model_cache`)
- [ ] **Step 6: 목록·캐시 만들기**, 확인: (가) 풀 random30 68 = 개입 52 + 성공 16, turn30 60 = 40 + 20; (나) 풀 둘 다 69.
- [ ] **Step 7: 커밋** `feat(data): build held-out lists and per-condition VIP caches for the rollout-scaling sweep` (목록 json 포함)

### Task 4: 짝을 지키는 부분집합과 job 목록

**Files:**
- Create: `projects/square_assembly/src/square_assembly/scripts/stg_scaling_sweep.py`
- Test: `projects/square_assembly/tests/test_stg_scaling.py`

**Interfaces:**
- Consumes: Task 3의 캐시(attrs `role`, `source_name`), 목록 json.
- Produces:
  - `order_names(pool: list[str], order: int) -> list[str]` — k 순 정렬 후 `RandomState(order).permutation`.
  - `select(order: list[str], eligible: set[str], solo: set[str], n: int, with_success: bool) -> list[str]` — `[x for x in order if x in eligible][:n]`, 성공 제외면 거기서 `solo`만 뺀다.
  - `n_grid(pool_size: int) -> list[int]` — `[0, 10, …] + [pool_size]`(중복 없이).
  - `CONDITIONS` — `(kind, pairing, with_success, obs_parts)` 16개: 입력 `all`에서 kind 3 × 성공 2 × 짝 (가) + kind 지연·즉시 × 성공 2 × 짝 (나), 그리고 지연·성공 포함·(가)에서 나머지 입력 6개.
  - `build_jobs(root: str, num_steps: int) -> tuple[list[dict], list[dict]]` — (job 목록, 논리 실행 목록). job = `{"id", "policy", "cache", "obs_parts", "seed", "train_episodes", "num_bins", "num_steps", "n_train_rollouts"}`; 논리 실행 = `{"policy", "kind", "pairing", "with_success", "obs_parts", "n", "order", "seed", "job"}`. n = 0은 지연 캐시로 통일해 조건끼리 공유된다.
  - CLI: `... stg_scaling_sweep plan --root R --num-steps S` → `R/jobs.json`.

- [ ] **Step 1: 실패하는 테스트**
  - `test_selection_nests_and_keeps_intervention_sets_paired` — n이 커지면 앞 집합 포함; 같은 n이면 성공 제외 = 성공 포함 − solo; 세 kind의 (가) 선택이 같은 원본 이름.
  - `test_n_grid_ends_at_the_pool` — `n_grid(68) == [0,10,…,60,68]`, `n_grid(60) == [0,…,60]`.
  - `test_build_jobs_counts_and_dedupes` — 가짜 풀(정책 둘)에서 논리 실행 수 = 조건·n·10의 합, n = 0 논리 실행은 입력별로 job 10개만 가리킨다, job id가 유일하다.
- [ ] **Step 2: 실행해 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 전체 테스트 통과 확인**
- [ ] **Step 5: 커밋** `feat(stg): select pairing-preserving rollout subsets and plan the scaling sweep jobs`

### Task 5: 평가 스크립트

**Files:**
- Create: `projects/square_assembly/src/square_assembly/scripts/eval_stg_scaling.py`
- Test: `projects/square_assembly/tests/test_stg_scaling.py`

**Interfaces:**
- Consumes: Task 3 평가 캐시, `eval_dstg.auroc`, `eval_dstg.reward_metrics`, `dino_stg_predictor.load_checkpoint`.
- Produces:
  - `unit_stats(d, nll, label, ep, t, episodes, window=8) -> dict[unit, dict[str, float]]` — 키 `abs/<set>`, `n/<set>`, `nll/<set>`(set = `policy_success`, `intv_delay`, `intv_immediate`, `intv_timeout`, `demo`), `fp`, `fp_w1`, `nb`, `fp_<branch>`, `nb_<branch>`, `hacky`, `branch_success`.
  - `RATIOS: dict[name, (num_keys, den_keys)]` — `success_mae`(①), `false_progress`(②), `false_progress_w1`, `mae/<set>`, `mae/intv`, `nll/<set>`, `false_progress/<branch>`.
  - `ratio_metrics(stats: dict[unit, dict]) -> dict[name, float]`
  - `bootstrap_ci(stats_list: list[dict[unit, dict]], strata: dict[unit, str], n_boot=1000, seed=0) -> dict[name, [lo, hi]]` — 예측기 여럿이면 같은 재추출에서 평균.
  - `frame_metrics(d, label, ep, t, episodes) -> dict` — AUROC, `reward_metrics` stride 1·8(정책 성공, 개입 구간, 데모).
  - CLI: `python -m square_assembly.scripts.eval_stg_scaling --predictor P --eval-cache C --out DIR` → `DIR/metrics.json`(점추정, 부트스트랩 구간, 단위 합계, 개수, `n_train_samples`), `DIR/pred.npz`(프레임별 d float16).

- [ ] **Step 1: 실패하는 테스트** (합성 배열)
  - `test_success_mae_pools_policy_success_and_intervention_tails_only` — 개입 에피소드의 `score_from` 앞 프레임과 데모는 ①에 안 들어간다; 값을 손으로 계산한 것과 같다.
  - `test_false_progress_uses_chunk_means_and_skips_successful_branches` — 실패 분기 하나는 d가 오르고(진전 아님) 하나는 내린다(거짓 진전) → 0.5; 야매 성공 분기·진짜 성공 분기는 분모에서 빠지고 `hacky`, `branch_success`로 센다; 끝 8프레임만 내리는 분기는 W=1과 W=8이 다르게 나온다.
  - `test_bootstrap_ci_brackets_the_point_and_is_reproducible` — 구간이 점추정을 포함하고, 같은 seed면 같다; 층 밖으로 재추출하지 않는다(단위가 하나뿐인 층은 그 값 그대로).
  - `test_auroc_ranks_failed_branch_frames_above_intervention_frames`.
- [ ] **Step 2: 실행해 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 전체 테스트 통과 확인**
- [ ] **Step 5: 커밋** `feat(stg): score scaling-sweep predictors on held-out success paths and failed branches`

### Task 6: 실행 묶음 — 기계 나눠 돌리기, 이어 하기, 결과 표·그림

**Files:**
- Modify: `projects/square_assembly/src/square_assembly/scripts/stg_scaling_sweep.py`
- Test: `projects/square_assembly/tests/test_stg_scaling.py`

**Interfaces:**
- Produces:
  - `shard(jobs: list[dict], shards: list[int], num_shards: int) -> list[dict]` — job 순서 i의 `i % num_shards ∈ shards`.
  - `run_job(job, root, python) -> bool` — `R/runs/<policy>/<id>/`에 `train_episodes.txt`, 학습(`train_dstg_vip`, hydra run dir = job 폴더, `use_wandb=false`), 평가, 로그, 소요 시간 `timing.json`. `metrics.json`이 있으면 건너뛴다. 실패하면 `metrics.json`을 남기지 않는다.
  - `aggregate(root) -> rows` — 논리 실행을 (정책, kind, pairing, 성공, 입력, n)으로 묶어 지표 평균·시드 표준편차·부트스트랩 구간·평균 학습 프레임 수 → `R/summary.csv`, `R/summary.json`, `R/figures/*.png`(정책별: 주 곡선, 짝 어블레이션, 입력 비교; x축 n과 학습 프레임 수).
  - CLI: `... run --root R --shards 0 1 --num-shards 5 --workers K [--limit N] [--only-ids ...]`, `... aggregate --root R`.

- [ ] **Step 1: 실패하는 테스트**
  - `test_shards_cover_every_job_once`.
  - `test_run_job_skips_finished_and_retries_failed` — 가짜 python(셸 스크립트)으로 성공·실패를 흉내 내 `metrics.json` 유무를 확인한다.
  - `test_aggregate_averages_seeds_and_reports_ci` — 가짜 metrics.json 두 개로 평균·표준편차·구간 열이 생긴다.
- [ ] **Step 2: 실행해 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 전체 테스트 통과 확인**
- [ ] **Step 5: 커밋** `feat(stg): run the scaling sweep in resumable shards and summarize it into tables and curves`

### Task 7: step 수 결정, 시험 실행, 시간 측정, 보고

**Files:**
- Modify: 이 플랜("플랜에서 정한 값과 근거" 1번 결과)
- Create (git 밖): `experiments/feat-stg-rollout-scaling/2026-11-20-stg-rollout-scaling.md`(결과 전에 가설 기록), `outputs/stg_rollout_scaling_20261120/phase1_report.md`

- [ ] **Step 1: 실험 기록에 가설을 먼저 적는다** — 성공 라벨만으로는 실패로 가는 상태가 직접 감독을 받지 않아 거짓 진전 비율이 n을 늘려도 0으로 가지 않을 수 있다; 고유수용 지름길; 개입 라벨 잡음으로 곡선이 단조롭지 않을 수 있다.
- [ ] **Step 2: 학습 곡선** — 학습 프레임이 가장 많은 두 조건(정책별)의 풀 전체로 `record_curves=true val_fraction=0.2 num_epochs=…`. 규칙대로 S를 정해 이 플랜 1번에 적는다.
- [ ] **Step 3: 시험 실행(로컬)** — 정책 둘 × (n = 0, 10, 풀 전체) × 지연·성공 포함·(가)·`all` 몇 개 + 새 입력 조합 하나, `run --only-ids`. 평가 결과와 aggregate까지 확인.
- [ ] **Step 4: 동시 실행 수 측정** — workers 1, 4, 8, 12(…)에서 같은 job 묶음의 처리량(job/시간), GPU 메모리, RAM.
- [ ] **Step 5: 전체 시간 추정** — 로컬 단독, 로컬 + Eddy.
- [ ] **Step 6: 보고서 작성**, 플랜 커밋 `docs(stg): record the chosen step count and the phase-1 smoke results in the plan`
