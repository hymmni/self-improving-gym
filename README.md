# Self-Improving Gym

Ghasemipour et al. 2025, *Self-Improving Embodied Foundation Models*(SI-EFM)의 pointmass 데모 재현에서 출발해, 같은 방법론(정책이 스스로 예측한 steps-to-go 감소분을 보상으로 삼는 self-improvement)을 서로 다른 환경·스택으로 확장해가는 연구 모노레포입니다. 방법론 요약과 설계 결정 이력은 `docs/ARCHITECTURE.md`·`docs/ADR.md`가 단일 진처입니다 — 아래는 이 레포를 처음 열었을 때 환경을 세팅하고 각 연구 트랙을 실행해보기 위한 안내입니다.

## 1. 디렉토리 구조

```
projects/
  pointmass/        # 1차 목표 — SI-EFM 원본 재현 (JAX/Haiku, 노트북 그대로 복사)
  grasp_carry/       # 은닉 물성 기반 파지·운반 (JAX/Haiku, pymunk 물리)
  square_assembly/   # robomimic Diffusion Policy + DDPO self-improvement (PyTorch, 별도 conda env)
docs/                # ARCHITECTURE.md·ADR.md(구조/결정 이력), superpowers/(플랜·스펙)
experiments/         # 실험 기록 (git 미추적, docs/RESEARCH.md)
scripts/             # 레포 유틸 (sync_remote.py, notion_sync.py 등) — 목록은 COMMANDS.md
AGENTS.md, .agents/, .harness/, .githooks/   # 에이전트 하네스 (agent-harness 템플릿)
docker/, docker-compose*.yml, DOCKER.md   # 두 스택(JAX/PyTorch)을 하나로 통합한 개발 컨테이너
references/          # 원본 참조 코드 (읽기 전용)
```

각 `projects/<name>/README.md`가 그 트랙의 구조·실행법을 담고 있고, 위 트리 전체의 최신 스냅샷은 `docs/ARCHITECTURE.md` 3절을 참고하세요.

## 2. 환경 설정

이 레포엔 서로 호환되지 않는 두 스택이 공존합니다: `pointmass`·`grasp_carry`는 JAX/Haiku, `square_assembly`는 PyTorch/robosuite/mujoco. 아래 두 방법 중 하나를 선택하세요.

### 방법 A — Docker (권장)

두 스택(JAX/PyTorch)의 충돌하는 의존성을 컨테이너 하나로 묶어서 씁니다. Docker가 처음이어도 아래 순서를 그대로 따라오면 됩니다.

**0) 최초 1회 준비**

```bash
cp .env.example .env        # WANDB_API_KEY 등 채우기 (없어도 대부분 기능은 동작)
sed -i "s/^DOCKER_UID=.*/DOCKER_UID=$(id -u)/; s/^DOCKER_GID=.*/DOCKER_GID=$(id -g)/" .env
```

**1) 이미지 빌드**

```bash
docker compose build
```

`docker/Dockerfile`을 읽어 JAX/PyTorch venv 2개가 든 이미지를 만듭니다. 처음 빌드는 패키지 다운로드 때문에 5~15분 정도 걸릴 수 있고, 이후엔 바뀐 줄부터만 다시 설치되니 훨씬 빠릅니다.

**2) 컨테이너 실행**

```bash
docker compose run --rm dev bash
```

- `dev`는 `docker-compose.yml`에 정의된 서비스 이름입니다.
- `--rm`은 "쉘에서 나가면(`exit`) 컨테이너를 자동으로 지운다"는 뜻입니다 — 지워지는 건 컨테이너뿐이고, 레포 코드는 호스트(지금 이 폴더)를 그대로 마운트해서 보는 것이라 나가도 작업 내용은 남습니다.
- 들어가면 컨테이너 안 셸(`/workspace`)입니다. 스택에 맞는 venv를 activate:

```bash
source /opt/venvs/jax/bin/activate     # pointmass, grasp_carry
source /opt/venvs/torch/bin/activate   # square_assembly
```

**3) 나가기 / 다시 들어가기**

- 나가기: `exit` (또는 Ctrl+D) — `--rm`이라 컨테이너가 삭제됩니다.
- 컨테이너가 이미 떠 있는 상태에서 셸을 하나 더 열고 싶으면: `docker compose exec dev bash`
- 다음에 다시 작업할 땐 `docker compose run --rm dev bash`를 그대로 다시 실행하면 됩니다 (이미지는 이미 빌드돼 있으니 바로 시작됩니다).

**4) 의존성을 바꿨거나 문제가 생겼을 때**

```bash
docker compose build                       # requirements.txt 등을 바꾼 뒤 반영
docker compose build --no-cache            # 캐시 없이 완전히 새로 빌드 (재빌드로도 안 풀릴 때)
docker rmi self-improving-gym/dev:latest   # 이미지 자체를 지움 (다음 build 때 처음부터 새로 받음)
```

GPU 서버에서 실행, VS Code Dev Containers 연동, X11 GUI(텔레옵/뷰어) 설정, 새 패키지 추가 규칙 등 상세는 `DOCKER.md`를 참고하세요.

### 방법 B — 로컬 conda (Docker를 못 쓸 때)

Docker를 설치할 수 없는 환경이라면 스택별로 직접 설치할 수 있습니다. 설치 대상이 늘어날수록(스택 2개, 버전 고정 다수) 방법 A보다 손이 많이 갑니다.

**JAX 스택** (`pointmass`, `grasp_carry`) — 레포 루트에서:
```bash
conda create -n self-improving-gym python=3.11 -y
conda activate self-improving-gym
pip install -r requirements.txt
```
GPU(CUDA) 환경을 전제로 `jax[cuda12]`가 설치됩니다. GPU가 없어도 두 트랙 모두 규모가 작아 CPU만으로 충분히 돌아갑니다(`jax[cpu]`로 교체). 버전 pin의 근거는 `experiments/2026-07-03_clean-repro.md` 참고. `grasp_carry` 실행 전엔 `PYTHONPATH=projects/grasp_carry/src`를 직접 설정해야 합니다(방법 A는 컨테이너가 이미 잡아둠).

**PyTorch 스택** (`square_assembly`) — 별도 env:
```bash
cd projects/square_assembly
conda create -n square_assembly python=3.10
conda activate square_assembly
pip install -r requirements.txt && pip install -e .
```
버전 고정 근거는 `projects/square_assembly/docs/plan.md`의 "M0 확정 결과" 참고.

실제 하드웨어 구성(GPU 유무 등)은 `docs/private/ENVIRONMENT.md`(git 미추적, 로컬 전용)에서 확인하세요.

## 3. 서브프로젝트 실행

| 프로젝트 | 방법론 / 스택 | 문서 |
|---|---|---|
| `projects/pointmass/` | SI-EFM 원본 재현 + 환경 교란(순간이동·외력·장애물) 관찰 도구, JAX/Haiku | [`projects/pointmass/README.md`](projects/pointmass/README.md) |
| `projects/grasp_carry/` | 은닉 물성(마찰·질량) 기반 파지·운반, JAX/Haiku + pymunk | [`projects/grasp_carry/README.md`](projects/grasp_carry/README.md) |
| `projects/square_assembly/` | robomimic 벤치마크 위 Diffusion Policy + DDPO self-improvement, PyTorch | [`projects/square_assembly/README.md`](projects/square_assembly/README.md) |

각 README가 실행 예시(학습/평가/시각화)와 코드 구성을 담고 있습니다. 예:

```bash
# pointmass — 노트북 하나로 전체 파이프라인(Stage 1 SFT → Stage 2 self-improvement)
jupyter lab projects/pointmass/pointmass_notebook.ipynb

# grasp_carry — 정책 학습
python -m grasp_carry.scripts.train.train_carry_actor --help

# square_assembly — Diffusion Policy 학습
python -m square_assembly.scripts.train task=square
```

## 4. 자주 쓰는 작업 (square_assembly)

옵션 전체는 `COMMANDS.md`와 각 스크립트 docstring에 있다. 여기는 자주 쓰는 순서대로 바로 복사해 쓰는 명령만 모았다.

### 4.1 컨테이너

```bash
xhost +local:docker            # 호스트에서, 재부팅 뒤 한 번 (수집 창 등 GUI용)
docker compose up -d           # 컨테이너를 띄워 둔다
docker compose exec dev bash   # 셸이 필요하면
```

이미지에 venv가 둘이라 PATH에 `python`이 없다. 아래 명령은 모두 레포 루트에서 `docker compose exec dev /opt/venvs/torch/bin/python -m ...`로 실행한다(줄여서 `$PY`로 적는다).

```bash
PY="docker compose exec dev /opt/venvs/torch/bin/python"
```

### 4.2 Quest 3S 준비 (재부팅할 때마다)

1. **PC-Service 켜기** — PC를 재부팅하면 꺼진다. 꺼져 있으면 Quest 앱 목록에 PC IP가 안 뜬다.
   ```bash
   bash /opt/apps/roboticsservice/runService.sh   # 호스트에서. 확인: pgrep -af RoboticsServiceProcess
   ```
2. Quest의 XRoboToolkit 앱에서 이 PC의 IP를 고른다. 안 뜨면 앱을 다시 열거나 IP를 직접 입력한다(Quest와 PC가 같은 네트워크여야 한다).
3. **모니터를 정면으로 본 채** 오른쪽 Meta 버튼을 길게 눌러 recenter한다.

### 4.3 데이터 수집 — `collect_square_scripted_intervention`

```bash
# Quest로 사람 시연만 (BC 초기 데모). --task can이면 Can
$PY -m square_assembly.scripts.collect_square_scripted_intervention \
    --mode quest --human-only --task square --episodes 20 --max-steps 700 \
    --out data/quest_square_demos.hdf5

# 정책을 돌리다 사람이 개입 (Tab으로 넘겨받기). --mode mouse면 마우스로 조종
$PY -m square_assembly.scripts.collect_square_scripted_intervention \
    --mode quest --base-ckpt <policy_epochN.pt> --episodes 50 --max-steps 700 \
    --out data/square_intv_r1.hdf5 [--stg-ckpt <predictor.pt>]

# 랜덤 지연 개입: Tab 뒤 U{0..max_steps−t_press−1} 스텝 지나 넘겨받는다(지연 중 Tab = 취소, 넘겨받은 뒤엔 돌려주지 않음)
$PY -m square_assembly.scripts.collect_square_scripted_intervention \
    --mode quest --base-ckpt <policy_epochN.pt> --episodes 50 --max-steps <데모 최대 길이×2> \
    --random-delay --out data/square_intv_delay.hdf5

# 이미 저장한 에피소드를 개입 시작 상태부터 다시 모으기
$PY -m square_assembly.scripts.collect_square_scripted_intervention \
    --mode mouse --base-ckpt <policy_epochN.pt> --out data/square_intv_r1.hdf5 --redo demo_113 demo_161
```

- `--episodes`는 저장할 **성공** 에피소드 수다. 같은 `--out`에 다시 실행하면 이어서 저장하고, 새로 쓰려면 `--overwrite`.
- `--stg-ckpt`를 주면 화면 오른쪽에 STG 예측 패널이 붙는다(VIP 예측기면 `-e HOME=/workspace/outputs/.model_cache`가 필요하다, DOCKER.md §6-1).
- `--quest-rot pose`면 손목 회전을 조이스틱 대신 컨트롤러 회전으로 따라간다. `--quest-pos-scale`은 손 이동 배율.

**키보드** (mouse·quest 공통)

| 키 | 동작 |
|---|---|
| `Tab` | 사람 ↔ 정책 전환 (`--human-only`에선 안 먹는다) |
| `s` | 일시정지 / 재개 |
| `←` | 직전 모드 전환점으로 되감기. 0.4초 안에 두 번 + `Enter`면 직전 저장 에피소드를 지우고 다시 모으기 |
| `b` | 1초 되감기 |
| `r` | 같은 배치로 에피소드 처음부터 |
| `q` | 에피소드 포기 (버린 배치는 다시 못 만나므로 보통 `r`을 쓴다) |
| `F11` | 전체화면 |

**Quest 오른쪽 컨트롤러**

| 입력 | 동작 |
|---|---|
| 중지 그립 쥐기 | 쥔 동안 팔이 손을 따라온다. 사람 모드에선 **쥔 동안만 시뮬레이터가 돈다** — 놓으면 멈추고 다시 쥐면 그 자리에서 이어간다 |
| 검지 트리거 | 그리퍼 닫기(당긴 동안) |
| 컨트롤러 수평 회전 | 손목 야우 (그립 쥔 동안) |
| 조이스틱 | 손목 롤·피치(헬기식, 놓으면 수직) |
| `A` | `Tab`과 같다 (넘겨받기, 랜덤 지연 모드에선 요청·취소) |
| `B` | `r`과 같다 (에피소드 처음부터) |

조이스틱 좌우로 앱이 시점을 돌리는 것은 자동으로 되돌린다(끄려면 `--quest-no-yaw-comp`). 쥔 물체를 정책에서 넘겨받을 때는 트리거를 당긴 채 `A`(또는 `Tab`).

### 4.4 데이터 합치기 — `merge_demo_hdf5`

수집은 세션마다 파일이 따로 생긴다. 학습 전에 하나로 합친다(기본은 성공 에피소드만, 실패도 넣으려면 `--keep-failures`). `env_args`는 robomimic 원본 데이터셋에서 가져온다.

```bash
$PY -m square_assembly.scripts.merge_demo_hdf5 \
    --sources data/quest_square_demos.hdf5 data/square_intv_r1.hdf5 \
    --env-args-from <robomimic square_image_v15.hdf5> \
    --out data/square_demos_intv_r1.hdf5
```

### 4.5 학습·평가

```bash
$PY -m square_assembly.scripts.train task=square use_wandb=true            # Diffusion Policy BC
$PY -m square_assembly.scripts.train_dstg policy_ckpt=<policy.pt> hdf5_path=<data.hdf5> out=<predictor.pt>   # STG 예측기
$PY -m square_assembly.scripts.train_si policy_ckpt=<policy.pt> dstg_ckpt=<predictor.pt> \
    algo=dppo iterations=50 episodes_per_iter=8 use_wandb=true out=<si.pt>  # 자기개선(RL)
$PY -m square_assembly.scripts.eval checkpoint_path=<policy.pt> num_episodes=50
```

학습은 `use_wandb=true`로 돌린다(`.env`에 `WANDB_API_KEY`).

### 4.6 테스트

```bash
docker compose exec -T dev bash -lc "cd /workspace && /opt/venvs/torch/bin/python -m pytest projects/square_assembly/tests -q"
```

## 5. 하네스 / 개발 워크플로우

에이전트 작업 규칙은 [agent-harness](https://github.com/hymmni/agent-harness) 템플릿(copier, `research` 옵션)으로 설치한 `AGENTS.md`와 `docs/RESEARCH.md`가 단일 진처입니다. `CLAUDE.md`는 `@AGENTS.md` 한 줄입니다. 하네스 파일은 직접 고치지 않고 `copier update`로 갱신합니다(명령은 `COMMANDS.md` 4절).

- clone마다 한 번 `python3 .harness/setup.py`를 실행합니다(git hook 연결, `.gitignore` 블록, superpowers 설치).
- main 반영은 PR로만 합니다(`main_via_pr: true`). 브랜치 이름은 `feat|fix|refactor|docs|chore/<주제>`입니다.
- release-please 릴리즈 PR이 열리려면 GitHub Settings → Actions → General에서 "Allow GitHub Actions to create and approve pull requests"를 켜야 합니다.

### 이 프로젝트 전용 규칙 (에이전트도 따른다)

- 연구·데이터 수집·시각화 스크립트 목록과 옵션은 `COMMANDS.md`에 있다.
- 서브프로젝트가 별도 스택·환경을 쓰면 `docs/ADR.md`에 근거를 남기고 독립 conda env로 분리한다.

## 6. 문서 지도

- `docs/ARCHITECTURE.md` — 디렉토리 구조·기술 스택 스냅샷
- `docs/ADR.md` — 결정-이유-트레이드오프 로그
- `docs/private/ENVIRONMENT.md` — 로컬 하드웨어 구성 (git 미추적)
- `docs/RESEARCH.md` — 연구 규칙 (실험 기록, 하드웨어, fork)
- `experiments/` — 실험 기록 (양식은 `.harness/LOG_TEMPLATE.md`)
- `docs/superpowers/` — 브레인스토밍 스펙·플랜 (하네스용)
