# Architecture Decision Records (ADR) - Robot Research

## 철학
- Base 구현체(Baseline)를 최대한 훼손하지 않으면서 새로운 모듈(Policy, Wrapper)을 확장하는 것을 최우선으로 합니다.
- 하드웨어별 분산 환경(Local/Server/Robot)을 고려하여 의존성 충돌을 피합니다.

---

### ADR-001: 작업 레포 단위의 독립 가상환경 적용
**결정**: 하네스는 작업 레포에 복사되어 함께 관리되며, 가상 환경(Conda/Venv)은 각 작업 레포 루트에서 독립적으로 구성합니다.
**이유**: 로봇 베이스라인마다 요구하는 PyTorch, Isaac Gym, CUDA 버전이 다르기 때문에, 레포(=프로젝트) 단위로 환경을 분리하여 충돌을 방지합니다.
**트레이드오프**: 하네스를 새 작업 레포에 적용할 때마다 환경 세팅을 별도로 수행해야 합니다.

### ADR-002: 대용량 데이터 및 모델 가중치 Git 추적 제외
**결정**: `.gitignore`를 통해 `.zarr`, `.hdf5`, `.pth` 등의 데이터를 전역 차단합니다.
**이유**: 작업 레포지토리가 기가바이트 단위로 무거워지는 것을 방지하고, 코드/텍스트 위주의 동기화만 수행하기 위함입니다.
**트레이드오프**: 학습된 데이터 및 체크포인트는 개발자가 수동으로 SSH/Rsync, 혹은 로컬 스토리지 볼륨 매핑을 통해 직접 관리해야 합니다.

### ADR-003: 원본 JAX 스택 유지 (PyTorch 포트 안 함)
**결정**: SI-EFM pointmass 노트북의 스택(JAX/Haiku/optax/dm_env/TensorFlow-data/tfp)을 그대로 사용합니다. CLAUDE.md의 기본 스택(PyTorch)을 이 프로젝트에는 적용하지 않습니다.
**이유**: 프로젝트의 목적이 논문 방법론의 **수치적 충실 재현**과 그 위에서의 경향 분석이기 때문입니다. 프레임워크 포팅은 재현성을 훼손할 수 있는 변인을 추가합니다.
**트레이드오프**: JAX 생태계 의존성(haiku, tfp 등)이 무겁고, 다른 PyTorch 프로젝트와 환경을 공유할 수 없습니다.

### ADR-004: 원본 복사(copy-as-is) 전략
**결정**: 클린 버전은 원본 코드를 **그대로 복사**하고 홈페이지 관련 파일(html/css/js/웹 assets)만 삭제하는 방식으로 구성합니다. 리팩토링·모듈 분해·리네이밍을 하지 않습니다.
**이유**: 재현 가능성 확보가 최우선입니다. 코드 변형을 최소화해야 원본과의 수치적 동일성이 보장됩니다.
**트레이드오프**: 노트북 형태의 코드라 기능 확장 시 재사용이 다소 불편하지만, 확장은 ADR-005의 비침습 방식으로 해결합니다.
**완화 (2026-07-12)**: 이 원칙 때문에 코드가 지나치게 비효율적·기형적으로 되는 지점은 **약간의 수정을 허용**합니다 (사용자 승인). 예: 노트북 코드를 모듈로 옮길 때 IPython 표시 코드 제거, 수치 로직과 무관한 부분 정리. 단, 수치 결과에 영향을 주는 수정은 여전히 금지하며, 수정 지점은 주석으로 표시합니다.

### ADR-005: 확장 기능은 원본 비침습(additive) 방식으로 추가
**결정**: 2차 목표의 환경 확장(순간이동, 외력, 수동 조종 관찰)은 복사된 원본 코드를 수정하지 않고, 서브클래스·래퍼·별도 스크립트로 추가합니다.
**이유**: 원본 그대로의 재현 기준선(baseline)을 항상 유지한 채로 확장 기능의 효과만 분리해 관찰하기 위함입니다.
**트레이드오프**: 원본 클래스 내부에 한 줄 추가하면 될 일도 상속/래핑 계층을 거쳐야 할 수 있습니다.

### ADR-006: 은닉 물성 기반 파지·운반 환경 `GraspCarry2D` 추가
**결정**: SI-EFM pointmass 재현(1차)·확장(2차)과 별개로, 은닉 물성 기반 파지·운반
환경 `GraspCarry2D`를 세 번째 방향으로 추가한다.
**이유**: pointmass 계열 환경은 전관측·결정론이라 steps-to-go 분포의 분산이
기댓값의 재표현에 불과함이 실측으로 확인되었다(`experiments/2026-07-21_sigma2-audit-and-partial-obs-redesign.md`).
분산·분위수의 의미를 검증하려면 관측이 결과를 미결정하는 환경이 필요하다.
**트레이드오프**: pymunk 물리 의존성이 추가되고, 원본 JAX 스택(ADR-003)과
프로세스가 분리된다(환경=pymunk, 학습=JAX).

### ADR-007: `square_assembly/` — PyTorch/robomimic 스택을 별도 서브프로젝트로 공존
**결정**: robomimic 벤치마크(square task) 위에서 Diffusion Policy를 학습·평가하는
외부 레포(`github.com/Leejw221/manipulation_simulator`)를 `.git`을 제거하고
레포의 `projects/square_assembly/` 서브디렉토리로 통합한다. ADR-003이 고정한 JAX/Haiku 스택과
별개로, `projects/square_assembly/`는 PyTorch/diffusers/robomimic/robosuite/mujoco 스택을 그대로
쓴다(독립 conda env `square_assembly`, `projects/square_assembly/environment.yml` 참고 — ADR-001의
"레포 단위 독립 가상환경" 원칙을 서브프로젝트 단위로 확장 적용).

**통합 범위 축소**: 원본 레포(~80파일, 10,645줄)는 diffusion policy 외에도
OpenVLA 학습/추론, SIRIUS/APO 가중치 연구, PICO VR 원격조작, 실물 Piper 로봇,
SARM 보상모델을 포함한다. 이 프로젝트가 필요로 하는 건 "square task를 학습하는
diffusion policy + 그 위에 DDPO self-improvement를 얹을 수 있는 최소 인프라"뿐이라,
diffusion policy·robomimic 데이터셋/env 어댑터·학습 러너만 가져왔다(정확한 목록은
`projects/square_assembly/src/square_assembly/factory.py`·`envs/robomimic/factory.py`의 통합 시 축소 주석
참고). 나머지는 원본 레포에서 필요할 때 추가로 가져올 수 있다.

**이유**: (1) 이 통합의 목적(DDPO/REINFORCE self-improvement)에는 이미 학습된
Diffusion Policy 구현이 필요하고, robomimic square는 표준 벤치마크라 그 위에서
검증하는 편이 처음부터 새 매니퓰레이션 환경을 만드는 것보다 빠르다. (2) Diffusion
Policy를 JAX로 다시 포팅하는 대신 PyTorch 구현을 그대로 쓰는 편이 원본과의 수치적
동일성(원본 체크포인트를 그대로 불러와 파인튜닝 가능)을 보장한다 — ADR-004의
"원본 그대로" 정신을 여기에도 적용한 것.

**트레이드오프**: 레포 안에 JAX(gym 본체)와 PyTorch(`projects/square_assembly/`) 두 스택이
공존한다 — 서로 import하지 않고 완전히 분리된 프로세스로 실행된다(GraspCarry2D가
env=pymunk/학습=JAX로 이미 분리했던 것과 같은 패턴, ADR-006 참고). `projects/square_assembly/`
내부 코드는 원본 레포의 구조·관례(hydra config, factory registry)를 그대로
따르며, ADR-004(원본 복사)·ADR-005(비침습 확장)를 준용해 가져온 파일은 원본
그대로 두고 DDPO 관련 확장은 새 파일로 추가한다. 단, registry 진입점
(`factory.py`, `envs/robomimic/factory.py`)은 가져오지 않은 모듈에 대한 import만
제거했다 — 이 두 곳은 예외적으로 트림 사실을 코드 주석에 명시했다.

**미해결**: 실제 conda env(`square_assembly`) 설치(torch+cuda/robosuite/mujoco/robomimic
v0.4.0)는 아직 하지 않았다 — 무거운 설치라 별도로 진행 여부를 확인한다. 학습
체크포인트는 통합 시점에 없고 추후 별도로 전달받는다.

### ADR-008: 렌더 결함은 코드 가드로 막고, 수집은 도커/네이티브 어느 쪽이든 허용한다
**결정**: MuJoCo 오프스크린 렌더가 초기화되지 않은 메모리를 돌려주는 결함(컨테이너: env 생성 후
첫 에피소드 100%, 호스트 네이티브: 약 3%)은 원인을 못 잡았으므로 **코드 가드**로 막는다 —
`make_eval_env`의 첫 에피소드 워밍업(`_burn_first_episode`) + 수집 스크립트의 3중 노이즈 가드
(DOCKER.md §4). 이 가드가 있으면 개입/텔레옵 수집은 도커 컨테이너든 pororo의 네이티브 conda env
`sig`든 어디서 해도 된다(둘 다 `projects/square_assembly/requirements.txt`로 만든다). 학습은
도커(ADR-007).

**이유**: 2026-09-18 하루 동안 도커·GPU 부하·X11·numpy를 차례로 의심했고 그때마다 단발 측정으로
잘못 결론냈다. 세 조건을 번갈아 30회 찍는 통제 실험으로 "컨테이너 첫 에피소드 결정적, 네이티브
드물게, numpy 무관"만 확정됐다. 노이즈 obs가 데이터에 섞이면 학습이 통째로 오염되므로, 원인
규명보다 "노이즈 프레임이 절대 저장·평가되지 않게" 하는 가드가 우선이다.

**트레이드오프**: 워밍업이 env 생성마다 0.1초와 reset 한 번을 더 쓴다(eval_seed는 그 뒤에 잡혀
영향 없음). 근본 원인은 열린 문제로 남는다 — 드라이버/glvnd/컨테이너 런타임을 바꿀 때 DOCKER.md
§4의 2-에피소드 점검으로 재확인한다. **교훈**: 렌더/드라이버처럼 시간에 따라 오락가락하는 현상은
조건을 번갈아 여러 번 찍기 전엔 결론내지 않는다.

### ADR-009: 자기개선 RL은 DPPO를 쓰되 DDPO-SF를 기본값으로 남긴다
**결정**: `train_si.py`에 `algo: ddpo_sf | dppo` 스위치를 두고, 라운드 1부터의 자기개선은
`algo=dppo`로 돌린다. 기본값은 `ddpo_sf`를 유지한다. DPPO 부품은
`policies/diffusion/dppo.py`(크리틱, GAE, 클립 대리 손실)이고, 로그확률 계산은
`ddpo.py`(`sample_with_trace`/`step_logp`)를 그대로 공유한다. 어드밴티지는 환경 MDP
쪽에서만(결정=청크 단위 GAE) 만들고, 한 결정의 디노이징 단계들은 그 값을 공유한다.

**이유**: `phases/4-diffusion-si/step4.md`는 DDPO-SF를 고른 근거로 SI-EFM Algorithm 1이
deadly triad의 두 꼭짓점(off-policy 재사용, 부트스트랩)을 의도적으로 배제한다는 점을 들었고,
그 판단 자체는 지금도 유효하다. 바꾸는 이유는 이론이 아니라 실측 비용이다 — square 에피소드
하나가 시뮬레이션 50초라, 배치를 한 번 쓰고 버리는 DDPO-SF는 iteration당 20에피소드 기준
20분 중 대부분을 롤아웃에 쓴다. DPPO의 `update_epochs`만큼의 재사용이 그 비용을 직접 줄인다.

**트레이드오프**: 논문 재현이 아니라 확장이 된다. 크리틱 오차가 어드밴티지에 실리고(부트스트랩),
비율이 튈 수 있다(재사용) — 그래서 `clip_frac`/`approx_kl`/`value_loss`/`adv_std`를 매 iteration
로그에 남기고, `max_grad_norm=1.0`을 기본으로 둔다. `advantage_norm`은 DDPO-SF 경로에선 논문대로
off가 기본이지만 DPPO 런에선 켠다(스모크 실측 adv_std=32). 두 알고리즘을 같은 스크립트·같은 지표
아래 두었으므로, 어느 쪽이 실제로 나은지는 같은 예측기·같은 시작 정책으로 대조하면 된다.
