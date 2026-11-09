# Agent Harness

## 🚀 세션 시작 체크리스트
새 세션을 시작할 때 반드시 수행하라:
1. `README.md`를 읽고 프로젝트 구조를 파악한다.
2. 가상환경이 있는지 확인한다(`venv/`, `.venv/`, `conda` 환경 등). 있으면 해당 환경에서 작업한다.
   - conda: `conda activate <env>` / venv: `source venv/bin/activate`
3. `docs/ARCHITECTURE.md`, `docs/ADR.md`를 읽고 현재 구조와 결정 이력을 파악한다. 문서에 "자리표시"가 남아 있으면 아직 채우지 않은 것이다. 사용자에게 알리고, 동의를 받으면 코드를 읽어 실제 내용으로 채운다.
4. `docs/RESEARCH.md`가 있으면 읽고 그 규칙도 따른다. 하네스를 `research` 옵션으로 설치한 경우에만 있다.
5. `git config core.hooksPath`가 `.githooks`가 아니면 사용자에게 알리고 동의를 받아 `python3 .harness/setup.py`를 실행한다.
6. 현재 브랜치의 플랜(`docs/superpowers/plans/`)이 있으면 체크되지 않은 task부터 이어 간다.
7. `continuation_plan.md`가 있으면 읽고 이전 세션의 중단 지점부터 이어서 시작한다.

## 🧪 작업 규칙
- **수정 범위 (Write Scope)**: 모든 소스 코드 수정은 이 작업 레포지토리 내부에서 수행합니다. (단, `references/`는 읽기 전용 — 아래 참조)
- **환경 관리 (Environment)**: 의존성은 프로젝트가 쓰는 파일(`pyproject.toml`, `requirements.txt`, `environment.yml`)에 추가하고, 이 파일들은 레포지토리 루트에서 관리합니다.
- **실행 환경 (Docker)**: `docker-compose.yml`이 있으면 실행·테스트는 `docker compose exec dev <명령>`으로 합니다. 에이전트는 컨테이너 밖에서 작업합니다.
- **참조 범위 (Read Scope)**: `references/` 폴더의 오픈소스 코드를 분석하여 로직을 이식하되, 수정은 절대 금지합니다.
- **출처 표기**: `references/`에서 코드를 가져올 경우 `# From: references/repo_name/file.py`와 같이 주석을 남기십시오.

## 📐 아키텍처 & 기술 스택
이 프로젝트의 스택·구조·결정 이력은 `docs/ARCHITECTURE.md`(현재 구조 스냅샷)와 `docs/ADR.md`(결정-이유-트레이드오프 로그)가 단일 진처다. 구조에 영향을 주는 작업 전에는 반드시 두 문서를 먼저 확인하라.

이 문서는 특정 기술 스택을 전제하지 않는다. 실제 채택 스택과 구조는 프로젝트마다 다르므로 두 문서에서 확인하라. 이 하네스를 새 프로젝트에 처음 적용할 때는 두 문서에 해당 프로젝트의 실제 스택과 구조를 채워 넣는다. 인프라 정보(서버 주소, PC 구성)는 두 문서에 쓰지 않고 git에서 제외된 `docs/private/`에 쓴다. 프로젝트 레포가 public일 수 있다. 인프라 정보가 필요하면 `docs/private/`에서 찾는다.

## 📝 개발 프로세스
- **기본 워크플로우 — 직접 세션 협업**: 새 설계나 구현 방향은 먼저 사용자와 논의해 확정한 뒤 진행한다. 작업 도중 설계 판단이 필요한 지점(접근 방식 선택, 시도가 실패했을 때의 다음 방향, 결과 해석)마다 먼저 확인받고, 사용자 모르게 재설계하지 않는다. 결과가 기대와 다르게 나오면 "안 됐다"로 뭉개지 말고, 무엇을 어떻게 시도했고 왜 그렇게 판단했는지 구체적으로 보고해 사용자가 직접 검증할 수 있게 한다.
- **상세 워크플로우**: 설계·플랜·실행 절차는 `harness` 스킬(`.agents/skills/harness/SKILL.md`)을 따른다.
- **큰 작업 시작 전**: 여러 task짜리 큰 작업을 시작하기 전에 대략적인 토큰 소모 규모(작음/보통/큼)를 사용자에게 알린다. 필요하면 각 에이전트의 공식 수단으로 사용량을 확인해 근거로 제시한다.
- **Commit Message**: Scoped Conventional Commits 사용. 타입은 `feat|fix|refactor|docs|chore|test|wip`만 쓴다. 다른 사람이 설정이나 코드를 바꿔야 하는 변경은 `feat(config)!:`처럼 콜론 앞에 `!`를 붙인다. **커밋 메시지(제목·본문)는 영어로 작성**한다. 괄호 안에 수정된 모듈 영역(`policy`, `env`, `data`, `config`, `harness` 등 베이스라인 이름이나 모듈)을 명시하고, **제목은 명사형이 아닌 서술형(동사 중심 문장)으로** 작성하며, **반드시 본문에 글머리기호(`-`)를 사용한 멀티라인 상세 설명을 추가**하라. 예시:
  ```
  feat(policy): add diffusion head to transformer backbone

  - replace MLP decoder with DDPM noise predictor to match paper Sec 3.2
  - expose num_diffusion_steps as Hydra config key (default 100)
  - update forward() signature: obs_seq → (action_pred, noise_pred)
  ```
- **세션 정리**: 작업 단위(태스크/phase/의미 있는 소결)가 끝날 때마다 컨텍스트가 충분히 쌓였는지 스스로 판단해 컨텍스트 정리나 새 세션 시작을 사용자에게 먼저 권장하라.

## 🌿 브랜치와 릴리즈
- GitHub Flow를 따른다. 브랜치는 `main`과 `feat|fix|refactor|docs|chore/<주제>`만 쓴다. 주제는 소문자, 숫자, `.`, `_`, `-`로 쓴다. main은 언제든 버전 태그를 달 수 있는 상태로 유지한다.
- 실패한 시도는 `git revert`한다.
- main에 합치기 전에 브랜치에 main을 병합하고 그 상태로 테스트한다. 테스트한 뒤 main이 바뀌었으면 다시 병합하고 테스트한다. push한 브랜치는 rebase하지 않는다. force push는 `--force-with-lease`도 쓰지 않는다.
- main 반영 방식은 `.copier-answers.yml`의 `main_via_pr`를 따른다. 파일이나 값이 없으면 `true`로 본다.
  - `true`: main 반영은 PR로만 한다. 에이전트가 PR을 만들고(`gh`가 없으면 링크 안내) 사람이 웹에서 일반 병합(merge commit)한다. 에이전트는 main에 직접 커밋이나 병합을 하지 않고, 공유 레포의 main에 push하지 않는다.
  - `false`: 사람이 로컬에서 `git merge --no-ff <브랜치>`로 main에 병합하고 push한다. 에이전트는 main에 커밋·병합·push하지 않고, 필요하면 실행할 명령을 사용자에게 알려 사용자가 터미널에서 직접 실행하게 한다.
  - 두 방식 모두 에이전트가 main에 커밋·병합·push하는 명령은 위험 명령 차단 스크립트가 막는다.
- 버전 태그 `vX.Y.Z`는 사람들에게 새 버전으로 올리라고 알릴 때 main에 단다. 버그 수정은 Z, 기능 추가는 Y, 다른 사람이 설정이나 코드를 바꿔야 하는 변경은 X를 올린다. 1.0.0 전에는 그런 변경도 Y를 올린다. 옛 버전은 따로 고치지 않는다. main에서 고치고 새 버전을 낸다.
  - `true`: release-please가 릴리즈 PR을 열어 두고 병합될 때마다 갱신한다. 사람이 릴리즈 PR을 병합하면 버전, `CHANGELOG.md`, 태그, GitHub Release가 만들어진다. 태그를 직접 달지 않는다.
  - `false`: 사람이 프로젝트의 버전 표기(`pyproject.toml` 등)를 고쳐 main에 반영한 뒤 `git tag -a vX.Y.Z -m "<바뀐 점>"`으로 태그를 달고 push한다. GitHub Release는 필요할 때만 만든다.
- 배포·실행 환경은 `git checkout vX.Y.Z`로 태그만 쓴다.
- 브랜치와 커밋 규칙은 `.githooks/`가 검사한다. hook을 우회하거나 `core.hooksPath`를 다른 값으로 바꾸지 않는다. 하네스 도입 전에 갈라져 `.githooks/`가 없는 브랜치에서는 main을 병합할 때까지 검사가 동작하지 않는다.

## 📦 작업 산출물
- `scratch/`: git에서 제외한다. 일회용 스크립트는 여기에만 쓰고, 계속 쓸 코드는 `scripts/`나 테스트로 옮긴다.

## 🐞 하네스 문제 보고
- 하네스 파일(`AGENTS.md`, `.agents/`, `.githooks/`, `.harness/`, 에이전트별 설정 폴더, `docs/RESEARCH.md`)의 버그나 개선점을 발견하면 사용자에게 알린다. 사용자가 동의하면 템플릿 레포에 이슈를 올린다.
- 템플릿 레포는 `.copier-answers.yml`의 `_src_path`다. 이 파일이 없으면 이 레포가 템플릿 레포다. `gh:owner/repo`면 `gh issue create -R owner/repo --label bug`(개선 제안은 `--label enhancement`)로 올리고, `gh`가 없으면 `https://github.com/owner/repo/issues/new/choose` 링크를 안내한다.
- 본문에는 문제, 재현 방법, 기대한 동작, 하네스 버전(`_commit`), 에이전트 종류를 적는다. 템플릿 레포는 public이므로 프로젝트 이름, 코드, 파일 경로, 인프라 정보는 넣지 않는다.
- 설치된 프로젝트(`.copier-answers.yml`이 있는 레포)에서는 하네스 파일을 직접 고치지 않는다. 고친 내용은 `copier update` 뒤에도 남아 템플릿의 수정과 겹치고, 다른 프로젝트에는 전달되지 않는다. 템플릿 레포에서는 직접 고친다.
- 문제를 피해 갈 때도 hook이나 차단 스크립트를 끄지 않는다(`--no-verify`, `core.hooksPath` 변경, 하네스 파일 수정). 규칙 안의 다른 방법을 쓰거나, 막힌 명령이 정당하면 사용자에게 알려 터미널에서 직접 실행하게 한다. 그런 방법이 없으면 멈추고 사용자의 결정을 따른다.
- 이슈가 템플릿에 반영되면 `copier update --defaults`로 받는다.

## 🤖 모델 선택 가이드
작업 복잡도에 맞는 모델 등급을 사용자에게 제안하라. 에이전트는 실행 중인 세션의 모델을 직접 바꿀 수 없으므로, 모델 선택은 세션 시작 전에 사용자가 한다. 등급별 모델 이름은 에이전트마다 다르다.

| 등급 | 적합한 작업 |
|---|---|
| **최고 난이도** | 복합 연구 설계, 장시간 자율 작업, 가장 까다로운 디버깅 |
| **설계·복잡한 구현** | 신규 아키텍처 설계, 복잡한 알고리즘 구현, 다단계 추론이 필요한 작업 |
| **일반 작업** | 일반 코딩, 리팩토링, 대부분의 day-to-day 작업 (기본값) |
| **단순 작업** | 단순 수정, 문서 작성, 빠른 조회성 작업 |

## 🛠️ 유틸리티 명령어
- `python3 .harness/setup.py` # clone마다 한 번 실행: git hook 연결, `.gitignore` 블록과 공유 파일 검사, 설치된 에이전트에 `superpowers` 설치
- `copier update --defaults` # 하네스 갱신. `--defaults`는 이전 답변을 그대로 쓴다. 없으면 터미널이 없는 에이전트 셸에서 실패한다. 사용자 동의를 받아 브랜치에서 실행하고, 이어서 `python3 .harness/setup.py`를 실행한 뒤 충돌 표시를 해결한다
