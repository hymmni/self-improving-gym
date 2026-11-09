# 연구 규칙

이 파일은 하네스를 `research` 옵션으로 설치한 프로젝트에만 있다. 에이전트는 `AGENTS.md`와 함께 이 규칙을 따른다.

## 하드웨어 환경
- 실제 하드웨어 구성(PC 대수, GPU 유무, 역할 분담)은 `docs/private/ENVIRONMENT.md`에 있다. 세션을 시작할 때 읽는다.
- `docs/private/ENVIRONMENT.md`가 없으면 사용자에게 하드웨어 구성을 물어 만든다. 이 파일은 git으로 전달되지 않는다. 프로젝트 레포가 public일 수 있어 인프라 정보를 분리했다.
- 디바이스를 하드코딩하지 않고 CPU/GPU fallback(`device = 'cuda' if torch.cuda.is_available() else 'cpu'` 등)을 유지한다.

## 실험 기록
- 의미 있는 변화(알고리즘 교체, 핵심 파라미터 변경)가 있을 때와 작업을 마칠 때마다 `experiments/`에 기록을 남긴다.
- 경로는 `experiments/<브랜치명의 / 를 - 로>/YYYY-MM-DD-주제.md`이고 양식은 `.harness/LOG_TEMPLATE.md`다. 브랜치마다 자기 폴더만 쓴다.
- `experiments/`는 git에서 제외한다. 기록을 개인 fork에 남길 때만 `git add -f`로 커밋하고, 공유 레포에는 넣지 않는다.

## fork에서 실험하기
- 실험은 개인 fork의 `feat/*`에서 한다. fork의 원격 이름은 `personal`로 둔다. `pre-push`는 `personal`로 가는 main push를 막지 않는다.
- 공유 레포에는 채택한 변경의 최소 diff만 올린다. 채택하지 않은 브랜치는 fork에 보존하고, 기록에 미채택 결론을 남긴다.
- 로봇 PC와 공용 학습 서버는 `git checkout vX.Y.Z`로 태그만 쓴다.

## 사람을 위한 안내

### 시작하기
1. 참고할 논문과 오픈소스 코드는 `references/`에 `git clone`한다. 에이전트는 읽기만 한다.
2. clone마다 `docs/private/ENVIRONMENT.md`에 그 PC가 속한 하드웨어 구성을 채운다.
3. 베이스라인 코드와 이식 목표를 에이전트에게 설명하고 작업을 시작한다.

### 참조와 이식 예시
- **분석**: "references/에 있는 오픈소스에서 시계열 데이터를 전처리하는 부분을 분석해줘."
- **이식**: "분석한 내용을 바탕으로 베이스라인 코드에 적용해줘. reference 출처는 주석으로 남겨줘."
- **기록**: 수정이 끝나면 `experiments/<브랜치명의 / 를 - 로>/`에 로그를 작성하게 한다.

### 데이터
- 대용량 데이터와 모델 파일(`*.pth`, `*.ckpt`, `*.zarr`, `*.hdf5`, `wandb/`, `outputs/` 등)은 `.gitignore`로 제외된다. 데이터셋은 심볼릭 링크(`ln -s`)로 외부 경로와 연결한다.
- 학습한 체크포인트는 SSH/rsync나 공유 스토리지로 직접 옮긴다.
