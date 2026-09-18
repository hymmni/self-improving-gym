# Docker 개발 환경

이 리포지토리에는 서로 호환되지 않는 두 개의 pip 의존성 스택(JAX/Haiku/Optax 루트 코드, PyTorch/robosuite/mujoco `projects/square_assembly/`)이 있습니다. Docker 이미지/컨테이너는 **하나**로 통합하고, 그 안에서 venv 2개로 스택을 나눕니다.

| venv | 대상 코드 | 스택 |
|---|---|---|
| `/opt/venvs/jax` | `projects/grasp_carry/` (`src/grasp_carry/scripts/`, `src/grasp_carry/` 등) | JAX + Haiku + Optax |
| `/opt/venvs/torch` | `projects/square_assembly/` | PyTorch + robosuite + mujoco |

## 상황별 명령어 요약

| 하고 싶은 것 | 명령어 | 비고 |
|---|---|---|
| VS Code로 코드 편집/디버깅 | `Dev Containers: Reopen in Container` | 인텔리센스·디버거·`claude` CLI 자동 세팅 |
| 터미널에서 학습/평가 스크립트만 빠르게 실행 | `docker compose run --rm dev bash` | 나가면(`exit`) 컨테이너 자동 삭제, 코드는 host에 남음 |
| 이미 떠 있는 컨테이너에 셸 하나 더 | `docker compose exec dev bash` | `.env`의 `DOCKER_UID`/`DOCKER_GID`(host UID/GID)로 들어감. root 아님 |
| GPU 서버에서 실행 | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm dev bash` | §3 참고 |
| GUI(mjviewer 등) 띄우기 | `xhost +local:docker` 후 평소처럼 `run` | §4 참고 |
| `requirements.txt` 등 의존성 변경 반영 | `docker compose build` | §6 참고 |
| 컨테이너 안에서 `claude` 쓰기 | 그냥 `claude` 실행 | 이미지에 baked-in(§0-1) — devcontainer든 plain run이든 동일 |
| **claude가 안 될 때** | 먼저 `Dev Containers: Rebuild Container` (그냥 Reopen 아님) | Reopen은 오래된(설정 바뀌기 전) 컨테이너를 그대로 재사용해서 실패할 수 있음. 그래도 안 되면 `docker compose build --no-cache` |

컨테이너는 리포지토리 전체를 `/workspace`에 bind-mount 합니다. 즉 컨테이너 안에서 코드를 고치는 게 아니라, 평소처럼 호스트(VSCode 등)에서 코드를 고치면 컨테이너에 바로 반영됩니다. **코드를 고쳐도 이미지 재빌드는 필요 없습니다.** 의존성(`requirements.txt`)을 바꿨을 때만 재빌드하면 됩니다.

## 0-1. VS Code로 작업할 때 (Dev Containers)

VS Code에 [Dev Containers 확장](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)을 설치한 뒤:

1. `Cmd/Ctrl+Shift+P` → `Dev Containers: Reopen in Container` (컨테이너 하나만 뜨므로 선택 메뉴 없음)
2. 컨테이너 안에서 `Cmd/Ctrl+Shift+P` → `File: Open Workspace from File...` → `self-improving-gym.code-workspace` 선택 (최초 1회만 하면 이후 그대로 유지됨)
3. 이제 창 하나에 폴더 루트(jax) + `square_assembly`(torch)가 각각 별도 루트로 열리고, 폴더별로 다른 Python 인터프리터(`.vscode/settings.json`, `projects/square_assembly/.vscode/settings.json`)가 자동 적용됩니다 — 어느 쪽 코드를 열어도 인텔리센스/디버거가 맞는 venv를 씀
4. 터미널은 각각 열어서 `source /opt/venvs/jax/bin/activate` / `source /opt/venvs/torch/bin/activate` 해두면 바로 실행 가능

컨테이너/창 재시작이 필요 없으니, 두 스택을 오가며 작업해도 VS Code를 하나만 켜두면 됩니다.

컨테이너 안에 Node.js + Claude Code CLI도 같이 들어있습니다(이미지 자체에 baked-in, `docker/Dockerfile` 참고 — devcontainer 전용 설정이 아니라 `docker compose run`으로 띄우는 학습용 컨테이너에도 동일하게 들어있음). VS Code 통합 터미널(컨테이너 안 터미널)에서 바로 `claude`를 실행하면, 코드 편집·python 실행이 전부 같은 셸에서 이뤄져 매 명령을 `docker compose exec`로 감쌀 필요가 없습니다. 로그인 정보는 호스트의 `~/.claude`와 `~/.claude.json`을 그대로 bind-mount해서 쓰므로(`docker-compose.yml`) — 자격증명은 `~/.claude/.credentials.json`에, 온보딩/프로젝트 신뢰 상태는 `~/.claude.json`에 나뉘어 있어 둘 다 마운트해야 함 — 호스트에서 이미 로그인돼 있으면 컨테이너 안에서 별도 로그인이 필요 없고, `--rm`으로 컨테이너가 지워지거나 이미지를 재빌드해도 로그인이 유지됩니다. 호스트에 `~/.claude`가 아직 없으면(Claude Code를 host에서 써본 적 없으면) 빈 폴더가 자동 생성되고 컨테이너 안에서 최초 1회 로그인하면 됩니다. 단 `~/.claude.json`은 파일 하나를 바로 마운트하는 거라, 호스트에 그 파일이 아예 없는 상태(=host에서 Claude Code를 한 번도 실행 안 한 상태)로 컨테이너를 띄우면 Docker가 그 경로에 빈 디렉터리를 만들어버려 컨테이너 안 Claude Code가 깨질 수 있음 — 이 경우 호스트에서 `claude`를 한 번 실행해 파일을 만든 뒤 컨테이너를 다시 띄우면 됨.

이 mount는 컨테이너 경로와 호스트 경로가 완전히 동일해야 합니다(`${HOME}/.claude`) — Claude Code 내부 설정(플러그인/마켓플레이스 등)이 절대경로를 그대로 저장하기 때문입니다. 그리고 컨테이너가 root로 돌면 그 안에서 새로 생기는 설정/세션 파일이 전부 root 소유로 호스트에 그대로 남아 `~/.claude`를 오염시킵니다 — `claude`는 이미지 어디서든(devcontainer든 `docker compose run`이든) 실행 가능하므로, 이 위험은 경로를 가리지 않습니다.

그래서 `docker-compose.yml`의 `dev` 서비스 자체가 root가 아니라 `.env`의 `DOCKER_UID`/`DOCKER_GID`(=호스트 사용자 UID/GID)로 돌게 고정돼 있습니다 — devcontainer, `docker compose run`, `docker compose exec` 전부 이 설정을 그대로 따릅니다. VS Code Dev Container는 여기에 더해 `common-utils` feature + `updateRemoteUserUID: true`로 실제 리눅스 유저 이름(`dev`)까지 만들어주지만(VS Code 통합 터미널·디버거가 유저 이름을 필요로 함), 핵심 오염 방지 자체는 `docker-compose.yml`의 UID/GID 고정에서 나옵니다.

## 0. 최초 1회 준비

```bash
cp .env.example .env
sed -i "s/^DOCKER_UID=.*/DOCKER_UID=$(id -u)/; s/^DOCKER_GID=.*/DOCKER_GID=$(id -g)/" .env
# .env 파일 열어서 WANDB_API_KEY도 채우기 (DOCKER_UID/GID는 위 sed가 자동으로 채움)
```

## 1. 빌드

```bash
docker compose build
```

## 2. 실행 (CPU — 지금 이 개인 PC)

```bash
docker compose run --rm dev bash
```

들어간 뒤 스택에 맞는 venv를 activate 합니다.

```bash
source /opt/venvs/jax/bin/activate     # 2D 블록 이송 등 루트 코드
source /opt/venvs/torch/bin/activate   # square_assembly
```

jax venv 기준 실행 예시: `python -m grasp_carry.scripts.train.train_carry_actor ...`

## 3. GPU 있는 학습 서버에서 실행할 때

서버에 [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) + [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)이 설치되어 있어야 합니다 (드라이버는 서버에 이미 있을 가능성이 높고, 툴킷만 추가 설치하면 됨).

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm dev bash
```

같은 이미지/코드가 그대로 GPU를 사용합니다. jax venv 쪽은 컨테이너 시작 시 GPU가 안 보이면 자동으로 `JAX_PLATFORMS=cpu`로 전환되고(entrypoint 스크립트), torch venv 쪽은 PyTorch가 `torch.cuda.is_available()`로 알아서 판단하므로 특별한 설정이 필요 없습니다.

## 4. GUI (텔레옵 수집, mjviewer) 쓰기

호스트(Linux)에서 컨테이너의 X11 접근을 한 번 허용해야 합니다:

```bash
xhost +local:docker
```

그 다음 평소처럼 `docker compose run` 하면 `DISPLAY`가 자동으로 전달됩니다 (`.env`의 `DISPLAY` 값 사용, 보통 호스트와 동일한 `:0` 등).

**GDM 로그인 화면을 거친 그래픽 세션(예: 원격 GPU 서버를 RustDesk/실물 모니터로 보는 경우)은
`DISPLAY`가 `:0`이 아니라 `:1`(또는 그 이상)일 수 있고, 인증 파일도 `~/.Xauthority`가 아니라
GDM이 관리하는 `/run/user/$(id -u)/gdm/Xauthority`에 있다** — 기본 `xhost +local:docker`가
"명령 안 먹힘"처럼 조용히 안 먹거나, 컨테이너에서 `cv2.imshow`/`mjviewer`가 "Invalid
MIT-MAGIC-COOKIE-1 key"로 죽는다면 이 케이스다(2026-09-09 GPU 서버에서 실측 — `who`로 실제
세션 번호를 확인). 그 경우 명시적으로 지정해야 합니다:

```bash
DISPLAY=:1 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority xhost +local:docker
docker compose run --rm -e DISPLAY=:1 dev bash   # 이후 run은 -e DISPLAY=:1만 오버라이드하면 됨
```

### ssh -X 포워딩으로 컨테이너 GUI 보기 (RustDesk 등 원격 데스크톱 없이)

순수 `ssh -X`만으로도 되지만, 2026-09-10 실측으로 세 가지를 순서대로 잡아야 했다 —
아래 증상이 보이면 해당 항목을 확인한다.

1. **`sshd_config`의 `X11UseLocalhost`가 `no`로 되어 있으면 안 된다.** `no`면 `DISPLAY`가
   `localhost:N.0`이 아니라 `<호스트명>:N.0` 형태로 잡히고, Docker 기본 브리지 네트워크에서
   그 호스트명이 해석되지 않거나(컨테이너 자신의 네트워크 namespace라 호스트를 못 찾음)
   TCP로 붙어도 인증 family가 안 맞아 죽는다. 기본값(`yes`, 또는 주석 처리)으로 되돌린다:
   ```bash
   sudo sed -i 's/^X11UseLocalhost no/X11UseLocalhost yes/' /etc/ssh/sshd_config
   sudo systemctl restart ssh   # 서비스명이 sshd가 아니라 ssh인 배포판이 많다(Ubuntu/Debian)
   ```
   (재시작해도 이미 붙어있는 세션은 안 끊긴다 — 새로 접속하는 세션부터 적용된다.)

2. **tmux를 거치면 `$DISPLAY`/쿠키가 스테일해질 수 있다.** 오래 떠 있던 tmux 세션에
   재접속(`tmux attach`)하면 그 pane의 `$DISPLAY`가 지금 이 ssh 연결이 아니라 그 pane이
   맨 처음 만들어졌을 때의 값을 그대로 들고 있다 — 디스플레이 번호가 그새 재사용되면서
   `~/.Xauthority`의 쿠키와 어긋나 `Invalid MIT-MAGIC-COOKIE-1 key`로 죽는다(로컬 클라이언트
   쪽 tmux도 마찬가지 — 클라이언트의 `ssh -X`도 자기 자신의 `$DISPLAY`를 참조해서 되돌려줄
   곳을 정하므로 로컬/서버 양쪽 tmux 모두 의심 대상이다). **tmux를 아예 안 거친 새 터미널로
   `ssh -X` 접속**해서 재현되는지 먼저 확인한다 — 이게 원인이면 그걸로 끝이다.

3. **이 서버의 sshd는 X11 forwarding에 유닉스소켓 파일을 안 만들고(`/tmp/.X11-unix/`에
   해당 디스플레이 번호가 안 보임) `xauth`에 등록되는 쿠키도 FamilyLocal
   (`<호스트명>/unix:N` 형태, `xauth list $DISPLAY`로 확인 가능)로만 발급한다.** 즉 컨테이너가
   호스트의 forwarding 포트(예: `127.0.0.1:6010`)에 실제로 TCP로 닿아야 하는데, Docker 기본
   브리지 네트워크는 컨테이너 자신의 루프백이 따로 있어 호스트의 루프백에 안 닿는다.
   `docker-compose.yml`의 `dev` 서비스에 `network_mode: "host"`를 이미 설정해뒀으므로(이
   레포에 다른 서비스가 없어 브리지 격리를 포기해도 트레이드오프가 거의 없음),
   `~/.Xauthority`만 추가로 마운트해서 `docker compose run`을 쓰면 된다(레포 전체 사용자가
   `~/.Xauthority`를 갖고 있진 않으므로 compose 파일 자체엔 이 마운트를 넣지 않았다 — 필요한
   사람만 `-v`로 얹는다):
   ```bash
   docker compose run --rm -e NUMBA_CACHE_DIR=/tmp/numba_cache \
     -v $HOME/.Xauthority:$HOME/.Xauthority:ro \
     dev bash
   ```

위 세 가지를 다 잡았는데도 안 되면, 원격 데스크톱(RustDesk 등)으로 서버의 **실제 로컬
세션**(`:1` 등, GDM 관리 — 위 GDM 문단 참고)에 붙는 쪽이 훨씬 간단하고 안정적이다.

### cv2.imshow가 멈춘다면: MuJoCo EGL 초기화 순서 문제다 (2026-09-12 실측, 중요)

**증상**: `cv2.imshow`가 첫 호출에서 영영 안 돌아온다. py-spy 네이티브 스택을 뜨면 Qt의
xcb 커넥션 초기화 중 확장 버전 질의(`xcb_shm_query_version`, `QT_XCB_NO_MITSHM=1`로 그걸
끄면 그다음 `xcb_xfixes_query_version`)에서 `xcb_wait_for_reply`에 박혀 있다. Ctrl+C도 안
먹고(네이티브 코드라), 겉보기엔 CPU 100%라 "뭔가 계산 중"처럼 보인다(실제론 OpenMP 스핀).

**원인**: X11/ssh/도커 문제가 아니다. **MuJoCo EGL 환경(`MUJOCO_GL=egl`, robosuite
`make_eval_env` 등)을 먼저 만들면, 그 뒤에 cv2(Qt/xcb)가 X 서버에 처음 붙을 때 데드락**이
난다(NVIDIA EGL/GL 라이브러리가 Xlib 잠금을 선점하는 것으로 보임). 이분 탐색으로 확인:
- 최소 프로세스에서 `cv2.imshow` → 정상
- torch CUDA 초기화 후 `cv2.imshow` → 정상
- `import robosuite`만, `import robomimic.utils.obs_utils`만 → 각각 정상
- robosuite EGL env 생성 후 `cv2.imshow` → **무한 정지**
- 이 프로젝트 모듈 임포트(`square_assembly.factory` / `utils.task_utils` /
  `runners.intervention_rollout`)만 해도 그 뒤 `cv2.imshow` → **무한 정지**
- **cv2 창을 먼저 열어두고** 그 임포트·EGL env 생성 → 그 뒤 `imshow` 반복도 전부 정상

**해결**: GUI 창을 robosuite/robomimic을 끌어오는 임포트보다 **먼저** 한 번 띄워라(빈 프레임 +
`waitKey(1)`이면 충분). `runners/scripted_intervention.py`의 모듈 함수 `open_window()`와,
그걸 무거운 임포트 앞에서 호출하려고 그 임포트들을 `run()` 안으로 내린
`scripts/collect_square_scripted_intervention.py`가 이 패턴의 예다.

**주의 — 2026-09-11에 이 문단에 적었던 "이 서버 sshd의 X11 forwarding이 확장 질의 응답을
구조적으로 못 돌려준다"는 진단은 틀렸다.** 같은 증상이 ssh를 전혀 안 거치는 RustDesk의 로컬
`:1` 화면에서도 똑같이 재현돼서 드러났다. ssh -X 경로 자체는 위 1~3번(`X11UseLocalhost yes`,
tmux 안 거침, `~/.Xauthority` 마운트)을 갖추면 정상일 가능성이 높다 — 다만 EGL 순서를 고친
뒤로 ssh -X를 다시 검증하진 않았다(RustDesk로 진행했기 때문).

### 렌더가 노이즈로 나온다: 도커 컨테이너 고유 문제 — 렌더·GUI 작업은 호스트 네이티브로 (2026-09-18 확정)

**증상**: 에러 없이 화면이 지지직거리고 정책이 갑자기 아무것도 못 한다. MuJoCo 오프스크린
렌더가 초기화되지 않은 메모리(노이즈) 또는 검은 화면을 돌려주며, 화면뿐 아니라 **정책이 보는
obs도 같이 노이즈**라 그대로 수집하면 데이터가 통째로 쓸모없다.

**정확한 서명**(2026-09-18 pororo·rupy 실측): `env.reset()` 직후 첫 프레임은 정상이고 **그 뒤
step 프레임부터 노이즈**다. 그래서 "reset 후 한 장" 점검은 고장을 통과시킨다 — 반드시 step
몇 번 뒤의 프레임을 봐야 한다. 대부분은 **env 생성 후 첫 에피소드**에서 나고 두 번째 reset부터
정상이지만(rupy 컨테이너 24/24 재현), 사람이 개입 중인 긴 에피소드 도중에도 간헐적으로 난다.

**도커 고유 문제다**: 같은 시각에 호스트 네이티브 환경(rupy `mani_sim`, 같은 mujoco 3.2.3·
robosuite 1.5.1·드라이버 595.84)과 컨테이너를 3분 간격으로 짝지어 돌리면 네이티브는 0/9,
컨테이너는 6/9 노이즈(`outputs/paired_probe.log`). 실험으로 배제한 것: GPU 부하(사용률 0%에
만든 컨테이너도 고장, 100%에서도 네이티브는 정상), DISPLAY 유무, TTY, 마운트 디렉터리/코드,
cv2 창 유무, X11 MIT-SHM. 컨테이너 안 완화책도 전부 무효: NVIDIA ICD만 사용
(`__EGL_VENDOR_LIBRARY_FILENAMES`), `mjr_readPixels` 앞 `glFinish`, `MUJOCO_EGL_DEVICE_ID=0`
(`outputs/probe_variants.log`, 각 6회). 남은 차이는 컨테이너의 glvnd 1.6.0(Debian) + 주입된
드라이버 라이브러리 조합뿐인데 원인까지는 못 잡았다.

**규칙(ADR-008)**: 렌더 품질이 데이터가 되는 작업(텔레옵/개입 수집, 화면 보며 하는 평가)은
**호스트 네이티브 환경**에서 한다. 도커는 학습(hdf5 입력)과 헤드리스 배치 작업용이다. 도커
안에서 어쩔 수 없이 env를 만들 때(학습 중 eval)는 `make_eval_env`가 첫 에피소드를 자동으로
버린다(`_burn_first_episode`).

**코드 쪽 방어**(collect_square_scripted_intervention.py): 시작 시 reset+5 step 프레임으로 점검하고
고장이면 env를 다시 만들며 최대 2분 대기(`wait_for_renderer`), 에피소드 도중 연속 3프레임 노이즈면
그 에피소드를 버리고 렌더가 돌아올 때까지 기다렸다 같은 번호로 재수집, 저장 시 프레임 거칠기를
다시 재서 `render_ok` 속성을 남기고 노이즈면 실패로 기록(병합에서 제외).

**직접 확인**(어디서든): reset 뒤 step을 20번 돌린 프레임의 이웃 픽셀 차이 평균 —
정상 84px 장면은 4~10, 노이즈는 30~130, 검은 화면은 ~0.

```bash
MUJOCO_GL=egl python -c "
import numpy as np, robosuite as suite
env = suite.make('NutAssemblySquare', robots='Panda', has_renderer=False, has_offscreen_renderer=True,
                 use_camera_obs=True, camera_names='agentview', camera_heights=84, camera_widths=84, control_freq=20)
for k in range(2):
    o = env.reset(); rs = []
    for _ in range(20):
        o, *_ = env.step(np.zeros(env.action_dim)); rs.append(np.abs(np.diff(o['agentview_image'].astype(int), axis=1)).mean())
    print(f'episode {k}: step frames max roughness {max(rs):.1f} ->', '정상' if max(rs) < 15 else '노이즈')
"
```

### cv2 창을 띄운 채 오프스크린 렌더 해상도를 키우면 죽는다 (2026-09-17 실측)

**증상**: `mujoco.FatalError: Default framebuffer is not complete, error 0x0` — 이어서
`AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'`.

robosuite 오프스크린 버퍼는 640x480(MJCF 기본)으로 잡힌다. 그보다 큰 렌더를 요청하면
`binding_utils.update_offscreen_size`가 `MjrContext`를 **다시 만드는데**, cv2(Qt) 창이 이미
떠 있는 프로세스에선 그 재생성이 EGL에서 실패한다. 같은 코드가 창 없이 헤드리스면 512도
정상이고, compose run 컨테이너에서 창을 띄우면 512는 죽고 480은 정상이었다.

→ 화면 표시용 렌더는 **480 이하**로 요청한다(정책 입력 84픽셀과 별개로 크게 보고 싶을 때).
collect_square_scripted_intervention.py의 `--display-size`가 이 상한을 강제한다.

- jax venv: matplotlib TkAgg 백엔드 사용 (이미지에 `python3-tk` 설치됨)
- torch venv: mujoco `mjviewer` 온스크린 창 사용 시 컨테이너 환경변수 `MUJOCO_GL`을 비워야 함 (compose 기본값은 `MUJOCO_GL=egl`, 헤드리스 학습/평가용). 온스크린이 필요하면:
  ```bash
  docker compose run --rm -e MUJOCO_GL= dev bash
  ```

## 5. 알려진 사소한 경고

- GPU 없는 환경에서 mujoco EGL 오프스크린 렌더러를 쓰면 프로그램 종료 시 `OpenGL.raw.EGL._errors.EGLError`가 찍힐 수 있음 — 렌더링 자체는 정상 동작하고, 소프트웨어 EGL 컨텍스트를 정리(`__del__`)하는 과정에서만 나는 무해한 경고. 실제 GPU 서버에서는 보통 안 뜸.

## 6. 새 패키지가 필요할 때

컨테이너 안에서 바로 `pip install <패키지>`를 해도 **당장은** 잘 동작합니다. 하지만 그건 그 컨테이너가 살아있는 동안만 유효합니다 — `--rm`으로 뜬 컨테이너는 나가는 순간 사라지고, `--rm` 없이 재사용해도 언젠가 컨테이너를 지우거나(`down`), 이미지를 재빌드하거나, GPU 서버 등 다른 머신에서 새로 빌드하면 그 설치는 흔적도 없이 사라집니다.

그래서 규칙:
1. 계속 쓸 패키지라고 판단되면 → 해당 스택의 `requirements.txt`(jax는 루트 `requirements.txt`, torch는 `projects/square_assembly/requirements.txt`)에 **버전을 명시해서** 추가
2. `docker compose build`로 재빌드해서 이미지에 영구 반영

이 순서를 안 지키면, 지금 `pymunk`/`cmake`가 빠져있던 것과 같은 "숨은 의존성" 문제가 또 생깁니다.

## 7. 자주 쓰는 팁

- 컨테이너가 이미 떠 있는 상태에서 셸을 하나 더 열고 싶으면: `docker compose exec dev bash`
- 의존성(`requirements.txt`) 수정 후에는: `docker compose build`
- 완전히 새로 빌드하고 싶으면: `docker compose build --no-cache`
