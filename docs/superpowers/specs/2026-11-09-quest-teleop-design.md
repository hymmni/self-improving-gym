# Quest 3S 컨트롤러 텔레옵 (Can·Square) 설계

## 왜 만드나

지금은 사람이 키보드·마우스(`runners/mouse_teleop.py`)로 조작한다. 이 방식은 자유도가 낮아서 두 가지 문제가 있다.

- 사람이 개입해 성공으로 마무리할 수 있는 실패 케이스가 일부 특수한 상황뿐이다.
- 사람이 만든 움직임 분포가 시연과 달라서 학습에 악영향을 줄까 걱정된다.

Meta Quest 3S 컨트롤러의 6자유도 손 움직임을 시뮬레이션 EE에 1:1로 전달하면, 사람이 자연스럽게 움직인 궤적을 그대로 모을 수 있다.

용도는 두 가지다.

1. **초기 BC 학습용 전문가 시연 수집**: 앞으로 PH 데이터는 쓰지 않는다. 초기 시연은 모두 사용자가 직접 Quest로 모은다.
2. **RL 중 사람 개입 수집**: 정책이 돌다가 사람이 넘겨받아 성공으로 마무리한다.

## 목표 / 비목표 / 실패 조건

**목표**
- Can과 Square 모두에서 Quest 컨트롤러로 EE 위치·자세 전체(6자유도)와 그리퍼를 조작할 수 있다.
- 기존 수집기(`scripts/collect_square_scripted_intervention.py`)를 그대로 쓴다. 시연 수집(`--human-only`)과 개입 수집(Tab 전환) 모두 Quest로 할 수 있다.
- 저장 포맷, 라벨(INTV/PREINTV), 되감기, 다시 모으기, STG 패널은 지금 하던 그대로 동작한다.

**비목표 (이번 범위가 아님)**
- 헤드셋 안으로 영상을 보내는 것(입체 영상 포함). 사용자는 PC 모니터에서 agentview 영상을 보며 조작한다. 깊이감이 부족해서 힘들 때만 나중에 검토한다.
- 키보드 조작(Tab, s, q, ←/b/r, Enter/Esc, F11)을 컨트롤러 버튼으로 옮기는 것. 키보드는 지금 그대로 둔다. 헤드셋을 쓰기로 하면 그때 옮긴다.
- 양팔 조작, 모션 트래커, 머리 추적(목).

**실패 조건**
- 저장된 액션과 실제로 실행된 액션이 다르면 안 된다. 클립 전 값을 저장하는 경우가 여기에 해당한다.
- 넘겨받거나 되감은 직후, 또는 그립을 다시 잡는 순간 팔이 튀면 안 된다.
- Quest 모드를 추가한 탓에 기존 마우스 모드나 오라클 모드가 깨지면 안 된다.

## 환경과 제약

- XRoboToolkit PC-Service는 이 PC의 `/opt/apps/roboticsservice`에 이미 설치되어 있다. 실행은 `bash /opt/apps/roboticsservice/runService.sh`로 한다.
- Quest에서는 XRoboToolkit 앱으로 이 PC의 IP에 접속한다.
- PC-Service는 SDK 클라이언트를 하나만 받는다. 새 프로세스가 접속하면 기존 프로세스의 입력이 끊긴다.
- Python SDK는 PyPI의 `xrobotoolkit-sdk==1.1.0`이다(GitHub `XR-Robotics/XRoboToolkit-PC-Service-Pybind`). abi3 wheel이라 이 레포의 Python 3.11에서도 쓸 수 있다. 의존성 파일에 추가한다.
- 실행 환경 규칙(AGENTS.md)에 따라 실행·테스트는 `docker compose exec dev`로 한다. 컨테이너는 `network_mode: host`다. 컨테이너 안의 SDK가 호스트의 PC-Service에 붙는지는 구현 중에 확인한다(아래 "확인할 것").
- 액션 공간은 robosuite OSC_POSE delta다(`goal_update_mode="achieved"`, `input_ref_frame="world"`). 각 차원은 [-1, 1]이고 1은 위치 5cm, 회전 0.5rad에 해당한다. 제어 주기는 20Hz다.
- 참고 구현은 `/home/hymm/Projects/manipulation_pipeline/src/flare/teleoperators/bi_piper_xr/`(실물 Piper XR 텔레옵)이다. 이 레포의 `references/`가 아니므로 코드를 가져오면 출처 주석을 단다.

## 사용자가 정한 것

1. **보는 화면**: PC 모니터의 agentview 카메라 영상. 손 이동 축은 agentview 시점에 맞춘다.
   - 손을 오른쪽으로 → 화면 오른쪽(월드 +y)
   - 손을 앞으로(모니터 쪽) → 화면 안쪽, 로봇 쪽(월드 −x)
   - 손을 위로 → 월드 +z
2. **클러치 방식**: 실물 로봇과 같다.
   - 그립을 0.9 이상 누르고 있는 동안만 로봇이 따라온다.
   - 그립을 잡는 순간의 컨트롤러 포즈와 EE 포즈를 기준으로 삼는다.
   - 목표 EE = 기준 EE + (컨트롤러 이동량 × scale)이다. 회전도 기준에서의 회전 차이를 그대로 적용해 6자유도 전부 따라간다.
3. **배율**: 1:1(scale 1.0)로 시작한다. 위치와 회전 배율을 CLI 옵션으로 조절할 수 있다.
4. **그리퍼**: 트리거로 조작하고 ±1로 이진화한다. 트리거는 그립과 관계없이 늘 동작한다.
   - 넘겨받는 순간(Tab, 되감기)에도 지금 트리거 상태를 바로 따른다. 마우스 모드가 이미 이렇게 정했다(2026-09-30 사용자 보고 반영).
   - 쥔 물체를 넘겨받을 때 떨어뜨리지 않으려면 트리거를 당긴 채 Tab을 누른다. *(Claude 판단: 마우스 모드의 결정을 그대로 따름)*
5. **키보드는 그대로**: 기존 마우스 모드의 창과 키 처리를 그대로 쓴다. 컨트롤러는 팔과 그리퍼에만 쓴다.
6. **액션 클립**: [-1, 1]로 클립한 값을 실행하고 같은 값을 저장한다. 이건 액션 공간 규약 때문이고, PH 분포를 맞추려는 목적이 아니다.
7. **Can 시연**: `--human-only`가 지금은 `square.yaml`로 고정되어 있다. 여기에 task 선택 옵션을 추가해 Can 시연도 모을 수 있게 한다.

## 구조

기존 `MouseTeleopIntervention`은 창, 키, 되감기, HUD를 맡고 있다. 제어 법칙은 `controller`(`reset()`, `take_over(state)`, `action(state)`)로 분리되어 있다. Quest는 이 controller 자리에만 새 구현을 넣는다.

- **새 클래스 `QuestTeleopController`**(새 파일 `runners/quest_teleop.py`): `MouseTeleopController`와 같은 세 메서드를 가진다.
  - `action(state)`
    1. 매 스텝 SDK에서 오른쪽 컨트롤러 포즈, 그립 값, 트리거 값을 읽는다.
    2. 1-Euro 필터를 거친다. 이 필터는 manipulation_pipeline에서 가져온다.
    3. 클러치 상태에 따라 목표 EE 포즈를 갱신한다. 그립을 놓는 순간 목표를 **지금 실제 EE 포즈**로 바꾸고, 그 자리를 붙잡는다. 접촉 중에 목표가 앞서 있어서 놓은 뒤에도 계속 밀어붙이는 일을 막기 위해서다. *(Claude 판단)* 그립을 잡고 있는 동안 목표가 실제 포즈보다 너무 앞서지 않게 상한을 둘지는 구현자가 sim에서 보고 정한다.
    4. 7차원 액션을 만든다.
       - 위치: `kp·(목표 − 현재) − kd·속도`를 5cm로 나누고 클립한다.
       - 회전: 현재 자세에서 목표 자세로 가는 axis-angle을 0.5rad로 나누고 클립한다.
       - 그리퍼: 트리거 값에서 ±1을 만든다.
  - `take_over(state)`: 클러치 기준을 비운다. 넘겨받거나 되감은 뒤에는 그립을 다시 잡아야 움직인다. 목표는 지금 EE 포즈로 둔다. 따라서 그립을 잡기 전에는 그 자리를 유지한다.
  - 마우스 전용 입력(`set_cursor`, `wheel`, `grip_pressed`, `z_up`/`z_down`)은 Quest 모드에서 아무 효과가 없어야 한다. 이걸 어떻게 막을지(빈 메서드 또는 intervention 쪽 분기)는 구현자가 정한다.
- **SDK 연결**: `xrt.init()`은 프로세스에서 한 번만 하고, 종료할 때 `close`한다. 첫 import는 Quest 모드에서만 한다. 그래야 SDK가 없는 환경에서도 마우스·오라클 모드가 돈다.
- **수집기 CLI**(`projects/square_assembly/src/square_assembly/scripts/collect_square_scripted_intervention.py`)
  - `--mode quest`를 추가한다. 마우스 모드와 같은 `MouseTeleopIntervention`을 쓰고 controller만 Quest로 바꾼다.
  - 지금 `mode == "mouse"`로 묶인 기능(human-only, redo, STG 패널, 카메라 열)은 quest에서도 그대로 동작해야 한다.
  - `--task {square,can}`을 추가한다(`--human-only`일 때 task 설정 선택). 정책이 있을 때는 지금처럼 체크포인트의 task를 쓴다.
  - Quest 옵션: `--quest-pos-scale`(1.0), `--quest-rot-scale`(1.0), `--quest-kp`, `--quest-kd`, `--quest-grip-threshold`, `--quest-trigger-threshold`(0.5).
- **의존성**: `xrobotoolkit-sdk==1.1.0`. square_assembly 프로젝트의 의존성 파일과 Docker 이미지에 넣는다.
- **사용 문서**: 수집기 docstring에 실행 순서를 적는다.
  1. PC-Service를 실행한다.
  2. Quest 앱으로 접속한다.
  3. **모니터를 정면으로 보고** 오른쪽 컨트롤러 Meta 버튼을 길게 눌러 recenter한다. 이걸 빼먹으면 축이 돌아간다.
  4. 수집기를 `--mode quest`로 실행한다.

## 구현자에게 맡기는 것

- **XR→월드 축 변환 행렬**: 위 "보는 화면" 방향이 나오도록 정한다. 실물 쪽 `R_headset_world`는 로봇 정면 기준이라 그대로 쓸 수 없다. 출발점은 OpenXR 좌표계(x 오른쪽, y 위, −z 앞)에서 `world = [z, x, y]`다. 실제 방향은 sim에서 손을 움직여 확인한다. 회전 차이도 같은 행렬로 월드 좌표계에 옮긴다(`M·ΔR·Mᵀ`).
- **SDK 쿼터니언 순서**: SDK 포즈는 `[x, y, z, qx, qy, qz, qw]`인지 확인하고 실물 코드와 맞춘다.
- **게인 기본값과 1-Euro 파라미터**: 마우스 모드(kp 2, kd 2)와 실물 쪽(min_cutoff 1.0, beta 0.01)에서 출발한다. sim에서 넘침과 지연을 보고 정한다.
- **회전 액션 상한**: 마우스 모드는 기본 상한이 0.4다(그 이상이면 위치가 밀린다는 실측이 있다). 같은 값으로 시작한다.
- **손목 관절 한계**: 첫 버전에서는 목표 회전을 손목 한계에 맞춰 자르지 않는다. 한계에 닿으면 팔이 거기서 멈출 뿐이다. 마우스 모드의 야우 제한은 야우 1축 전용이라 6자유도에 그대로 쓸 수 없다. 사용자가 실제로 조작해 보고 문제가 되면 그때 추가한다. *(Claude 판단)* 다만 지금 손목 각도가 한계에 얼마나 가까운지를 HUD에 보여 줄지는 구현자가 정한다.
- **헬퍼 재사용**: `rot_delta_toward` 같은 기존 헬퍼를 쓸지, 새로 짤지는 구현자가 정한다.
- **화면 구성**: 화면에 맵을 그대로 띄울지 정한다. 키 도움말(`_HELP`)도 Quest 모드에 맞게 바꾼다.

## 확인할 것 (구현 중)

- 컨테이너 안에서 `xrobotoolkit_sdk`가 호스트의 PC-Service에 붙는지 확인한다. 안 붙으면 Quest 수집기만 호스트의 `self-improving-gym` conda 환경에서 실행한다(2026-11-09 사용자 승인). 이 경우 SDK를 그 conda 환경에 설치하고, 실행 방법을 docstring에 적는다.
- 실제 Quest 조작 검증은 사용자가 한다. 구현자는 컨트롤러 입력을 가짜로 넣는 테스트로 다음을 확인한다.
  - 클러치를 잡는 순간 튀지 않는다.
  - 1:1 이동이 맞다.
  - 그립을 놓으면 그 자리에 머문다.
  - take_over 뒤에는 그립을 다시 잡아야 움직인다.
  - 액션이 클립된다.
  - 축 방향이 맞다.
