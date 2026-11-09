# Quest 3S 컨트롤러 텔레옵 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 수집기에 `--mode quest`를 추가한다. 이 모드에서는 Quest 3S 오른쪽 컨트롤러로 Can·Square의 EE를 6자유도로 조작해 시연과 개입을 모은다.

**Architecture:** `MouseTeleopIntervention`(창·키·되감기·HUD)은 그대로 쓰고 controller 자리만 바꾼다. 새 `QuestTeleopController`는 `MouseTeleopController`를 상속한다. HUD가 읽는 속성은 물려받고 `take_over`/`action`만 덮어쓴다. SDK 읽기는 주입 가능한 `reader`로 분리해서 헤드셋 없이도 테스트할 수 있게 한다.

**Tech Stack:** Python 3.11, robosuite 1.5.1 OSC_POSE(delta, `input_ref_frame="world"`), `xrobotoolkit-sdk==1.1.0`, scipy `Rotation`, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-quest-teleop-design.md`

## Global Constraints

- 액션은 7차원 `[dx, dy, dz, rx, ry, rz, grip]`다. 위치 1.0은 0.05m(`_POS_SCALE`), 회전 1.0은 0.5rad(`_ROT_SCALE`)에 해당한다. 모든 성분은 [-1, 1] 안에 있어야 하고, 저장값과 실행값이 같아야 한다.
- 그리퍼 값은 ±1만 쓴다. 트리거 값이 `trigger_threshold`(기본 0.5) 이상이면 +1(닫힘)이다. 트리거는 그립(클러치)과 관계없이 늘 동작한다.
- 클러치는 그립 값이 `grip_threshold`(기본 0.9) 이상인 동안 켜진다.
- 위치 배율 `pos_scale`과 회전 배율 `rot_scale`의 기본값은 1.0(1:1)이다.
- XR→월드 축은 다음과 같다. 출발값 `M = [[0,0,1],[1,0,0],[0,1,0]]`은 OpenXR 기준이며, Task 3에서 실측으로 확정한다.
  - 손 오른쪽 → 월드 +y
  - 손 앞쪽(모니터 쪽) → 월드 −x
  - 손 위쪽 → 월드 +z
- SDK import는 Quest 모드에서만 한다. SDK가 없어도 mouse·oracle 모드와 기존 테스트가 돌아야 한다.
- manipulation_pipeline에서 가져온 코드에는 `# From: /home/hymm/Projects/manipulation_pipeline/src/flare/teleoperators/bi_piper_xr/bi_piper_xr.py` 주석을 단다.
- 실행·테스트는 `docker compose exec dev ...`로 한다. 컨테이너에서 PC-Service에 안 붙는 경우에만 Quest 수집기를 호스트 `self-improving-gym` conda 환경에서 실행한다(사용자 승인).
- 커밋 메시지는 영어, scoped conventional 형식이고 본문은 `-` 글머리표로 쓴다.

## Review Focus

1. **넘겨받기·되감기 직후**: `take_over` 다음 첫 `action`의 위치·회전 성분은 0 근처여야 하고, 그립을 잡은 채 넘겨받아도 튀면 안 된다. → Task 1 `test_take_over_requires_regrip`
2. **그립 잡는 순간 컨트롤러가 멀리 있음**: 잡기 전 손 위치와 관계없이 기준이 잡는 순간으로 정해져야 하고, 액션은 0이어야 한다. → Task 1 `test_clutch_engage_does_not_jump`
3. **SDK가 잘못된 포즈를 줌**: 연결 전 0벡터, 쿼터니언 노름 0 같은 경우다. 예외로 죽지 않고 클러치가 꺼진 것처럼 제자리를 유지해야 한다. → Task 1 `test_invalid_pose_holds`
4. **회전 180° 근처**: rotvec 필터가 ±π를 넘을 때 회전 액션의 부호가 뒤집혀 튀면 안 된다. → Task 1 `test_rotation_near_pi_is_continuous`
5. **Quest 모드에서 마우스를 움직이거나 휠을 굴림**: 목표가 바뀌거나 OS 포인터를 빼앗으면 안 된다. → Task 2 `test_mouse_input_disabled_ignores_motion_and_wheel`

---

### Task 1: `QuestTeleopController` 제어 법칙

**Files:**
- Create: `projects/square_assembly/src/square_assembly/runners/quest_teleop.py`
- Test: `projects/square_assembly/tests/test_quest_teleop.py`

**Interfaces:**
- Consumes:
  - `MouseTeleopController`(`runners/mouse_teleop.py`)
  - `_POS_SCALE`, `_ROT_SCALE`, `yaw_of`(`runners/square_oracle.py`)
  - state dict: `{"grip": (3,) 월드 EE 위치, "R": (3,3) EE 회전행렬(월드), "wrist": ...}`. 형식은 `read_privileged_state` 기준이다.
- Produces:
  - `QuestTeleopController(MouseTeleopController)`
    - `__init__(self, reader=None, pos_scale=1.0, rot_scale=1.0, kp=2.0, kd=2.0, pos_cap=1.0, rot_cap=0.4, grip_threshold=0.9, trigger_threshold=0.5, axes=M, euro_min_cutoff=1.0, euro_beta=0.01, euro_d_cutoff=1.0, dt=0.05)`
    - `reader`: `() -> (pose7: np.ndarray [x,y,z,qx,qy,qz,qw], grip: float, trigger: float)`. None이면 `xrt_reader()`를 쓴다.
  - `xrt_reader()`: 첫 호출 때 `xrobotoolkit_sdk`를 import하고 `init()`한다. 반환은 오른쪽 컨트롤러 포즈, 그립, 트리거를 읽는 함수다.
  - `close_xrt()`: init했으면 `xrt.close()`를 부른다.
  - `take_over(state)`: 클러치 기준을 비운다. 목표는 현재 EE로 둔다. 다음에 그립을 새로 누를 때까지(그립 값이 threshold 아래로 한 번 내려갔다가 다시 올라올 때까지) 움직이지 않는다.
  - `action(state) -> np.ndarray(7, float32)`
  - `set_cursor`, `wheel`: 아무 일도 하지 않는다(no-op, `wheel`은 False 반환).
  - HUD용 속성 `target_xy`, `target_yaw`, `target_z`를 목표 포즈로 채운다.

- [ ] **Step 1: 실패하는 테스트 작성** (`FakeReader`는 테스트 안에서 `(pose7, grip, trigger)`를 큐로 내주는 클래스다. 상태는 `_state(pos, R)` 헬퍼로 만든다. 필터 지연 때문에 기대값이 흐려지지 않도록, 필터를 보는 테스트가 아니면 `euro_min_cutoff=1e6`으로 필터를 사실상 끈다.)

```python
M = np.array([[0,0,1],[1,0,0],[0,1,0]])

def test_clutch_engage_does_not_jump():   # 잡기 전 손이 어디 있든 잡는 첫 스텝 액션 [:6] == 0
def test_one_to_one_position():           # 잡은 뒤 XR +x(오른쪽)로 0.01m → 목표 = 시작 + [0, 0.01, 0]
                                          #   kd=0, kp=1이면 a[:3] ≈ [0, 0.01/_POS_SCALE, 0]
def test_axes_forward_is_minus_world_x(): # XR −z로 0.01m → a[0] < 0, |a[1]|,|a[2]| ≈ 0
def test_pos_scale_halves_motion():       # pos_scale=0.5 → 목표 이동 0.005m
def test_rotation_follows_controller():   # 잡은 뒤 XR y축(위) 기준 +0.1rad 회전 → a[3:6] ≈ M@[0,0.1,0]/_ROT_SCALE
                                          #   (kp 영향 없음 — 회전은 남은 각 그대로를 rot_cap으로 클립)
def test_release_holds_actual_pose():     # 잡고 0.05m 이동 후 놓음, state의 EE는 0.02m만 감
                                          #   → 다음 액션 a[:3] ≈ 0 (목표가 실제 위치로 바뀜, kd=0)
def test_action_is_clipped():             # 0.5m 이동 → a[:3]은 pos_cap(1.0) 이하, a[3:6]은 rot_cap 이하
def test_trigger_binary_and_independent_of_grip():
                                          # grip=0, trigger=0.6 → a[6] == 1.0; trigger=0.4 → -1.0
def test_take_over_requires_regrip():     # 그립 0.95 유지한 채 take_over → 손을 움직여도 a[:6] == 0
                                          #   그립 0으로 뗐다가 다시 0.95 → 그때부터 따라감
def test_invalid_pose_holds():            # pose 쿼터니언 전부 0 → 예외 없음, a[:6] == 0
def test_rotation_near_pi_is_continuous():# 기준에서 XR z축으로 3.0 → 3.2rad까지 0.05씩 돌림
                                          #   연속 스텝의 a[3:6] 부호가 뒤집히지 않는다
def test_set_cursor_and_wheel_are_noops():
```

- [ ] **Step 2: 실패 확인**

Run: `docker compose exec dev pytest projects/square_assembly/tests/test_quest_teleop.py -v`
Expected: FAIL (`ModuleNotFoundError: square_assembly.runners.quest_teleop`)

- [ ] **Step 3: `quest_teleop.py` 구현**
  - `OneEuroFilter`는 manipulation_pipeline에서 그대로 가져오고 출처 주석을 단다.
  - XR 포즈를 월드로 옮긴다: `p_w = M @ p`, `R_w = M @ R_xr @ M.T`. `R_xr`는 scipy `Rotation.from_quat([qx,qy,qz,qw])`로 만든다.
  - 위치는 1-Euro 필터에 통과시킨다.
  - 회전은 클러치 기준 대비 상대 회전을 rotvec으로 바꿔 필터에 넣는다. 절대 rotvec이 아니라 상대 회전이므로 클러치를 잡는 순간 0에서 시작하고, ±π 문제는 Review Focus 4의 테스트로 잡는다.
  - 목표 계산:
    - 위치: `target_p = ref_ee_p + pos_scale·(p_w − ref_ctrl_p)`
    - 회전: `target_R = Rotation.from_rotvec(rot_scale·filtered_rotvec) * ref_ee_R`
  - 액션 계산:
    - 위치: `clip((kp·(target_p − p) − kd·v)/_POS_SCALE, ±pos_cap)`. 여기서 `v`는 직전 스텝 대비 EE 이동량이다.
    - 회전: `clip((target_R·R⁻¹).as_rotvec()/_ROT_SCALE, ±rot_cap)`

- [ ] **Step 4: 통과 확인**

Run: `docker compose exec dev pytest projects/square_assembly/tests/test_quest_teleop.py projects/square_assembly/tests/test_mouse_teleop.py -v`
Expected: 전부 PASS

- [ ] **Step 5: 커밋** — `feat(teleop): add Quest controller law with grip clutch and 1:1 pose mapping`

### Task 2: 수집기에 `--mode quest`와 `--task` 추가

**Files:**
- Modify: `projects/square_assembly/src/square_assembly/runners/mouse_teleop.py` (`MouseTeleopIntervention.__init__`, `_on_motion`, `_on_wheel`, 좌클릭 바인딩, `_HELP`)
- Modify: `projects/square_assembly/src/square_assembly/scripts/collect_square_scripted_intervention.py` (`run()`의 mode 분기, `main()`의 CLI, 모듈 docstring)
- Test: `projects/square_assembly/tests/test_quest_teleop.py`에 추가

**Interfaces:**
- Consumes: Task 1의 `QuestTeleopController`, `close_xrt`
- Produces:
  - `MouseTeleopIntervention(..., mouse_input=True)`: False면 마우스 이동, 휠, 좌클릭을 무시한다. 자석 기능과 포인터 warp도 하지 않는다. 키 도움말 둘째 줄은 `"[s] pause   [q] give up   Quest: grip=move  trigger=gripper"`로 바뀐다.
  - `run(..., mode in {"oracle","mouse","quest"}, task="square", quest=None)`
    - `"quest"`는 mouse와 같은 경로를 탄다. 지금 `mode == "mouse"`로 걸려 있는 조건은 모두 `mode in ("mouse", "quest")`로 바꾼다.
    - quest일 때는 `QuestTeleopController(**quest)`와 `mouse_input=False`를 쓴다. 종료 시 `close_xrt()`를 부른다.
    - `human_only`일 때는 task 설정을 `configs/task/{task}.yaml`에서 읽는다.
  - CLI 옵션
    - `--mode quest`
    - `--task {square,can}`(기본 square, `--human-only`에서만 쓰고 정책이 있으면 체크포인트의 task를 쓴다)
    - `--quest-pos-scale 1.0`, `--quest-rot-scale 1.0`, `--quest-kp 2.0`, `--quest-kd 2.0`, `--quest-grip-threshold 0.9`, `--quest-trigger-threshold 0.5`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_mouse_input_disabled_ignores_motion_and_wheel():
    # MouseTeleopIntervention(fake_env, controller=QuestTeleopController(reader=FakeReader(...)),
    #                         mouse_input=False, state_fn=lambda: _state(...))
    # _on_motion(100, 100), _on_wheel(3) 호출 → controller.target_xy / target_yaw 변화 없음, 예외 없음
def test_human_only_task_cfg_selects_can():
    # 수집기의 task 설정 로더(run()에서 뽑아낸 load_human_only_task_cfg(task) -> DictConfig)
    # "can" → 결과 env_name이 PickPlaceCan, "square" → NutAssemblySquare
```
(`fake_env`는 `test_mouse_teleop.py`에 이미 있는 방식을 따른다. 없으면 `state_fn`과 `raw` 속성만 있는 SimpleNamespace로 만든다.)

- [ ] **Step 2: 실패 확인** — `docker compose exec dev pytest projects/square_assembly/tests/test_quest_teleop.py -v`. 새 테스트 두 개가 FAIL해야 한다.
- [ ] **Step 3: 구현** — 위 Interfaces대로 구현한다. `load_human_only_task_cfg(task)`는 수집기 모듈의 최상위 함수로 둔다. 모듈 docstring에 Quest 사용 절차를 추가한다.
  1. `bash /opt/apps/roboticsservice/runService.sh`
  2. Quest XRoboToolkit 앱에서 이 PC의 IP를 선택한다.
  3. **모니터를 정면으로 본 채** 오른쪽 Meta 버튼을 길게 눌러 recenter한다.
  4. `--mode quest [--human-only --task can]`으로 실행한다. 그립을 누른 동안 팔이 따라오고, 트리거로 그리퍼를 조작한다. 쥔 물체를 넘겨받을 때는 트리거를 당긴 채 Tab을 누른다.
- [ ] **Step 4: 통과 확인** — `docker compose exec dev pytest projects/square_assembly/tests/ -q`. 기존 테스트를 포함해 전부 PASS해야 한다.
- [ ] **Step 5: 커밋** — `feat(teleop): add --mode quest and --task to the intervention collector`

### Task 3: SDK 의존성과 실기 연결 확인

**Files:**
- Modify: `projects/square_assembly/requirements.txt`. 기존 PICO 주석 블록에 있는 "xrobotoolkit_sdk: PyPI 아님" 설명이 틀렸으므로 고치고, `xrobotoolkit-sdk==1.1.0`을 추가한다. 이 주석이 언급하는 `pico_intervention`은 이제 없으니 그 문장도 고친다.
- Modify (필요시): `projects/square_assembly/src/square_assembly/runners/quest_teleop.py`의 `M` 기본값

**Interfaces:**
- Consumes: Task 1·2 전부
- Produces: 사용자가 실제 Quest로 조작할 수 있는 상태, 그리고 실측으로 확정한 `M`

- [ ] **Step 1: 설치** — 이미지를 재빌드한다(`docker compose build dev && docker compose up -d dev`). 재빌드가 오래 걸리거나 막히면 사용자에게 알리고, 컨테이너 안에서 `pip install xrobotoolkit-sdk==1.1.0`으로 먼저 확인한다. Expected: `docker compose exec dev python -c "import xrobotoolkit_sdk"`가 성공한다.
- [ ] **Step 2: 연결 확인 (사용자 필요)** — 사용자에게 PC-Service 실행과 Quest 앱 접속을 요청한다. 그다음 `docker compose exec dev python -c "import xrobotoolkit_sdk as x, time; x.init(); time.sleep(1); print(x.get_right_controller_pose(), x.get_right_grip(), x.get_right_trigger()); x.close()"`를 실행한다.
  - Expected: 0이 아닌 포즈가 나오고, 그립을 쥐면 1에 가까운 값이 나온다.
  - 실패하면 호스트에서 `/home/hymm/miniconda3/envs/self-improving-gym/bin/pip install xrobotoolkit-sdk==1.1.0` 뒤 같은 명령을 다시 실행한다. 성공하면 docstring의 실행 예시를 호스트 conda 기준으로 바꾸고 사용자에게 보고한다.
- [ ] **Step 3: 축 실측 (사용자 필요)** — 사용자가 recenter한 뒤 컨트롤러를 오른쪽, 앞, 위로 차례대로 움직인다. 이때 SDK 원시 위치 변화와 `M @ Δp`를 출력하는 일회용 스크립트를 `scratch/`에 둔다. 결과가 각각 월드 +y, −x, +z가 되도록 `M`을 확정한다. 바뀌면 Task 1의 축 테스트 기대값도 함께 고친다.
- [ ] **Step 4: 실조작 확인 (사용자)** — `--mode quest --human-only --task square`와 `--task can`으로 각각 한 에피소드씩 모으게 한다. 사용자가 다음을 확인한다.
  - 1:1 감각이 맞는지
  - 클러치를 잡을 때 튀지 않는지
  - 회전 방향이 맞는지
  - 저장이 되는지
  게인과 필터가 불편하다는 피드백이 오면 CLI 기본값만 조정한다.
- [ ] **Step 5: 커밋** — `feat(teleop): pin xrobotoolkit-sdk and confirm Quest axis mapping`
