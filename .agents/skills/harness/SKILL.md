---
name: harness
description: 이 프로젝트의 개발 워크플로우. 새 구현 작업을 설계·계획·실행할 때 사용한다.
---

이 프로젝트는 Agent Harness를 사용한다. 아래 워크플로우에 따라 작업을 진행하라.

---

## 워크플로우

### A. 탐색

`docs/` 하위 문서(ARCHITECTURE, ADR, 있으면 RESEARCH 등)를 읽고 프로젝트의 아키텍처·설계 의도를 파악한다.

### B. 논의

구현을 위해 구체화하거나 기술적으로 결정해야 할 사항이 있으면 사용자에게 제시하고 논의한다.
설계 전반에 브레인스토밍이 필요하면 `superpowers:brainstorming`을 먼저 쓴다.

### C. 플랜 작성

사용자가 구현 계획 작성을 지시하면 `superpowers:writing-plans` 스킬로
`docs/superpowers/plans/YYYY-MM-DD-<feature>.md`에 checkbox task 플랜을 작성한다.

### D. 실행

작업 전에 규칙에 맞는 브랜치를 만든다. 플랜은 현재 세션에서
`superpowers:executing-plans`나 `superpowers:subagent-driven-development`로 실행하고
끝난 step은 플랜에서 체크한다. 어떤 방식을 쓸지는 실행 전에 사용자에게 묻는다.
세션이 끊기면 새 세션이 체크되지 않은 task부터 이어 간다.

### E. 병합

main 반영은 `AGENTS.md`의 main 반영 방식(`main_via_pr`)을 따른다. task 커밋은 압축하지 않고
그대로 보존한다.

### 세션 정리

task/플랜 완료처럼 의미 있는 작업 단위가 끝날 때마다 컨텍스트가 충분히 쌓였는지
스스로 판단해 컨텍스트 정리나 새 세션 시작을 사용자에게 먼저 권장하라.
