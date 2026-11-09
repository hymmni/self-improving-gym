#!/usr/bin/env python3
"""
PreToolUse 훅 — 위험한 셸 명령을 차단한다. 세 에이전트가 공용으로 쓴다.

| 도구              | 명령 필드                  | 차단 방식                      |
| Claude Code·Codex | tool_input.command         | stderr + exit 2                |
| Antigravity       | toolCall.args.CommandLine  | stdout의 deny JSON + exit 0    |

입력을 읽지 못하면 통과시키지 않고 차단한다 — 훅이 조용히 무력화되는 것을 막기 위해서다.
어느 도구의 입력인지 알 수 없으므로 이때는 두 방식을 모두 쓴다.

에이전트가 main에 커밋·병합·push하는 명령도 막는다. 이 스크립트는 에이전트의 명령에만 걸리므로,
사람과 에이전트를 구분하지 못하는 git hook 대신 여기서 막는다. 사람은 터미널에서 직접 실행한다.
"""

from __future__ import annotations   # Python 3.8에서도 import된다. 깨지면 훅이 차단 없이 통과된다.

import json
import os
import re
import shlex
import subprocess
import sys

# 사고를 막는 문자열 매칭이다. 스크립트 경유나 공백이 든 `git -C "a b"`는 못 잡고, 커밋 메시지 안의
# ` -n` 같은 글자는 잘못 잡는다. 잘못 잡히면 메시지를 파일로 넘기는 `git commit -F`를 쓴다.
GIT = r"\bgit(?:\s+-c\s+\S+|\s+--[\w-]+(?:=\S+)?)*\s+"   # git -C dir, -c k=v, --git-dir=x
SAME = r"[^;&|\n]*"                                      # 같은 셸 명령 안
DANGEROUS = re.compile(
    rf"\brm(?={SAME}\s(-[a-z]*r|--recursive))(?={SAME}\s(-[a-z]*f|--force)){SAME}"
    rf"|{GIT}push\b{SAME}\s(--force|-[a-z]*f[a-z]*\b|\+\S)"
    rf"|{GIT}reset\b{SAME}\s--hard"
    rf"|{GIT}clean\b{SAME}\s(-[a-z]*f|--force)"
    r"|DROP\s+TABLE"
    rf"|\bgit\b{SAME}\s--no-verify\b"
    rf"|{GIT}commit\b{SAME}\s-[a-z]*n[a-z]*\b"
    rf"|\bgit\b{SAME}\bunset\S*\s{SAME}core\.hooksPath"
    rf"|\bgit\b{SAME}core\.hooksPath="
    rf"|\bgit\b{SAME}core\.hooksPath[ \t]+(?!\.githooks[ \t]*($|[;&|)\n]))(?![012]?>|#|[;&|)])\S"
    r"|\bgh\s+pr\s+merge\b",
    re.IGNORECASE,
)


# main 보호. 명령을 셸 구분자로 나눠 순서대로 따라가며 `cd`, `git -C`, 브랜치 전환을 반영한다.
# 스크립트 경유, 서브셸 안의 cd, 변수로 만든 경로는 못 따라간다.
SEGMENTS = re.compile(r"&&|\|\||[;&|\n]")
CHANGES_BRANCH = {"commit", "merge", "cherry-pick", "revert", "rebase", "am"}
NEW_BRANCH = {"-c", "-C", "-b", "-B", "--create", "--force-create", "--orphan"}
MAIN_REFS = ("main", "refs/heads/main")


def git_in(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=10)


def current_branch(repo: str) -> str | None:
    """커밋이 없거나 브랜치가 아니면 None. 첫 커밋은 main에서 해야 하므로 막지 않는다."""
    try:
        if git_in(repo, "rev-parse", "-q", "--verify", "HEAD").returncode != 0:
            return None
        return git_in(repo, "symbolic-ref", "-q", "--short", "HEAD").stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def is_local_branch(repo: str, name: str) -> bool:
    try:
        return git_in(repo, "rev-parse", "-q", "--verify", f"refs/heads/{name}").returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def words(segment: str) -> list[str]:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()
    while tokens and re.match(r"\w+=", tokens[0]):   # `VAR=value git ...`
        tokens.pop(0)
    return tokens


def changes_main(command: str, cwd: str) -> str | None:
    if "git" not in command:
        return None
    here = cwd
    branches: dict[str, str | None] = {}   # 이 명령 안에서 레포마다 체크아웃된 브랜치

    def branch(repo):
        if repo not in branches:
            branches[repo] = current_branch(repo)
        return branches[repo]

    for segment in SEGMENTS.split(command):
        tokens = words(segment)
        if len(tokens) >= 2 and tokens[0] == "cd":
            here = os.path.normpath(os.path.join(here, os.path.expanduser(tokens[1])))
            continue
        if not tokens or tokens[0] != "git":
            continue
        repo, i = here, 1
        while i < len(tokens) and tokens[i].startswith("-"):   # git -C dir, -c k=v, --no-pager
            if tokens[i] == "-C" and i + 1 < len(tokens):
                repo = os.path.normpath(os.path.join(here, os.path.expanduser(tokens[i + 1])))
            i += 2 if tokens[i] in ("-C", "-c") else 1
        if i >= len(tokens):
            continue
        sub, args, shown = tokens[i], tokens[i + 1:], segment.strip()
        if sub in ("switch", "checkout"):
            created = [args[j + 1] for j, arg in enumerate(args[:-1]) if arg in NEW_BRANCH]
            target = args[0] if args and not args[0].startswith("-") else None
            if created:
                branches[repo] = created[0]
            elif target and (sub == "switch" or is_local_branch(repo, target)):   # checkout은 파일일 수 있다
                branches[repo] = target
        elif sub in CHANGES_BRANCH and branch(repo) == "main":
            return shown
        elif sub == "push":
            if {"--all", "--mirror"} & set(args):
                return shown
            refs = [arg for arg in args if not arg.startswith("-")][1:]   # 첫 단어는 원격 이름
            if any(ref.split(":")[-1] in MAIN_REFS for ref in refs):   # +main은 force push 규칙이 막는다
                return shown
            if branch(repo) == "main" and (not refs or "HEAD" in refs):
                return shown
    return None


def deny_json(reason: str) -> str:
    return json.dumps({"decision": "deny", "reason": reason})


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        antigravity = "toolCall" in payload
        command = (payload["toolCall"]["args"]["CommandLine"] if antigravity
                   else payload["tool_input"]["command"])
        dangerous = DANGEROUS.search(command)
        cwd = (payload["toolCall"]["args"].get("Cwd") if antigravity else payload.get("cwd")) or os.getcwd()
    except Exception as e:
        reason = f"BLOCKED: 훅 입력을 읽지 못했습니다 ({e!r})."
        print(deny_json(reason))
        print(reason, file=sys.stderr)
        return 2
    if dangerous:
        reason = f"BLOCKED: 위험한 명령어가 감지되었습니다: {dangerous.group(0)}"
    else:
        main_change = changes_main(command, str(cwd))
        if not main_change:
            return 0
        reason = (f"BLOCKED: 에이전트는 main에 커밋·병합·push하지 않습니다: {main_change.strip()}"
                  " — 필요하면 사용자가 터미널에서 직접 실행합니다.")
    if antigravity:
        print(deny_json(reason))
        return 0
    print(reason, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
