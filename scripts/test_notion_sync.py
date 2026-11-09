"""notion_sync의 증분 동기화를 가짜 Notion(메모리 리스트)으로 점검한다: python3 scripts/test_notion_sync.py"""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import notion_sync as ns

store, counter, calls = [], itertools.count(), []


def fake(method, path, body=None):
    calls.append(method)
    if method == "GET":
        return {"results": [dict(b) for b in store], "has_more": False}
    if method == "DELETE":
        store[:] = [b for b in store if b["id"] != path.rsplit("/", 1)[1]]
    elif path.endswith("/children"):
        new = [{**b, "id": f"b{next(counter)}"} for b in body["children"]]
        at = next((i + 1 for i, b in enumerate(store) if b["id"] == body.get("after")), 0 if "after" not in body else len(store))
        store[at:at] = new
        return {"results": new}
    else:
        b = next(b for b in store if b["id"] == path.rsplit("/", 1)[1])
        b.update(body)
    return {}


ns.call = fake


def texts():
    return [b[b["type"]]["rich_text"][0]["text"]["content"] for b in store]


def sync(md, old):
    blocks = ns.to_blocks(md)
    ids = ns.incremental("P", old, blocks) if old else None
    if ids is None:
        ids = ns.full_rewrite("P", blocks)
    assert texts() == [b[b["type"]]["rich_text"][0]["text"]["content"] for b in blocks], texts()
    return [(i, ns.h(b), b["type"]) for i, b in zip(ids, blocks)]


state = sync("# A\nb\nc\nd", None)
keep = {s[0] for s in state}
calls.clear()
state = sync("# A\nb\nc\nd", state)
assert calls == [], "변경 없으면 호출 없음"
state = sync("# A\nb\nX\nd", state)  # 중간 수정: 같은 종류는 제자리 수정
assert {s[0] for s in state} == keep
state = sync("# A\nb\nX\nnew1\nnew2\nd", state)  # 중간 삽입
state = sync("# A\nb\nd", state)  # 삭제
state = sync("# A\n- b\nd", state)  # 종류 변경(문단 -> 목록)
state = sync("first\n# A\n- b\nd", state)  # 맨 앞 삽입은 전체 재작성으로 폴백
print("ok")
