"""로컬 마크다운 한 파일을 Notion 페이지 본문으로 덮어쓴다 (단방향: 로컬 -> Notion).

    python3 scripts/notion_sync.py docs/paper/rita_wip_ko.md [--page <url|id>] [--dry-run] [--full]

처음(또는 --full)에는 본문을 전부 다시 쓰고, 이후에는 <md>.notion.json에 저장한 블록 해시와 비교해 바뀐 블록만
수정/삽입/삭제한다 (안 바뀐 블록과 거기 달린 댓글은 그대로). Notion 쪽 직접 편집은 감지하지 않는다.

토큰: 환경변수 NOTION_TOKEN 또는 레포 루트 .env의 NOTION_TOKEN (Notion integration을 만들고
대상 페이지 우상단 ... > Connections에서 그 integration을 추가해야 접근된다).
지원 문법: # ## ###, 이미지 ![캡션](상대경로.png), "- ", "1. ", "> ", 한 줄 $$수식$$, **굵게**, $인라인수식$, `코드`, 표/HTML 주석은 건너뜀.
# ponytail: 줄 하나 = 블록 하나. 줄바꿈 이어쓰기·중첩 목록·표는 필요해지면 추가.
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid

API = "https://api.notion.com/v1"
DEFAULT_PAGE = "4040681a278d80a589d4d4d5a16421af"  # RiTA WIP 페이지


def token():
    t = os.environ.get("NOTION_TOKEN")
    if not t and os.path.exists(".env"):
        for line in open(".env"):
            if line.startswith("NOTION_TOKEN="):
                t = line.split("=", 1)[1].strip().strip("\"'")
    if not t:
        sys.exit("NOTION_TOKEN이 없다 (환경변수 또는 .env).")
    return t


def call(method, path, body=None):
    req = urllib.request.Request(
        API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token()}", "Notion-Version": "2022-06-28", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> {e.code}: {e.read().decode()}")


def rich(text):
    """**굵게**, $수식$, `코드`를 rich_text 조각으로 나눈다."""
    out = []
    for part in re.split(r"(\*\*.+?\*\*|\$[^$]+\$|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**"):
            out.append({"type": "text", "text": {"content": part[2:-2]}, "annotations": {"bold": True}})
        elif part.startswith("$"):
            out.append({"type": "equation", "equation": {"expression": part[1:-1]}})
        elif part.startswith("`"):
            out.append({"type": "text", "text": {"content": part[1:-1]}, "annotations": {"code": True}})
        else:
            out.append({"type": "text", "text": {"content": part}})
    return out


def block(kind, text):
    return {"object": "block", "type": kind, kind: {"rich_text": rich(text)}}


def to_blocks(md, base="."):
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    blocks = []
    for line in md.splitlines():
        s = line.strip()
        if not s or s.startswith("|"):
            continue
        if m := re.fullmatch(r"\$\$(.+)\$\$", s):
            blocks.append({"object": "block", "type": "equation", "equation": {"expression": m[1].strip()}})
        elif m := re.fullmatch(r"!\[(.*)\]\((.+)\)", s):  # 이미지: 올릴 때 파일을 업로드해 id로 바꾼다
            path = os.path.join(base, m[2])
            tag = hashlib.sha1(open(path, "rb").read()).hexdigest()
            blocks.append({"object": "block", "type": "image", "image": {
                "type": "file_upload", "file_upload": {"id": f"@{path}:{tag}"}, "caption": rich(m[1])}})
        elif m := re.match(r"(#{1,3}) (.*)", s):
            blocks.append(block(f"heading_{len(m[1])}", m[2]))
        elif m := re.match(r"[-*] (.*)", s):
            blocks.append(block("bulleted_list_item", m[1]))
        elif m := re.match(r"\d+\. (.*)", s):
            blocks.append(block("numbered_list_item", m[1]))
        elif m := re.match(r"> ?(.*)", s):
            blocks.append(block("quote", m[1]))
        else:
            blocks.append(block("paragraph", s))
    return blocks


def h(b):
    return hashlib.sha1(json.dumps(b, sort_keys=True).encode()).hexdigest()


def upload(path):
    """로컬 이미지를 Notion file upload로 올려 id를 돌려준다."""
    name = os.path.basename(path)
    fid = call("POST", "/file_uploads", {"filename": name, "content_type": "image/png"})["id"]
    bd = uuid.uuid4().hex
    body = (f"--{bd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + open(path, "rb").read() + f"\r\n--{bd}--\r\n".encode()
    req = urllib.request.Request(f"{API}/file_uploads/{fid}/send", data=body, method="POST", headers={
        "Authorization": f"Bearer {token()}", "Notion-Version": "2022-06-28",
        "Content-Type": f"multipart/form-data; boundary={bd}"})
    try:
        urllib.request.urlopen(req).read()
    except urllib.error.HTTPError as e:
        sys.exit(f"upload {name} -> {e.code}: {e.read().decode()}")
    return fid


def resolve(b):
    if b["type"] == "image" and b["image"]["file_upload"]["id"].startswith("@"):
        path = b["image"]["file_upload"]["id"][1:].rsplit(":", 1)[0]
        return {**b, "image": {**b["image"], "file_upload": {"id": upload(path)}}}
    return b


def append(page, blocks, after=None):
    """blocks를 page 아래(after 뒤)에 붙이고 새 블록 id 목록을 돌려준다."""
    ids = []
    for i in range(0, len(blocks), 100):
        chunk = [resolve(b) for b in blocks[i:i + 100]]
        body = {"children": chunk, **({"after": after} if after else {})}
        got = [x["id"] for x in call("PATCH", f"/blocks/{page}/children", body)["results"]]
        if len(got) != len(chunk):  # 응답이 부모의 자식 전체일 때를 대비해 after 뒤 조각만 취한다
            j = got.index(after) + 1 if after in got else 0
            got = got[j:j + len(chunk)]
        ids += got
        after = got[-1]
    return ids


def full_rewrite(page, blocks):
    cursor = None
    while True:
        r = call("GET", f"/blocks/{page}/children?page_size=100" + (f"&start_cursor={cursor}" if cursor else ""))
        for b in r["results"]:
            call("DELETE", f"/blocks/{b['id']}")
        if not r["has_more"]:
            break
        cursor = r["next_cursor"]
    return append(page, blocks)


def incremental(page, old, blocks):
    """old = [(id, hash, type)] (마지막 동기화 상태). 바뀐 블록만 수정/삽입/삭제하고 새 id 목록을 돌려준다.
    같은 블록은 건드리지 않아 그 블록의 댓글이 유지된다. 맨 앞 삽입은 API 제약으로 None(전체 재작성)."""
    new_h = [h(b) for b in blocks]
    ops = difflib.SequenceMatcher(None, [x[1] for x in old], new_h, autojunk=False).get_opcodes()
    ids = []
    for t, i1, i2, j1, j2 in ops:
        if t == "equal":
            ids += [x[0] for x in old[i1:i2]]
            continue
        olds, news = old[i1:i2], blocks[j1:j2]
        n = 0
        if t == "replace":  # 같은 종류끼리 짝지어 제자리 수정
            while n < min(len(olds), len(news)) and news[n]["type"] == olds[n][2] != "image":
                bid, nb = olds[n][0], news[n]
                call("PATCH", f"/blocks/{bid}", {nb["type"]: nb[nb["type"]]})
                ids.append(bid)
                n += 1
        for bid, *_ in olds[n:]:
            call("DELETE", f"/blocks/{bid}")
        if news[n:]:
            if not ids:
                return None
            ids += append(page, news[n:], after=ids[-1])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("--page", default=DEFAULT_PAGE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--full", action="store_true", help="상태 파일을 무시하고 전체 재작성")
    a = ap.parse_args()
    page = re.findall(r"[0-9a-f]{32}", a.page.replace("-", ""))[-1]
    blocks = to_blocks(open(a.md, encoding="utf-8").read(), os.path.dirname(a.md))
    if a.dry_run:
        print(json.dumps(blocks, ensure_ascii=False, indent=1))
        return
    state_path = a.md + ".notion.json"
    old = None
    if not a.full and os.path.exists(state_path):
        st = json.load(open(state_path))
        if st["page"] == page:
            old = [tuple(x) for x in st["blocks"]]
    ids = incremental(page, old, blocks) if old else None
    mode = "incremental"
    if ids is None:
        ids, mode = full_rewrite(page, blocks), "full"
    json.dump({"page": page, "blocks": [[i, h(b), b["type"]] for i, b in zip(ids, blocks)]}, open(state_path, "w"))
    print(f"synced {len(blocks)} blocks ({mode}) -> {page}")


if __name__ == "__main__":
    main()
