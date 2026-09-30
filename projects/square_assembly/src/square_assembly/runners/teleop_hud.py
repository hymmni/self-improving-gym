"""텔레옵 창의 HUD(이름표·모드 알약·키캡·게이지) — PIL로 안티에일리어싱해 반투명 오버레이에 그린다.
cv2 Hershey 글씨는 작게 쓰면 뭉개지고 반투명·둥근 모서리가 없어서 HUD만 여기로 뺐다(2026-09-30)."""

from functools import lru_cache

from PIL import ImageFont

# 모드 색(RGB) — 맵의 실제 그리퍼 색과 같게 맞춘다(mouse_teleop._draw_map).
POLICY = (46, 204, 113)
HUMAN = (255, 99, 72)
PAUSED = (255, 190, 50)
TEXT = (236, 239, 243)
DIM = (150, 156, 166)
PANEL = (14, 16, 20, 170)


@lru_cache(maxsize=None)
def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:  # 폰트가 없는 환경(테스트 등) — Pillow 내장 스케일 폰트
        return ImageFont.load_default(size=size)


def pill(draw, xy, text, fill, color=(20, 20, 20), size=18, bold=True, pad=(12, 6)):
    """둥근 알약 하나. 오른쪽 끝 x를 돌려준다."""
    f = font(size, bold)
    x, y = xy
    w = draw.textlength(text, font=f)
    h = size + 2 * pad[1]
    draw.rounded_rectangle((x, y, x + w + 2 * pad[0], y + h), radius=h // 2, fill=fill)
    draw.text((x + pad[0], y + h / 2), text, font=f, fill=color, anchor="lm")
    return x + w + 2 * pad[0]


def row(draw, xy, items, size=15):
    """한 줄에 (종류, 글자)를 이어 그린다: "key"는 키캡, "text"는 설명(흐린 색), "value"는 밝은 값.
    끝 x를 돌려준다."""
    x, y = xy
    for kind, text in items:
        f = font(size, bold=kind == "key")
        w = draw.textlength(text, font=f)
        if kind == "key":
            draw.rounded_rectangle((x, y - size * 0.75, x + w + 12, y + size * 0.75), radius=5,
                                   outline=(200, 205, 215, 200), width=1, fill=(255, 255, 255, 22))
            draw.text((x + 6, y), text, font=f, fill=TEXT, anchor="lm")
            x += w + 12 + 6
        else:
            draw.text((x, y), text, font=f, fill=TEXT if kind == "value" else DIM, anchor="lm")
            x += w + (18 if kind == "text" else 6)
    return x
