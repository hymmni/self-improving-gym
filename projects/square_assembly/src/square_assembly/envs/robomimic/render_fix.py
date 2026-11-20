"""robosuite 오프스크린 렌더가 GC 시점에 깨지는 결함을 고친다 (2026-11-20 원인 확정).

robosuite(1.5.1) 기본 renderer("mjviewer")의 hard reset은 옛 MjSim을 정리하지 않고 새 것으로 바꿔 끼운다. 옛 sim과
그 렌더 컨텍스트(MjRenderContext)는 서로를 참조하는 순환이라 파이썬 GC가 **임의의 시점**(대개 다음 에피소드 도중)에
치우는데, 그때 MjRenderContext.__del__이
  1. `con.free()` — 지금 붙어 있는 **새** 컨텍스트에서 옛 GL 객체 번호(프레임버퍼·텍스처)를 지운다. 두 컨텍스트는
     같은 순서로 객체를 만들어 번호가 겹치므로 새 env의 렌더 버퍼가 지워진다.
  2. `gl_ctx.free()` — `eglReleaseThread()`로 새 컨텍스트까지 떼어 낸다.
그 뒤 렌더는 아무것도 그리지 못하고 read_pixels의 `np.empty`(초기화 안 된 메모리)가 이미지로 나온다 — 줄무늬·검은
화면·노이즈. 다음 reset이 새 컨텍스트를 붙이면 회복된다. gc.collect()를 부르면 결정적으로 재현된다
(scratch/egl_gc_probe.py). DOCKER.md §4의 "첫 에피소드 노이즈"와 에피소드 도중 노이즈가 모두 이것이다.

고침: 해제는 자기 컨텍스트를 붙인 채 하고 원래 붙어 있던 컨텍스트를 되돌려 놓는다. 렌더·픽셀 읽기 직전마다 자기
컨텍스트를 붙인다(env가 둘 이상인 프로세스에서도 남의 컨텍스트에 그리지 않게).
"""

import sys

_INSTALLED = False


def _free_in_own_context(rc):
    con, ctx = rc.__dict__.get("con"), rc.__dict__.get("gl_ctx")
    if ctx is None:
        return
    egl = sys.modules.get("robosuite.renderers.context.egl_context")
    if egl is not None and isinstance(ctx, egl.EGLGLContext):
        if not ctx._context:
            return
        E, own = egl.EGL, ctx._context
        prev = E.eglGetCurrentContext()
        E.eglMakeCurrent(egl.EGL_DISPLAY, E.EGL_NO_SURFACE, E.EGL_NO_SURFACE, own)
        if con is not None:
            con.free()
        keep = prev and prev.address != own.address   # 자기 자신이 붙어 있었으면 떼고 끝낸다
        E.eglMakeCurrent(egl.EGL_DISPLAY, E.EGL_NO_SURFACE, E.EGL_NO_SURFACE, prev if keep else E.EGL_NO_CONTEXT)
        E.eglDestroyContext(egl.EGL_DISPLAY, own)
        ctx._context = None
        return
    # osmesa/glfw: 원래 컨텍스트를 되돌리진 못하지만 다음 렌더가 자기 것을 다시 붙인다
    ctx.make_current()
    if con is not None:
        con.free()
    ctx.free()


def install():
    """robosuite MjRenderContext를 고친 동작으로 바꾼다. 여러 번 불러도 한 번만 적용된다."""
    global _INSTALLED
    if _INSTALLED:
        return
    from robosuite.utils import binding_utils as B

    render, read_pixels = B.MjRenderContext.render, B.MjRenderContext.read_pixels

    def render_own(self, *a, **k):
        self.gl_ctx.make_current()
        return render(self, *a, **k)

    def read_pixels_own(self, *a, **k):
        self.gl_ctx.make_current()
        return read_pixels(self, *a, **k)

    def __del__(self):
        try:
            _free_in_own_context(self)
        except Exception:   # 인터프리터 종료 중엔 EGL 모듈이 먼저 사라져 있을 수 있다
            pass

    B.MjRenderContext.render = render_own
    B.MjRenderContext.read_pixels = read_pixels_own
    B.MjRenderContext.__del__ = __del__
    _INSTALLED = True
