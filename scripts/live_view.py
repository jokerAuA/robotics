#!/usr/bin/env python3
"""评估 / 推理过程的实时可视化窗口。

它只是一个"贴图窗口"，**不改变 MuJoCo 的渲染后端**：
  * MuJoCo 继续用 EGL 在 GPU 上离屏渲染（快）；
  * 每渲染出一帧 RGB，就用 Tkinter 贴到窗口里（走 X11 / VNC 桌面）。
因此它不依赖 X 上的 OpenGL 加速 —— 即使桌面是 llvmpipe 软件渲染（VNC 常见）也不卡。

为什么不用 cv2.imshow / mujoco.viewer：
  * 本机 OpenCV 是 headless 构建（GUI: NONE），没有 imshow；
  * X 桌面只有软件 OpenGL，开 GLFW 3D 窗口会极慢。
而 tkinter + Pillow 在 ai312 环境里现成可用，不需要装任何包。

配合 `scripts/eval.py --view` 使用。
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

DEFAULT_TITLE = "SmolVLA + LIBERO —— 实时推理"


def _to_uint8(img: Any) -> np.ndarray:
    """把任意相机图像规整成 uint8 HWC。"""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))  # CHW -> HWC
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    if a.size and float(a.max()) <= 1.0 + 1e-6:
        a = a * 255.0
    return np.clip(a, 0.0, 255.0).astype(np.uint8)


def render_env_frames(env: Any, index: int = 0) -> list[np.ndarray]:
    """取第 index 个 sub-env 的画面，返回 [agentview, wrist, ...]。

    已按可视化惯例做上下 + 左右翻转（与 lerobot `LiberoEnv.render()` 保持一致）。
    """
    sub_envs = getattr(env, "envs", None)

    # 1) SyncVectorEnv：直接问底层 robosuite 要全部相机（能得到 2 路画面）
    if sub_envs is not None:
        try:
            sub = sub_envs[index]
        except Exception:
            sub = None
        if sub is not None:
            try:
                raw = sub._env.env._get_observations()  # noqa: SLF001
                names = getattr(sub, "camera_name", None)
                if isinstance(names, str):  # "a,b" 形式
                    names = [c.strip() for c in names.split(",") if c.strip()]
                elif isinstance(names, (list, tuple)):  # lerobot 已解析成 list
                    names = [str(c).strip() for c in names]
                else:
                    names = []
                frames = [np.asarray(raw[n])[::-1, ::-1] for n in names if n in raw]
                if frames:
                    return frames
            except Exception:
                pass
            try:
                return [np.asarray(sub.render())]
            except Exception:
                return []

    # 2) AsyncVectorEnv 兜底：只能拿到 LiberoEnv.render() 的一路画面
    call = getattr(env, "call", None)
    if call is not None:
        try:
            frames = list(call("render"))
            if frames:
                return [np.asarray(frames[index])]
        except Exception:
            pass
    return []


class LiveView:
    """一个 Tkinter 实时窗口。必须在主线程创建。"""

    def __init__(
        self,
        title: str = DEFAULT_TITLE,
        scale: float = 1.0,
        max_fps: float = 25.0,
    ) -> None:
        import tkinter as tk

        from PIL import Image, ImageDraw, ImageTk

        self._tk = tk
        self._Image = Image
        self._ImageDraw = ImageDraw
        self._ImageTk = ImageTk

        self._scale = max(0.2, float(scale))
        self._min_dt = 1.0 / max(1.0, float(max_fps))
        self._last = 0.0
        self._photo = None

        self.closed = False
        self.frames_shown = 0

        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="black")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.image_label = tk.Label(self.root, bg="black", bd=0, highlightthickness=0)
        self.image_label.pack()
        self.status_label = tk.Label(
            self.root,
            bg="black",
            fg="#7CFC00",
            anchor="w",
            justify="left",
            font=("DejaVu Sans Mono", 10),
        )
        self.status_label.pack(fill="x")
        self.root.update()

    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        self.closed = True

    def show(self, frame: Any, caption: str = "") -> bool:
        """显示一帧。返回 False 表示窗口已被关闭（调用方可以忽略，不必中断评估）。"""
        if self.closed:
            return False

        now = time.monotonic()
        if now - self._last < self._min_dt:  # 限帧，别把评估拖慢
            return True
        self._last = now

        img = self._Image.fromarray(_to_uint8(frame))
        if self._scale != 1.0:
            img = img.resize(
                (max(1, int(img.width * self._scale)), max(1, int(img.height * self._scale))),
                self._Image.BILINEAR,
            )

        if caption:
            draw = self._ImageDraw.Draw(img)
            draw.rectangle([0, 0, img.width, 15], fill=(0, 0, 0))
            draw.text((4, 2), caption, fill=(0, 255, 0))

        try:
            self._photo = self._ImageTk.PhotoImage(img)
            self.image_label.configure(image=self._photo)
            self.status_label.configure(text=caption or " ")
            self.root.update()
        except Exception:
            self.closed = True
            return False

        self.frames_shown += 1
        return True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass
