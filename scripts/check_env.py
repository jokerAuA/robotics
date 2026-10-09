#!/usr/bin/env python3
"""SmolVLA + LIBERO 真实环境自检。

替代已废弃的 verify_installation.py（那个只查 import，毫无意义）。
本脚本会真建 LIBERO 环境、真跑 step、真导入 SmolVLA 策略。

用法：
    bash scripts/run.sh scripts/check_env.py     # 推荐：自动用 ai312 解释器
    python scripts/check_env.py                  # 需先 conda activate ai312
"""
from __future__ import annotations

import os

# 必须在 import mujoco / torch 之前设置
os.environ.setdefault("MUJOCO_GL", "egl")
# 自检时不限制可见 GPU，以便报告全部设备（评估脚本才限定单卡）
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import importlib  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f"  —— {detail}" if detail else ""))


def section(n: int, title: str) -> None:
    print(f"\n{'=' * 62}\n {n}. {title}\n{'=' * 62}")


# ------------------------------------------------------------------ 1
section(1, "Python / 解释器")
v = sys.version_info
check("Python >= 3.12", v >= (3, 12), f"{v.major}.{v.minor}.{v.micro}  {sys.executable}")

# ------------------------------------------------------------------ 2
section(2, "PyTorch / CUDA")
try:
    import torch

    check("torch 可导入", True, f"{torch.__version__}  (编译于 CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        check(
            "CUDA 可用",
            False,
            "torch 的 CUDA 版本高于驱动支持的上限。\n"
            "       → nvidia-smi 右上角看驱动支持的 CUDA 上限，让 torch 的 CUDA 版本 <= 该上限\n"
            "       → cu13x 需要驱动 >= 580；cu12x 只需 >= 525\n"
            "       → 本机可用：pip install \"torch==2.11.0+cu126\" \"torchvision==0.26.0+cu126\" "
            "--index-url https://download.pytorch.org/whl/cu126",
        )
    else:
        n = torch.cuda.device_count()
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
        check("CUDA 可用", True, f"{n} 张卡  {names}")
        a = torch.randn(256, 256, device="cuda")
        _ = a @ a
        torch.cuda.synchronize()
        check("GPU 张量运算", True, "矩阵乘法 OK")
except Exception as e:  # noqa: BLE001
    check("torch 可导入", False, f"{type(e).__name__}: {e}")

# ------------------------------------------------------------------ 3
section(3, "关键依赖包")
for mod in [
    "lerobot", "transformers", "accelerate", "mujoco", "robosuite",
    "libero", "bddl", "num2words", "safetensors", "datasets", "gymnasium",
]:
    try:
        m = importlib.import_module(mod)
        check(mod, True, str(getattr(m, "__version__", "?")))
    except Exception as e:  # noqa: BLE001
        check(mod, False, f"{type(e).__name__}: {str(e)[:70]}")

# ------------------------------------------------------------------ 4
section(4, "LIBERO 配置")
cfg = Path(os.path.expanduser("~/.libero/config.yaml"))
if cfg.exists():
    check("~/.libero/config.yaml 存在", True, str(cfg))
    try:
        import yaml

        d = yaml.safe_load(cfg.read_text())
        missing = [k for k in ("bddl_files", "init_states") if not Path(d.get(k, "")).exists()]
        check(
            "bddl_files / init_states 路径有效",
            not missing,
            ("缺失: " + ", ".join(missing)) if missing else "OK",
        )
    except Exception as e:  # noqa: BLE001
        check("配置可解析", False, f"{type(e).__name__}: {e}")
else:
    check(
        "~/.libero/config.yaml 存在",
        False,
        "缺失会让 import libero.libero 弹交互式 input() → 非交互环境直接 EOFError。\n"
        "       修复命令（回答 N = 使用默认路径即可）：\n"
        "       echo \"N\" | python -c \"import libero.libero\"\n"
        "       等价于 README.md「安装」第 5 步。",
    )

# ------------------------------------------------------------------ 5
section(5, "LIBERO 仿真（真建环境 + 真跑 step）")
try:
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark("libero_spatial")()
    task = suite.get_task(0)
    # 注意：task.problem 是占位符字符串 "Libero"，不是路径！
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.reset()
    check("创建环境 + reset", True, task.name[:44])

    from lerobot.envs.libero import get_task_init_states

    init = get_task_init_states(suite, 0)
    env.set_init_state(init[0])
    check("set_init_state", True, f"init_states shape {init.shape}")

    for _ in range(3):
        obs, _r, _done, _info = env.step(np.zeros(7, dtype=np.float32))
    check(
        "连续 step",
        True,
        f"agentview_image {obs['agentview_image'].shape} {obs['agentview_image'].dtype}",
    )
    env.close()
except Exception as e:  # noqa: BLE001
    check("LIBERO 仿真可用", False, f"{type(e).__name__}: {str(e)[:90]}")

# ------------------------------------------------------------------ 6
section(6, "SmolVLA 策略")
try:
    from lerobot.policies.smolvla import (  # noqa: F401
        SmolVLAPolicy,
        make_smolvla_pre_post_processors,
    )

    check("SmolVLAPolicy 可导入", True)
    check("make_smolvla_pre_post_processors 可导入", True)
except Exception as e:  # noqa: BLE001
    check("SmolVLA 可导入", False, f"{type(e).__name__}: {str(e)[:90]}")

# ------------------------------------------------------------------ 总结
print(f"\n{'=' * 62}\n 自检总结\n{'=' * 62}")
for n, ok, _ in RESULTS:
    print(f"  {'✅' if ok else '❌'} {n}")

failed = [n for n, ok, _ in RESULTS if not ok]
if failed:
    print(f"\n❌ {len(failed)} 项未通过：{', '.join(failed)}")
    print("   请对照上面的提示修复，常见原因见 README.md「常见问题」。")
    raise SystemExit(1)

print("\n✅ 全部通过。可以运行： bash scripts/run.sh scripts/eval.py")
