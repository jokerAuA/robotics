#!/usr/bin/env python3
"""SmolVLA + LIBERO 评估入口。

已在 RTX 4090 / torch 2.11.0+cu126 上实测通过：
  libero_spatial 全部 10 个任务 x 1 episode → 成功率 80%（8/10），总耗时 93.6 s（≈9.4 s/episode）。

用法：
    bash scripts/run.sh scripts/eval.py                              # libero_spatial，每任务 1 episode
    bash scripts/run.sh scripts/eval.py --suite libero_object -n 10  # 指定套件 + episode 数
    bash scripts/run.sh scripts/eval.py --task-ids "[0]"             # 只跑 task 0
    CUDA_VISIBLE_DEVICES=2 bash scripts/run.sh scripts/eval.py       # 指定显卡
    LEROBOT_POLICY=lerobot/pi05_libero bash scripts/run.sh scripts/eval.py   # 换别的策略
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SmolVLA + LIBERO 评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--suite", default="libero_spatial", choices=SUITES, help="评估套件")
    p.add_argument("-n", "--episodes", type=int, default=1, help="每个任务的 episode 数")
    p.add_argument("--task-ids", default="", help='只跑指定任务，如 "[0]" 或 "[0,1,2]"；留空=该套件全部')
    p.add_argument(
        "--policy",
        default=os.environ.get("LEROBOT_POLICY", "lerobot/smolvla_libero"),
        help="策略权重（HF repo id 或本地目录）",
    )
    p.add_argument("--device", default="cuda", help="推理设备")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output-dir", default=None, help="默认 outputs/eval_<suite>")
    return p.parse_args()


def build_eval_args(a: argparse.Namespace) -> list[str]:
    """组装 lerobot-eval 的命令行参数。"""
    args = [
        f"--policy.path={a.policy}",
        "--env.type=libero",
        f"--env.task={a.suite}",
        f"--eval.batch_size={a.batch_size}",
        f"--eval.n_episodes={a.episodes}",
        "--eval.use_async_envs=false",
        f"--policy.device={a.device}",
        f"--output_dir={a.output_dir or f'outputs/eval_{a.suite}'}",
    ]

    # SmolVLA 检查点继承 lerobot/smolvla_base（SO-100 真机）的输入签名：
    #   3 路相机 camera1/camera2/camera3 + state 6 维
    # 而 LIBERO 提供的是 2 路相机 image/image2 + state 8 维
    # → 必须「相机改名」+「补一个空相机」，否则报 Feature mismatch
    if "smolvla" in a.policy:
        args.append(
            "--env.camera_name_mapping="
            '{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
        )
        args.append("--policy.empty_cameras=1")

    if a.task_ids:
        args.append(f"--env.task_ids={a.task_ids}")

    return args


def find_lerobot_eval() -> str:
    """优先用与当前解释器同目录的 lerobot-eval，避免环境串了。"""
    sibling = Path(sys.executable).with_name("lerobot-eval")
    if sibling.is_file():
        return str(sibling)
    return shutil.which("lerobot-eval") or ""


def main() -> int:
    a = parse_args()

    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("MUJOCO_GL", "egl")  # 服务器无显示器 → EGL 无头渲染
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")  # 默认用第 3 张显卡（0/1/2/3）

    # 检查当前环境是否有 lerobot-eval，避免跑到错误的 conda 环境里
    eval_bin = find_lerobot_eval()
    if not eval_bin:
        print(
            f"❌ 当前环境里没有 lerobot-eval。\n"
            f"   正在使用的解释器：{sys.executable}\n"
            f"   lerobot 只装在 ai312 环境里，请改用：\n"
            f"     bash scripts/run.sh scripts/eval.py          # 推荐\n"
            f"     $HOME/miniconda3/envs/ai312/bin/python scripts/eval.py",
            file=sys.stderr,
        )
        return 1

    out_dir = a.output_dir or f"outputs/eval_{a.suite}"
    cmd = [eval_bin, *build_eval_args(a)]

    print("=" * 46)
    print(f" 套件       : {a.suite}")
    print(f" 任务       : {a.task_ids or '（该套件全部任务）'}")
    print(f" 每任务回合 : {a.episodes}")
    print(f" 策略       : {a.policy}")
    print(f" 显卡       : {os.environ['CUDA_VISIBLE_DEVICES']}  (MUJOCO_GL={os.environ['MUJOCO_GL']})")
    print(f" Python     : {sys.executable}")
    print(f" 输出       : {out_dir}")
    print("=" * 46)
    print(" ".join(cmd))
    print()

    rc = subprocess.call(cmd)
    if rc == 0:
        print(f"\n✅ 评估结束。\n   指标 : {out_dir}/eval_info.json\n   视频 : {out_dir}/videos/")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
