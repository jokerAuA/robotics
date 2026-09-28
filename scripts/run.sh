#!/usr/bin/env bash
# ============================================================
# 通用启动壳 —— 只做三件事：
#   ① 切到项目根目录（让 --output_dir 等相对路径有意义）
#   ② 设无头渲染环境变量（服务器无显示器）
#   ③ 用 ai312 解释器执行你指定的 Python 脚本
# ============================================================
# 用法：
#   bash scripts/run.sh scripts/check_env.py
#   bash scripts/run.sh scripts/eval.py --suite libero_object --episodes 10
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run.sh scripts/eval.py
#   LEROBOT_PYTHON=/别的/解释器 bash scripts/run.sh scripts/eval.py
# ============================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PY="${LEROBOT_PYTHON:-$HOME/miniconda3/envs/ai312/bin/python}"

exec "$PY" "$@"
