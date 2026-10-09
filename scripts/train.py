#!/usr/bin/env python3
"""SmolVLA + LIBERO 训练入口（微调）。

与 scripts/eval.py 完全对称的薄封装：只做四件事
    ① 切到项目根目录（让 --output_dir / --dataset.root 等相对路径有意义）
    ② 设无头渲染环境变量 + 指定显卡
    ③ 拼出 lerobot-train 的命令行（含 SmolVLA 特有的「相机键名对齐」）
    ④ 用子进程调 lerobot-train

用法：
  ── ① 快速自检（不训练 / 一分钟内）──────────────────────────────
    bash scripts/run.sh scripts/train.py --dry-run
        # 只打印将要执行的 lerobot-train 命令，不启动训练
    bash scripts/run.sh scripts/train.py --tmp-run
        # 冒烟：4 步 / batch 2 / 只用 episode [0,1,2,3]；验证链路能跑通
    bash scripts/run.sh scripts/train.py --official-recipe --tmp-run
        # 冒烟「官方配方」这条代码路径（会解冻全参数、走路线 B，但只跑 4 步）
    bash scripts/run.sh scripts/train.py --tmp-run --batch-size 32
        # 冒烟但用真实 batch —— 想量真实显存占用就用这个（2026-10-07 修了此处的静默覆盖 bug）

  ── ② 正式微调（推荐；本机 RTX 4090 48G 实测 ≈ 4 小时）──────────
    bash scripts/run.sh scripts/train.py --official-recipe
        # 一键复刻 lerobot/smolvla_libero：25000 步 / batch 32 / 全参数微调 / 路线 B
        # 实测：num_learnable_params=392,904,096（总数 450,046,176）、mem_gb≈28、
        #       smp/s≈56（≈1.72 step/s）；loss 30 步内 3.006 → 0.978
        # 输出目录自动生成 outputs/train/<日期>/<时间>_smolvla_libero
        #   （时间戳目录天然避开 preflight 的「output_dir 已存在」拦截，也不会覆盖旧实验）

  ── ③ 改超参 / 指定资源（显式参数永远优先于 --official-recipe）────
    ... --official-recipe --batch-size 16            # 显存紧张时降 batch（见 ⑤ 末尾）
    ... --official-recipe --steps 50000 --save-freq 10000
    ... --official-recipe --lr 5e-5                  # lr=0 表示不改，沿用权重里的预设
    ... --official-recipe -n 8                       # DataLoader worker 数
    ... --official-recipe --video-backend pyav       # torchcodec 解不开视频时改 pyav
    ... -g 1 --official-recipe                       # 写死显卡；不带 -g 则自动挑空闲显存最多的卡
    ... --job-name my_exp --output-dir outputs/train/my_exp
        # 注意：output_dir 已存在会被 preflight 拦住（lerobot 会抛 FileExistsError）
    ... --official-recipe --policy.train_state_proj=false
        # 任意 lerobot-train 参数都能追加在末尾原样透传；重复时 draccus 取最后一个
    # 说明：预设优先级 = 显式 CLI 参数 > --official-recipe > --tmp-run > argparse 默认值

  ── ④ 相机键路线 ──────────────────────────────────────────────
    默认 = 路线 A：--policy.input_features=null --policy.output_features=null
        （特征从数据集推断，键名保持 image/image2，评估时零附加参数）
    ... --rename-map
        # 路线 B（= 官方做法）：image/image2 → camera1/camera2；
        # 评估时 scripts/eval.py 会自动补 --env.camera_name_mapping
    ... --rename-map '{"observation.images.image": "observation.images.camera1"}'
        # 自带值 = 自定义改名表
    ... --rename-map --empty-cameras 0
        # 路线 B 下补齐几路空相机；不给则按权重声明的相机数自动推断（官方是 0）

  ── ⑤ 长时间训练（几十分钟~几小时）：必须脱离终端 ──────────────
    前台跑的风险（已实测踩过）：关掉 VS Code / SSH 断开 / Ctrl-C /
    **在同一个终端里再敲别的命令** 都会把训练一起带走 —— 那次连日志文件都没来得及创建。

    # 方式 1：nohup（最省事）
    cd /mnt/data/home/wpj/projects/robotics
    PYTHONUNBUFFERED=1 nohup bash scripts/run.sh scripts/train.py --official-recipe > outputs/train_official.log 2>&1 &
    echo "PID=$!"
      · 一定要重定向到文件：脱离终端后 stdout 已无处可写
      · 一定要 PYTHONUNBUFFERED=1：否则重定向到文件后 Python 会块缓冲，
        tail -f 看着像卡死（几 KB 才蹦一次）
      · 重定向里的相对路径由**当前 shell** 解析，所以先 cd 到项目根（或写绝对路径）

    # 方式 2：tmux（更推荐：既能脱离终端，又能随时回到现场看实时输出）
    tmux new -s train
    bash scripts/run.sh scripts/train.py --official-recipe 2>&1 | tee outputs/train_official.log
    #   Ctrl-b d 脱离；tmux attach -t train 回来

    # 看进度
    tail -f outputs/train_official.log
    grep -o "step:[0-9]* " outputs/train_official.log | tail -1
    ls outputs/train/<日期>/<时间>_smolvla_libero/checkpoints/     # 每 save_freq 步一个

    # 停止训练 —— 不要用 kill 启动器的 PID！
    pgrep -af lerobot-train        # 先看真正在跑的是谁
    pkill -f lerobot-train         # 再停它（不行再用 kill -9 指定 PID）
    # 原因：scripts/run.sh 用 exec 起 python（PID 不变），而 train.py 内部是
    #   subprocess.call([lerobot-train, ...]) —— 训练是**另一个 PID 的子进程**。
    #   只 kill 启动器的话，lerobot-train 会被 init 收养、继续跑。

    # 显存不够怎么办（本版本没有梯度检查点开关）
    #   · SmolVLAConfig 里**没有** gradient_checkpointing 字段（模型卡里的通用模板不适用，
    #     传了会被 draccus 拒绝）
    #   · --accelerator.activation_checkpointing 在 configs/train.py:359 是明确未接线的
    #     占位符，传了直接 ValueError
    #   ⇒ 实际手段只有两个：① 降 --batch-size（batch 32≈28GB，减半≈减半）；
    #     ② 退回 --policy.train_expert_only=true（可训参数回到 100M、显存≈7.7GB，
    #        但那就不是官方配方了）

  ── ⑥ 续训（被中断 / 想加长训练）───────────────────────────────
    bash scripts/run.sh scripts/train.py --resume-from outputs/train/<日期>/<时间>_<job>/checkpoints/last/train_config.json
      · 只传 --resume=true + --config_path，其余参数（--policy/--steps/--batch-size/
        --output-dir…）一律忽略 —— 配置整体来自 checkpoint 里的 train_config.json。
        这是必需的：只要给了 --policy.path，lerobot 就永远走不到续训分支。
      · 确实要改的项，追加在命令末尾原样透传，例如：
        ... --resume-from <train_config.json> --steps 50000

  ── ⑦ 训练完 → 评估 ──────────────────────────────────────────
    bash scripts/run.sh scripts/eval.py --policy <job>/checkpoints/last/pretrained_model -n 10
    bash scripts/run.sh scripts/eval.py --policy <job>/checkpoints/0005000/pretrained_model -n 1
        # 中间检查点先看趋势（10 任务 × 1 回合 ≈ 3 分钟），别等 4 小时
    # 参照系：lerobot/smolvla_libero 官方权重 80%(8/10)；base 与 4/30 步产物均为 0%

    python scripts/train.py --help                                # 全部参数

为什么必须处理「相机键名」（本脚本存在的主要理由）：
    `lerobot/smolvla_base` 的 config.json 把输入签名写死成
        observation.images.camera1 / camera2 / camera3 + observation.state[6]
    而 `lerobot/libero` 数据集给的是
        observation.images.image / image2 + observation.state[8] + action[7]
    lerobot 的 make_policy 只在策略 config 的 input_features **为空**时才从数据集推断，
    而 smolvla_base 的 input_features 是非空的 → camera1/2/3 会被保留下来。
    结果 modeling_smolvla.prepare_images() 在 batch 里一个相机都找不到，直接抛
        ValueError: All image features are missing from the batch.
    两条官方解法（都在 lerobot 自带文档/测试里验证过）：

    路线 A（本脚本默认）—— 官方 PEFT 文档「Training SmolVLA on the libero dataset」同款：
        --policy.input_features=null --policy.output_features=null
        让特征从数据集推断，键名保持 image/image2 → 评估时零配置，直接配合 --env.type=libero。
    路线 B（加 --rename-map）—— 对齐 lerobot/smolvla_libero 血统：
        --rename_map='{"observation.images.image": "observation.images.camera1", ...}'
        代价是评估/仿真评测要同时给 --env.camera_name_mapping（scripts/eval.py 会自动加）。

维度差异（state 6 vs 8、action 6 vs 7）不需要处理：
    SmolVLA 的 state_proj / action_in_proj / action_out_proj 都是
    Linear(max_state_dim=32, ·) 且有 pad_vector()，6/7/8 维差异被 padding 吸收；
    output_features 无论如何都会被数据集覆盖；归一化统计量在训练时会被
    lerobot_train.py 强制覆写成数据集的 stats（跳过 checkpoint 里的旧 stats）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_POLICY = "lerobot/smolvla_base"
DEFAULT_DATASET = "lerobot/libero"
DEFAULT_DATASET_ROOT = "datasets/libero"

# 路线 B 的默认改名表：LIBERO 数据集键 → smolvla_base 期望的键
SMOLVLA_LIBERO_RENAME_MAP = json.dumps(
    {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }
)

# argparse 的默认值（预设靠它判断「用户是否显式改过」）
ARG_DEFAULTS = {"steps": 20000, "batch_size": 32, "save_freq": 2000, "log_freq": 50, "num_workers": 4}
# `--tmp-run` 想改成的值
TMP_RUN_OVERRIDES = {"steps": 4, "batch_size": 2, "save_freq": 2, "log_freq": 1}

# ---------------------------------------------------------------------------
# 官方配方（--official-recipe）：一键复刻 `lerobot/smolvla_libero`
# ---------------------------------------------------------------------------
# 来源：Hub 上 `lerobot/smolvla_libero` 的 **train_config.json** —— 作者把训练配置连同
#       权重一起传了，所以配方是公开且权威的：
#       https://huggingface.co/lerobot/smolvla_libero/raw/main/train_config.json
# 为什么以它为准：这是唯一在本仓库实测能打到 80%（8/10）的 SmolVLA 检查点
#       （见 outputs/eval_baseline.log）；而 base（0 步）与只训 4 步的产物都是 0%。
#
# 实测 diff 过 base 与官方权重的 config.json，二者**只有 4 处不同**，
# 另外 3 项（optimizer_lr / betas / eps / wd / grad_clip / warmup / decay_lr /
# resize_imgs_with_padding / chunk_size）本来就完全一致 ⇒ 只需覆盖下面这几项。
OFFICIAL_RECIPE: dict[str, int] = {
    "steps": 25000,  # 官方 25000（脚本默认 20000）
    "batch_size": 32,
    "save_freq": 5000,
    "log_freq": 200,
    "num_workers": 4,
}
# 官方 train_config.json 里 policy.empty_cameras 就是 0（第 3 路相机缺失，不补零图）
OFFICIAL_EMPTY_CAMERAS = 0

# 必须显式覆盖的 policy 字段。为什么必须我们传：这几项会被
# `--policy.path=models/smolvla_base` **带进来**，而 lerobot 不会用数据集去纠正它们
# （configs/train.py:217 的 `from_pretrained(path, cli_overrides=...)` 只是把 CLI 覆写
#  叠在「加载进来的权重配置」之上，所以我们给了才会生效）。
OFFICIAL_POLICY_OVERRIDES: tuple[str, ...] = (
    # base 的 config.json 里这两项都是 True ⇒ 可训参数只有 100M / 450M：
    #   train_expert_only=True     → smolvlm_with_expert.py:157 把**整个 VLM** 冻住
    #                                （self.vlm.eval() + 全部 requires_grad=False），
    #                                只剩 action expert 与几个投影层可训（实测
    #                                num_learnable_params=99880992）；
    #   freeze_vision_encoder=True → 再额外冻 vision_model。
    # 官方两项都是 False = **全参数微调**（只冻 lm_head / 最后一两层 text layer）。
    "--policy.freeze_vision_encoder=false",
    "--policy.train_expert_only=false",
    # base 自带 30000，而我们只跑 25000 步 ⇒ cosine 退火在结束前走不到 decay_lr。
    # 官方是 25000/25000 对齐的。
    "--policy.scheduler_decay_steps=25000",
)


# argparse 的短选项 → dest 名（长选项靠 lstrip("-")+replace("-","_") 自动归一化）
OPTION_ALIASES = {"n": "num_workers", "g": "gpu"}


def explicit_option_names() -> set[str]:
    """返回「用户在命令行上真的敲了」的选项名（已归一化掉前导 - 与 =value）。

    为什么不能靠「值 == argparse 默认值」来判断是否显式：
        `--tmp-run --batch-size 32` 里的 32 恰好等于默认值，按值比较会被判定成
        「没传」，于是被 --tmp-run 静默压回 2 —— 而这正是「想用真实 batch 量显存」
        时最不能接受的（会测出一个假的 1.7GB）。
    """
    names: set[str] = set()
    for tok in sys.argv[1:]:
        if not tok.startswith("-"):
            continue
        name = tok.split("=", 1)[0].lstrip("-").replace("-", "_")
        names.add(OPTION_ALIASES.get(name, name))
    return names


def effective_policy_flag(a: argparse.Namespace, name: str) -> str | None:
    """取 `--policy.<name>` 的**最终生效**值（后出现的覆盖先出现的）；没出现则 None。

    为什么需要：`--official-recipe` 自己会追加 `--policy.freeze_vision_encoder=false` 等，
    用户又可以在命令末尾再传一次同名字段透传（draccus 取最后一个）。于是命令行里会出现
    “同一个字段两次、值还不一样”。打印摘要时必须报告真正生效的那个，否则会误导
    （实测踩过：实际跑的是 expert-only，摘要却写“全参数微调”）。
    """
    tokens = (*OFFICIAL_POLICY_OVERRIDES, *a.extra) if a.official_recipe else tuple(a.extra)
    prefix = f"--policy.{name}="
    val = None
    for tok in tokens:
        if tok.startswith(prefix):
            val = tok[len(prefix) :]
    return val


def resolve_policy(policy: str) -> tuple[str, str]:
    """把 HF repo id 换成项目内已下好的本地目录（如果存在）。

    为什么要换：`--policy.path=lerobot/smolvla_base` 是 **repo id**，lerobot 会去
    HF 缓存里解析权重；而缓存里那份 `models--lerobot--smolvla_base` 很可能只缓存了
    config.json（`download_assets.py` 是把权重落在**项目内** `models/` 的），
    于是它会把已经下好的 900MB **重新下一遍**（实测 788kB/s，还要走 xet，能卡死）。
    直接指向本地目录则完全不碰网络。
    """
    if Path(policy).is_dir():
        return policy, "本地目录"

    local = PROJECT_ROOT / "models" / Path(policy).name
    if (local / "config.json").is_file() and (local / "model.safetensors").is_file():
        return str(local.relative_to(PROJECT_ROOT)), f"本地目录（由 {policy} 解析而来）"

    return policy, "HF repo id（走 HF 缓存，缺权重会联网下载）"


def pick_idle_gpu() -> tuple[str, str]:
    """按「空闲显存最多」自动选一张卡。

    为什么必须自动选：这台机器是**多人共用**的。实测 GPU 0 上跑着别的用户
    （`lxc`）一个已持续 5 天的 `unet-mipinn.py`，占 33.6/48 GB；
    写死 `-g 0` 会让训练在剩下那点空间里直接 OOM。
    注意：4 张卡是 **RTX 4090 48GB 版**（49140 MiB），不是 24GB。
    查不到 nvidia-smi 时退回 0（并说明原因）。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except Exception:  # noqa: BLE001
        return "0", "（nvidia-smi 不可用，退回 0 —— 若 OOM 请用 -g 显式指定空卡）"

    best_idx, best_free = None, -1
    for line in out.strip().splitlines():
        try:
            idx, free = (s.strip() for s in line.split(","))
            if int(free) > best_free:
                best_idx, best_free = idx, int(free)
        except ValueError:
            continue

    if best_idx is None:
        return "0", "（nvidia-smi 输出无法解析，退回 0）"
    return best_idx, f"（自动选空闲显存最多的卡：{best_free / 1024:.1f} GiB 可用）"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SmolVLA 在 LIBERO 数据集上的微调（封装 lerobot-train）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--policy", default=DEFAULT_POLICY, help="基线权重（HF repo id 或本地目录）")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="数据集 repo id（本地数据集也必填，只作标识）")
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="数据集在本地的目录（留空=交给 lerobot 自己下）")
    p.add_argument("--steps", type=int, default=20000, help="优化步数（不是 epoch）")
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="每卡 micro-batch。4090 48G 上实测 batch 32 已把 GPU 跑满（122 samples/s），"
        "再往上只涨显存不涨速度（batch 64 需 14GB 而吞吐相同）",
    )
    p.add_argument("--save-freq", type=int, default=2000, help="每多少步存一个 checkpoint（最后一步必存）")
    p.add_argument("--log-freq", type=int, default=50, help="每多少步打一次日志")
    p.add_argument("--lr", type=float, default=1e-4, help="--policy.optimizer_lr；设 0 表示不改（用权重里的预设）")
    p.add_argument("-n", "--num-workers", type=int, default=4, help="DataLoader worker 数")
    p.add_argument(
        "--video-backend",
        default="torchcodec",
        choices=("torchcodec", "pyav"),
        help="视频解码后端；torchcodec 更快，装不上/加载失败就换 pyav",
    )
    p.add_argument(
        "-g",
        "--gpu",
        default=None,
        help="用哪张卡；默认自动选空闲显存最多的卡（这台机器多人共用，别写死 0）",
    )
    p.add_argument("--device", default="cuda", help="--policy.device")
    p.add_argument("--output-dir", default=None, help="默认交给 lerobot：outputs/train/<日期>/<时间>_<job_name>")
    p.add_argument("--job-name", default="smolvla_libero", help="任务名（参与默认 output_dir 命名）")
    p.add_argument(
        "--rename-map",
        nargs="?",
        const=SMOLVLA_LIBERO_RENAME_MAP,
        default=None,
        help="走路线 B：把数据集相机键改名为 camera1/camera2。不带值=用内置 LIBERO 改名表",
    )
    p.add_argument(
        "--empty-cameras",
        type=int,
        default=None,
        help="补齐策略期望的空相机位数（路线 B 下通常要 1）；默认由脚本按权重声明自动推断",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        metavar="TRAIN_CONFIG",
        help="续训：指向 <ckpt>/checkpoints/last/train_config.json（与 --policy.path 互斥）",
    )
    p.add_argument(
        "--official-recipe",
        action="store_true",
        help="一键复刻官方 lerobot/smolvla_libero 的微调配方：25000 步 / batch 32 / "
        "全参数微调（解冻 VLM）/ 路线 B（rename_map）。显式传入的选项优先于本预设",
    )
    p.add_argument("--tmp-run", action="store_true", help="冒烟测试：4 步 / batch 2 / 只取 4 个 episode")
    p.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")

    args, extra = p.parse_known_args()
    args.extra = extra  # 未知参数原样透传给 lerobot-train
    return args


def infer_declared_cameras(policy: str) -> list[str] | None:
    """读出基线权重 config.json 里声明的相机名（去掉 observation.images. 前缀）。

    仅用于路线 B 自动推断 --policy.empty_cameras。读不到时返回 None（不猜测）。
    """
    try:
        if Path(policy).is_dir():
            cfg_path = Path(policy) / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            cfg_path = Path(hf_hub_download(repo_id=policy, filename="config.json"))
        cfg = json.loads(Path(cfg_path).read_text())
    except Exception:  # noqa: BLE001
        return None

    prefix = "observation.images."
    names = [
        k[len(prefix) :]
        for k in (cfg.get("input_features") or {})
        if isinstance(k, str) and k.startswith(prefix)
    ]
    real = [n for n in names if not n.startswith("empty_camera_")]
    return real or None


def build_train_args(a: argparse.Namespace) -> tuple[list[str], str | None]:
    """组装 lerobot-train 的命令行参数。返回 (参数表, 提示信息)。

    续训（--resume-from）走**完全不同的分支**，见下面注释。
    """
    # 预设优先级：显式 CLI 参数 > --official-recipe > --tmp-run > argparse 默认值。
    explicit = explicit_option_names() & set(ARG_DEFAULTS)

    def apply_preset(values: dict) -> None:
        for k, v in values.items():
            if k not in explicit:
                setattr(a, k, v)

    if a.official_recipe:
        apply_preset(OFFICIAL_RECIPE)
        if a.rename_map is None:
            a.rename_map = SMOLVLA_LIBERO_RENAME_MAP  # 官方走路线 B
        if a.empty_cameras is None:
            a.empty_cameras = OFFICIAL_EMPTY_CAMERAS

    if a.tmp_run:
        # 放在预设**之后**：于是 `--official-recipe --tmp-run` 会把步数压回 4 步，
        # 但保留官方配方的其他项（全参数微调 / 路线 B）—— 这正是"小规模验证官方配方"想要的效果。
        apply_preset(TMP_RUN_OVERRIDES)

    # ---- 续训：只给 --resume=true + --config_path，其余一律不传 ----
    # 两条硬性理由（都实测/查源码确认）：
    #   1) `TrainPipelineConfig._resolve_pretrained_from_cli()` 的优先级是
    #      「reward_model.path > policy.path > resume」。只要传了 `--policy.path`，
    #      就永远走不到 `_resolve_resume_checkpoint()`，续训根本不生效。
    #      上游官方续训测试（lerobot/Makefile:test-act-ete-train-resume）也只给这两行。
    #   2) 其余参数（--steps / --batch_size / --output_dir / --save_freq …）会**覆盖**
    #      checkpoint 里 train_config.json 记录的值。用脚本的默认值去覆盖很危险：
    #      比如原实验跑了 50000 步，默认 --steps=20000 会把它截断。
    if a.resume_from:
        args = ["--resume=true", f"--config_path={a.resume_from}"]
        note = (
            "续训模式：只传 --resume=true + --config_path，\n"
            "              其余选项（--policy/--steps/--batch-size/--output-dir…）都会被忽略，\n"
            "              因为本次配置整体来自 checkpoint 的 train_config.json。\n"
            "              确实需要改的，请追加在命令末尾原样透传。"
        )
        return args + list(a.extra), note

    out_dir = a.output_dir
    if out_dir is None:
        if a.tmp_run:
            # 冒烟目录固定，方便脚本化的"训练→评估"一条龙；已存在就加序号，避免误删上次结果
            base = Path("outputs/train") / "_smoke"
            n = 0
            while (PROJECT_ROOT / f"{base}{'' if n == 0 else f'_{n}'}").exists():
                n += 1
            out_dir = str(base) + ("" if n == 0 else f"_{n}")
        else:
            stamp = dt.datetime.now()
            out_dir = f"outputs/train/{stamp:%Y-%m-%d}/{stamp:%H-%M-%S}_{a.job_name}"

    args = [
        f"--policy.path={a.policy}",
        f"--policy.device={a.device}",
        "--policy.push_to_hub=false",  # 别误传上 Hub
        f"--dataset.repo_id={a.dataset}",
        f"--dataset.video_backend={a.video_backend}",
        f"--batch_size={a.batch_size}",
        f"--steps={a.steps}",
        f"--save_freq={a.save_freq}",
        f"--log_freq={a.log_freq}",
        f"--num_workers={a.num_workers}",
        "--env_eval_freq=0",  # 训练中的仿真评测先关掉（字段名不是 eval_freq）
        f"--job_name={a.job_name}",
        f"--output_dir={out_dir}",
    ]
    if a.dataset_root:
        args.append(f"--dataset.root={a.dataset_root}")
    if a.tmp_run:
        # 注意：不要加 shell 引号。我们走 subprocess(list) 不经 shell，
        # 引号会原样变成值的一部分 → draccus 解析 `"[0,1,2,3]"` 会失败。
        args.append("--dataset.episodes=[0,1,2,3]")
    if a.lr > 0:
        args.append(f"--policy.optimizer_lr={a.lr}")

    if a.rename_map:
        # 路线 B：把数据集相机键改名为权重期望的键
        args.append(f"--rename_map={a.rename_map}")
        n_declared = infer_declared_cameras(a.policy)
        n_empty = a.empty_cameras
        if n_empty is None:
            n_empty = max(0, (len(n_declared) - 2)) if n_declared else 0
        if n_empty:
            args.append(f"--policy.empty_cameras={n_empty}")
    else:
        # 路线 A：让特征从数据集推断（官方 PEFT 文档同款）
        args.append("--policy.input_features=null")
        args.append("--policy.output_features=null")

    if a.official_recipe:
        # 放在 a.extra **之前**：想覆盖其中某项，可在命令行末尾再透传一次同名字段
        # （draccus 对重复的 --policy.xxx 取最后一个）。
        args += list(OFFICIAL_POLICY_OVERRIDES)

    return args + list(a.extra), None


def find_lerobot_train() -> str:
    """优先用与当前解释器同目录的 lerobot-train，避免环境串了。"""
    sibling = Path(sys.executable).with_name("lerobot-train")
    if sibling.is_file():
        return str(sibling)
    return shutil.which("lerobot-train") or ""


def preflight(a: argparse.Namespace, argv: list[str]) -> None:
    """提前用友好文案拦住三类必挂的情况，别等 lerobot 报晦涩错误。"""
    if a.dataset_root and not a.resume_from:
        root = PROJECT_ROOT / a.dataset_root
        info = root / "meta" / "info.json"
        if not info.is_file():
            raise SystemExit(
                f"❌ 数据集不存在或不完整：{root}\n"
                f"   缺 {info.relative_to(PROJECT_ROOT)}。\n"
                f"   先下载：bash scripts/run.sh scripts/download_assets.py --only dataset\n"
                f"   （注意：HF 缓存里那份很可能缺 videos/，训练必须要有 mp4）"
            )
        videos = root / "videos"
        if not videos.is_dir() or not any(videos.rglob("*.mp4")):
            raise SystemExit(
                f"❌ 数据集缺视频：{root}/videos 下没有 mp4。\n"
                f"   训练是按 mp4 逐帧解码取图的，没有视频跑不起来。\n"
                f"   补下：bash scripts/run.sh scripts/download_assets.py --only dataset"
            )

    if a.resume_from:
        return  # 续训时 output_dir 交给 lerobot 自动创建

    for tok in argv:
        if not tok.startswith("--output_dir="):
            continue
        out = PROJECT_ROOT / tok.split("=", 1)[1]
        if out.is_dir():
            raise SystemExit(
                f"❌ 输出目录已存在：{out}\n"
                f"   lerobot 对已存在的 output_dir 会直接抛 FileExistsError（防止覆盖旧实验）。\n"
                f"   请改 --output-dir，或删掉该目录，或改用 --resume-from 续训。"
            )


def main() -> int:
    a = parse_args()

    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("MUJOCO_GL", "egl")  # 服务器无显示器；只在开了仿真评测时才用得上
    # xet 传输在本机的 HTTP 代理下会挂死（且不报错，只是永远不动）→ 强制走传统 HTTP。
    # 与 scripts/download_assets.py 保持一致。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    policy_arg, policy_src = resolve_policy(a.policy)
    a.policy = policy_arg

    if a.gpu is None:
        a.gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        if a.gpu:
            gpu_note = "（来自环境变量 CUDA_VISIBLE_DEVICES）"
        else:
            a.gpu, gpu_note = pick_idle_gpu()
    else:
        gpu_note = "（来自 -g，未做空闲检查）"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu  # 必须在起 lerobot-train 之前设好

    # 这里的lerobot-train是 lerobot 仓库里自带的可执行文件（不是 pip 安装的 lerobot 库里的 train.py），它会默认调用"lerobot.scripts.lerobot_train:main"
    train_bin = find_lerobot_train()
    if not train_bin:
        print(
            f"❌ 当前环境里没有 lerobot-train。\n"
            f"   正在使用的解释器：{sys.executable}\n"
            f"   lerobot 只装在 ai312 环境里，请改用：\n"
            f"     bash scripts/run.sh scripts/train.py          # 推荐\n"
            f"     $HOME/miniconda3/envs/ai312/bin/python scripts/train.py",
            file=sys.stderr,
        )
        return 1

    argv, resume_note = build_train_args(a)
    preflight(a, argv)
    cmd = [train_bin, *argv]

    if a.resume_from:
        route = f"续训 ← {a.resume_from}"
    elif a.rename_map:
        route = "B（rename_map → camera1/camera2）"
    else:
        route = "A（input_features=null，特征从数据集推断）"
    out_dir = next((t.split("=", 1)[1] for t in argv if t.startswith("--output_dir=")), None)

    if a.official_recipe:
        # 必须报告**真正生效**的冻结策略：见 effective_policy_flag 的注释。
        plan = "全参数微调"
        if effective_policy_flag(a, "train_expert_only") == "true":
            plan = "只训 action expert（整个 VLM 被冻住）"
        elif effective_policy_flag(a, "freeze_vision_encoder") == "true":
            plan = "冻结视觉编码器，其余全参数"
        print(f" 配方       : 官方 smolvla_libero（25000 步 / 路线 B）+ {plan}")
    print("=" * 62)
    print(f" 模式       : {'冒烟测试' if a.tmp_run else ('续训' if a.resume_from else '正式训练')}")
    print(f" 基线权重   : {a.policy}  [{policy_src}]")
    print(f" 数据集     : {a.dataset}" + (f"  (root={a.dataset_root})" if a.dataset_root else ""))
    print(f" 相机键路线 : {route}")
    if a.resume_from:
        print(" 步数/批大小: 由 checkpoint 的 train_config.json 决定")
    else:
        print(f" 步数/批大小: {a.steps} 步 / batch {a.batch_size} / 每 {a.save_freq} 步存档")
    print(f" 显卡       : {a.gpu} {gpu_note}   (MUJOCO_GL={os.environ['MUJOCO_GL']}, backend={a.video_backend})")
    print(f" Python     : {sys.executable}")
    print(f" 输出       : {out_dir or '由 lerobot 自动创建 outputs/train/<日期>/<时间>_resume'}")
    if a.extra:
        print(f" 透传参数   : {' '.join(a.extra)}")
    if resume_note:
        print(f" ⚠️  {resume_note}")
    print("=" * 62)
    print(" ".join(cmd))
    print()

    if a.dry_run:
        print("（--dry-run：只打印，不执行）")
        return 0

    hint_dir = out_dir or "<lerobot 自动创建的目录>"
    print(f"⏳ 开始训练…（Ctrl-C 可提前中断；中断后用 --resume-from {hint_dir}/checkpoints/last/train_config.json 续训）\n")
    rc = subprocess.call(cmd)

    if rc == 0:
        print(f"\n✅ 训练结束。\n"
              f"   权重 : {hint_dir}/checkpoints/last/pretrained_model/\n"
              f"   评估 : bash scripts/run.sh scripts/eval.py "
              f"--policy {hint_dir}/checkpoints/last/pretrained_model -n 10")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
