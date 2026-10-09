# SmolVLA + LIBERO

基于 [LeRobot](https://github.com/huggingface/lerobot) 在 **LIBERO** 基准上微调与评估 **SmolVLA**，
并复现全部消融实验。

## 结果速览

单卡 RTX 4090，LIBERO 每任务 1 回合（10 回合/套件）：

| 实验 | 训练配置 | 可训 / 总参数 | libero_spatial | libero_goal |
| --- | --- | --- | --- | --- |
| `lerobot/smolvla_base` 零样本 | — | — | 0/10 | — |
| 冒烟产物 | 4 步 | 100M / 450M | 0/100 | — |
| pilot | 30 步 | 393M / 450M | 0/10 | — |
| **全参数微调（主线）** | 25000 步 | 393M / 450M | **7/10（70%）** | — |
| **只训 action expert（消融）** | 25000 步 | 100M / 450M | 6/10（60%） | 7/10（70%） |
| `lerobot/smolvla_libero`（官方对照） | — | 393M / 450M | 8/10（80%） | — |

结论：

1. 零样本不可用，必须微调；25000 步是本仓库验证过的有效训练量。
2. 在数据、步数完全相同的前提下，**解冻整个 VLM 做全参数微调** 比只训 action expert
   高 10pp，代价是 2.2 倍训练时间（4h vs 1.5h）。

> 表中结果为单次评测，置信区间较宽，仅用于快速对照；正式结论请用 `-n 10` 复评。

## 环境要求

| 项 | 要求 |
| --- | --- |
| 系统 | Linux（无头 EGL 渲染） |
| GPU | NVIDIA，驱动 ≥ 525；显存 ≥ 8GB（expert-only）/ ≥ 28GB（全参数） |
| Python | 3.12（conda 环境名 `ai312`） |
| 磁盘 | ≥ 25GB |

## 安装

```bash
git clone https://github.com/jokerAuA/robotics.git
cd robotics

# 1) 创建 Python 环境
conda create -n ai312 python=3.12 -y
conda activate ai312

# 2) PyTorch（必须 cu126，原因见 requirements-ai312.txt 顶部）
pip install "torch==2.11.0+cu126" "torchvision==0.26.0+cu126" \
  --index-url https://download.pytorch.org/whl/cu126

# 3) 其余依赖
pip install -r requirements-ai312.txt

# 4) LeRobot 源码（版本 0.6.2，PyPI 未发布，必须从源码装）
git clone https://github.com/huggingface/lerobot.git lerobot
git -C lerobot checkout e595b7902714ba51f91e47523f66f89c5181b649
pip install -e lerobot

# 5) 初始化 LIBERO 配置（缺失会导致 import 时报 EOFError）
echo "N" | python -c "import libero.libero"
```

## 数据与权重

```bash
python scripts/download_assets.py          # LIBERO 数据集 ~1.9GB + SmolVLA 权重 ~0.9GB
python scripts/download_assets.py --check  # 校验完整性
```

数据集与权重分别落在 `datasets/`、`models/`（已 gitignore，需自行下载）。

> 以下命令均在项目根目录、`ai312` 环境下执行。
> 不使用 `conda activate` 时，可用 `bash scripts/run.sh <脚本>` 代替 `python <脚本>`
> （自动使用 `~/miniconda3/envs/ai312/bin/python`，可用环境变量 `LEROBOT_PYTHON` 覆盖）。

## 复现消融实验

按顺序执行以下步骤，即可复现「结果速览」中的全部数字。

### 步骤 1 · 环境自检与链路冒烟

```bash
python scripts/check_env.py    # 真建 LIBERO 环境、真跑一次 step()

# 4 步 / batch 2 / 8 个样本，验证「训练 → checkpoint → 评估」链路（几秒完成）
python scripts/train.py --official-recipe --tmp-run --job-name _smoke
```

### 步骤 2 · 基线：零样本评估（预期 0%）

```bash
python scripts/eval.py --policy models/smolvla_base \
  --suite libero_spatial -n 1 --output-dir outputs/eval_base
```

预期 `outputs/eval_base/eval_info.json` 中 `overall.pc_success = 0.0`。

### 步骤 3 · 主线训练：全参数微调 25000 步（约 4 小时）

```bash
nohup python scripts/train.py --official-recipe \
  --job-name fullparam_25k --output-dir outputs/train/fullparam_25k \
  > outputs/train_fullparam_25k.log 2>&1 &
```

> 长训练必须脱离终端（nohup / tmux），否则 SSH 断开或关闭编辑器会中断训练。

`--official-recipe` 展开为官方 `lerobot/smolvla_libero` 的配方：
steps 25000 / batch 32 / save_freq 5000 / log_freq 200 / num_workers 4，
相机键改名（`rename_map`），全参数微调
（`--policy.freeze_vision_encoder=false --policy.train_expert_only=false --policy.scheduler_decay_steps=25000`）。

日志对照值：

| 指标 | 期望值 |
| --- | --- |
| `num_learnable_params` | 392,904,096（总 450,046,176） |
| `mem_gb` | ≈ 28.3 |
| `smp/s` | ≈ 56 |
| 末步 | `loss ≈ 0.31`，`lr = 2.5e-06` |

### 步骤 4 · 评估主线模型（预期 70%）

```bash
python scripts/eval.py \
  --policy outputs/train/fullparam_25k/checkpoints/last/pretrained_model \
  --suite libero_spatial -n 1 --output-dir outputs/eval_fullparam_25k
```

### 步骤 5 · 消融训练：只训 action expert 25000 步（约 1.5 小时）

与步骤 3 相同，仅在**命令末尾**追加两个参数，其余配置不变：

```bash
nohup python scripts/train.py --official-recipe \
  --job-name expert_only_25k --output-dir outputs/train/expert_only_25k \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true \
  > outputs/train_expert_only_25k.log 2>&1 &
```

> 两个参数必须放末尾：`--official-recipe` 会先注入 `=false`，同名字段后者覆盖前者。
> 这样两次训练的唯一变量就是冻结策略。

日志对照值：`num_learnable_params = 99,880,992`、`mem_gb ≈ 7.7`、`smp/s ≈ 120`。

### 步骤 6 · 评估消融模型（预期 60%）

```bash
python scripts/eval.py \
  --policy outputs/train/expert_only_25k/checkpoints/last/pretrained_model \
  --suite libero_spatial -n 1 --output-dir outputs/expert_only_25k
```

### 步骤 7 · 换套件复评同一权重（预期 70%）

```bash
python scripts/download_assets.py --only assets   # libero_goal 需要额外 3D 资产，缺失时补

python scripts/eval.py \
  --policy outputs/train/expert_only_25k/checkpoints/last/pretrained_model \
  --suite libero_goal -n 1 --output-dir outputs/expert_only_25k_goal
```

### 步骤 8 · 汇总结果

```bash
python - <<'PY'
import glob, json
for f in sorted(glob.glob("outputs/*/eval_info.json")):
    o = json.load(open(f))["overall"]
    print(f"{f:<44s} {o['n_success']:>3d}/{o['n_episodes']:<3d} "
          f"{o['pc_success']:>5.1f}%  {o['eval_ep_s']:>6.2f} s/ep")
PY
```

### 自定义消融

在 `--official-recipe` 基础上，把要改的项追加在命令末尾即可：

| 改什么 | 参数 |
| --- | --- |
| 训练步数 | `--steps 10000` |
| 冻结视觉编码器 | `--policy.freeze_vision_encoder=true` |
| 不训 state 投影 | `--policy.train_state_proj=false` |
| 学习率 / batch | `--lr 5e-5` / `--batch-size 16` |
| 数据子集 | `--dataset.episodes=[0,1,2,3]`（不要加引号） |
| 其他 `lerobot-train` 参数 | `-- <参数>`，如 `-- --accelerator.mixed_precision=bf16` |

## 评估

```bash
python scripts/eval.py                            # 默认权重 lerobot/smolvla_libero
python scripts/eval.py --suite libero_object -n 10
python scripts/eval.py --policy <权重目录或 HF id> -n 10
python scripts/eval.py --task-ids "[0,1]"         # 只跑指定任务
```

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--policy` | 权重（HF repo id 或本地目录） | `lerobot/smolvla_libero` |
| `--suite` | `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` / `libero_90` | `libero_spatial` |
| `-n / --episodes` | 每任务回合数 | 1 |
| `--task-ids` | 任务编号，如 `"[0,1]"` | 该套件全部 |
| `--output-dir` | 输出目录 | `outputs/eval_<套件>/<日期>/<时间>_n<N>` |
| `-g / --gpu` | 指定显卡 | 自动选空闲显存最多的卡 |
| `--view` | 实时弹窗显示推理过程（需图形界面） | 关 |

结果写入 `<输出目录>/eval_info.json`（含逐任务结果与 `video_paths`）
和 `<输出目录>/videos/`（每任务一段 rollout 视频）。

## 训练

```bash
python scripts/train.py --official-recipe                        # 复刻官方配方（步骤 3）
python scripts/train.py --steps 50000 --batch-size 16
python scripts/train.py --tmp-run                                # 冒烟测试
python scripts/train.py --dry-run                                # 只打印将执行的命令
python scripts/train.py --resume-from <ckpt>/checkpoints/last/train_config.json   # 续训
```

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--steps` | 优化步数 | 20000 |
| `--batch-size` | 每卡 batch（4090 48G 实测 32 为吞吐甜点） | 32 |
| `--save-freq` / `--log-freq` | 存档 / 日志间隔 | 2000 / 50 |
| `--lr` | 学习率（`--policy.optimizer_lr`） | 1e-4 |
| `-n / --num-workers` | DataLoader worker 数 | 4 |
| `-g / --gpu` | 指定显卡 | 自动选空闲显存最多的卡 |
| `--official-recipe` | 官方配方预设（25000 步 / 全参数 / rename_map） | 关 |
| `--rename-map` | 相机键改名路线（默认走特征推断） | 关 |

产物：`outputs/train/<job_name>/checkpoints/<步数>/pretrained_model/`
（`last` 指向最新 checkpoint；`train_config.json` 记录该次训练全部超参，续训依赖它）。

## 目录结构

```
robotics/
├── README.md
├── TRAIN_FLOW.md           # 训练侧原理（接口适配、冻结开关、官方配方解析）
├── DATA_FLOW.md            # 评估链路讲解
├── requirements-ai312.txt  # 依赖清单与安装顺序
├── scripts/
│   ├── run.sh              # 薄壳：切目录 + 设 MUJOCO_GL + 用 ai312 解释器执行
│   ├── check_env.py        # 环境自检
│   ├── download_assets.py  # 下载数据集 / 权重 / LIBERO 3D 资产（幂等，支持 --check）
│   ├── train.py            # 训练入口（封装 lerobot-train）
│   ├── eval.py             # 评估入口（封装 lerobot-eval）
│   └── live_view.py        # eval.py --view 的显示窗口
├── lerobot/                # LeRobot 源码（安装步骤 4 自行 clone，已 gitignore）
├── datasets/               # 数据集（download_assets.py 下载，已 gitignore）
├── models/                 # 基线权重（download_assets.py 下载，已 gitignore）
└── outputs/                # 评估结果与训练 checkpoint（已 gitignore）
```

## 常见问题

- **找不到 `lerobot-eval` / `lerobot-train`**：解释器不对，先 `conda activate ai312`
  （`which python` 确认），或漏了 `pip install -e lerobot`。
- **`import libero` 抛 `EOFError`**：缺 `~/.libero/config.yaml`，执行「安装」第 5 步。
- **训练抛 `FileExistsError`**：`--output-dir` 已存在，换目录或改用 `--resume-from`。
- **续训没生效**：必须用 `--resume-from <train_config.json>`，且不能同时传 `--policy`。
- **报 `All image features are missing from the batch`**：权重声明的相机键与数据不一致；
  训练时加 `--rename-map`，评估侧由 `scripts/eval.py` 自动处理。
- **HF 下载长时间停在 0 字节**：代理网络下 xet 传输可能挂起，脚本已默认设置
  `HF_HUB_DISABLE_XET=1`；下载日志中的 timeout / resume 提示属正常续传。
- **评估结果与视频对不上**：以 `eval_info.json` 的 `video_paths` 为准；
  固定 `--output-dir` 时脚本会先清理上一次的产物再跑。

## 参考

- LeRobot 文档：`lerobot/docs/source/libero.mdx`、`lerobot/docs/source/smolvla.mdx`
- 官方配方来源：`https://huggingface.co/lerobot/smolvla_libero/raw/main/train_config.json`
