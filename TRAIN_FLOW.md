# SmolVLA + LIBERO 后训练（微调）全记录

> 配套阅读：`DATA_FLOW.md` 讲的是**评估侧**（机器人怎么"考试"）；本文讲的是**训练侧**（怎么把 base 训成能考 70 分的考生）。
> 所有数字都取自实际落盘的 `train_config.json`、训练日志、`eval_info.json`，不是估算。

---

## 0. 结论速查

| 权重 | 训练量 | 可训参数 | 成功率 | `eval_ep_s` |
|---|---|---|---|---|
| `lerobot/smolvla_base`（0 步） | — | — | 0/10 | 14.32 |
| 早期 4 步产物（只训 expert） | 8 样本 | 100M | 0/100 | 26.48 |
| 30 步（全参数） | 960 样本 | 393M | 0/10 | 15.31 |
| **全参数 25000 步** | 80 万样本 | 393M | **7/10（70%）** | **10.04** |
| expert-only 25000 步 | 80 万样本 | 100M | 6/10（60%） | — |
| 官方 `lerobot/smolvla_libero` | — | — | 8/10（80%） | 9.91 |

**一句话**：拿 LIBERO 全量 1693 条轨迹，把 `smolvla_base` 的输入接口对齐、并用数据集自己的归一化统计量换掉它自带的 SO-100 统计量，然后**解冻整个 VLM 做全参数微调**：25000 步 / batch 32 / lr 1e-4 余弦退火，单卡 4090 跑 4 小时 → 70%。

---

## 1. 问题是怎么定位的

最初的产物成功率 0%，逐层排除后才找到真因：

| 排查 | 结论 |
|---|---|
| 评估链路有没有问题？ | 没有。同一条评估命令跑官方权重就是 80% |
| 模型没微调行不行？ | 不行。`smolvla_base` 零样本 0/10（实测） |
| 训练量够不够？ | 不够。用 `--tmp-run` 只训了 **4 步 / 8 个样本**（那是冒烟测试，不是训练） |
| **训练范围对不对？** | **不对 —— 这是真因**：`smolvla_base` 的 config 里 `freeze_vision_encoder=true` + `train_expert_only=true`，会把**整个 VLM 冻住**，只剩下 100M 可训（`num_learnable_params=99880992`）。官方微调那份权重两项都是 `false` |

另外两个直接证据：

- `smolvla_base` 的归一化层里**只有 `so100-blue/so100-red/so100.buffer.action.*`（6 维）**，**连 `observation.state` 的统计量都没有** → 它连动作尺度都是 SO-100 的，零样本必然废。
- 官方文件明写：`smolvla_base/README.md`「**Intended use: Base model to fine tune** on your specific use case」；`lerobot/docs/source/smolvla.mdx:31`「SmolVLA is a base model, so fine-tuning on your own data is required」。

---

## 2. 官方配方（权威来源）

作者把训练配置连权重一起传上了 Hub，所以配方是公开的：

```
https://huggingface.co/lerobot/smolvla_libero/raw/main/train_config.json
```

| 项 | 官方值 |
|---|---|
| 初始权重 | `policy.pretrained_path = lerobot/smolvla_base`（不是从零） |
| 数据 | `lerobot/libero` **全量**（`episodes = null`） |
| 相机键 | `rename_map = {image→camera1, image2→camera2}`（= 路线 B），`empty_cameras = 0` |
| **冻结策略** | `freeze_vision_encoder = false`、`train_expert_only = false`、`train_state_proj = true` |
| 步数 / batch | **25000** / 32，`num_workers = 4`，`seed = 1000` |
| 调度 | cosine + warmup 1000，`scheduler_decay_steps = 25000`，`decay_lr = 2.5e-6` |
| 优化器 | adamw lr 1e-4，betas (0.9, 0.95)，eps 1e-8，wd 1e-10，grad_clip 10 |
| 保存 / 日志 | `save_freq = 5000`，`log_freq = 200`，`env_eval_freq = 0`（训练中不评测） |
| 图像增强 | 定义了 ColorJitter / RandomAffine，但 `image_transforms.enable = false`（**未启用**） |

实测 diff 过 base 与官方权重的 `config.json`，**只有 4 处不同**（上面加粗那几项），其余（lr / betas / eps / wd / grad_clip / warmup / decay_lr / resize / chunk）本来就一致 ⇒ 只需覆盖这几项。

### 已固化为脚本预设

```bash
bash scripts/run.sh scripts/train.py --official-recipe
```

`scripts/train.py` 里的三个常量：
- `OFFICIAL_RECIPE` = steps 25000 / batch 32 / save_freq 5000 / log_freq 200 / num_workers 4
- `OFFICIAL_EMPTY_CAMERAS` = 0
- `OFFICIAL_POLICY_OVERRIDES` = `--policy.freeze_vision_encoder=false`、`--policy.train_expert_only=false`、`--policy.scheduler_decay_steps=25000`

> 为什么要自己显式传这几项：它们会被 `--policy.path=models/smolvla_base` **带进来**，而 lerobot 不会用数据集去纠正
> （`lerobot/src/lerobot/configs/train.py:217` 的 `from_pretrained(path, cli_overrides=...)` 只是把 CLI 覆写叠在**加载进来的权重配置**之上）。

---

## 3. 用了哪些数据

| 项 | 值 |
|---|---|
| 数据集 | `lerobot/libero`（v3.0，LIBERO 仿真，Franka Panda） |
| 规模 | **1693 episodes / 273465 frames / 40 个任务 / fps 10** |
| 用了多少 | **全量**（`dataset.episodes = null`）→ 25000 × 32 = 80 万样本，实际 **2.93 个 epoch** |
| 输入 | `observation.images.image` + `image2`（256×256×3，**存的是 mp4**）\| `observation.state`（**8 维**） |
| 输出 | `action`（**7 维** = 6 DoF delta EE + gripper） |
| 语言 | `meta/tasks.parquet` 的任务描述 |
| 统计量 | `meta/stats.json` |
| 视频解码 | `torchcodec` |

---

## 4. 接口适配：一共动了 5 处

`smolvla_base` 的"出厂签名"和 LIBERO 的"数据签名"对不上：

| # | 差在哪 | 怎么适配 | 生效位置 |
|---|---|---|---|
| 1 | **相机键名**：base 声明 `camera1/camera2/camera3`，数据是 `image/image2` | `rename_map` 改名对齐（路线 B）。第 3 路没有数据 → `empty_cameras = 0`，**不补零图** | `processor/rename_processor.py:27` |
| 2 | **图像尺度**：期望 512×512、像素 [-1,1] | `resize_with_pad` → 512×512，[0,1] → [-1,1]（SigLIP 要求） | `modeling_smolvla.py:336 prepare_images()` |
| 3 | **state 维度**：数据 8 维 vs base 6 维 | `pad_vector(state, 32)` 补零 → `state_proj = Linear(32, hidden)` | `modeling_smolvla.py:411-414`、`:510` |
| 4 | **action 维度**：数据 7 维 vs base 6 维 | `pad_vector(action, 32)` → `action_in/out_proj = Linear(32,·)`，输出再切回 7 | `modeling_smolvla.py:417-419`、`:513-514` |
| 5 | **数值尺度（最关键）**：base 自带 SO-100 的统计量 | 训练时**强制用数据集 stats 覆写**归一化层（`MEAN_STD`），跳过 checkpoint 旧 stats | `lerobot_train.py:532 / 581 / 604` |

> 维度差（6/7/8）靠 padding 吸收；第 5 条才是"零样本必废"的直接原因。

### 附带两个"陈旧元数据"陷阱（都不用管）
- 官方权重和我们产物的 `config.json` 里 `observation.state` 都写 `[6]`，但**真正生效的是 8 维的归一化统计量** —— 说明书印错一行字，零件是对的。
- 原始图像 360×360（仿真）vs 数据集 256×256（mp4）vs 模型 512×512：**名字必须对上，尺寸不用**。

---

## 5. 全参数微调到底做了什么

### 5.1 三个开关的真实含义（`smolvlm_with_expert.py:150 set_requires_grad()`）

| 开关 | base 默认 | 官方 / 我们 | 效果 |
|---|---|---|---|
| `freeze_vision_encoder` | `True` | **`False`** | 视觉编码器解冻 |
| `train_expert_only` | `True` | **`False`** | 整个 VLM 解冻（除下面 3 样） |
| `train_state_proj` | `True` | `True` | state 投影层可训 |

`train_expert_only=False` 时**只冻 3 样**：`vlm.lm_head` + `text_model.norm.weight` + **`text_model.layers.15.`（仅最后一层）**。
（`num_expert_layers=0` 在构造时被解析成 `= num_vlm_layers = 16`，所以那个 `%.` 条件不成立，只冻一层。）

### 5.2 参数账（从 `model.safetensors` 头部实测）

| 模块 | 参数量 | 冻结 | 可训 |
|---|---:|---:|---:|
| ① `vision_model`（SigLIP，12 层） | 86,433,024 | 0 | **86,433,024** |
| ② `connector`（视觉→语言空间） | 11,796,480 | 0 | **11,796,480** |
| ③ `text_model` 第 0~14 层 | 147,484,800 | 0 | **147,484,800** |
| ③ `text_model` 第 **15** 层 | 9,832,320 | 9,832,320 ❄ | 0 |
| ④ `text_model.embed_tokens` | 47,308,800 | 0 | **47,308,800** |
| ⑤ `text_model.norm` | 960 | 960 ❄ | 0 |
| ⑥ `vlm.lm_head` | 47,308,800 | 47,308,800 ❄ | 0 |
| ⑦ `lm_expert`（action expert，16 层） | 98,245,840 | 0 | **98,245,840** |
| ⑧ policy 投影层 | 1,635,152 | 0 | **1,635,152** |
| **合计** | **450,046,176** | **57,142,080** | **392,904,096** |

最后一行与训练日志的 `num_learnable_params=392,904,096` **完全一致** ⇒ 可训比例 **87%**。

**为什么偏偏冻那 57.1M**：`lm_head` + 最终 norm + 最后一层是"输出文本"那条路的末端，而 SmolVLA 不出文本（动作由 action expert 的 `action_out_proj` 产出）⇒ 对 loss 没贡献，冻掉等于白省梯度与优化器状态（源码注释：「To avoid unused params issue with distributed training」）。

### 5.3 "冻结"在 PyTorch 里的确切含义

1. `requires_grad=False` → 反向图不经过它，**不保存中间激活** ⇒ 省显存省算力
2. 优化器只收到可训参数 ⇒ 没有 Adam 一阶/二阶动量（8 字节/参数）
3. 权重**一个 bit 都不变**（不是变小，是不动）；解冻则每 step 都被改写
4. 冻结模块通常还被 `.eval()` ⇒ dropout / 归一化行为被钉死
5. **梯度仍能穿过冻结层往回传** ⇒ 冻第 15 层不影响 0~14 层学习。冻的不是"路"，是"路上的几颗螺丝"

### 5.4 解冻之后谁被改造了

| 模块 | 本质变化 |
|---|---|
| 视觉编码器 86.4M | SigLIP 原本学的是**自然图像**；LIBERO 是 **MuJoCo 渲染**（纯色桌面、低纹理、固定光照）→ 可重学仿真域特征。**这是最大的一处改造** |
| connector 11.8M | 视觉 token 投影进语言空间，跨模态对齐被重塑 |
| text_model 0~14 层 + embed_tokens 194.8M | LIBERO 的模板化长指令被重新语境化 |
| action expert 98.2M | 通过 **逐层** cross-attention 读 VLM 每一层的 K/V（`smolvlm_with_expert.py:forward_cross_attn_layer`：expert layer *i* 取 VLM layer *i* 的 KV）⇒ VLM 一变它的输入分布就变，**必须一起训** |
| policy 投影 1.6M | state 8→32 维后的 `state_proj`、7↔32 维的 `action_in/out_proj` |

### 5.5 代价（实测，batch 32）

| | expert-only | 全参数 | 比值 |
|---|---|---|---|
| 可训参数 | 99,880,992 | 392,904,096 | 3.9× |
| 显存 | 7.67 GB | 28.30 GB | 3.7× |
| 吞吐 | 117 smp/s | 56 smp/s | 2.1× |
| 25000 步墙钟 | 2h29m | 4h03m | 1.6× |

多出的 ~20 GB：**① 激活**（整条 VLM/视觉编码器的中间结果都要留到 backward 结束）是大头，**② 梯度 + Adam 动量**约 4.7 GB。

---

## 6. 训练执行

```bash
# 方式 1：nohup（脱离终端，推荐）
cd /mnt/data/home/wpj/projects/robotics
PYTHONUNBUFFERED=1 nohup bash scripts/run.sh scripts/train.py --official-recipe > outputs/train_official.log 2>&1 &
echo "PID=$!"

# 方式 2：tmux（可随时 attach 看实时输出）
bash scripts/run.sh scripts/train.py --official-recipe 2>&1 | tee outputs/train_official.log
```

两个"必须"：**`PYTHONUNBUFFERED=1`**（否则重定向到文件后 Python 块缓冲，`tail -f` 看着像卡死）、**重定向路径由当前 shell 解析**（所以先 `cd` 到项目根）。

实测：25000 步 / **4h03m** / 1.71 step/s / `mem_gb` 稳定 28.29 / loss **3.006 → 0.309** / `lr` 精确退火到 **2.5e-06** / `grdn` 1.37。

日志（每 `log_freq=200` 步一条，`loss` 就在这里）：

```bash
grep -o "step:[0-9KM]*\|loss:[0-9.]*\|mem_gb:[0-9.]*" outputs/train_official.log | paste - - - | sed -n '1~10p'
```

---

## 7. 评估

```bash
bash scripts/run.sh scripts/eval.py --policy <job>/checkpoints/last/pretrained_model -n 10
```

`eval.py` 会**读权重 config.json** 自动补评估侧参数（因为走路线 B）：

```
--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}
--policy.empty_cameras=1
```

结果：**7/10 = 70%**，`eval_ep_s = 10.04`（官方 9.91，失败的那些是 14~26 s/回合 —— 瞎动到跑满 280 步）。

---

## 8. 对照实验：expert-only vs 全参数

唯一变量 = 冻结策略，其余（数据 / 步数 / batch / lr / 调度 / seed）完全一致。

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expert-only | ✗ | ✓ | ✗ | ✓ | ✗ | **✗** | ✓ | ✓ | ✓ | ✓ | 6/10 |
| 全参数 | ✗ | ✓ | ✗ | ✓ | ✗ | **✓** | ✓ | ✓ | ✓ | ✓ | 7/10 |

**10 道题里 9 道结果完全一致，唯一差别是 task 5 一个回合。** 两边失败的是同一批难题（0、2、4）——那是题目难，不是能力差。

统计上：60% CI95 = [31.3%, 83.2%]，70% CI95 = [39.7%, 89.2%]，大面积重叠，Fisher p≈1.0。
**n=10 只能排除 30 个点以上的差距**，所以"6 vs 7"目前无法解释为"解冻有用"，也无法解释为"没用"。

（想分辨 10 个点需要约 350 回合/边；`-n 20` 的 200 回合能分辨约 13 个点。另外 LIBERO 初始摆放固定，但流匹配采样带随机性 ⇒ 可以用**同一模型重跑一次**来校准噪声基底，3 分钟就能判定 6 vs 7 有没有意义。）

**实践结论**：把 expert-only 当默认迭代配置（便宜 1.6~2.1×，质量差异测不出）；只在最终对标官方时用全参数。

---

## 9. 踩过的坑（都已在脚本里修掉）

| # | 坑 | 表现 | 处理 |
|---|---|---|---|
| 1 | `--tmp-run` 被当成训练 | 4 步 / 8 样本 → 0% | 冒烟测试 ≠ 训练，正式跑用 `--official-recipe` |
| 2 | base 的冻结默认值 | 只训 100M，误以为在"全参数微调" | `--official-recipe` 显式覆盖 |
| 3 | base 的归一化统计量是 SO-100 的 | 零样本必废、动作尺度全错 | 训练时强制用数据集 stats |
| 4 | 相机键不对齐 | `ValueError: All image features are missing from the batch.` | 路线 A（推断）或路线 B（`rename_map`） |
| 5 | **调度器自动缩放** | 小步数跑时 lr 迅速降到 `decay_lr`，像 bug | `optim/schedulers.py:136-155`：`steps < decay_steps` 时按比例缩放 warmup/decay。正式跑 25000 == 25000 ⇒ 不缩放，与官方一致 |
| 6 | **评估输出目录会串** | `-n 1` 跑完目录里仍有 10 个视频，9 个全错 | 旧的固定目录 + lerobot-eval **从不清理旧文件** ⇒ 把两次不同权重的产物混在一个文件夹。已改：`eval.py` 默认时间戳目录；显式 `--output-dir` 时先清 `eval_info.json`/`videos`/`recordings`。**判断产物以 `eval_info.json` 的 `video_paths` 为准** |
| 7 | `$!` 不是训练进程 | `kill $!` 后训练还在跑 | `run.sh` 用 `exec` 起 python，而 `train.py` 内部是 `subprocess.call([lerobot-train,...])` ⇒ 训练是**另一个 PID 的子进程**。用 `pgrep -af lerobot-train` + `pkill -f lerobot-train` |
| 8 | 前台跑长训练 | 关编辑器/断线/Ctrl-C/在同终端敲别的命令 → 训练被打断（曾连日志都没建出来） | 必须 `nohup` 或 `tmux` |
| 9 | 预设优先级不能靠"值==默认值"判断 | `--tmp-run --batch-size 32` 里的 32 恰好等于默认值 → 被静默压成 2 | 改成扫 `sys.argv`（`explicit_option_names()`） |
| 10 | 没有梯度检查点开关 | 模型卡里的 `--policy.gradient_checkpointing` 传了会报错 | `SmolVLAConfig` 无此字段；`--accelerator.activation_checkpointing` 是未接线占位符（`configs/train.py:359` 直接 `ValueError`）⇒ 只能降 `--batch-size` |
| 11 | 摘要谎报冻结策略 | 跑 expert-only 时摘要写"全参数微调" | 新增 `effective_policy_flag()`，报告**最终生效**值 |
| 12 | 训练日志不在 run 目录里 | 找日志找错地方 | 日志路径取决于 `> ...` 重定向，由调用方决定 |

---

## 10. 从零复现（3 条命令）

```bash
cd /mnt/data/home/wpj/projects/robotics

# ① 下资产（数据集 1.9GB + 基线权重 907MB，落到项目内）
bash scripts/run.sh scripts/download_assets.py

# ② 训练（≈4 小时，单卡 4090 48G）
PYTHONUNBUFFERED=1 nohup bash scripts/run.sh scripts/train.py --official-recipe > outputs/train_official.log 2>&1 &

# ③ 评估（10 任务 × 10 回合 ≈ 17 分钟）
bash scripts/run.sh scripts/eval.py \
  --policy outputs/train/<日期>/<时间>_smolvla_libero/checkpoints/last/pretrained_model -n 10
```

---

## 11. 产物与文件地图

```
outputs/train/<日期>/<时间>_smolvla_libero/
└── checkpoints/
    ├── 005000/ 010000/ 015000/ 020000/ 025000/     ← 每 save_freq 步一个
    │   ├── pretrained_model/                       ★ 评估/发布用这一层
    │   │   ├── config.json                            策略签名
    │   │   ├── model.safetensors                      权重本体（约 900MB）
    │   │   ├── train_config.json                      ★ 完整配置快照 → 续训靠它
    │   │   ├── policy_pre/postprocessor.json          管线定义
    │   │   ├── policy_*_normalizer/unnormalizer_processor.safetensors   ★ 归一化统计量
    │   │   └── tokenizer/
    │   └── training_state/                         ← 只有续训才需要
    └── last → 025000
```

| 本仓库改过的文件 | 改动 |
|---|---|
| `scripts/train.py` | 新增 `--official-recipe` 预设；`explicit_option_names()` 修预设优先级；`effective_policy_flag()` 修摘要谎报；「用法」段扩写成 7 个场景（含长训练的 nohup/tmux、停止方法、显存手段、续训） |
| `scripts/eval.py` | 默认输出目录改为带时间戳；显式 `--output-dir` 时先清旧产物（`clear_stale_outputs()`） |

---

## 12. 接下来值得做的

1. **校准噪声基底**（3 分钟）：拿同一份权重、同一设置重跑一次评估。若它自己在 6~8 之间摆动，那 70% vs 60% 的对比当场作废。
2. **换套件测泛化**（信息量最高）：`--suite libero_object` / `libero_goal` / `libero_10`。spatial 上 60~70% 的模型在 goal 上可能差很多。
3. **要严格回答"解冻值不值"**：两边各 `-n 20~30`（200~300 回合），才够分辨 10~13 个点的差距。
4. **迭代提速**：日常调参一律用 expert-only（`--policy.train_expert_only=true --policy.freeze_vision_encoder=true` 放在命令末尾即可覆盖预设）。
5. **想再往上冲**：加步数（`--steps 50000 --save-freq 10000`，注意同时 `--policy.scheduler_decay_steps=50000`）、或换更大的 VLM 骨干（`policy.vlm_model_name`）。
