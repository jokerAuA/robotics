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
    bash scripts/run.sh scripts/eval.py --view --task-ids "[0]" -n 1        # 有显示器：实时看推理过程

    # 显式指定输出目录（会先清掉该目录里上次评估的产物，保证干净覆盖）
    bash scripts/run.sh scripts/eval.py --policy <ckpt> -n 10 --output-dir outputs/eval_smolvla25k_n10

关于输出目录（2026-10-07 改）：
    默认 = outputs/eval_<套件>/<日期>/<时间>_n<每任务回合数>
        （带时间戳 → 每次评估天然独立，不同权重的结果不会混在一起）
    显式给 --output-dir 时，会先删除该目录下的 eval_info.json / videos/ / recordings/
        再启动评估；因为 lerobot-eval **只覆盖它本轮要写的文件、不清理多余的旧文件**。
    踩过的坑：旧默认目录是固定的 outputs/eval_<套件>，用 -n 1 跑完后目录里仍留着上次
        -n 10 的 eval_episode_1..9.mp4，看起来像“每个任务生成了 10 个视频、其中 9 个全错”，
        实际是两次不同权重的产物混在一起（25000 步 70% 的 episode_0 + 9-30 那天 4 步 0% 的 1..9）。
    要判断某次评估到底产出了哪些视频，**以 <输出目录>/eval_info.json 的 video_paths 为准**，
        不要去数 videos/ 里的文件个数。

关于 --view（实时可视化）：
    默认（不加 --view）是"无头"跑法：MUJOCO_GL=egl，全程不开窗口，结束后看 outputs/eval_*/videos/*.mp4。
    加上 --view 后，会**额外**弹一个窗口，把每一步的画面实时显示到你的 X11 / VNC 桌面。
    注意：MuJoCo 仍然用 EGL 在 GPU 上离屏渲染，窗口只是把图像贴出来，
    所以不会因为桌面是软件 OpenGL（VNC）而变慢。
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import json
import sys
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# LIBERO 环境的两个原始相机（--env.camera_name_mapping 的键）
LIBERO_ENV_VIEWS = ("agentview_image", "robot0_eye_in_hand_image")
# LiberoEnv 的**默认**输出键：pixels/agentview_image → observation.images.image
# 权重若就声明 image/image2，则环境原生行为已经匹配，不需要任何映射参数。
LIBERO_DEFAULT_VIEWS = ("image", "image2")

# lerobot-eval 会往 output_dir 里写这些（见 lerobot_eval.py:824/865 与 recording 分支）。
# 清理旧产物时**只动这几样**，绝不整目录删。
EVAL_ARTIFACTS = ("eval_info.json", "videos", "recordings")


def default_output_dir(suite: str, episodes: int) -> str:
    """默认输出目录：outputs/eval_<套件>/<日期>/<时间>_n<每任务回合数>。

    为什么带时间戳（2026-10-07 实测踩过）：
        旧默认是**固定**目录 outputs/eval_<套件>，而 lerobot-eval 只覆盖它本轮要写的
        文件、**不删除多余的旧文件**。于是 `-n 1` 跑完，目录里仍躺着上次 `-n 10` 留下的
        eval_episode_1..9.mp4 —— 看起来像「每个任务生成了 10 个视频、其中 9 个全错」，
        实际是把两次不同权重的产物混在一起看了。
    末尾的 `_n<回合数>` 是刻意留的：一眼能区分「-n 1 的抽样体检」和「-n 10 的正式复测」。
    """
    stamp = dt.datetime.now()
    return f"outputs/eval_{suite}/{stamp:%Y-%m-%d}/{stamp:%H-%M-%S}_n{episodes}"


def clear_stale_outputs(out_dir: Path) -> list[str]:
    """清掉目标目录里上一次评估留下的产物，返回被清掉的名字（空列表 = 本来就没有）。

    为什么必须清：lerobot-eval 不做任何清理，`-n 1` 会在 `-n 10` 的残骸上叠加，
    导致「视频个数对不上 / 两个模型的结果混在一个目录里」。

    只删 EVAL_ARTIFACTS 点名的几项、**不整目录删**：
    `--output-dir` 完全可能被误指向 outputs/train 甚至项目根，整删太危险。
    """
    removed: list[str] = []
    for name in EVAL_ARTIFACTS:
        p = out_dir / name
        if not p.exists():
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        removed.append(name)
    return removed


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
    p.add_argument(
        "-g",
        "--gpu",
        default=None,
        help="用哪张卡；默认自动选空闲显存最多的卡（这台机器多人共用，别写死 0）",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--output-dir",
        default=None,
        help="输出目录。默认 = outputs/eval_<套件>/<日期>/<时间>_n<回合数>（带时间戳，不串）；"
        "显式指定时会先清掉该目录里上次评估的产物（eval_info.json/videos/recordings）再跑",
    )
    # ---- 实时可视化（有显示器 / VNC 桌面时用）----
    p.add_argument(
        "--view",
        action="store_true",
        help="弹出实时仿真窗口，逐步显示推理画面（需要 X11 显示，如 VNC 桌面）",
    )
    p.add_argument("--view-scale", type=float, default=1.0, help="窗口放大倍数")
    p.add_argument("--view-fps", type=float, default=25.0, help="窗口最大刷新率，防止拖慢评估")
    p.add_argument(
        "--view-display",
        default=None,
        help="X11 DISPLAY，如 :2；默认沿用环境变量 DISPLAY",
    )
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

    # 相机键名必须与权重声明的 input_features 对齐，否则 prepare_images() 会抛
    #   ValueError: All image features are missing from the batch.
    # 判定依据只能是权重 config.json 里**真实声明**的键名。
    # 不能用 `"smolvla" in a.policy` 这种字符串匹配：
    #   - 自己训练的产物路径形如 outputs/train/smolvla_libero/... → 会被误加 camera1/camera2
    #     （若那份权重是用 image/image2 训练的，加了反而报 Feature mismatch）
    #   - 路径形如 outputs/train/my_run/... → 又不加（那份权重若期望 camera1/camera2 就会挂）
    cameras = policy_declared_cameras(a.policy)
    if cameras is None:
        # 读不到 config.json（离线 / 路径不存在）→ 退回旧的启发式，但明确告警
        if "smolvla" in a.policy:
            print(
                "⚠️  读不到权重 config.json，按旧启发式假定需要 camera1/camera2 映射。\n"
                "    若评估报 Feature mismatch，请显式检查权重的 input_features。",
                file=sys.stderr,
            )
            cameras = ["camera1", "camera2", "camera3"]
    if cameras:
        args += camera_mapping_args(cameras)

    if a.task_ids:
        args.append(f"--env.task_ids={a.task_ids}")

    return args


def policy_declared_cameras(policy: str) -> list[str] | None:
    """读权重 config.json 里声明的相机名（去掉 observation.images. 前缀与空相机位）。

    本地目录直接读；HF repo id 走 hf_hub_download（命中缓存则不发请求）。
    读不到返回 None —— 交给调用方决定退路，不在这里瞎猜。
    """
    try:
        if Path(policy).is_dir():
            cfg_path = Path(policy) / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            cfg_path = Path(hf_hub_download(repo_id=policy, filename="config.json"))
        cfg = json.loads(cfg_path.read_text())
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


def camera_mapping_args(cameras: list[str]) -> list[str]:
    """把 LIBERO 的两个视角对齐到权重声明的相机名，并补齐多余的空相机位。

    三种情况：
      - 权重声明 image/image2（例如在本仓库用路线 A 微调出来的）→ 环境默认行为就匹配，零参数
      - 权重声明 camera1/camera2/camera3（例如 lerobot/smolvla_libero）→ 改名 + 补 1 个空相机
      - 声明不足 2 路 → 没法对齐，只告警不硬拼参数
    """
    targets = sorted(cameras)[:2]  # camera1 → 第三人称，camera2 → 手腕
    if len(targets) < 2:
        print(
            f"⚠️  权重只声明了 {cameras}，无法与 LIBERO 的两路视角对齐，评估大概率失败。",
            file=sys.stderr,
        )
        return []

    out: list[str] = []
    if tuple(targets) != LIBERO_DEFAULT_VIEWS:
        out.append(f"--env.camera_name_mapping={json.dumps(dict(zip(LIBERO_ENV_VIEWS, targets)))}")

    n_empty = len(cameras) - len(targets)  # 环境提供不了的多余相机位 → 补空
    if n_empty:
        out.append(f"--policy.empty_cameras={n_empty}")
    return out


def pick_idle_gpu() -> tuple[str, str]:
    """按「空闲显存最多」自动选一张卡（与 scripts/train.py 里的同名函数一致）。

    这台机器多人共用：实测 GPU 0 上跑着别的用户持续 5 天的任务、占 33.6/48 GB。
    写死卡号会在别人占满时直接 OOM。4 张卡是 **RTX 4090 48GB 版**。
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
        return "0", "（nvidia-smi 不可用，退回 0）"

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


def resolve_policy(policy: str) -> tuple[str, str]:
    """把 HF repo id 换成项目内已下好的本地目录（如果存在）。

    为什么：`--policy.path=lerobot/smolvla_base` 是 **repo id**，lerobot 会去 HF 缓存
    解析权重；而缓存里那份可能只缓存了 config.json（权重是 download_assets.py
    落在项目内 models/ 的），于是会把已下好的 900MB 重新下一遍。
    （本函数与 scripts/train.py 里的同名函数刻意保持一致。）
    """
    if Path(policy).is_dir():
        return policy, "本地目录"

    local = PROJECT_ROOT / "models" / Path(policy).name
    if (local / "config.json").is_file() and (local / "model.safetensors").is_file():
        return str(local.relative_to(PROJECT_ROOT)), f"本地目录（由 {policy} 解析而来）"

    return policy, "HF repo id（走 HF 缓存，缺权重会联网下载）"


def find_lerobot_eval() -> str:
    """优先用与当前解释器同目录的 lerobot-eval，避免环境串了。"""
    sibling = Path(sys.executable).with_name("lerobot-eval")
    if sibling.is_file():
        return str(sibling)
    return shutil.which("lerobot-eval") or ""


def run_with_view(
    eval_argv: list[str],
    suite: str,
    scale: float = 1.0,
    fps: float = 25.0,
    display: str | None = None,
) -> int:
    """在**当前进程**里跑 lerobot-eval，并额外弹一个实时窗口。

    为什么要在当前进程里跑：
        lerobot 的评估循环自带钩子 `lerobot_eval.rollout(..., render_callback=...)`，
        但它由 CLI 内部固定传入，外部没法注入。这里把模块里的 `rollout` 换成包装版：
        先调用 lerobot 原有的 render_callback（负责录视频），再把同一帧喂给窗口。
    注意：MuJoCo 仍然是 EGL 离屏渲染，窗口只是贴图，不影响 GPU 性能。
    """
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    if display:
        os.environ["DISPLAY"] = display
    if not os.environ.get("DISPLAY"):
        print(
            "❌ --view 需要一个 X11 显示（例如 VNC 桌面 :2）。\n"
            "   请在有桌面的会话里运行，或显式指定：--view-display :2",
            file=sys.stderr,
        )
        return 1

    import numpy as np

    from live_view import LiveView, render_env_frames

    try:
        import lerobot.scripts.lerobot_eval as le
    except Exception as e:  # noqa: BLE001
        print(f"❌ 无法导入 lerobot 评估模块：{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    try:
        view = LiveView(title=f"实时推理 —— {suite}", scale=scale, max_fps=fps)
    except Exception as e:  # noqa: BLE001
        print(
            f"❌ 创建窗口失败：{type(e).__name__}: {e}\n"
            f"   当前 DISPLAY={os.environ.get('DISPLAY')}；可试试 --view-display :2",
            file=sys.stderr,
        )
        return 1

    orig_rollout = le.rollout
    stats: dict = {"frame": 0, "task": "", "shape": None}

    def on_frame(env) -> None:
        frames = render_env_frames(env)
        if not frames:
            return
        stats["frame"] += 1
        if stats["frame"] % 32 == 1:  # 任务名不必每帧都查（有 IPC 开销）
            try:
                stats["task"] = str(env.call("task_description")[0])
            except Exception:
                pass
        # 把两路相机（第三人称 + 手腕）横向拼起来，看得更清楚
        image = frames[0] if len(frames) == 1 else np.concatenate(frames, axis=1)
        if stats["shape"] is None:
            stats["shape"] = tuple(image.shape)
        caption = f"frame {stats['frame']:>5d}  |  {stats['task'][:64]}"
        view.show(image, caption)

    def rollout_with_view(*args, **kwargs):
        inner = kwargs.get("render_callback")

        def combined(env):
            if inner is not None:
                inner(env)  # 保留 lerobot 自带的录制逻辑
            on_frame(env)

        kwargs["render_callback"] = combined
        return orig_rollout(*args, **kwargs)

    le.rollout = rollout_with_view

    old_argv = sys.argv
    sys.argv = ["lerobot-eval", *eval_argv]
    rc = 0
    try:
        le.main()
    except SystemExit as e:
        rc = int(e.code or 0)
    except KeyboardInterrupt:
        print("\n⏹  已手动中断（Ctrl-C）")
    except Exception as e:  # noqa: BLE001
        print(f"\n❌ 评估过程出错：{type(e).__name__}: {e}", file=sys.stderr)
        rc = 1
    finally:
        sys.argv = old_argv
        le.rollout = orig_rollout
        if stats["frame"]:
            size = f"，首帧尺寸 {stats['shape'][1]}x{stats['shape'][0]}" if stats["shape"] else ""
            print(f"\n🖥  实时窗口共推送 {view.frames_shown} 帧{size}")
        elif view.closed:
            print("\n🖥  窗口被提前关闭，未推送画面")
        else:
            print("\n⚠️  没有从环境取到画面（窗口未收到帧）", file=sys.stderr)
        view.close()

    return rc


def main() -> int:
    a = parse_args()

    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("MUJOCO_GL", "egl")  # 服务器无显示器 → EGL 无头渲染
    # xet 传输在本机的 HTTP 代理下会挂死（且不报错，只是永远不动）→ 强制走传统 HTTP
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    if a.gpu is None:
        a.gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        if a.gpu:
            gpu_note = "（来自环境变量 CUDA_VISIBLE_DEVICES）"
        else:
            a.gpu, gpu_note = pick_idle_gpu()
    else:
        gpu_note = "（来自 -g）"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu  # 必须在起 lerobot-eval 之前设好

    a.policy, policy_src = resolve_policy(a.policy)

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

    # 目录必须在 build_eval_args 之前定好（它读的就是 a.output_dir）
    a.output_dir = a.output_dir or default_output_dir(a.suite, a.episodes)
    out_dir = a.output_dir
    removed = clear_stale_outputs(PROJECT_ROOT / out_dir)
    cmd = [eval_bin, *build_eval_args(a)]

    print("=" * 46)
    print(f" 套件       : {a.suite}")
    print(f" 任务       : {a.task_ids or '（该套件全部任务）'}")
    print(f" 每任务回合 : {a.episodes}")
    print(f" 策略       : {a.policy}  [{policy_src}]")
    print(f" 显卡       : {a.gpu} {gpu_note}  (MUJOCO_GL={os.environ['MUJOCO_GL']})")
    print(f" Python     : {sys.executable}")
    print(f" 输出       : {out_dir}")
    if removed:
        print(f" 旧产物     : 已清理 {'、'.join(removed)}（否则会和本次结果混在一个目录里）")
    print("=" * 46)
    print(" ".join(cmd))
    print()

    if a.view:
        print("🖥  已开启实时窗口（画面显示在 DISPLAY=%s 的桌面上）" % os.environ.get("DISPLAY"))
        print("   关闭窗口不会中断评估；Ctrl-C 可提前结束。\n")
        rc = run_with_view(
            cmd[1:],
            suite=a.suite,
            scale=a.view_scale,
            fps=a.view_fps,
            display=a.view_display,
        )
    else:
        rc = subprocess.call(cmd)

    if rc == 0:
        print(f"\n✅ 评估结束。\n   指标 : {out_dir}/eval_info.json\n   视频 : {out_dir}/videos/")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
