#!/usr/bin/env python3
"""下载训练/评测所需的资产：LIBERO 数据集 + SmolVLA 基线权重 + LIBERO 3D 资产。

为什么需要这个脚本：
    0) LIBERO 的 3D 资产（`~/.cache/libero/assets`，评测时由 libero 包按需下载）
       它自己的"已下载"判定是**假检查**：只看 4 个物体目录**在不在**、不看里面
       有没有文件。代理中途掐断后目录都在、内容只剩一个 black_book，于是此后
       每次都打印「Assets already downloaded」直接跳过，评测到 `env.reset()`
       才 `FileNotFoundError: .../turbosquid_objects/wine_rack/wine_rack.xml`
       （2026-10-08 实测：远端 585 个文件，本地缺 142 个）。
       本脚本按**远端文件清单**逐文件核对，并且只补缺的那几个。
    1) `models/` 是空占位，权重实际由 HF 缓存提供；但 `lerobot/smolvla_base`
       本地缓存里**只有 config.json，没有权重**，微调前必须补下。
    2) `lerobot/libero` 数据集本地缓存里**只有 meta/ 和 data/，没有 videos/**，
       而训练是按 mp4 逐帧解码取图的 —— 不补下会直接失败。
    3) 已有缓存落在 `$HF_HOME/hub/`，而 lerobot 默认下载走
       `$HF_LEROBOT_HOME/hub/`（= `~/.cache/huggingface/lerobot/hub`），是**另一处**，
       所以不显式指定 --dataset.root 会重复下载。
    本脚本用 `snapshot_download(local_dir=...)` 把两份资产都落进**项目内**，
    路径稳定、自包含，便于 .gitignore 统一管理。

与 _archive/scripts/download_models.py 的区别（那个已废弃）：
    - 删掉了 `resume_download=True` —— 该参数在 huggingface_hub 1.x 已被移除，
      本机装的是 1.33.0，照抄旧脚本会直接 TypeError。
    - 不自动加载 `.env`。`.env` 里写的是 `HF_ENDPOINT=https://hf-mirror.com`，
      但本机实际生效的下载通道是 `~/.bashrc` 里的 `http_proxy=127.0.0.1:1080` 直连官方；
      自动加载 `.env` 反而会把通道切到镜像站，属于"配置与行为不一致"的坑。
      需要换端点请显式传 `--endpoint`。

用法：
    bash scripts/run.sh scripts/download_assets.py                  # 数据集 + 权重 + LIBERO 资产
    bash scripts/run.sh scripts/download_assets.py --only model     # 只下权重
    bash scripts/run.sh scripts/download_assets.py --only assets    # 只补 LIBERO 3D 资产
    bash scripts/run.sh scripts/download_assets.py --check          # 只体检，不下载
    python scripts/download_assets.py --help                        # 查看全部参数
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# 允许直接 `python scripts/download_assets.py` 时 import huggingface_hub
# （不依赖 lerobot，所以任何装过 huggingface_hub 的解释器都能跑）
try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ImportError:  # pragma: no cover
    print(
        "❌ 找不到 huggingface_hub。\n"
        "   请用 ai312 解释器运行：bash scripts/run.sh scripts/download_assets.py",
        file=sys.stderr,
    )
    raise SystemExit(1) from None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- 资产清单 ----------------------------------------------------------------
DATASET = {
    "key": "dataset",
    "repo_id": "lerobot/libero",
    "repo_type": "dataset",
    "local_dir": Path("datasets/libero"),
    "what": "LIBERO 训练集（1693 episodes / 273465 frames / 40 tasks / fps 10）",
    "size": "约 1.9 GB（大头是两个相机的 mp4）",
}
MODEL = {
    "key": "model",
    "repo_id": "lerobot/smolvla_base",
    "repo_type": "model",
    "local_dir": Path("models/smolvla_base"),
    "what": "SmolVLA 基线权重（450M，VLM 主干另走 HF 缓存）",
    "size": "约 0.9 GB",
}
# LIBERO 的 3D 场景/物体/贴图。⚠️ **不在项目内**：路径由 libero 包写死成
# `~/.cache/libero/assets`（libero/libero/__init__.py:get_assets_path）。
LIBERO_ASSETS = {
    "key": "assets",
    "repo_id": "lerobot/libero-assets",
    "repo_type": "dataset",
    "local_dir": Path("~/.cache/libero/assets"),
    "what": "LIBERO 3D 场景/物体/贴图（libero_goal 要用的 wine_rack 等 turbosquid 物体在这）",
    "size": "约 400 MB（585 个文件）",
    # 只补缺、不整仓重下：中断后往往只缺几个文件，整仓重下要几百 MB（见 ensure_asset）
    "incremental": True,
}
ASSETS = [DATASET, MODEL, LIBERO_ASSETS]

# 权重目录存在这些文件才算下全（config.json 单独存在不算下全 —— 旧缓存就是这种情况）
MODEL_SENTINELS = ("config.json", "model.safetensors")


# ---- 小工具 ------------------------------------------------------------------
def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def dir_size(path: Path) -> int:
    """目录总大小（只统计真实文件，跳过符号链接，避免 HF 缓存重复计数）。"""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def ok(cond: bool) -> str:
    return "✅" if cond else "❌"


def asset_root(a: dict) -> Path:
    """资产落地目录。`~`/绝对路径原样展开 —— LIBERO 资产在 `~/.cache` 里，不在项目内。"""
    p = Path(str(a["local_dir"])).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---- 完整性判定 --------------------------------------------------------------
def expected_parquet_files(root: Path, info: dict) -> dict[str, set[Path]]:
    """从 `meta/episodes/**` 索引表反推「应该存在哪些分片文件」。

    为什么非这么做不可：只检查 `videos/<key>/` 下「有任何 mp4」是**不够的**。
    LIBERO 有 74 个视频分片，代理中途掐断时可能只缺 1 个；弱检查会误报「已下全」，
    然后训练在解码时才炸 —— 或者更糟：静默少训几个 episode。
    索引表 `meta/episodes/*.parquet` 里有 `videos/<key>/{chunk_index,file_index}`
    和 `data/{chunk_index,file_index}`，据此可以算出精确的文件清单。
    """
    out: dict[str, set[Path]] = {}
    ep_files = sorted(root.glob("meta/episodes/**/*.parquet"))
    if not ep_files:
        return out

    try:
        import pandas as pd

        df = pd.concat([pd.read_parquet(p) for p in ep_files])
    except Exception:  # noqa: BLE001
        return out

    # 数值分片：data/chunk-000/file-000.parquet
    if "data/chunk_index" in df.columns:
        tpl = root / info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        out["data"] = {
            Path(str(tpl).format(chunk_index=int(c), file_index=int(f)))
            for c, f in zip(df["data/chunk_index"], df["data/file_index"])
        }

    # 视频分片：videos/<key>/chunk-000/file-000.mp4
    for key in info.get("features", {}):
        ci, fi = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
        if ci not in df.columns or fi not in df.columns:
            continue
        tpl = root / "videos" / key / "chunk-{chunk_index:03d}" / "file-{file_index:03d}.mp4"
        out[key] = {
            Path(str(tpl).format(chunk_index=int(c), file_index=int(f)))
            for c, f in zip(df[ci], df[fi])
        }
    return out


def dataset_status(root: Path) -> tuple[bool, list[str]]:
    """判定数据集是否下全。

    不能只看 meta/info.json —— meta/ 和 data/ 早就下了，缺的往往是 videos/。
    所以按 `meta/episodes` 索引表逐片核对，精确到「缺第几个分片」。
    """
    problems: list[str] = []

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return False, [f"缺 {info_path.relative_to(root)}（连 meta 都没有）"]

    try:
        info = json.loads(info_path.read_text())
    except Exception as e:  # noqa: BLE001
        return False, [f"meta/info.json 解析失败：{type(e).__name__}: {e}"]

    # 索引表本身
    ep_files = sorted(root.glob("meta/episodes/**/*.parquet"))
    if not ep_files:
        problems.append("缺 meta/episodes/**/*.parquet（分片索引表）")
    if not (root / "meta" / "stats.json").is_file():
        problems.append("缺 meta/stats.json（归一化统计量；缺了训练会静默不归一化）")

    expected = expected_parquet_files(root, info)
    if not expected:
        # 索引表读不出来 → 退回弱检查，并说明为什么
        video_keys = [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]
        for key in video_keys:
            if not any((root / "videos" / key).rglob("*.mp4")):
                problems.append(f"缺 videos/{key}/*.mp4")
        if not any((root / "data").rglob("*.parquet")):
            problems.append("缺 data/**/*.parquet")
    else:
        for name, paths in expected.items():
            missing = [p for p in sorted(paths) if not p.is_file()]
            if missing:
                kind = "视频分片" if name != "data" else "数值分片"
                shown = ", ".join(str(p.relative_to(root)) for p in missing[:3])
                more = f" 等 {len(missing)} 个" if len(missing) > 3 else ""
                problems.append(f"缺 {len(missing)}/{len(paths)} 个{kind}：{shown}{more}")

    if problems:
        return False, problems

    ep = info.get("total_episodes", "?")
    fr = info.get("total_frames", "?")
    n_vid = len([k for k in expected if k != "data"])
    return True, [f"{ep} episodes / {fr} frames / {n_vid} 路相机 × 全部分片就绪（v3.0）"]


def model_status(root: Path) -> tuple[bool, list[str]]:
    missing = [f for f in MODEL_SENTINELS if not (root / f).is_file()]
    if missing:
        return False, [f"缺 {', '.join(missing)}"]
    size = human(dir_size(root))
    return True, [f"config.json + model.safetensors 就绪（{size}）"]


def libero_assets_missing(root: Path) -> list[str] | None:
    """远端文件清单 − 本地实有文件 = 还缺哪些；返回 None 表示拿不到远端清单。

    为什么非跟远端逐文件比不可：LIBERO 自带的判定是**假检查** ——
    `libero/utils/download_utils.py:download_assets_from_huggingface()` 只判断
    `articulated_objects/ stable_scanned_objects/ turbosquid_objects/ stable_hope_objects/`
    这 4 个**目录存在**，根本不看里面有没有文件。于是代理掐断后（目录都在、内容只剩
    一个 black_book）此后每次都打印「Assets already downloaded」直接跳过，
    评测要 `env.reset()` 时才炸 —— 「弱检查一路绿灯 → 跑挂」比没有检查更糟。
    """
    try:
        files = [
            f
            for f in HfApi().list_repo_files(repo_id=LIBERO_ASSETS["repo_id"], repo_type="dataset")
            if not f.startswith(".")
        ]
    except Exception:  # noqa: BLE001
        return None

    # 0 字节也算缺：中断会留下空占位文件（实测踩到过）
    return [f for f in files if not (root / f).is_file() or (root / f).stat().st_size == 0]


def libero_assets_status(root: Path) -> tuple[bool, list[str]]:
    missing = libero_assets_missing(root)
    if missing is None:
        return False, ["拿不到远端文件清单（离线 / 代理不通）→ 无法判断是否下全，联网后重跑"]
    if missing:
        by_dir = Counter(f.split("/")[0] for f in missing)
        group = "、".join(f"{k} {v} 个" for k, v in by_dir.most_common(4))
        shown = "、".join(missing[:3])
        more = " 等" if len(missing) > 3 else ""
        return False, [f"缺 {len(missing)} 个文件（{group}）：{shown}{more}"]
    return True, [f"{human(dir_size(root))}，与远端清单逐文件一致"]


STATUS_FN = {"dataset": dataset_status, "model": model_status, "assets": libero_assets_status}
# 能算出「精确缺哪些文件」的资产 → 下载时只补这几个（见 ensure_asset）
MISSING_FN = {"assets": libero_assets_missing}


# ---- 下载 --------------------------------------------------------------------
def _download_files(a: dict, root: Path, files: list[str], *, endpoint: str | None) -> bool:
    """逐个补下指定文件；单个文件失败**不打断**整批 —— 代理会随机掐断大文件。

    为什么不用整仓 `snapshot_download`：本机代理反复掐断连接，整仓重下要么白等
    几百 MB、要么卡在同一个文件上；逐个下 + 重试能把损失限制在单个文件。
    另外 `hf_hub_download` 对已存在的文件会直接跳过，重跑天然是幂等的。
    """
    failed: list[tuple[str, str]] = []
    for i, name in enumerate(files, 1):
        for attempt in range(3):
            try:
                hf_hub_download(
                    repo_id=a["repo_id"],
                    filename=name,
                    repo_type=a["repo_type"],
                    local_dir=str(root),
                    endpoint=endpoint,
                )
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    failed.append((name, f"{type(e).__name__}: {str(e)[:100]}"))
                else:
                    time.sleep(3)
        if i % 50 == 0 or i == len(files):
            print(f"   补下 {i}/{len(files)}（失败 {len(failed)}）")

    if failed:
        print(f" ❌ 仍有 {len(failed)} 个文件没下下来：", file=sys.stderr)
        for name, err in failed[:5]:
            print(f"     {name} → {err}", file=sys.stderr)
        print("     重跑本脚本可继续补（已下好的会跳过）。", file=sys.stderr)
        return False
    return True


def ensure_asset(a: dict, *, endpoint: str | None, force: bool) -> bool:
    root = asset_root(a)
    complete, detail = STATUS_FN[a["key"]](root)

    print(f"\n{'=' * 62}")
    print(f" {a['repo_id']}  →  {a['local_dir']}")
    print(f"{'=' * 62}")
    print(f" 用途   : {a['what']}")
    print(f" 体积   : {a['size']}")
    for line in detail:
        print(f" {ok(complete) if not force else '🔄'} {line}")

    if complete and not force:
        print(" → 已下全，跳过。（想强制重下加 --force）")
        return True

    print(f"\n → 开始下载…（repo_type={a['repo_type']}）")
    root.mkdir(parents=True, exist_ok=True)

    if a.get("incremental"):
        missing = MISSING_FN[a["key"]](root)
        if missing is None:
            print(" ⚠️ 拿不到远端文件清单（网络 / 代理？）→ 退回整仓重下")
        elif not missing:
            print(" → 远端清单与本地一致，无需补下。")
            return True
        else:
            return _download_files(a, root, missing, endpoint=endpoint)

    try:
        snapshot_download(
            repo_id=a["repo_id"],
            repo_type=a["repo_type"],
            local_dir=str(root),
            endpoint=endpoint,
        )
    except Exception as e:  # noqa: BLE001
        print(f"\n❌ 下载失败：{type(e).__name__}: {e}", file=sys.stderr)
        print(
            "   排查：网络/代理是否可用（本机靠 ~/.bashrc 的 http_proxy=127.0.0.1:1080）；\n"
            "         或显式换端点：--endpoint https://hf-mirror.com",
            file=sys.stderr,
        )
        return False

    complete, detail = STATUS_FN[a["key"]](root)
    for line in detail:
        print(f" {ok(complete)} {line}")
    return complete


# ---- 体检（不下载）-----------------------------------------------------------
def check_only() -> int:
    print("=" * 62)
    print(" 训练资产体检（只查不下载）")
    print("=" * 62)
    all_ok = True
    for a in ASSETS:
        root = asset_root(a)
        complete, detail = STATUS_FN[a["key"]](root)
        all_ok &= complete
        print(f"\n{ok(complete)} {a['repo_id']}  →  {a['local_dir']}")
        for line in detail:
            print(f"     {line}")
        if complete:
            print(f"     体积 {human(dir_size(root))}")

    # 顺带确认 VLM 主干在 HF 缓存里（SmolVLA 需要它，通常早已下好）
    backbone = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    cached = (hub / f"models--{backbone.replace('/', '--')}").is_dir()
    note = f"已在 HF 缓存（{hub}）" if cached else "不在 HF 缓存，训练时会自动下载"
    print(f"\n{'✅' if cached else '⚠️ '} VLM 主干 {backbone}\n     {note}")

    print()
    return 0 if all_ok else 1


# ---- 入口 --------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="下载 SmolVLA 微调所需的数据集与基线权重",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--only",
        choices=("dataset", "model", "assets", "both"),
        default="both",
        help="只下载其中一项（assets = LIBERO 的 3D 场景/物体/贴图，不在项目内）",
    )
    p.add_argument("--check", action="store_true", help="只体检，不下载")
    p.add_argument("--force", action="store_true", help="即使已下全也重下")
    p.add_argument(
        "--endpoint",
        default=None,
        help="HF 端点；留空=用 huggingface_hub 默认（默认读 HF_ENDPOINT 环境变量）。不要依赖 .env",
    )
    return p.parse_args()


def main() -> int:
    a = parse_args()

    os.chdir(PROJECT_ROOT)  # 让相对路径 local_dir 有意义

    if a.check:
        return check_only()

    if a.endpoint:
        os.environ["HF_ENDPOINT"] = a.endpoint

    targets = [x for x in ASSETS if a.only == "both" or x["key"] == a.only]

    print("=" * 62)
    print(" SmolVLA 资产下载（数据集 / 权重 / LIBERO 3D 资产）")
    print("=" * 62)
    print(f" 项目根 : {PROJECT_ROOT}")
    print(f" 端点   : {a.endpoint or os.environ.get('HF_ENDPOINT') or '（huggingface_hub 默认）'}")
    print(f" 代理   : http_proxy={os.environ.get('http_proxy', '（未设置）')}")
    print(f" 范围   : {a.only}")
    print(f" Python : {sys.executable}")

    results = {x["key"]: ensure_asset(x, endpoint=a.endpoint, force=a.force) for x in targets}

    print(f"\n{'=' * 62}")
    print(" 下载总结")
    print(f"{'=' * 62}")
    for x in targets:
        print(f" {ok(results[x['key']])} {x['repo_id']:26} → {x['local_dir']}")

    if not all(results.values()):
        print("\n❌ 有资产没下全，请重跑或加 --force。")
        return 1

    print("\n✅ 资产就绪。下一步：")
    print("   冒烟测试   bash scripts/run.sh scripts/train.py --tmp-run")
    print("   正式训练   bash scripts/run.sh scripts/train.py -g 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
