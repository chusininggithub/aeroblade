"""杂项小工具：目录创建、设备选择、配置落盘。"""

import json
import os
import sys
from pathlib import Path

import torch


def resolve_model_source(repo_id: str) -> str:
    """把不可达的 HF repo id 映射到本地等价的模型目录。

    stabilityai/stable-diffusion-2-base 与 stabilityai/stable-diffusion-2-1-base
    在 HuggingFace 上是 gated 仓库，hf-mirror 无法代理（返回 401 "Invalid
    username or password"），国内也没有别的可达官方源，故改从 ModelScope 取
    同一份权重放本地。

    注意这里只改「从哪里加载权重」，**不改 repo_id 字符串本身**：下游数据表
    的 repo_id 列必须保持原始写法，才能和作者预计算的结果
    （data/precomputed/*.pickle）按 repo_id 对齐做 groupby。

    映射由环境变量 AEROBLADE_REPO_DIRS 提供（JSON）:
        {"stabilityai/stable-diffusion-2-base": "E:/.../stable-diffusion-2-base"}
    未设置或解析失败时原样返回 repo_id，行为与上游一致。
    """
    raw = os.environ.get("AEROBLADE_REPO_DIRS")
    if not raw:
        return repo_id
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError:
        return repo_id
    if not isinstance(mapping, dict):
        return repo_id
    return mapping.get(repo_id, repo_id)


def compile_net(model: torch.nn.Module) -> torch.nn.Module:
    """在支持的平台上编译模型，否则原样返回。

    torch.compile 目前只有 Linux 可用：Windows 缺 Triton 后端，调用会直接抛
    ``RuntimeError: Windows not yet supported for torch.compile``。编译只是性能
    优化、不改变数值结果，所以这里静默回退到 eager 执行，让实验能在 Windows 上
    跑通（代价是慢一些，不替换掉原有逻辑）。
    """
    if sys.platform == "win32":
        return model
    try:
        return torch.compile(model)
    except Exception:
        return model


def safe_mkdir(directory: Path) -> None:
    """Ask before using an existing directory."""
    # 为什么需要这一步：重建结果按参数哈希分目录缓存，若目录已存在说明
    # 之前跑过一次；这里让用户确认，避免误覆盖别人的实验产物。
    if directory.exists():
        response = input(
            f"Directory '{str(directory)}' exists, continue? (y/n) "
        ).lower()
        if response not in ["yes", "y"]:
            # 注意：这里是直接 exit()，不是抛异常，所以调用方无法捕获。
            exit()
    directory.mkdir(parents=True, exist_ok=True)


def device() -> str:
    """Return 'cuda' if available, 'cpu' otherwise"""
    # 全仓库统一从这里取设备，避免各模块各自判断导致 CPU/GPU 混用。
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_config(config: dict, directory: Path) -> None:
    """Write config text file to specified directory."""
    # 把参数写成 config.json 放到缓存目录旁边，方便事后回溯某个哈希目录
    # 到底是哪组参数跑出来的。值统一 str() 化，因为里面有 Path、Dataset 等
    # 不可直接 JSON 序列化的对象。
    with open(directory / "config.json", "w") as f:
        json.dump({key: str(value) for key, value in config.items()}, f)
