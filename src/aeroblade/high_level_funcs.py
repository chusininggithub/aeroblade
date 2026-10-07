"""把「重建 + 距离」串成一条流水线，直接产出可以喂给 notebooks 的 DataFrame。

experiments/01_detect.py 与 02_analyze_patches.py 都只做参数解析，真正的
四层嵌套循环（扰动变换 × 数据目录 × AE 模型 × 距离指标）在这里。

产出的一张长表每行是「某数据集里某张图、在某个扰动下、用某个 AE 重建、
用某个指标算出的距离」。所有指标都会被取负（见 distances.py），
所以「越大越像生成图」。
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as tf
from tqdm import tqdm

from aeroblade.complexities import complexity_from_config
from aeroblade.data import ImageFolder
from aeroblade.distances import distance_from_config
from aeroblade.image import compute_reconstructions
from aeroblade.transforms import transform_from_config


def compute_distances(
    dirs: list[Path],
    transforms: list[str | Callable],
    repo_ids: list[str],
    distance_metrics: list[str],
    amount: Optional[int],
    reconstruction_root: Path,
    seed: int,
    batch_size: int,
    num_workers: int,
    compute_max: bool = True,
    **distance_kwargs,
) -> pd.DataFrame:
    """Compute distances between original and reconstructed images."""
    # 进度条的总步数 = 四层循环的总迭代次数，方便估时（这四层是乘法关系）。
    # set up progress bar
    pbar = tqdm(
        desc="PROGRESS (compute_distances)",
        total=len(transforms) * len(dirs) * len(repo_ids) * len(distance_metrics),
    )

    distances = []

    # iterate over transforms
    for transform_config in transforms:
        # "clean" 表示不做扰动，此时不加任何 transform（保持默认的
        # 张量转换），这样能保证 clean 一路的结果和只跑检测时完全一致。
        if transform_config != "clean":
            # 顺序很重要：先做扰动（在 PIL/张量上），再统一转成 float32 [0,1]。
            transform = tf.Compose(
                [
                    # 允许直接传已构造好的 transform 对象，而不仅是配置字符串。
                    transform_from_config(transform_config)
                    if isinstance(transform_config, str)
                    else transform_config,
                    tf.ToImage(),
                    tf.ToDtype(torch.float32, scale=True),
                ]
            )

        # iterate over directories
        for dir in dirs:
            # 每个数据集单独建 dataset：amount 限流在这个目录内生效。
            if transform_config != "clean":
                ds = ImageFolder(dir, amount=amount, transform=transform)
            else:
                ds = ImageFolder(dir, amount=amount)

            # iterate over autoencoder repo_ids
            for repo_id in repo_ids:
                # 重建只和 (数据、AE、扰动) 有关，与距离指标无关，
                # 所以放在距离指标的循环外面——一次重建，多个指标复用。
                rec_paths = compute_reconstructions(
                    ds,
                    repo_id=repo_id,
                    output_root=reconstruction_root,
                    seed=seed,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                # 重建图存成 PNG，用默认 transform 读入即可；
                # 注意扰动已经固化在重建图里，这里不能再施加扰动。
                ds_rec = ImageFolder(rec_paths)

                # iterate over distance metrics
                for dist_metric in distance_metrics:
                    dist_dict, files = distance_from_config(
                        dist_metric,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        **distance_kwargs,
                    ).compute(
                        ds_a=ds,
                        ds_b=ds_rec,
                    )
                    # 一个配置可能返回多个指标（layer=-1 时逐层返回），
                    # 所以这里再遍历一次字典，每个指标一行记录。
                    for dist_name, dist_tensor in dist_dict.items():
                        if not distance_kwargs.get("spatial", False):
                            # 非空间模式已经是 (N,1,1,1)，压成 (N,) 方便存 CSV。
                            # 空间模式保持 4D，因为 patch 级结果要单独存成张量。
                            dist_tensor = dist_tensor.squeeze(1, 2, 3)
                        # 长表结构：每一维配置都是普通列，方便后面 groupby。
                        df = pd.DataFrame(
                            {
                                "dir": str(dir),
                                "image_size": int(ds[0][0].shape[-1]),
                                "repo_id": repo_id,
                                "transform": transform_config,
                                "distance_metric": dist_name,
                                "file": files,
                                "distance": list(dist_tensor.numpy()),
                            }
                        )
                        distances.append(df)
                    # 进度条只在最内层更新，因为最内层才是单位工作量。
                    pbar.update()

    distances = pd.concat(distances)

    # 论文的最终检测器是「多个 AE 取最大分数」，这里顺手把 max 也算出来，
    # 当作一个虚拟的 repo_id="max" 追加到表里，下游直接 groupby 就能取用。
    # determine maximum distance over all repo_ids for each file
    if compute_max:
        maxima = []
        for group_keys, group_df in distances.groupby(
            # 海象运算符把列名列表存下来，后面重建行时要用同一组列。
            group_cols := ["dir", "image_size", "transform", "distance_metric"],
            # sort=False 保留原顺序，保证 CSV 行序稳定可复现。
            sort=False,
        ):
            # 同一张图在多个 AE 下的分数取 max（axis=0 支持 patch 级的
            # (num_patches, 1, H, W) 张量逐元素取最大）。
            max_values = group_df.groupby("file").apply(
                lambda df: np.stack(df.distance).max(axis=0)
            )
            max_df = {col: key for col, key in zip(group_cols, group_keys)}
            max_df.update(
                {
                    "repo_id": "max",
                    "file": max_values.index.values,
                    "distance": max_values.values,
                }
            )
            maxima.append(pd.DataFrame(max_df))

        distances = pd.concat([distances, *maxima]).sort_values("dir", kind="stable")
    distances = distances.reset_index(drop=True)
    return distances


def compute_complexities(
    dirs: list[Path],
    transforms: list[str],
    complexity_metrics: list[str],
    amount: Optional[int],
    patch_size: Optional[int],
    patch_stride: Optional[int],
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Compute distances between original and reconstructed images."""
    # 结构上 compute_distances 的简化版：没有 AE 维度和距离维度，
    # 因为复杂度只取决于图像本身，不需要重建。产出表用于 5.4 节的散点图。
    # set up progress bar
    pbar = tqdm(
        desc="PROGRESS (compute_complexities)",
        total=len(transforms) * len(dirs) * len(complexity_metrics),
    )

    complexities = []

    # iterate over transforms
    for transform_config in transforms:
        if transform_config != "clean":
            # 这里不像 compute_distances 那样允许传 Callable，
            # 因为复杂度实验只跑配置字符串形式的扰动。
            transform = tf.Compose(
                [
                    transform_from_config(transform_config),
                    tf.ToImage(),
                    tf.ToDtype(torch.float32, scale=True),
                ]
            )

        # iterate over directories
        for dir in dirs:
            if transform_config != "clean":
                ds = ImageFolder(dir, amount=amount, transform=transform)
            else:
                ds = ImageFolder(dir, amount=amount)

            # iterate over complexity metrics
            for comp_metric in complexity_metrics:
                comp_dict, files = complexity_from_config(
                    comp_metric,
                    patch_size=patch_size,
                    patch_stride=patch_stride,
                    batch_size=batch_size,
                    num_workers=num_workers,
                ).compute(
                    ds=ds,
                )
                for comp_name, comp_tensor in comp_dict.items():
                    df = pd.DataFrame(
                        {
                            "dir": str(dir),
                            "transform": transform_config,
                            "complexity_metric": comp_name,
                            # 局部复杂度时每个 patch 一行（file 重复出现），
                            # 后续分析按 file 分组到同一张图上。
                            "file": files,
                            "complexity": list(comp_tensor.numpy()),
                        }
                    )
                    complexities.append(df)
                pbar.update()

    complexities = pd.concat(complexities).reset_index(drop=True)
    return complexities
