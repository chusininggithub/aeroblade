"""
Compute patch-wise distances and complexities.

===========================================================================
论文 5.4 节的入口：把距离和复杂度都算到 patch 粒度，再对齐成一张表，
用于回答「AEROBLADE 到底是靠图像复杂度/平滑度在判别，还是靠 AE 重建误差」。

跑法（README）：
    python experiments/02_analyze_patches.py
    # 用作者预计算的真实图结果，避免 LAION 原图失效：
    python experiments/02_analyze_patches.py \
        --precomputed-real-dist data/precomputed/02_default_real_dist.pickle \
        --precomputed-real-compl data/precomputed/02_default_real_compl.pickle

产物：output/02/<experiment-id>/combined_dist_compl.parquet
每一行是「某图的某个 patch」的 (距离, 复杂度)，对应散点图里的一个点。

关键实现细节：--patch-size/--patch-stride 是在**原图**分辨率上定义的
（默认 128/64），但 LPIPS 第 2 层的特征图已经下采样了 4 倍，
所以下面组合时要按 factor 把 patch 尺寸换算到特征图分辨率。
"""

import argparse
from pathlib import Path

import pandas as pd
from aeroblade.high_level_funcs import compute_complexities, compute_distances
from aeroblade.image import extract_patches
from aeroblade.misc import safe_mkdir, write_config


def main(args):
    output_dir = Path("output/02") / args.experiment_id
    safe_mkdir(output_dir)
    write_config(vars(args), output_dir)

    # compute distances, eventually load precomputed distances for real images
    # 注意与 01_detect.py 的差别：这里用 --dirs（真实图只是列表里的一项，
    # 需要从列表里 remove 掉），而 01 用的是 real-dir + fake-dirs 两个参数。
    # .copy() 是必要的：否则 remove 会改到 argparse 的默认列表对象，
    # 第二次复用 dirs 时真实图就凭空消失了。
    if args.precomputed_real_dist is not None:
        dirs = args.dirs.copy()
        dirs.remove(Path("data/raw/real"))
    else:
        dirs = args.dirs.copy()
    distances = compute_distances(
        dirs=dirs,
        transforms=args.transforms,
        repo_ids=args.repo_ids,
        distance_metrics=args.distance_metrics,
        amount=args.amount,
        reconstruction_root=args.reconstruction_root,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # spatial=True：保留空间维度，得到每个 patch 的分数而不是整图一个数。
        spatial=True,
    )
    if args.precomputed_real_dist is not None:
        distances = pd.concat([distances, pd.read_pickle(args.precomputed_real_dist)])
    # 统一 dir 的路径分隔符为 posix 风格。生成图的目录由本次运行的 WindowsPath
    # 拼出（反斜杠），预计算的 pickle 则来自 Linux（正斜杠）；不归一会导致下面
    # 用 row.dir 匹配时对不上，而且下游 paper.get_nice_name() 认的是 "data/raw/"。
    distances["dir"] = distances["dir"].astype(str).str.replace("\\", "/", regex=False)

    # compute complexities, eventually load precomputed complexities for real images
    # 复杂度同样要从 dirs 里排除真实图（若用预计算结果）。
    if args.precomputed_real_compl is not None:
        dirs = args.dirs.copy()
        dirs.remove(Path("data/raw/real"))
    else:
        dirs = args.dirs.copy()
    complexities = compute_complexities(
        dirs=dirs,
        transforms=args.transforms,
        complexity_metrics=args.complexity_metrics,
        amount=args.amount,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.precomputed_real_compl is not None:
        complexities = pd.concat(
            [complexities, pd.read_pickle(args.precomputed_real_compl)]
        )
    # 同 distances：两批数据的分隔符风格不同，统一成 posix 才能按 row.dir 匹配。
    complexities["dir"] = (
        complexities["dir"].astype(str).str.replace("\\", "/", regex=False)
    )

    def _combine_distance_and_complexity(
        row: pd.Series, complexity_metric: str
    ) -> pd.Series:
        """把一行「整图距离张量」重新切块，并与该图对应的复杂度拼到一行。"""
        # row.distance 是 (1, 1, h, w) 的特征图分辨率距离（lpips_vgg_2 为 h=w=H/4）。
        # factor = 原图边长 / 特征图边长，用来把 128/64 换算成特征图上的 32/16。
        factor = row.image_size // row.distance.shape[-1]
        patches = extract_patches(
            array=row.distance[None],
            size=args.patch_size // factor,
            stride=args.patch_stride // factor,
        )
        # 结果形状 (1, num_patches, 1, ps, ps)：对通道与空间取平均得到每个
        # patch 一个标量。再取负号还原成「距离」（compute_distances 里统一
        # 取过负，这里为了和复杂度的语义一致改回正值，即越小越像生成图）。
        patch_distances = -patches.mean(axis=(2, 3, 4)).flatten()  # back to positive
        # 复杂度是预先按同一套 patch_size/stride 算好的，直接查出来用。
        # 两边 patch 数量必须一致，否则散点图会错位（见下面 to_parquet 前的检查思路）。
        patch_complexities = complexities.query(
            "dir == @row.dir and file == @row.file and transform == @row['transform'] and complexity_metric == @complexity_metric"
        )["complexity"].item()
        # 用 concat 保留原行的其他列（dir/repo_id/transform/...），
        # 只把 distance 换成 patch 级的、并加上 complexity。
        out = pd.Series(
            [complexity_metric, patch_distances, patch_complexities],
            index=["complexity_metric", "distance", "complexity"],
        )
        return pd.concat([row.drop("distance"), out])

    # combine distances and complexities (of patches) to new dataframe
    # 每个复杂度指标的 patch 网格相同，但列名要区分，所以按指标循环一次。
    combined = []
    for complexity_metric in args.complexity_metrics:
        combined.append(
            distances.apply(
                _combine_distance_and_complexity,
                axis=1,
                complexity_metric=complexity_metric,
            )
        )
    combined = pd.concat(combined)

    # store result
    # 注意这里没有 "file"：因为一个 file 现在有多行（每个 patch 一行）。
    categoricals = [
        "dir",
        "image_size",
        "repo_id",
        "transform",
        "distance_metric",
        "complexity_metric",
    ]
    combined[categoricals] = combined[categoricals].astype("category")
    combined.to_parquet(output_dir / "combined_dist_compl.parquet")

    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-id", default="default")

    # images
    parser.add_argument("--precomputed-real-dist", type=Path)
    parser.add_argument("--precomputed-real-compl", type=Path)
    # 与 01_detect.py 不同：这里真实图是列表的第一项。
    parser.add_argument(
        "--dirs",
        type=Path,
        nargs="+",
        default=[
            Path("data/raw/real"),
            Path("data/raw/generated/CompVis-stable-diffusion-v1-1-ViT-L-14-openai"),
            Path("data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai"),
            Path(
                "data/raw/generated/stabilityai-stable-diffusion-2-1-base-ViT-H-14-laion2b_s32b_b79k"
            ),
            Path(
                "data/raw/generated/kandinsky-community-kandinsky-2-1-ViT-L-14-openai"
            ),
            Path("data/raw/generated/midjourney-v4"),
            Path("data/raw/generated/midjourney-v5"),
            Path("data/raw/generated/midjourney-v5-1"),
        ],
    )
    parser.add_argument("--amount", type=int)
    parser.add_argument("--transforms", nargs="*", default=["clean"])

    # autoencoder
    # 与 01 一样是三个 AE（历史默认值；本脚本只用一个也没问题）。
    parser.add_argument(
        "--repo-ids",
        nargs="+",
        default=[
            "CompVis/stable-diffusion-v1-1",
            "stabilityai/stable-diffusion-2-base",
            "kandinsky-community/kandinsky-2-1",
        ],
    )

    # distance
    # 注意默认是 lpips_vgg_2（单层第 2 层），而不是 01 里的 -1：
    # 这里要的是「论文选出的最佳单层」的空间热力图。
    parser.add_argument(
        "--distance-metrics",
        nargs="+",
        default=[
            "lpips_vgg_2",
        ],
    )

    # complexity
    # jpeg_50：质量 50 的 JPEG 每像素字节数。
    parser.add_argument(
        "--complexity-metrics",
        nargs="+",
        default=[
            "jpeg_50",
        ],
    )
    # patch 尺寸在原图分辨率下定义；128 的窗口、64 的步长（重叠一半）。
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-stride", type=int, default=64)

    # technical
    parser.add_argument(
        "--reconstruction-root", type=Path, default="data/reconstructions"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
