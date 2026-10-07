"""
Compute reconstruction distances and detection/attribution results.

===========================================================================
论文 5.2 / 5.5 的主实验入口：检测性能（detection）与来源归属（attribution）。

跑法见 README：
    python experiments/01_detect.py                              # 5.2 主表
    python experiments/01_detect.py --experiment-id robustness --amount 250 \
        --transforms clean jpeg_90 ...                           # 鲁棒性
    python experiments/01_detect.py --experiment-id distance_metric_ablation \
        --distance-metrics lpips_vgg_0 lpips_alex_0 ...          # 距离指标消融

产生两个文件（在 output/01/<experiment-id>/ 下）：
    distances.parquet       —— 每个 (数据集, 图像, 扰动, AE, 距离指标) 一行的长表
    detection_results.csv   —— 每个 (生成数据集, 扰动, AE, 距离指标) 的 AP 与 TPR@5%FPR
    attribution_results.csv —— 各 AE 单独判对、且与 max 一致的样本比例（来源归属）

注意：这里是「一图一分数」的判定（重建距离），与真实图像是否可用预计算
结果是解耦的——LAION 原图版权受限会随时间失效，所以提供
--precomputed-real-dist 直接复用作者算好的真实图距离。
"""

import argparse
from pathlib import Path

import pandas as pd
from aeroblade.evaluation import tpr_at_max_fpr
from aeroblade.high_level_funcs import compute_distances
from aeroblade.misc import safe_mkdir, write_config
from sklearn.metrics import average_precision_score


def main(args):
    # 每个 experiment-id 一个输出目录，方便同一脚本跑多组配置而不互相覆盖。
    output_dir = Path("output/01") / args.experiment_id
    safe_mkdir(output_dir)
    # 保存本次运行的全部参数，便于复现（notebook 也会读它）。
    write_config(vars(args), output_dir)

    # compute distances, eventually load precomputed real distances
    # 有预计算的真实图距离时，就不必（也无法）处理真实图目录。
    if args.precomputed_real_dist is not None:
        dirs = args.fake_dirs
    else:
        dirs = [args.real_dir] + args.fake_dirs

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
    )

    if args.precomputed_real_dist is not None:
        # 预计算结果是一个 pickle 过的 DataFrame，列结构与上面完全一致，
        # 直接纵向拼接即可。
        distances = pd.concat([distances, pd.read_pickle(args.precomputed_real_dist)])

    # store distances
    # 统一 dir 的路径分隔符为 posix 风格。
    # 为什么必须做：生成图的 dir 由本次运行的 WindowsPath 拼出（反斜杠），而
    # --precomputed-real-dist 里的真实图来自作者在 Linux 上生成的 pickle（正斜杠）。
    # 两者直接做字符串比较会全部失配——真实图一张都取不到，AP 会退化成 1.0 而不报错。
    distances["dir"] = distances["dir"].astype(str).str.replace("\\", "/", regex=False)

    # 把几个低基数的列转成 category：parquet 里能显著压缩体积，
    # 后续 groupby 也更快。
    categoricals = [
        "dir",
        "image_size",
        "repo_id",
        "transform",
        "distance_metric",
        "file",
    ]
    distances[categoricals] = distances[categoricals].astype("category")
    distances.to_parquet(output_dir / "distances.parquet")

    # compute detection results
    # 二分类：真实图标签 0，生成图标签 1；分数是取负后的重建距离
    # （越大越像生成图），所以正类分数天然更高，无需再翻转。
    detection_results = []
    for (transform, repo_id, dist_metric), group_df in distances.groupby(
        ["transform", "repo_id", "distance_metric"], sort=False, observed=True
    ):
        # 每个生成数据集单独和同一批真实图比一次（一 vs 一，而不是多分类）。
        # @ 是 pandas query 的「引用外部变量」语法。
        # 用 as_posix()：与上面归一化过的 dir 列保持同一种分隔符。
        y_score_real = group_df.query("dir == @args.real_dir.as_posix()").distance.values
        for fake_dir in args.fake_dirs:
            y_score_fake = group_df.query("dir == @fake_dir.as_posix()").distance.values
            # 真实图在前、生成图在后，标签顺序与分数顺序一致。
            y_score = y_score_real.tolist() + y_score_fake.tolist()
            y_true = [0] * len(y_score_real) + [1] * len(y_score_fake)
            # AP：不依赖阈值，衡量整体排序质量（论文主指标）。
            ap = average_precision_score(y_true=y_true, y_score=y_score)
            # TPR@5%FPR：实际部署更关心的「误报率受限时的召回」。
            tpr5fpr = tpr_at_max_fpr(y_true=y_true, y_score=y_score, max_fpr=0.05)
            detection_results.append(
                {
                    # 同样用 posix：paper.get_nice_name() 是靠 "data/raw/" 这个
                    # 子串来识别数据集的，Windows 的反斜杠会让它全部匹配失败，
                    # 下游 notebook 的表格 pivot 会直接 KeyError。
                    "fake_dir": Path(fake_dir).as_posix(),
                    "transform": transform,
                    "repo_id": repo_id,
                    "distance_metric": dist_metric,
                    "ap": ap,
                    "tpr5fpr": tpr5fpr,
                }
            )
    # stable 排序保证 CSV 行序确定，方便和论文表格逐行核对。
    pd.DataFrame(detection_results).sort_values("fake_dir", kind="stable").to_csv(
        output_dir / "detection_results.csv"
    )

    # compute attribution results
    # 来源归属：不只判断「是不是生成图」，还要猜「是哪个模型生成的」——
    # 哪个 AE 的重建误差最大，就认为图来自对应模型。这里统计「单个 AE 的
    # 判定与多 AE 取 max 的判定一致」的样本比例（即该 AE 是最佳 AE 的概率）。
    attribution_results = []
    for (dir, transform, dist_metric), group_df in distances.groupby(
        ["dir", "transform", "distance_metric"], sort=False, observed=True
    ):
        for repo_id, repo_id_df in group_df.groupby(
            "repo_id", sort=False, observed=True
        ):
            # "max" 是 high_level_funcs 造出来的虚拟行，跳过它本身。
            if repo_id == "max":
                continue
            # 浮点相等比较在这里是安全的：max 行就是由这些值原样取 max 得到的，
            # 没有经过额外的数值运算。
            matches = (
                repo_id_df.distance.values
                == group_df.query("repo_id == 'max'").distance.values
            )
            # 一致率 = 该 AE 就是「最佳 AE」的图像比例。
            fraction = matches.sum() / len(repo_id_df)
            attribution_results.append(
                {
                    "dir": dir,
                    "transform": transform,
                    "distance_metric": dist_metric,
                    "repo_id": repo_id,
                    "fraction": fraction,
                }
            )
    pd.DataFrame(attribution_results).sort_values("dir", kind="stable").to_csv(
        output_dir / "attribution_results.csv"
    )

    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()
    # 一次运行的全部产物都放在 output/01/<experiment-id>/ 下。
    parser.add_argument("--experiment-id", default="default")

    # images
    # 真实图像的预计算距离；给了它就不再需要 data/raw/real（LAION 图可能已失效）。
    parser.add_argument("--precomputed-real-dist", type=Path)
    parser.add_argument("--real-dir", type=Path, default="data/raw/real")
    # 七个生成数据集：4 个开源模型 + 3 个 Midjourney 版本。
    # 目录名里带着生成时使用的 CLIP 模型，因为提示词是用它抽的。
    parser.add_argument(
        "--fake-dirs",
        type=Path,
        nargs="+",
        default=[
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
    # 每个目录只取前 N 张（不指定则全用）。子集实验靠它控制耗时。
    parser.add_argument("--amount", type=int)
    # 默认只跑 clean；鲁棒性实验在这里传一串扰动配置。
    parser.add_argument("--transforms", nargs="*", default=["clean"])
    # 重建结果（按参数哈希分目录）的根目录，也是最大的磁盘占用来源。
    parser.add_argument(
        "--reconstruction-root", type=Path, default="data/reconstructions"
    )

    # autoencoder
    # 三个 AE 分别重建、分数取 max —— 这就是论文的完整检测器。
    # 注意是 AE 所属的模型 id，不是生成这些图的模型。
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
    # 默认 lpips_vgg_-1：vgg 骨干、-1 表示把所有层都算出来（notebook 里再挑）。
    parser.add_argument(
        "--distance-metrics",
        nargs="+",
        default=[
            "lpips_vgg_-1",
        ],
    )

    # technical
    # seed 固定，保证 VAE 隐空间采样与扰动采样可复现。
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
