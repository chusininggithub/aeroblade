"""
Evaluate deeper reconstructions (more than just AE).

===========================================================================
论文 5.5 节「Using Deeper Reconstructions」的入口。

前面所有实验都只用 AE 往返（一次 encode→decode）。这里改为「DDIM 反演
k 步 → 再退回 k 步」的更深重建，研究一个问题：如果重建本身变得更粗糙
（失真更大、更「不像原图」），AEROBLADE 还能不能区分真实与生成？
结论是能，而且 k 越大差距往往越明显。

跑法（README）：
    python experiments/03_deeper_reconstructions.py --experiment-id deeper_sd15 \
        --real-dir data/raw/real \
        --fake-dir data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai \
        --repo-id runwayml/stable-diffusion-v1-5
    # SD2.1 同理，换 --experiment-id/--fake-dir/--repo-id 即可
    # 预计算真实图距离：--precomputed-real-dist data/precomputed/03_deeper_sd15_real_dist.pickle

产物：output/03/<experiment-id>/detection_results.csv
（按 k = num_reconstruction_steps 给出 AP 与 TPR@5%FPR。）

注意：每个 k 都是一次完整的扩散往返（比 01 的纯 AE 慢几十倍），
而且结果按参数哈希缓存到 --reconstruction-root（默认
data/deeper_reconstructions），第二次跑同一 k 直接读盘。
"""

import argparse
from pathlib import Path

import pandas as pd
from aeroblade.data import ImageFolder
from aeroblade.distances import distance_from_config
from aeroblade.evaluation import tpr_at_max_fpr
from aeroblade.image import compute_deeper_reconstructions
from aeroblade.misc import safe_mkdir, write_config
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def main(args):
    output_dir = Path("output/03") / args.experiment_id
    safe_mkdir(output_dir)
    write_config(vars(args), output_dir)

    distances = []

    # 与 01 同样的取舍：真实图可用预计算结果时就不再遍历真实目录。
    if args.precomputed_real_dist is not None:
        dirs = [args.fake_dir]
    else:
        dirs = [args.real_dir, args.fake_dir]

    # compute distances
    # 这里没有沿用 high_level_funcs.compute_distances，因为它不含「深重建」
    # 这一维；所以本脚本自己写循环：目录 × 重建步数 × 距离指标。
    for dir in dirs:
        # 进度条按「几个 k」计数（每个 k 都很贵，按 k 显示更直观）。
        pbar = tqdm(
            desc="PROGRESS (deeper_reconstructions)",
            total=len(args.num_reconstruction_steps),
        )
        ds = ImageFolder(dir, amount=args.amount)

        # iterate over number of reconstruction steps
        # k 从小到大：k=1 最接近纯 AE 重建，k=50 是一整轮扩散。
        for num_rec in args.num_reconstruction_steps:
            rec_paths = compute_deeper_reconstructions(
                ds,
                repo_id=args.repo_id,
                output_root=args.reconstruction_root,
                num_inference_steps=args.num_inference_steps,
                num_reconstruction_steps=num_rec,
            )
            # 重建图文件名与原图一致（stem 保留），所以距离计算能自动配对。
            ds_rec = ImageFolder(rec_paths)

            # iterate over distance metrics
            for dist_metric in args.distance_metrics:
                # 这里显式 spatial=False：整图一个分数，不需要 patch 级结果。
                dist_dict, files = distance_from_config(
                    dist_metric,
                    spatial=False,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                ).compute(
                    ds_a=ds,
                    ds_b=ds_rec,
                )
                # 与 compute_distances 一样：一个配置可能返回多层，
                # 逐层各存一行。注意这里没做「多 AE 取 max」——本实验
                # 只研究单个 AE 的重建深度影响。
                for dist_name, dist_tensor in dist_dict.items():
                    dist_tensor = dist_tensor.squeeze(1, 2, 3)
                    df = pd.DataFrame(
                        {
                            "dir": str(dir),
                            "num_reconstruction_steps": num_rec,
                            "distance_metric": dist_name,
                            "file": files,
                            "distance": list(dist_tensor.numpy()),
                        }
                    )
                    distances.append(df)
                pbar.update()

    distances = pd.concat(distances)

    # load precomputed real distances
    # 预计算文件必须是「用同样的 repo_id / k / 指标」算出来的，
    # 否则这里的拼接会静默产生错误结论。
    if args.precomputed_real_dist is not None:
        distances = pd.concat([distances, pd.read_pickle(args.precomputed_real_dist)])

    # compute detection results
    # 分组维度少了 AE（本脚本只有一个 repo_id），多了重建步数 k。
    detection_results = []
    for (num_rec, dist_metric), group_df in distances.groupby(
        ["num_reconstruction_steps", "distance_metric"], sort=False
    ):
        y_score_real = group_df.query("dir == @args.real_dir.__str__()").distance.values
        y_score_fake = group_df.query("dir == @args.fake_dir.__str__()").distance.values
        y_score = y_score_real.tolist() + y_score_fake.tolist()
        y_true = [0] * len(y_score_real) + [1] * len(y_score_fake)
        ap = average_precision_score(y_true=y_true, y_score=y_score)
        tpr5fpr = tpr_at_max_fpr(y_true=y_true, y_score=y_score, max_fpr=0.05)
        detection_results.append(
            {
                # 字符串化，避免 Path 对象进 CSV 时带平台相关前缀。
                "fake_dir": str(args.fake_dir),
                "num_reconstruction_steps": num_rec,
                "distance_metric": dist_metric,
                "ap": ap,
                "tpr5fpr": tpr5fpr,
            }
        )
    pd.DataFrame(detection_results).to_csv(output_dir / "detection_results.csv")

    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()

    # 默认是 sd15：README 里两条命令分别用 deeper_sd15 / deeper_sd21。
    parser.add_argument("--experiment-id", default="sd15")

    # images
    parser.add_argument("--precomputed-real-dist", type=Path)
    parser.add_argument("--real-dir", type=Path, default=Path("data/raw/real"))
    # 只有一个生成数据集（一次研究一个「生成模型 vs 真实图」的对比）。
    parser.add_argument(
        "--fake-dir",
        type=Path,
        default=Path(
            "data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai"
        ),
    )
    # 默认只取 250 张：每个 k 都要跑完整扩散往返，全量太慢。
    parser.add_argument("--amount", type=int, default=250)

    # reconstruction
    # --repo-id 这里是完整的 diffusers 模型 id（要拿它的 UNet 做反演），
    # 与 01/02 里 --repo-ids 的用法一致但用途不同：01/02 只取 VAE。
    parser.add_argument(
        "--repo-id",
        default="runwayml/stable-diffusion-v1-5",
    )
    # 采样总步数；k 必须 <= 它。
    parser.add_argument("--num-inference-steps", type=int, default=50)
    # 要评估的 k 值列表：1,2,4,8,16,32,50（指数间隔，覆盖从浅到最深）。
    parser.add_argument(
        "--num-reconstruction-steps",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 50],
    )

    # distance
    # -1：把所有层都算出来（notebook 里再挑最佳层）。
    parser.add_argument(
        "--distance-metrics",
        nargs="+",
        default=[
            "lpips_vgg_-1",
        ],
    )

    # technical
    # 注意根目录与 01/02 不同（data/deeper_reconstructions），
    # 避免把两种重建结果混在一起。
    parser.add_argument(
        "--reconstruction-root", type=Path, default="data/deeper_reconstructions"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
