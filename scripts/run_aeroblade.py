"""Compute the AE reconstruction distance.

中文说明（新增，英文原文保留）
================================
本脚本是 AEROBLADE 的开箱即用入口（对应 README 的 Quickstart）：给定若干图片或
若干图片目录，用 LDM（Stable Diffusion / Kandinsky）自带的 VAE 做 encode→decode
重建，再计算「原图 vs 重建图」的距离，输出逐图分数。

为什么重建误差能判断真假图（领域含义）
--------------------------------------
这些 VAE 是在真实图像上训练出来的，真实照片对它的隐空间而言是分布外（OOD）样本，
encode→decode 之后信息损失大、重建误差大；而 LDM 生成的图恰好落在同一个 VAE 的
「舒适区」内，重建误差更小。所以距离越小 → 越可能是 AI 生成图。
注意 aeroblade/distances.py 的 _postprocess 里对距离取了负号，因此最终分数是
「越大越像生成图」，这也是 README 里说「我们保存的是负距离，所以最好的 AE 记为 max」
的原因。

什么时候运行
------------
只想对少量图片快速拿分数时用它；要复现论文表格请走 experiments/01_detect.py，
两者底层调用的都是 high_level_funcs.compute_distances。

读取
----
--files-or-dirs 指定的图片文件或图片目录，默认用仓库自带的 example_images/
（real.png 是真图，其余是各模型的生成图，正好可以对照真假分数差异）。

写出
----
<output-dir>/distances.csv                     逐图、逐 AE、逐距离度量的距离表
<output-dir>/reconstructions/<参数哈希>/1/*.png 各 AE 的重建图。
重建目录名是 compute_reconstructions 对参数做哈希得到的，一旦算过就会被复用，
所以调参重跑不会白白重算一遍重建。
"""

import argparse
from pathlib import Path

from aeroblade.high_level_funcs import compute_distances
from aeroblade.misc import safe_mkdir


def main(args):
    # create output directory
    # 建输出目录。safe_mkdir 在目录已存在时会先交互问一句 (y/n)，
    # 避免把上一轮的 distances.csv / 重建图无声覆盖掉。
    safe_mkdir(args.output_dir)

    # compute distances
    # 一次跑完「变换 × 目录 × AE × 距离度量」的笛卡尔积：
    # 每个 AE 先把所有图重建一遍，再逐张算原图与重建图的距离。
    distances = compute_distances(
        dirs=args.files_or_dirs,
        # 只跑 "clean"，即不做 JPEG 压缩 / 模糊 / 裁剪 / 加噪等扰动。
        # 扰动版本（论文的鲁棒性实验）由 experiments/01_detect.py 的 --transforms 指定。
        transforms=["clean"],
        repo_ids=args.autoencoders,
        distance_metrics=[args.distance_metric],
        # None 表示不限制张数，目录里有多少张就全用；
        # 论文实验用 --amount 截断样本量来控制耗时。
        amount=None,
        # 重建图落在 --output-dir 下的 reconstructions/，而不是实验用的 data/reconstructions。
        reconstruction_root=args.output_dir / "reconstructions",
        # seed 固定为 1，且与 experiments/ 里的默认值一致：VAE 编码时的隐变量采样
        # 依赖随机数生成器，固定住才能让同一张图的分数可复现（对齐论文表格数字）。
        seed=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # save and display results
    # 距离以负数形式落盘（分数越大越像生成图）。表里除了各个 AE 的行，还有
    # repo_id == "max" 的行：它取同一张图在所有 AE 上的最大距离，对应论文中
    # 「多个 AE 取最大」的集成检测策略，也是归属（attribution）实验里判断
    # 「这张图最可能出自哪个生成模型」的依据。
    distances.to_csv(args.output_dir / "distances.csv", index=False)
    print(distances)
    print(f"\nSaving distances to {args.output_dir / 'distances.csv'}.\nDone.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the AE reconstruction distances for images or directories."
    )
    # --files-or-dirs：待检测的图片文件或图片目录，可传多个（nargs="+"）。
    # 同一目录内的所有图片必须尺寸一致，否则无法堆成一个 batch 送给 VAE。
    # 默认值是 example_images/ 下的示例图：real.png 为真实照片，
    # SD1-1 / SD1-5 / SD2-1 / KD2-1 为对应模型生成图，MJ4 / MJ5 / MJ5-1 为 Midjourney。
    parser.add_argument(
        "--files-or-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("example_images/real.png"),
            Path("example_images/SD1-1.png"),
            Path("example_images/SD1-5.png"),
            Path("example_images/SD2-1.png"),
            Path("example_images/KD2-1.png"),
            Path("example_images/MJ4.png"),
            Path("example_images/MJ5.png"),
            Path("example_images/MJ5-1.png"),
        ],
        help="Paths to images or directories containing images. All images in a directory should have the same dimensions.",
    )
    # --output-dir：输出根目录，distances.csv 和 reconstructions/ 子目录都在它下面。
    parser.add_argument(
        "--output-dir", type=Path, default="aeroblade_output", help="Output directory."
    )
    # --autoencoders：HuggingFace 上的 LDM 仓库名，脚本只取其中的 VAE 做重建。
    # 默认三个即论文使用的 SD1.1 / SD2 / KD2.1；换成非默认模型时，
    # aeroblade/image.py 里取 vae（或 VQModel 的 movq）的逻辑可能需要适配。
    parser.add_argument(
        "--autoencoders",
        nargs="+",
        default=[
            "CompVis/stable-diffusion-v1-1",  # SD1
            "stabilityai/stable-diffusion-2-base",  # SD2
            "kandinsky-community/kandinsky-2-1",  # KD2.1
        ],
        help="HuggingFace model name of an LDM to use for reconstruction. Non-default models might need adaptation.",
    )
    # --distance-metric：重建图与原图之间的距离度量，默认 lpips_vgg_2（VGG 第 2 层），
    # 这是论文实验中表现最好的配置。lpips_vgg_0 是原始 LPIPS 定义（所有层求和），
    # 1~5 表示只取对应那一层，-1 表示把所有层都输出成多列 lpips_vgg_i 便于再挑选。
    parser.add_argument(
        "--distance-metric",
        default="lpips_vgg_2",
        choices=[
            "lpips_vgg_0",  # sum of all layers, original LPIPS definition
            "lpips_vgg_1",  # first layer
            "lpips_vgg_2",  # second layer
            "lpips_vgg_3",  # third layer
            "lpips_vgg_4",  # fourth layer
            "lpips_vgg_5",  # fifth layer
            "lpips_vgg_-1",  # returns all layers
        ],
        help="Distance metric to use.",
    )
    # --num-workers / --batch-size：DataLoader 的进程数与批大小，默认都是 1（最省显存）。
    # 显存不够就调小 batch-size，纯 CPU 环境建议保持 1。
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
