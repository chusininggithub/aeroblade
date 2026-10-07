"""Create AE reconstructions.

中文说明（新增，英文原文保留）
================================
把某个目录下的图片用指定的 LDM 自带的 VAE 全部重建一遍并存成 png，是一个
「数据准备 / 调试」用的小脚本：重建结果落盘后可被实验脚本反复复用，
从而避免每次都重算重建（AEROBLADE 的主要开销就在重建上）。

什么时候运行
------------
调试单个 (图片目录, AE) 组合的重建质量时运行，默认值是 debug/real → debug/reconstructions
这样的最小配置；正式复现论文请用 experiments/01_detect.py，它会调用
high_level_funcs.compute_distances，内部走同一个 compute_reconstructions。

命名约定
--------
输出目录名是 f"{图片目录名}-{AE 仓库名(斜杠换成短横线)}"，例如
data/raw/generated/CompVis-stable-diffusion-v1-1-ViT-L-14-openai 表示
「用 SD1.1 生成、prompt 由 ViT-L-14/openai 抽取」。这里沿用同一套命名习惯，
让目录名本身就记录了用的是哪个 AE，便于和生成数据一一对照。

读取
----
--dir 指定目录下的图片（.png / .jpg / .jpeg / .webp，由 aeroblade.data.ImageFolder 筛选）。

写出
----
<output-root>/<目录名>-<AE 名>/ 下的重建图 png，文件名与原图 stem 保持一致。
"""

import argparse
from pathlib import Path

from aeroblade.data import ImageFolder
from aeroblade.image import compute_reconstructions


def main(args):
    # 输出目录名 = 图片目录名 + AE 仓库名（"/" → "-"，因为 "/" 不能做目录名）。
    # 这样 data/raw/real 用 SD1.1 重建就落在 reconstructions/real-CompVis-stable-diffusion-v1-1，
    # 一眼能看出配置来源。
    output_dir = args.output_root / f'{args.dir.name}-{args.repo_id.replace("/", "-")}'

    # ImageFolder 会把目录里的图片按扩展名筛选并读成 float32 的 [0,1] 张量；
    # 同一目录内图片尺寸必须一致，否则无法组成 batch（AEROBLADE 要求整目录同分辨率）。
    ds = ImageFolder(args.dir)

    # 与 compute_distances 内部调用的是同一个函数。这里显式传了 output_dir，
    # 所以不会走「按参数哈希生成子目录」的分支——那条分支是留给实验脚本做缓存复用的。
    # 重建流程：归一化到 [-1,1] → VAE encode 采样隐变量 → decode → 反归一化到 [0,1] → 存 png。
    compute_reconstructions(
        ds,
        repo_id=args.repo_id,
        output_dir=output_dir,
        # seed 默认 1，与 experiments/ 保持一致：VAE 编码时隐变量是从对角高斯里
        # 采样出来的，种子不同重建就不同，固定住才能复现论文数字。
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    # --dir：待重建的图片目录。默认 debug/real 是调试用的小样本目录。
    parser.add_argument("--dir", type=Path, default="debug/real")
    # --repo-id：用哪个 LDM 的 VAE 做重建，默认 SD1.1。
    parser.add_argument("--repo-id", default="CompVis/stable-diffusion-v1-1")
    # --output-root：输出根目录，真正的重建图在其下的 <目录名>-<AE 名>/ 里。
    parser.add_argument("--output-root", type=Path, default="debug/reconstructions")
    # --seed：隐变量采样种子，默认 1（与论文实验一致）。
    parser.add_argument("--seed", type=int, default=1)
    # --batch-size / --num-workers：批大小与加载进程数，显存不足时把 batch-size 调小。
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
