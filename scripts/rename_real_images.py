"""Rename LAION-5B images s.t. filenames correspond to prompt IDs.

中文说明（新增，英文原文保留）
================================
img2dataset 下载 LAION-5B 时是按「分片 + 分片内序号」命名的（例如
tmp/laion/00000/00000000.png、00000001.png……），而 prompt CSV 和生成图是以
9 位 prompt ID 命名的。本脚本按 metadata parquet 的索引把序号翻译成 prompt ID，
重命名后真实图片的文件名就与 prompt / 生成图对齐了。

为什么必须对齐文件名
--------------------
AEROBLADE 评测要把「真实图 + 用同一 prompt 生成的假图」配成一组，
aeroblade/distances.py 的 Distance.compute 是直接比较两个数据集的**文件名**
来配对的（不一致会直接报错），所以真图必须和生成图同名。

什么时候运行
------------
只在准备真实图像数据集时运行一次，位于 README 的 img2dataset 命令之后、
运行 experiments/01_detect.py 之前（见 README「Real Images」一节）。

读取
----
--metadata-path 指向的 parquet（行内容为图片 URL / 文本等元信息，索引为 prompt ID），
以及 --image-dir 下 img2dataset 产出的 *.png。

写出
----
重命名后的图片，落在 --output-dir（默认 data/raw/real），文件名为 9 位零填充的 prompt ID。
"""

import argparse
from pathlib import Path

import pandas as pd
from aeroblade.misc import safe_mkdir
from tqdm import tqdm


def main(args):
    safe_mkdir(args.output_dir)
    # parquet 每行对应一张 LAION 图；它的 index 保存的是 9 位 prompt ID，
    # 而 img2dataset 写盘用的是行序，两者靠本脚本建立映射。
    metadata = pd.read_parquet(args.metadata_path)
    # 只遍历 .png：README 里的 img2dataset 命令用了 --encode_format "png"，
    # 且 img2dataset 的下载顺序与 parquet 行序一致，所以 sorted 后序号即行号。
    image_files = sorted(args.image_dir.glob("*.png"))
    for file in tqdm(image_files, desc="Renaming files"):
        # 文件名（不含扩展名）是分片内的序号，例如 00000000.png → 0。
        # 注意：这只在第一个分片（00000）成立；若有多分片，需要再加上分片偏移量。
        idx = int(file.stem)
        new_idx = metadata.index[idx]
        # Path.rename 是「移动」而不是复制：文件会从 tmp/laion 里消失，
        # 因此重复运行前要先确认原目录还在（或重新下载）。
        # :09 是 9 位零填充，与 prompt CSV 的 image_id、生成图文件名保持一致。
        file.rename(args.output_dir / f"{new_idx:09}.png")


def parse_args():
    parser = argparse.ArgumentParser()
    # --metadata-path：真实图像元数据 parquet，默认 data/raw/real/real_metadata.parquet
    # （README 说明需从 Zenodo 下载；其 index 即 prompt ID）。
    parser.add_argument(
        "--metadata-path", type=Path, default="data/raw/real/real_metadata.parquet"
    )
    # --image-dir：img2dataset 的输出分片目录，默认 tmp/laion/00000（第一个分片）。
    parser.add_argument("--image-dir", type=Path, default="tmp/laion/00000")
    # --output-dir：重命名后的图片落点，默认 data/raw/real，正是 experiments/01_detect.py
    # 的 --real-dir 默认值，因此重命名完即可直接跑检测实验。
    parser.add_argument("--output-dir", type=Path, default="data/raw/real")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
