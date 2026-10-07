"""数据集读取。

统一把「一堆图片路径」包装成 torch Dataset：输入既可以是目录也可以是
文件列表，输出 (张量图像, 路径字符串) 二元组。路径字符串很重要，因为
距离计算要靠文件名把「原图」和「重建图」配对。
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
import torchvision.transforms.v2 as tf
from PIL import Image
from torchvision.datasets import VisionDataset

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


class ImageFolder(VisionDataset):
    """
    Dataset for reading images from a list of paths, directories, or a mixture of both.
    """

    def __init__(
        self,
        paths: Union[list[Path], Path],
        # 默认变换：转成张量并把像素归一到 [0,1] 的 float32
        #（ToDtype(scale=True) 负责 /255）。这是各距离指标的输入约定。
        # 注意这里是在默认参数里构造对象，属于本仓库的原样写法，保持不动。
        transform: Optional[Callable] = tf.Compose(
            [tf.ToImage(), tf.ToDtype(torch.float32, scale=True)]
        ),
        amount: Optional[int] = None,
    ) -> None:
        # 允许直接传单个 Path，内部统一成列表处理。
        self.paths = [paths] if isinstance(paths, Path) else paths
        self.transform = transform
        self.amount = amount

        # 目录 → 展开成文件列表；文件 → 直接加入。目录与文件可以混着传。
        self.img_paths = []
        for path in self.paths:
            if path.is_dir():
                for file in read_files(path):
                    if file.suffix.lower() in IMG_EXTENSIONS:
                        self.img_paths.append(file)
                        # amount 用于「只用前 N 张」的子集实验；一旦凑够就
                        # 跳出，保证同一目录下取到的总是同一批图（可复现）。
                        if (
                            self.amount is not None
                            and len(self.img_paths) == self.amount
                        ):
                            break
            else:
                self.img_paths.append(path)

        # 明确报错而不是静默少用几张图：真实图像子集一旦不足，实验结果
        # 会被悄悄改写，而论文要求各数据集样本数一致。
        if self.amount is not None and len(self.img_paths) < self.amount:
            raise ValueError("Number of images is less than 'amount'.")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[str, float]]:
        # 强制 RGB：生成数据集里可能混有灰度图或带 alpha 的 PNG，
        # 通道数不一致会让 VAE 编码直接崩。
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # 返回路径字符串而不是索引，是为了让下游按文件名做配对和缓存。
        return img, str(self.img_paths[idx])

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Paths: {self.paths}")
        body.append(f"Transform: {repr(self.transform)}")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


def read_files(path: Path) -> list[Path]:
    # 排序是为了让遍历顺序确定：重建输出的文件名、缓存哈希、CSV 行序
    # 都依赖这个顺序，乱序会让实验结果无法逐位复现。
    return sorted(path.iterdir())
