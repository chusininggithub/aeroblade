"""图像复杂度指标。

论文用「固定质量下 JPEG 编码后的每像素字节数」来衡量一张图有多复杂
（纹理越多、越不可压缩 → 值越大）。5.4 节用它画重建误差 vs 复杂度的
散点图，说明：生成图和真实图的复杂度分布重叠，但重建误差仍然可分——
即 AEROBLADE 不是靠「生成图更简单/更平滑」这种捷径在判别。
"""

import abc
from pathlib import Path
from typing import Any, Optional

import torch
from joblib.memory import Memory
from torch.utils.data import DataLoader
from torchvision.io import encode_jpeg
from torchvision.transforms.v2.functional import convert_image_dtype
from tqdm import tqdm

from aeroblade.data import ImageFolder
from aeroblade.image import extract_patches

# 与 distances.py 同样的模式：把纯函数的结果缓存到本地 cache/ 目录。
# 复杂度计算要跑上千张图的 JPEG 编解码，重复调参时缓存能省下大量时间。
mem = Memory(location="cache", compress=("lz4", 9), verbose=0)


class Complexity(abc.ABC):
    """Base class for all complexity metrics."""

    @torch.no_grad()
    def compute(self, ds: ImageFolder) -> tuple[dict[str, torch.Tensor], list[str]]:
        """
        Compute complexity of dataset.
        """

        # 与 Distance.compute 一样：返回「指标字典 + 文件名列表」，
        # 文件名用于把所有指标对齐到同一行 CSV。
        files = [Path(f).name for f in ds.img_paths]
        result = self._compute(ds=ds)
        return self._postprocess(result), files

    @abc.abstractmethod
    def _compute(self, ds: ImageFolder) -> Any:
        """Metric-specific computation."""
        pass

    @abc.abstractmethod
    def _postprocess(self, result: Any) -> dict[str, torch.Tensor]:
        """Post-processing step, that maps result into dictionary."""
        pass


# 被缓存的是这个模块级纯函数，而不是类方法：joblib 只对可哈希的输入做键，
# 方法上的 self 往往不可哈希，所以距离/复杂度都是这个写法。
# num_workers 不影响结果，因此从缓存键里忽略掉。
@mem.cache(ignore=["num_workers"])
def _compute_jpeg(
    ds: ImageFolder, quality: int, patch_size: int, patch_stride: int, num_workers: int
) -> torch.Tensor:
    # batch_size 固定为 1：每张图的 patch 数量可能不同（图像尺寸不同），
    # 无法拼成一个规则 batch。
    dl = DataLoader(ds, batch_size=1, num_workers=num_workers)

    image_results = []
    for tensor, _ in tqdm(dl, desc="Computing JPEG complexity", total=len(dl)):
        if patch_size is None:
            # 不做分块：整张图算一个「patch」（全局复杂度）。
            patches = [tensor[0]]
        else:
            # 分块：得到局部复杂度，用于 5.4 节的空间对比图。
            patches = extract_patches(
                array=tensor, size=patch_size, stride=patch_stride
            )[0]

        patch_results = []
        for patch in patches:
            # 编码后的字节数就是「这张（块）图有多难压」的度量。
            nbytes = len(
                encode_jpeg(convert_image_dtype(patch, torch.uint8), quality=quality)
            )
            patch_results.append(nbytes)
        # float16 是为了省内存：patch 数量可能上万。
        image_results.append(torch.tensor(patch_results, dtype=torch.float16))
    # 除以像素数做面积归一化，否则大图必然字节更多，指标就退化成「图像尺寸」。
    # 注：patch 是循环变量泄漏到这里的，即最后处理的那个 patch 的形状；
    # 同一批图的 patch 尺寸一致，所以这里等价于用统一的分块大小归一化。
    return torch.stack(image_results) / (patch.shape[1] * patch.shape[2])  # normalize


class JPEG(Complexity):
    def __init__(
        self,
        quality: int = 50,
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None,
        num_workers: int = 0,
    ) -> None:
        """
        quality: JPEG quality to use
        """
        # 默认 50：质量越低，字节数对纹理复杂度的敏感度越高（区分度更好）。
        self.quality = quality
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_workers = num_workers

    def _compute(self, ds: ImageFolder) -> Any:
        return _compute_jpeg(
            ds=ds,
            quality=self.quality,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            num_workers=self.num_workers,
        )

    def _postprocess(self, result: Any) -> dict[str, torch.Tensor]:
        # 指标名带上质量参数，这样同一张图跑多个质量时 CSV 里不会互相覆盖。
        return {f"jpeg_{self.quality}": result}


def complexity_from_config(
    config: str, patch_size: int, patch_stride: int, batch_size: int, num_workers: int
) -> Complexity:
    """Parse config string and return matching complexity metric."""
    # 目前只实现了 JPEG 复杂度，配置形如 "jpeg_50"。
    # batch_size 参数保留是为了和 distance_from_config 签名一致（当前未使用）。
    if config.startswith("jpeg"):
        _, quality = config.split("_")
        return JPEG(
            quality=int(quality),
            patch_size=patch_size,
            patch_stride=patch_stride,
            num_workers=num_workers,
        )
    else:
        raise NotImplementedError(f"No matching complexity metric for {config}.")
