"""鲁棒性实验用的图像扰动。

论文 5.5 节要检验「生成图被压缩/裁剪/加噪后，检测器还灵不灵」，这里的
四个变换就是那四类扰动。所有变换都同时接受 PIL 图和张量（张量约定取值
[0,1]，与 data.ImageFolder 的输出一致），并且都在原地保持尺寸不变——
因为检测要求原图与重建图逐像素可比。

配置字符串的格式是 `<名字>_<参数>`，例如 jpeg_90、blur_1.0、crop_0.8、
noise_0.15，由 transform_from_config 解析。
"""

from typing import Any, Callable, List, Sequence, Union

import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as tf
from PIL import Image
from torchvision.io import decode_jpeg, encode_jpeg


def transform_from_config(config: str) -> Callable:
    """Parse config string and return matching transform."""
    # 只按第一个下划线切分：参数本身可能带小数点（如 "blur_1.0"）。
    name, param = config.split("_")
    try:
        param = int(param)
    except ValueError:
        # 整数质量/档位用 int，sigma/factor 这类用 float。
        param = float(param)
    # 用类名小写匹配，省掉一张手工维护的名字映射表。
    for transform in [JPEG, Crop, Blur, Noise]:
        if name == transform.__name__.lower():
            return transform(param)
    else:
        # 注意：for 后面跟 else 表示「循环正常走完都没 return」才执行。
        raise NotImplementedError(f"No matching transform for {config}.")


class JPEG:
    """JPEG 重压缩。quality 越小压缩越狠，生成图的高频伪影越容易被抹掉。"""

    def __init__(
        self,
        quality: int | tuple[int, int],
    ) -> None:
        # 允许传区间，从而在区间内随机采样——避免「只测了 q=90 这一个点」
        # 的过拟合式结论。
        if isinstance(quality, int):
            quality = (quality, quality)
        self.quality = quality

    def _get_params(self, quality_min: int, quality_max: int) -> int:
        # high 是开区间，所以 +1 才能让 quality_max 有机会被取到。
        return torch.randint(low=quality_min, high=quality_max + 1, size=())

    def __call__(
        self, img: Union[torch.Tensor, Image.Image]
    ) -> Union[torch.Tensor, Image.Image]:
        quality = self._get_params(
            quality_min=self.quality[0], quality_max=self.quality[1]
        )
        t_img = img
        if not isinstance(img, torch.Tensor):
            if not F._is_pil_image(img):
                raise TypeError(f"img should be PIL Image or Tensor. Got {type(img)}")
            t_img = F.to_tensor(img)
        # 编解码只在 uint8 上有意义，所以先把 [0,1] 浮点还原成 0-255 整型。
        t_img = F.convert_image_dtype(t_img, torch.uint8)

        # 真正走一遍 JPEG 编解码，而不是用近似滤波模拟——这样才复现了
        # 真实社交平台/转发链路上的压缩伪影。
        output = decode_jpeg(encode_jpeg(t_img, quality=quality))
        output = F.convert_image_dtype(output)

        # 输入是 PIL 就还回 PIL，保持调用方类型不变。
        if not isinstance(img, torch.Tensor):
            output = F.to_pil_image(output, mode=img.mode)
        return output


class Crop:
    """中心裁剪后再缩放回原尺寸（相当于数字变焦）。

    factor=0.8 表示先裁掉外围 20%，再放大回来——这会破坏生成图原有的
    噪声/纹理统计，是四类扰动里最能削弱检测的一类。
    """

    def __init__(
        self,
        factor: float | tuple[float, float],
    ) -> None:
        if isinstance(factor, float):
            factor = (factor, factor)
        self.factor = factor

    def _get_params(self, factor_min: float, factor_max: float) -> float:
        # 不用 torch.rand 是因为需要指定区间且要拿到 python float。
        return torch.empty(1).uniform_(factor_min, factor_max).item()

    def __call__(
        self, img: Union[torch.Tensor, Image.Image]
    ) -> Union[torch.Tensor, Image.Image]:
        factor = self._get_params(factor_min=self.factor[0], factor_max=self.factor[1])
        # 张量是 CHW、PIL 是 WH，两种取尺寸方式不同。
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        cropped_width, cropped_height = (
            round(width * factor),
            round(height * factor),
        )
        cropped = F.center_crop(img, output_size=(cropped_height, cropped_width))
        # 必须缩放回去：下游要求同目录所有图尺寸一致，否则无法成 batch。
        resized = F.resize(cropped, size=(height, width))
        return resized


class Blur(tf.GaussianBlur):
    """高斯模糊。sigma 越大越糊；kernel_size=9 足够覆盖论文用到的 sigma 范围。"""

    def __init__(
        self,
        sigma: float | tuple[float, float],
        kernel_size: float = 9,
    ) -> None:
        super().__init__(kernel_size=kernel_size, sigma=sigma)


class Noise:
    """加高斯白噪声（在 [0,1] 像素值上加，再截断回 [0,1]）。"""

    def __init__(self, std: float | tuple[float]) -> None:
        if isinstance(std, float):
            std = (std, std)
        self.std = std

    def _get_params(self, std_min: float, std_max: float) -> float:
        return torch.empty(1).uniform_(std_min, std_max).item()

    def __call__(
        self, img: Union[torch.Tensor, Image.Image]
    ) -> Union[torch.Tensor, Image.Image]:
        std = self._get_params(std_min=self.std[0], std_max=self.std[1])
        t_img = img
        if not isinstance(img, torch.Tensor):
            if not F._is_pil_image(img):
                raise TypeError(f"img should be PIL Image or Tensor. Got {type(img)}")
            t_img = F.to_tensor(img)

        # 加噪后再 clamp：越界值直接截掉，等价于饱和，而不是回绕。
        output = torch.clamp(t_img + torch.randn(t_img.size()) * std, min=0.0, max=1.0)

        if not isinstance(img, torch.Tensor):
            output = F.to_pil_image(output, mode=img.mode)
        return output


class RandomChoiceN(tf.RandomChoice):
    """随机选 n 个变换并顺序叠加（论文里的组合扰动，如 jpeg+blur+noise）。"""

    def __init__(
        self,
        transforms: Sequence[Callable[..., Any]],
        p: List[float] | None = None,
        n: int = 1,
    ) -> None:
        super().__init__(transforms, p)
        self.n = n

    def forward(self, *inputs: Any) -> Any:
        # 与父类的区别：父类只抽 1 个，这里不放回地抽 n 个再 Compose 起来。
        indices = torch.multinomial(torch.tensor(self.p), self.n)
        transform = tf.Compose([self.transforms[idx] for idx in indices])
        return transform(*inputs)
