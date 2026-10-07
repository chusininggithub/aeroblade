"""原图与 AE 重建图之间的距离（= AEROBLADE 的检测分数）。

调用链：experiments/01_detect.py → high_level_funcs.compute_distances
→ distance_from_config(...).compute(原图数据集, 重建图数据集)。

两个约定必须记住：

1. **结果取负号**。论文想要「分数越高 = 越像生成图」，而距离越小才越像
   生成图，所以所有指标在 _postprocess 里统一取负。这就是 README 里说
   「best AE is denoted by max」的原因。
2. **同名文件配对**。原图与重建图是分别两个 ImageFolder，靠文件名一一
   对应；重建时保留了原文件名，所以这里只校验文件名是否对齐。

批量计算被拆成「模块级纯函数 + joblib 缓存」，因为缓存键需要可哈希；
类方法带 self 无法可靠哈希，而纯函数的参数（数据集、指标名、batch_size
等）是可哈希的。
"""

import abc
import json
import warnings
from pathlib import Path
from typing import Any, Optional

import lpips
import pyiqa
import torch
import torch.nn.functional as F
from joblib.memory import Memory
from torch.utils.data import DataLoader
from tqdm import tqdm

from aeroblade.data import ImageFolder
from aeroblade.misc import compile_net, device

# 本地磁盘缓存（cache/ 目录，lz4 压缩）。跑距离指标时动辄要把上千张图过一遍
# 深度网络，缓存是「调参不重复算」的关键；注意缓存目录会变得很大。
mem = Memory(location="cache", compress=("lz4", 9), verbose=0)


class Distance(abc.ABC):
    """Base class for all distance metrics."""

    @torch.no_grad()
    def compute(
        self,
        ds_a: ImageFolder,
        ds_b: ImageFolder,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """
        Compute distance between two datasets with matching filenames.
        """
        # 配对用的是「文件名 + 后缀」，因为重建图统一存成 .png，而原图可能
        # 是 .jpg/.webp，后缀本来就可能不同（下面会给出警告而不是报错）。
        files_a = [Path(f).name for f in ds_a.img_paths]
        files_b = [Path(f).name for f in ds_b.img_paths]
        if files_a != files_b:
            # 退一步只比 stem（不含后缀）。
            files_a_stems = [Path(f).stem for f in ds_a.img_paths]
            files_b_stems = [Path(f).stem for f in ds_b.img_paths]
            if files_a_stems != files_b_stems:
                raise ValueError("ds_a and ds_b should contain matching files.")
            else:
                # stem 相同但后缀不同时，逐位比较仍然是「同序对齐」的，
                # 所以放行；风险是用同一张图的两种格式冒充两张图。
                warnings.warn(
                    "ds_a and ds_b contain files with different file endings. Make sure that is does not cause issues, e.g., you should not have different images with the same name but different file endings."
                )

        result = self._compute(
            ds_a=ds_a,
            ds_b=ds_b,
        )
        return self._postprocess(result), files_a

    @abc.abstractmethod
    def _compute(self, ds_a: ImageFolder, ds_b: ImageFolder) -> Any:
        """Distance-specific computation."""
        pass

    @abc.abstractmethod
    def _postprocess(self, result: Any) -> dict[str, torch.Tensor]:
        """Post-processing step that maps result into dictionary."""
        pass


class _PatchedLPIPS(lpips.LPIPS):
    """Patched version of LPIPS which returns layer-wise output without upsampling.

    为什么要打补丁：官方 lpips 包只返回「各层加权后求和」的标量，而本论文
    需要 (a) 逐层的分数（层数是最好的超参数之一），(b) 保留空间维度的
    patch 级分数（5.4 节的空间热力图）。官方实现在 spatial 模式下会把每层
    上采样到同一分辨率再相加，那样就丢了「这一层专属的粒度」；这里改为
    原样返回未上采样的层输出，需要上采样时由调用方决定。
    """

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if (
            normalize
        ):  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            # 我们的数据管道里像素是 [0,1]，LPIPS 期望 [-1,1]。
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        # version=="0.1" 才需要 scaling_layer（首层 1x1 卷积的遗留 bug 修正）。
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1))
            if self.version == "0.1"
            else (in0, in1)
        )
        # 把两张图分别过骨干网络，取出每个 stage 的特征图。
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            # LPIPS 的关键细节：先按通道做 L2 归一化，再算平方差，
            # 这样比直接算像素差更接近人眼感知。
            feats0[kk], feats1[kk] = (
                lpips.normalize_tensor(outs0[kk]),
                lpips.normalize_tensor(outs1[kk]),
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            # lpips=True：每层差异再过一个可学习的 1x1 卷积（官方预训练权重）。
            if self.spatial:
                # 不上采样，保留各层原始分辨率；res 用于求和，
                # res_no_up 是逐层输出（本次调用真正要的东西）。
                res_no_up = [self.lins[kk](diffs[kk]) for kk in range(self.L)]
                res = [
                    lpips.upsample(res_no_up[kk], out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    lpips.spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
                    for kk in range(self.L)
                ]
                res_no_up = res
        else:
            # lpips=False：不做 1x1 卷积，直接对通道求和。
            if self.spatial:
                res_no_up = [diffs[kk].sum(dim=1, keepdim=True) for kk in range(self.L)]
                res = [
                    lpips.upsample(res_no_up[kk], out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    lpips.spatial_average(
                        diffs[kk].sum(dim=1, keepdim=True), keepdim=True
                    )
                    for kk in range(self.L)
                ]
                res_no_up = res

        # 官方 LPIPS 的「总和」定义：各层简单相加（权重已经含在 lins 里）。
        val = 0
        for layer in range(self.L):
            val += res[layer]

        if retPerLayer:
            # 注意返回的是未上采样的 res_no_up，各层形状不同。
            return (val, res_no_up)
        else:
            return val


# LPIPS 骨干每次构造都要从磁盘读一遍（vgg16 528MB、alexnet 244MB），
# 而 compute_distances 的每一轮 (transform, dir, repo_id, metric) 都会重新
# 走到这里——不缓存的话，光反复加载骨干就能占掉相当一部分时间。同一种
# model_kwargs 只建一次后复用；只影响速度，不影响数值。
_LPIPS_MODELS: dict[str, _PatchedLPIPS] = {}


def _get_lpips_model(model_kwargs: dict) -> _PatchedLPIPS:
    key = json.dumps(model_kwargs, sort_keys=True, default=str)
    model = _LPIPS_MODELS.get(key)
    if model is None:
        with warnings.catch_warnings():
            # lpips 内部会打一些无关的 UserWarning，静音以保持日志干净。
            warnings.simplefilter("ignore")
            # spatial=True：要 patch 级结果，供后面的空间分析使用。
            model = _PatchedLPIPS(spatial=True, **model_kwargs).to(device())
        # 只编译骨干网络（net），不编译整个 LPIPS 包装，避免编译 lins 等小算子。
        compile_net(model.net)
        _LPIPS_MODELS[key] = model
    return model


# batch_size/num_workers 只影响速度不影响数值，从缓存键里排除，
# 这样换 batch size 重跑能直接命中缓存。
@mem.cache(ignore=["batch_size", "num_workers"])
def _compute_lpips(
    ds_a: ImageFolder,
    ds_b: ImageFolder,
    model_kwargs: dict,
    batch_size: int,
    num_workers: int,
):
    # 两个 dataloader 各自只用 num_workers//2，总 worker 数与原设定一致
    # （两个 loader 同时在跑，各占一半才不会超订）。
    dl_a = DataLoader(dataset=ds_a, batch_size=batch_size, num_workers=num_workers // 2)
    dl_b = DataLoader(dataset=ds_b, batch_size=batch_size, num_workers=num_workers // 2)

    model = _get_lpips_model(model_kwargs)

    # 多留一个槽位放「各层加权求和」的总分，索引 0 是总分，1..L 是逐层。
    # 层数 = 1 + 骨干 stage 数（vgg 为 5，所以共 6 个 key）。
    lpips_layers = [[] for _ in range(1 + len(model.chns))]
    for (tensor_a, _), (tensor_b, _) in tqdm(
        zip(dl_a, dl_b),
        desc="Computing LPIPS",
        total=len(dl_a),
    ):
        # datasets 返回的第二个元素是路径，这里不需要，所以用 _ 接住。
        sum_batch, layers_batch = model(
            tensor_a.to(device()),
            tensor_b.to(device()),
            retPerLayer=True,
            normalize=True,
        )
        
        # 统一搬到 CPU 并存 float16：patch 级结果是 (B,1,H,W) 的大张量，
        # 全用 float32 留在显存里很容易 OOM。
        lpips_layers[0].append(sum_batch.to(device="cpu", dtype=torch.float16))
        for i, layer_result in enumerate(layers_batch):
            lpips_layers[i + 1].append(
                layer_result.to(device="cpu", dtype=torch.float16)
            )

    # 每个槽位内按 batch 维拼接，得到该层的全部结果。
    lpips_layers = [torch.cat(lpips_layer) for lpips_layer in lpips_layers]
    return lpips_layers


class LPIPS(Distance):
    """From Zhang et al., The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, 2018"""

    def __init__(
        self,
        net: str = "vgg",
        layer: int = -1,
        spatial: bool = False,
        output_size: Optional[int] = None,
        concat_layers_and_flatten: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        """
        net: backbone to use from ['alex', 'vgg', 'squeeze']
        layer: layer to return, -1 returns all layers
        spatial: whether to return scores for each patch
        output_size: resize output to this size (only applicable if spatial=True)
        """
        # 论文实验结论：vgg 骨干 + 第 2 层 LPIPS 效果最好（配置 "lpips_vgg_2"）。
        self.net = net
        self.layer = layer
        self.spatial = spatial
        self.output_size = output_size
        self.concat_layers_and_flatten = concat_layers_and_flatten
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _compute(self, ds_a: ImageFolder, ds_b: ImageFolder) -> list[torch.Tensor]:
        """Use pure function to enable caching."""
        return _compute_lpips(
            ds_a=ds_a,
            ds_b=ds_b,
            model_kwargs={"net": self.net},
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _postprocess(self, result: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Handle layer selection and resizing."""
        # 逐层拆成字典：layer=-1 表示把所有层都导出（距离指标消融实验
        # 就是这么一次算出全部层再挑的，省去重复前向）。
        out = {}
        if self.layer == -1:
            for i, tensor in enumerate(result):
                # 取负号：把「距离」变成「分数」，越大越像生成图。
                # 索引 0 是各层总和，所以 key 0 相当于官方 LPIPS 距离的负值。
                out[f"lpips_{self.net}_{i}"] = -tensor
        else:
            out[f"lpips_{self.net}_{self.layer}"] = -result[self.layer]

        for layer, tensor in out.items():
            if not self.spatial:
                # 非空间模式：对 H,W 求平均得到每张图一个标量（keepdim 保持 4D，
                # 便于和空间模式共用下游的 squeeze/resize 逻辑）。
                out[layer] = tensor.mean((2, 3), keepdim=True)
            elif self.output_size is not None:
                # 不同层的分辨率不同；要拼成统一网格就得先插值到同一尺寸。
                # float32 中间计算、再转回 float16，避免半精度插值掉精度。
                out[layer] = F.interpolate(
                    tensor.to(dtype=torch.float32),
                    size=self.output_size,
                    mode="bilinear",
                    antialias=True,
                ).to(dtype=torch.float16)

        if (
            self.concat_layers_and_flatten
            and self.layer == -1
            and self.spatial
            and self.output_size is not None
        ):
            # 消融实验用：把所有层的空间分数拼成一个特征向量，
            # 相当于一种「多层感知特征」的距离。
            out = {
                f"lpips_{self.net}_flat": torch.cat(
                    [tensor.flatten(start_dim=1) for tensor in out.values()], dim=1
                )
            }

        return out


@mem.cache(ignore=["batch_size", "num_workers"])
def _compute_pyiqa_distance(
    ds_a: ImageFolder,
    ds_b: ImageFolder,
    metric_name: str,
    batch_size: int,
    num_workers: int,
    **metric_kwargs,
):
    # 与 LPIPS 相同的双 loader 折半策略。
    dl_a = DataLoader(dataset=ds_a, batch_size=batch_size, num_workers=num_workers // 2)
    dl_b = DataLoader(dataset=ds_b, batch_size=batch_size, num_workers=num_workers // 2)

    # pyiqa 按名字返回现成的指标实现：psnr / ssimc / ms_ssim / dists 等。
    metric = pyiqa.create_metric(metric_name, **metric_kwargs)

    out = []
    for (tensor_a, _), (tensor_b, _) in tqdm(
        zip(dl_a, dl_b),
        desc=f"Computing {metric_name}",
        total=len(dl_a),
    ):
        out_tensor = metric(tensor_a, tensor_b).to(device="cpu", dtype=torch.float16)
        # PSNR 这类指标会返回 0 维标量，补一个维度以便后面统一 cat / 看 shape。
        if out_tensor.ndim == 0:
            out_tensor = out_tensor.unsqueeze(0)
        out.append(out_tensor)
    return torch.cat(out)


class PyIQADistance(Distance):
    """pyiqa 里现成指标的统一包装。"""

    def __init__(
        self,
        metric_name: str,
        batch_size: int,
        num_workers: int,
        **metric_kwargs,
    ) -> None:
        self.metric_name = metric_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metric_kwargs = metric_kwargs

    def _compute(self, ds_a: ImageFolder, ds_b: ImageFolder) -> Any:
        return _compute_pyiqa_distance(
            ds_a=ds_a,
            ds_b=ds_b,
            metric_name=self.metric_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # **self.metric_kwargs,  # this should be done on case by case basis
            # 上面这行被注释掉是有意为之：不同指标的 kwargs 语义不同，
            # 不能盲目透传，需要时再逐个显式加上。
        )

    def _postprocess(self, result: Any) -> dict[str, torch.Tensor]:
        # 有些指标（如 LPIPS 类）是「越小越好」，必须翻符号，
        # 才能满足「分数越大越像生成图」的统一约定。
        if pyiqa.DEFAULT_CONFIGS[self.metric_name].get("lower_better", False):
            result *= -1
        # 补齐到 4D，让下游可以不加判断地做 squeeze(1,2,3) 或插值。
        result = result[(...,) + (None,) * (4 - result.ndim)]  # make sure output is 4D
        return {self.metric_name: result}


def distance_from_config(
    config: str,
    batch_size: int = 1,
    num_workers: int = 1,
    **kwargs,
) -> Distance:
    """Parse config string and return matching distance."""
    # 配置字符串约定：
    #   "lpips_<net>_<layer>"（如 lpips_vgg_2）→ 本仓库的 LPIPS 实现
    #   其它名字（psnr、ssimc、ms_ssim、dists、lpips_alex_0 …）→ 交给 pyiqa
    # 额外的空间/patch 参数通过 **kwargs 透传给 LPIPS。
    if config.startswith("lpips"):
        _, net, layer = config.split("_")
        distance = LPIPS(
            net=net,
            layer=int(layer),
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
    else:
        distance = PyIQADistance(
            metric_name=config,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
    return distance
