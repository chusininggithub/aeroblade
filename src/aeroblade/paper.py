"""论文绘图与命名的辅助函数（只被 notebooks 使用，不影响检测流程）。

这里的东西都是「让图长得像 CVPR 论文」用的：数据集简写、LaTeX 字体、
按单栏/双栏宽度设置 figsize 等。
"""

import re
from typing import Literal, Optional

import matplotlib.pyplot as plt
import torch

# 论文图表的固定顺序：所有图都用同一个横轴/图例顺序，方便对照。
DATASET_ORDER = ["SD1.1", "SD1.5", "SD2.1", "KD2.1", "MJ4", "MJ5", "MJ5.1", "Real"]


def get_nice_name(input: str) -> str:
    """把内部路径/模型 id 转成论文图表里使用的简写。"""
    # datasets
    # 数据集目录名里带一堆前缀（模型名 + CLIP 模型名），这里按关键字
    # 抽取人类可读的版本号，例如 .../runwayml-stable-diffusion-v1-5-... → SD1.5。
    if "data/raw/" in input:
        if "real" in input:
            return "Real"
        elif "stable-diffusion" in input:
            version = re.search("(\d)-(\d)", input)
            return f"SD{version.group(1)}.{version.group(2)}"
        elif "kandinsky" in input:
            version = re.search("(\d)-(\d)", input)
            return f"KD{version.group(1)}.{version.group(2)}"
        elif "midjourney" in input:
            version = re.search("-v(.*)", input)
            return f"MJ{version.group(1).replace('-', '.')}"
    # AEs
    # 自编码器：图上用 SD1/SD2 这类短名，避免表头撑爆列宽。
    elif input == "CompVis/stable-diffusion-v1-1":
        return "SD1"
    elif input == "stabilityai/stable-diffusion-2-base":
        return "SD2"
    elif input == "kandinsky-community/kandinsky-2-1":
        return "KD2.1"

    # distance metrics
    # 距离指标：lpips_vgg_0 → LPIPS，lpips_vgg_2 → LPIPS₂（用 LaTeX 下标）；
    # 括号里标注使用的骨干网络，因为不同骨干的绝对值不可直接比较。
    elif input.startswith("lpips"):
        if "alex" in input:
            net = " (AlexNet)"
        elif "squeeze" in input:
            net = " (SqueezeNet)"
        else:
            net = ""
        if input.endswith("0"):
            return "LPIPS" + net
        else:
            return f"LPIPS$_{input[-1]}$" + net
    elif input in (
        # 海象运算符：顺手把 pyiqa 指标名到图表名的映射建成 metric_dict。
        metric_dict := {
            "dists": "DISTS",
            "psnr": "PSNR",
            "ssimc": "SSIM",
            "ms_ssim": "MS-SSIM",
        }
    ):
        return metric_dict[input]
    else:
        return input


def configure_mpl() -> None:
    """统一 matplotlib 样式，使其匹配 CVPR 模板（LaTeX 排版 + 单栏宽度）。"""
    plt.rcdefaults()
    params = {
        # 用真正的 LaTeX 渲染文本，图里的公式才能和正文一致。
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amssymb} \usepackage{amsmath}",
        "font.family": "serif",
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 6,
        "legend.handlelength": 1.0,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.labelpad": 2.0,
        "xtick.major.pad": 1.0,
        "ytick.major.pad": 1.0,
        "lines.linewidth": 0.75,
        "lines.markersize": 2,
    }

    plt.rcParams.update(params)

    figure_params = {
        "figure.dpi": 300,
        "figure.constrained_layout.use": True,
        "axes.grid": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.0,
    }

    plt.rcParams.update(figure_params)

    set_figsize()


def set_figsize(
    format: Literal["single", "double"] = "single",
    # 默认高宽比为黄金分割：论文里默认用这个比例出图。
    ratio: float = 2 / (1 + 5**0.5),
    factor: float = 1.0,
    nrows: int = 1,
    ncols: int = 1,
) -> None:
    """
    Set width and height of figure.

    :param width: Width of figure, single or double column.
    :param ratio: Ratio between width and height (height = width * ratio).
        Defaults to golden ratio.
    :param factor: Scaling factor for both width and height.
    :param nrows, ncols: Number of rows/columns if subplots are used.
    """
    # 单位是 TeX 的 pt（这里是 1/100 pt 记法）：单栏约 3.3 英寸、
    # 双栏约 6.9 英寸，对应 CVPR 模板的栏宽。
    if format == "single":
        width = 237.13594
    elif format == "double":
        width = 496.85625
    else:
        raise ValueError

    # 子图越多行、越少列，单个子图越高。
    height = width * ratio * (nrows / ncols)
    # 100/7227 是 (1/100 pt) → 英寸的换算系数。
    factor = 100 / 7227 * factor
    plt.rcParams["figure.figsize"] = width * factor, height * factor


def colorbar(mappable):
    """在图的右侧追加一个与主坐标轴等高的 colorbar（不挤压主图）。"""
    # 局部 import：这两个模块只有画图时才需要，避免拖慢非绘图场景的导入。
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    # 恢复「当前坐标轴」，否则后续绘图会画到 cax 上去。
    plt.sca(last_axes)
    return cbar


# 出图用 600 dpi 且关掉网格：这是给 camera-ready 用的尺寸与观感。
@plt.rc_context({"figure.dpi": 600, "axes.grid": False})
def plot_tensor(
    image: torch.Tensor,
    overlay: Optional[torch.Tensor] = None,
    alpha: float = 0.9,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_cbar: bool = True,
    ax=None,
):
    """显示一张 CHW 张量；overlay 用来把热力图半透明叠在原图上。"""
    if ax is None:
        ax = plt.gca()
    # matplotlib 要 HWC，所以 permute；下面的 colorbar 只是为了让主图
    # 按色值范围自适应并被 remove 掉（即只为建立 norm，不显示图例）。
    img = ax.imshow(image.permute(1, 2, 0))
    cbar = colorbar(img)
    cbar.remove()
    if overlay is not None:
        # vmin/vmax 由调用方显式给定，才能在多张图之间使用同一个色标区间。
        ol = ax.imshow(overlay, alpha=alpha, vmin=vmin, vmax=vmax)
        if show_cbar:
            colorbar(ol)
    # 隐扩散检测的图不需要像素坐标刻度。
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
