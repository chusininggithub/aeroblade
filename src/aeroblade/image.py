"""AEROBLADE 的核心一步：用 AE（自编码器）重建图像。

重建 = 「图像 → VAE 编码到隐空间 → 立刻解码回来」，中间不含任何扩散
采样步骤、不含提示词、不需要训练。整个过程就是一次 VAE 的 encode/decode
往返，得到 x' ≈ x；x 与 x' 的距离就是检测分数。

关键点：
- 生成图因为本来就落在该 AE 的分布内，往返失真小；真实图失真大。
- 隐空间采样用固定 seed 的 generator，保证可复现（重建结果会被缓存）。
- 中间结果按「参数哈希」分目录存盘，第二次跑同一配置直接读盘。
"""

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForImage2Image
from diffusers.models import VQModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from joblib.hashing import hash
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2.functional import to_pil_image
from tqdm import tqdm

from aeroblade.data import ImageFolder, read_files
from aeroblade.inversion import create_pipeline
from aeroblade.misc import (
    compile_net,
    device,
    resolve_model_source,
    safe_mkdir,
    write_config,
)


# 进程内的 AE 缓存。compute_distances 的循环是 transform → dir → repo_id，
# 同一个 AE 会随目录被反复加载（7 个目录 × 3 个 AE = 21 次），而每次
# from_pretrained 都要把整个管线读进来，Kandinsky 那条要几分钟，纯属浪费。
# 实际上这里只用管线里的 AE（vae / movq），三个 AE 一共不到 1GB，缓存代价
# 很小。缓存的是 AE 而非整个管线，所以不改变任何数值结果，只省掉重复加载。
_AE_CACHE: dict[str, torch.nn.Module] = {}


def _load_ae(repo_id: str) -> torch.nn.Module:
    """取出（并缓存）某个 repo 的自编码器。"""
    if repo_id in _AE_CACHE:
        return _AE_CACHE[repo_id]

    # set up pipeline
    # 用 image2image 管线只是为了借用它的 VAE 和调度器配置，
    # 真正的推理只用下面的 ae.encode/decode，不跑 UNet。
    pipe = AutoPipelineForImage2Image.from_pretrained(
        resolve_model_source(repo_id),
        torch_dtype=torch.float16,
        use_safetensors=True,
        # Kandinsky 2.1 没有 fp16 变体权重，只有它需要普通权重。
        variant="fp16" if "kandinsky-2" not in repo_id else None,
    )

    # extract AE
    # 不同模型的 AE 属性名不同：SD 系列叫 vae，Kandinsky 用 MOVQ。
    if hasattr(pipe, "vae"):
        ae = pipe.vae
        if hasattr(pipe, "upcast_vae"):
            # SD 系列会把 VAE 保持 float32 以保证解码质量，
            # 但 fp16 权重加载后需要显式 upcast 回来。
            pipe.upcast_vae()
    elif hasattr(pipe, "movq"):
        ae = pipe.movq
    # 只编译 AE（本流程的重头），可以省掉不少时间。
    ae = compile_net(ae)
    # 缓存前搬到 CPU：显存只有 6GB，三个 AE 同时驻留 GPU 会 OOM。
    ae.to("cpu")
    _AE_CACHE[repo_id] = ae
    # 管线其余部分（unet / text_encoder / prior 等）这里用不到，直接释放。
    # 注意：光 del 引用是不够的——diffusers 的管线内部有循环引用，必须显式
    # gc.collect() 才会真正回收。Kandinsky 那条的 prior 有 5GB+，漏掉的话会
    # 一直占着内存，把 15GB 的机器拖进换页（实测主进程涨到 8.7GB、速度从
    # 2 秒/张掉到 10 秒/张）。
    del pipe
    gc.collect()
    return ae


# 纯推理，显存占用大，关掉梯度记录。
@torch.no_grad()
def compute_reconstructions(
    ds: Dataset,
    repo_id: str,
    output_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    iterations: int = 1,
    seed: int = 1,
    batch_size: int = 1,
    num_workers: int = 1,
) -> list[Path]:
    """Compute AE reconstructions and save them in a unique directory."""
    # 至少要有一个输出位置：output_root 用于自动生成哈希子目录，
    # output_dir 用于精确指定（复现实验时用得上）。
    if output_root is None and output_dir is None:
        raise ValueError("Either output_root or output_dir must be specified.")
    if output_root is not None and output_dir is not None:
        # 两者都给时以 output_dir 为准，并明确告知，避免用户以为写到了 root 下。
        print("Ignoring output_root since output_dir is specified.")

    # 缓存键只包含真正影响结果的参数（数据、AE、输出根目录、seed）；
    # iterations 会作为路径的一级子目录拼在后面。
    arg_dict = {"ds": ds, "repo_id": repo_id, "output_root": output_root, "seed": seed}
    if output_dir is None:
        # create output directory based on hashed arguments if not specified
        output_dir = output_root / hash(arg_dict) / str(iterations)

    # load files if output directory already exists, compute otherwise
    # 两个条件同时满足才算「缓存命中」：目录存在 **且** 文件数量与数据集一致。
    # 只判目录存在是不够的——上一次可能中途被打断，留下半份结果。
    if not (
        output_dir.exists()
        and len(reconstruction_paths := read_files(output_dir)) == len(ds)
    ):
        # safe_mkdir 会在目录已存在时向用户确认（半份结果的场景）。
        safe_mkdir(output_dir)
        # 把参数写到哈希目录的上一级，事后能知道这个哈希对应什么配置。
        write_config(arg_dict, output_dir.parent)

        # if more than one iteration, recursively load previous iterations
        # 多次重建 = 在上一轮的重建图上再重建一次（5.5 节「更深的
        # 重建」用扩散反演，这里是把同一个 AE 反复用）。
        if iterations > 1:
            previous_paths = compute_reconstructions(
                ds=ds,
                repo_id=repo_id,
                output_root=output_root,
                iterations=iterations - 1,
                seed=seed,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            # 关键：继承原 dataset 的 transform 与 amount，否则第二轮
            # 会丢掉扰动、或者取到不同数量的图。
            ds = ImageFolder(
                paths=previous_paths, transform=ds.transform, amount=ds.amount
            )

        # 取（或复用）自编码器；缓存的 AE 放在 CPU，用前搬到 GPU。
        ae = _load_ae(repo_id).to(device())
        # 解码 dtype 从 AE 参数里读实际值，而不是假设 float16：
        # upcast_vae 之后 VAE 是 float32，但 latents 仍是 fp16，需要显式转换。
        decode_dtype = next(iter(ae.post_quant_conv.parameters())).dtype

        # reconstruct
        # 固定 seed：encode 时要从隐空间后验里采样，不固定就不可复现。
        generator = torch.Generator().manual_seed(seed)
        reconstruction_paths = []
        for images, paths in tqdm(
            DataLoader(ds, batch_size=batch_size, num_workers=num_workers),
            desc=f"Reconstructing with {repo_id}.",
        ):
            # normalize
            # 数据集给的是 [0,1]，VAE 期望 [-1,1]。
            images = images.to(device(), dtype=ae.dtype) * 2.0 - 1.0

            # encode
            # retrieve_latents 是 diffusers 的工具：对普通 VAE 走后验采样
            # （用上面的 generator 保证确定），对 VQ 模型走量化编码。
            latents = retrieve_latents(ae.encode(images), generator=generator)

            # decode
            if isinstance(ae, VQModel):
                # 关键细节：force_not_quantize=True 表示解码时**不做**码本
                # 最近邻替换（不做真正量化）。因为要衡量的正是「隐表示能否
                # 无损往返」，让量化误差混进来会掩盖真实的分布外程度。
                reconstructions = ae.decode(
                    latents.to(decode_dtype), force_not_quantize=True, return_dict=False
                )[0]
            else:
                reconstructions = ae.decode(
                    latents.to(decode_dtype), return_dict=False
                )[0]

            # de-normalize
            # [-1,1] → [0,1] 并截断，PIL 只能存这个范围。
            reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)

            # save
            # 保留原文件的 stem（换成 .png）：下游靠文件名把原图与重建图配对。
            for reconstruction, path in zip(reconstructions, paths):
                reconstruction_path = output_dir / f"{Path(path).stem}.png"
                to_pil_image(reconstruction).save(reconstruction_path)
                reconstruction_paths.append(reconstruction_path)
        print(f"Images saved to {output_dir}.")
    return reconstruction_paths


@torch.no_grad()
def compute_deeper_reconstructions(
    ds: Dataset,
    repo_id: str,
    output_root: Path,
    num_inference_steps: int,
    num_reconstruction_steps: int,
) -> list[Path]:
    """
    Compute reconstructions with AE and some inversion steps and save them in a
    unique directory.

    5.5 节的「更深的重建」：先用 DDIM 反演把图像推到带噪的隐状态，再往回
    去噪若干步。反演步数越多，重建图离原图越远，能看出不同 AE 的粗糙
    程度（重建越「深」、失真越大，检测越容易）。
    """
    # create output directory based on hashed arguments
    arg_dict = {
        "ds": ds,
        "repo_id": repo_id,
        "output_root": output_root,
        "num_inference_steps": num_inference_steps,
        "num_reconstruction_steps": num_reconstruction_steps,
    }
    output_dir = output_root / hash(arg_dict)

    # load files if output directory already exists, compute otherwise
    # 与上面相同的缓存命中判断（存在 + 数量一致）。
    if not (
        output_dir.exists()
        and len(reconstruction_paths := read_files(output_dir)) == len(ds)
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        write_config(arg_dict, output_dir.parent)

        # set up pipeline
        # 这条路径需要完整的扩散管线（含 BLIP 生成提示词），
        # 所以不用 AutoPipelineForImage2Image，而是自己组装。
        pipe = create_pipeline(sd_model_ckpt=repo_id, use_blip_only=True)

        # reconstruct
        reconstruction_paths = []
        # batch_size 固定为 1：整条反演流程内部按单张图处理。
        for _, paths in tqdm(
            DataLoader(ds, batch_size=1),
            desc=f"Reconstructing with {repo_id}.",
        ):
            path = paths[0]
            # 反演流程内部处理的是 512×512 的图，这里显式统一尺寸。
            img = Image.open(path).convert("RGB").resize((512, 512))
            # 提示词由 BLIP 自动生成（use_blip_only=True），与原始生成
            # 提示词无关——检测器不需要知道真实提示词。
            rec = pipe.compute_reconstruction(
                img,
                reconstruction_steps=num_reconstruction_steps,
                num_inference_steps=num_inference_steps,
            )

            # save
            reconstruction_path = output_dir / f"{Path(path).stem}.png"
            rec.save(reconstruction_path)
            reconstruction_paths.append(reconstruction_path)
        print(f"Images saved to {output_dir}.")
    return reconstruction_paths


def extract_patches(
    array: np.ndarray | torch.Tensor, size: int, stride: int
) -> np.ndarray | torch.Tensor:
    """
    Split 4D tensor into (overlapping) spatial patches.
    Output shape is batch_size x num_patches x num_channels x patch_size x patch_size

    用 F.unfold 实现滑窗取块：它本身输出 (B, C*size*size, num_patches)，
    所以先转置（mT）让 patch 维在中间，再 reshape 成五维。
    stride < size 时块之间重叠——空间热力图希望平滑一些，所以重叠是常态。
    """
    # 统一走张量，最后再按输入类型还回去（保持调用方拿到的类型不变）。
    if isinstance(array, np.ndarray):
        is_ndarray = True
        array = torch.from_numpy(array)
    else:
        is_ndarray = False
    # unfold 只支持 4D（N,C,H,W）输入。
    if array.ndim != 4:
        raise ValueError("array must be 4D.")
    patches = F.unfold(array, kernel_size=size, stride=stride).mT.reshape(
        array.shape[0], -1, array.shape[1], size, size
    )
    if is_ndarray:
        patches = patches.numpy()
    return patches
