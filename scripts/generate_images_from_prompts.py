"""Generate images from prompts.

中文说明（新增，英文原文保留）
================================
读取一个 prompt CSV（通常由 extract_prompts.py 从真实图片反推得到），用指定的
LDM 逐条生成图片并落盘。它是构建 AEROBLADE 数据集的两步中的第二步：
真图 → extract_prompts.py 得 prompt → 本脚本生成假图，
于是真假图共享同一个 image_id，可以严格配对比较。

什么时候运行
------------
只在构建数据集时运行（一次性、很慢），不在检测/评测阶段运行。

读取
----
--prompt-file 指定的 CSV，必须有 image_id 和 prompt 两列。

写出
----
<output-root>/<AE 仓库名(斜杠换短横线)>-<prompt 文件名去扩展名>/<image_id>.png
例如 data/raw/generated/CompVis-stable-diffusion-v1-1-ViT-L-14-openai/。
目录名的后半段就是「prompt 由哪个 CLIP 抽取」，与 experiments/01_detect.py 里
--fake-dirs 的默认路径一一对应。
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from aeroblade.misc import device, safe_mkdir
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm

# 允许 FP32 矩阵乘法走 TF32（降低尾数精度换取速度）。生成阶段只是造数据，
# 不要求比特级复现，因此可以放开；这是全局开关，影响本进程后续所有 CUDA 运算。
torch.backends.cuda.matmul.allow_tf32 = True


def main(args):
    # 输出目录名 = 模型名 + prompt 文件名（不含扩展名）。
    # 这样 data/raw/generated/CompVis-stable-diffusion-v1-1-ViT-L-14-openai
    # 一眼就能读出「SD1.1 生成、prompt 来自 ViT-L-14/openai」。
    output_dir = (
        args.output_root / f'{args.repo_id.replace("/", "-")}-{args.prompt_file.stem}'
    )
    safe_mkdir(output_dir)

    # set up pipeline
    # AutoPipelineForText2Image 会根据仓库配置自动选对 pipeline 类（SD / Kandinsky 等）。
    # torch_dtype=float16 + variant="fp16" 用半精度权重加载，512×512 的 SD 才能在单卡上跑得动；
    # safety_checker=None 关掉 NSFW 检查器——否则被判为 NSFW 的图会被替换成纯黑图，
    # 静默污染数据集（论文数据里也确实包含 NSFW 内容）。
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.repo_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None,
    ).to(device())
    # Kandinsky 2.x 的 pipeline 与 diffusers 的 CPU offload 不兼容，会直接报错，
    # 因此对它跳过这一步；其余模型开启后显存不够时会把权重临时换到 CPU。
    if "kandinsky-2" not in args.repo_id:
        pipe.enable_model_cpu_offload()

    # load prompts
    # 强制 image_id 按字符串读入：这些 ID 是 9 位数字（如 000001234），
    # 交给 pandas 推断会变成整数并丢掉前导零，生成的文件名就对不上真实图了。
    prompts = pd.read_csv(args.prompt_file, dtype={"image_id": str})

    # fix generator for reproducibity
    # 固定随机数生成器以保证可复现：同一个 prompt 每次都生成同一张图。
    # 默认 seed=42，注意它与检测阶段的 seed=1 是两回事（生成与重建各自独立）。
    generator = torch.Generator().manual_seed(args.seed)

    # generate and save images
    # 按 batch_size 把 prompt 表切成若干子表。整个循环共用同一个 generator，
    # 随机数按 batch 顺序消耗，所以 batch_size 变了采样结果也会变——
    # 想完全复现某批数据，必须连 batch_size 一起固定。
    prompt_batches = [
        prompts.iloc[i : i + args.batch_size]
        for i in range(0, len(prompts), args.batch_size)
    ]
    for prompt_batch in tqdm(prompt_batches):
        image_batch = pipe(
            prompt=prompt_batch["prompt"].tolist(),
            generator=generator,
        ).images

        # 文件名用 prompt 文件里的 image_id（统一存 png），
        # 与 rename_real_images.py 处理后的真实图重名，distances.py 才能按键名
        # 把真图与重建图/生成图配对，进而计算重建距离。
        for image, image_id in zip(image_batch, prompt_batch["image_id"]):
            image.save(output_dir / f"{image_id}.png")


def parse_args():
    parser = argparse.ArgumentParser()
    # --prompt-file：prompt CSV 路径，需要含 image_id 与 prompt 两列；
    # 默认 debug/debug_prompts.csv 是调试用的小样本。
    parser.add_argument(
        "--prompt-file", type=Path, default=Path("debug/debug_prompts.csv")
    )
    # --output-root：输出根目录，生成图在其下的 <模型名>-<prompt 文件名>/ 里。
    parser.add_argument("--output-root", type=Path, default="debug/generated")
    # --repo-id：用哪个生成模型。四个候选正是论文数据集里使用的模型；
    # 想换模型需要同时在 --prompt-file 里提供该模型文本编码器抽出的 prompt。
    parser.add_argument(
        "--repo-id",
        choices=[
            "kandinsky-community/kandinsky-2-1",
            "stabilityai/stable-diffusion-2-1-base",
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-1",
        ],
        default="CompVis/stable-diffusion-v1-1",
    )
    # --batch-size：每次喂给 pipeline 的 prompt 数，默认 2（生成阶段显存占用高，故比检测小）；
    # 它会改变随机数消耗顺序，进而影响结果，复现时不要随手改。
    parser.add_argument("--batch-size", type=int, default=2)
    # --seed：生成用随机种子，默认 42（与检测阶段的 seed=1 无关）。
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
