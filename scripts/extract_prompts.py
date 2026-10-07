"""
Extract prompts from images using CLIP Interrogator.
Adapted from https://github.com/pharmapsychotic/clip-interrogator/blob/main/run_cli.py.

中文说明（新增，英文原文保留）
================================
用 CLIP Interrogator 给图片「反推」出一句文字 prompt，常用于把真实图片
（LAION-5B）转成 prompt，再用 generate_images_from_prompts.py 拿同一批 prompt
去生成假图，从而得到 prompt 对齐的「真图 / 假图」配对数据——这是 AEROBLADE
评测里真假图对照实验的数据基础。

为什么要把 image_id 写进 CSV
----------------------------
CSV 只有 image_id 和 prompt 两列，image_id 取图片文件名（不含扩展名）。
生成脚本会用这个 ID 当输出文件名，于是生成图与真实图文件名一致，
distances.py 就能按键名把两张图配成一对来算重建距离。

输出命名与「不要覆盖」的约定
----------------------------
输出文件名是 <--clip 里的斜杠换成短横线>.csv，例如 ViT-L-14/openai → ViT-L-14-openai.csv。
这个后缀正是 data/raw/generated/CompVis-stable-diffusion-v1-1-ViT-L-14-openai
里那一段的来源，用来记录 prompt 是哪个 CLIP 抽的。
脚本在目标文件已存在时直接抛 FileExistsError：换配置重抽会悄悄改掉整个数据集，
所以宁可报错让人显式处理。

什么时候运行
------------
只在构建数据集时运行（一次性、很慢）。--clip 必须与后续生成模型的文本编码器匹配：
SD1 系列用 ViT-L-14/openai，SD2 系列用 ViT-H-14/laion2b_s32b_b79k，
这样 prompt 的分布才贴近模型训练时见过的文本分布。

读取
----
-f/--folder 指定的图片目录，或 -i/--image 指定的单张图片（本地路径或 http(s) URL）。

写出
----
<--output 目录>/<CLIP 名>.csv，列为 image_id, prompt：
- 单图模式只把 prompt 打到 stdout，不写文件；
- 目录模式批量处理，每个文件一行。
"""

import argparse
import csv
import os

import requests
import torch
from clip_interrogator import Config, Interrogator, list_clip_models
from PIL import Image
from tqdm import tqdm


def inference(ci, image, mode):
    # 三种问询策略的封装：best 最慢但质量最好（默认），classic 用经典 prompt 模板，
    # 其余一律走 fast（更少的 beam/更短的候选，速度优先）。
    # 统一在这里 convert("RGB")，因为 PNG 可能带 alpha 通道，CLIP 只接受三通道。
    image = image.convert("RGB")
    if mode == "best":
        return ci.interrogate(image)
    elif mode == "classic":
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)


def main():
    parser = argparse.ArgumentParser()
    # -c/--clip：用哪个 CLIP 模型反推 prompt。两个候选分别对应两代 SD 的文本编码器：
    # ViT-L-14/openai 供 SD1，ViT-H-14/laion2b_s32b_b79k 供 SD2。
    parser.add_argument(
        "-c",
        "--clip",
        default="ViT-L-14/openai",
        choices=[
            "ViT-L-14/openai",  # SD1
            "ViT-H-14/laion2b_s32b_b79k",  # SD2
        ],
        help="name of CLIP model to use",
    )
    # -d/--device：推理设备，auto 表示有 CUDA 就用 CUDA，否则退回 CPU（会很慢）。
    parser.add_argument(
        "-d", "--device", default="auto", help="device to use (auto, cuda or cpu)"
    )
    # -f/--folder：批量模式，图片所在目录（与 -i 互斥）。
    parser.add_argument("-f", "--folder", help="path to folder of images")
    # -i/--image：单图模式，本地图片路径或 http(s) URL（与 -f 互斥），结果只打印。
    parser.add_argument("-i", "--image", help="image file or url")
    # -m/--mode：问询策略，对应 Interrogator 的三种方法（best / classic / fast）。
    parser.add_argument("-m", "--mode", default="best", help="best, classic, or fast")
    # -o/--output：输出目录（必填），CSV 会写到它下面。
    parser.add_argument("-o", "--output", help="output directory", required=True)
    # --lowvram：显存吃紧时改用 CLIP Interrogator 的低显存配置（推理解析度/缓存策略更省显存）。
    parser.add_argument(
        "--lowvram", action="store_true", help="Optimize settings for low VRAM"
    )

    args = parser.parse_args()
    # 既没给目录也没给图片时打印帮助并以 1 退出；两个都给同样报错退出。
    if not args.folder and not args.image:
        parser.print_help()
        exit(1)

    if args.folder is not None and args.image is not None:
        print("Specify a folder or batch processing or a single image, not both")
        exit(1)

    # define output path
    # 输出路径带上 CLIP 名（斜杠换成短横线），使「prompt 是哪来的」可追溯；
    # 已存在则直接抛错，防止不同配置的 prompt 互相覆盖。
    csv_path = os.path.join(args.output, args.clip.replace("/", "-") + ".csv")
    if os.path.exists(csv_path):
        raise FileExistsError

    # validate clip model name
    # 提前校验模型名，避免跑到一半才发现 CLIP 名字拼错（list_clip_models 是本地注册表）。
    models = list_clip_models()
    if args.clip not in models:
        print(f"Could not find CLIP model {args.clip}!")
        print(f"    available models: {models}")
        exit(1)

    # select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU. Warning: this will be very slow!")
    else:
        device = torch.device(args.device)

    # generate a nice prompt
    # quiet=True 关掉 Interrogator 自己的进度输出，只用外层的 tqdm；
    # 低显存模式通过 apply_low_vram_defaults() 施加。
    config = Config(device=device, clip_model_name=args.clip, quiet=True)
    if args.lowvram:
        config.apply_low_vram_defaults()
    ci = Interrogator(config)

    # process single image
    # 单图模式：支持 URL（requests 流式取回字节流交给 PIL）和本地路径两种情况，
    # 结果直接打印，不落盘——方便临时查一张图的 prompt。
    if args.image is not None:
        image_path = args.image
        if str(image_path).startswith("http://") or str(image_path).startswith(
            "https://"
        ):
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        if not image:
            print(f"Error opening image {image_path}")
            exit(1)
        print(inference(ci, image, args.mode))

    # process folder of images
    # 批量模式：只挑 .jpg/.png/.webp 三种扩展名（注意不含 .jpeg）。
    # 先用列表存下每张图的 prompt，再统一写 CSV，保证 files / prompts 顺序一一对应。
    elif args.folder is not None:
        if not os.path.exists(args.folder):
            print(f"The folder {args.folder} does not exist!")
            exit(1)

        files = [
            f
            for f in os.listdir(args.folder)
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".webp")
        ]
        prompts = []
        for file in tqdm(files):
            image = Image.open(os.path.join(args.folder, file)).convert("RGB")
            prompt = inference(ci, image, args.mode)
            prompts.append(prompt)

        # 只在确实抽到 prompt 时才写文件：避免留下一张空 CSV，
        # 而空 CSV 会让后续 generate_images_from_prompts.py 生成不出任何图还找不到原因。
        # newline="" 与 utf-8 是 csv 模块在 Windows 上的标准写法（防止空行/编码问题）；
        # image_id 写入的是不带扩展名的文件名，供生成脚本还原成 <image_id>.png。
        if len(prompts):
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                w.writerow(["image_id", "prompt"])
                for file, prompt in zip(files, prompts):
                    w.writerow([os.path.splitext(os.path.basename(file))[0], prompt])

            print(f"\n\n\n\nGenerated {len(prompts)} and saved to {csv_path}, enjoy!")


if __name__ == "__main__":
    main()
