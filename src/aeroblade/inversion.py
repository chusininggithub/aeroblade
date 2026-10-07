"""DDIM 反演 + 部分去噪，用于论文 5.5 节的「更深的重建」。

思路：
1. invert() 把图像沿 DDIM 反演方向推到带噪的隐状态 x_T'（相当于「加密」成噪声），
   反演步数由 invert_steps 控制，可以只反演一部分。
2. denoise() 从某个中间时刻开始正向去噪回图像（默认从 x_T 走到 x_0，
   也可用 denoise_from/denoise_steps 只走一段）。

于是 compute_reconstruction 就是「反演 k 步 → 再往回 k 步」，往返越深、
重建图越模糊，检验的是「AE 重建质量越差，AEROBLADE 是否同样有效」。

文件里保留了 diffusers 官方 pix2pix-zero 管线的大量注释级代码（包括被
注释掉的旧实现），我们只加中文注释，不动原逻辑。
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from clip_interrogator import Config, Interrogator
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    StableDiffusionPix2PixZeroPipeline,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.utils import BaseOutput
from diffusers.utils.logging import disable_progress_bar
from torchvision.transforms.functional import pil_to_tensor
from transformers import BlipForConditionalGeneration, BlipProcessor

from aeroblade.misc import resolve_model_source

# diffusers 内部会打很多加载日志，实验脚本要输出自己的进度条，这里全局关掉。
disable_progress_bar()


class BLIPCaptioner:
    """用 BLIP 给图像生成描述，充当后续反演/去噪所需的提示词。

    检测流程里我们并不知道原图的真实提示词；用 BLIP 现编一个「差不多的」
    就够——这也说明 AEROBLADE 并不依赖提示词还原得准不准。
    """

    def __init__(self, captioner_ckpt="Salesforce/blip-image-captioning-large"):
        self.captioner_ckpt = captioner_ckpt
        self.caption_processor = BlipProcessor.from_pretrained(self.captioner_ckpt)
        self.caption_generator = BlipForConditionalGeneration.from_pretrained(
            self.captioner_ckpt,
            # low_cpu_mem_usage=True
        )
        # 字幕模型固定跑在 cuda 上；与扩散模型共享显存时靠下面的
        # 「用完就搬回去」来错峰。
        self._execution_device = torch.device("cuda")

    @torch.no_grad()
    def generate_caption(self, image):
        """Generates caption for a given image."""
        # text="" 表示只做图像描述（非条件生成），BLIP 会从 <bos> 开始生成。
        text = ""

        # 记录当前所在设备，推理结束后归还，避免长期占用显存。
        prev_device = self.caption_generator.device

        device = self._execution_device
        inputs = self.caption_processor(image, text, return_tensors="pt").to(
            device=device, dtype=self.caption_generator.dtype
        )
        self.caption_generator.to(device)
        outputs = self.caption_generator.generate(**inputs, max_new_tokens=128)

        # offload caption generator
        self.caption_generator.to(prev_device)

        # batch_decode + 取 [0]：每次只描述一张图。
        caption = self.caption_processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0]
        return caption


class CLIPInterrogator:
    """用 CLIP Interrogator 生成提示词（比 BLIP 更长、更像 SD 风格提示词）。

    这是 create_pipeline 的默认字幕器；BLIP 是 use_blip_only=True 时的备选。
    """

    def __init__(self, clip_model_name="ViT-L-14/openai"):
        self.interrogator = Interrogator(
            Config(clip_model_name=clip_model_name, quiet=True)
        )

    @torch.no_grad()
    def generate_caption(self, image):
        # 直接转发，接口与 BLIPCaptioner 保持一致（鸭子类型）。
        return self.interrogator.interrogate(image)


@dataclass
class InversionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        latents (`torch.FloatTensor`)
            inverted latents tensor
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    # 同时返回隐状态和解码图：距离既可以在像素域算，也可以在隐空间算
    #（见 compute_stepwise_reconstruction_distance 的 use_latent）。
    latents: torch.FloatTensor
    images: Union[List[PIL.Image.Image], np.ndarray]


class StableDiffusionPipelinePartialInversion(StableDiffusionPix2PixZeroPipeline):
    """在 pix2pix-zero 管线上加了「部分反演 / 部分去噪」的能力。

    继承 pix2pix-zero 是为了复用它的 auto_corr_loss / kl_divergence /
    get_epsilon（反演时的正则化项：把预测出的噪声约束成近似标准正态，
    否则 DDIM 反演的噪声会越来越不像高斯，往返重建就不稳）。
    """

    # silent 控制是否打印选中的时间步；批量跑实验时设为 True 免得刷屏。
    silent = True

    @torch.no_grad()
    def denoise(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,  #
        denoise_from: int = 0,  # which of the selected timesteps to start denoising from, by default from the very beginning (x_T)
        denoise_steps: int = None,  # how many steps to run the denoising for, if None, the until the very end (x_0)
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_latent: bool = False,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        # 不指定尺寸时按 UNet 的默认采样尺寸 × VAE 下采样倍率（512）。
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        # 注意：这是上游代码遗留的调试 print，保持原样不动。
        print(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )
        StableDiffusionPipeline.check_inputs(
            self,
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        # batch 大小从 prompt 推断；只给 embeddings 时看 embeddings 的第 0 维。
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # >1 才启用 CFG（此时 UNet 前向要跑「有条件+无条件」两份）。
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        # 注意用的是 _encode_prompt（pix2pix 版本，支持 token 级加权），
        # 与下面 invert() 里的 encode_prompt 不同。
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        # 去噪用正向调度器（DDIMScheduler），时间步从大到小。
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        # print(f"---- num_channels_latents: {num_channels_latents}")
        # 可以传入已有的带噪 latents（这就是「部分去噪」的入口：从反演
        # 得到的中间状态接着往下采样）。
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # print(f"Normal timesteps: {timesteps}")
        # denoise_from：从第几个时间步开始（0 = 从最噪的 x_T 开始）。
        # denoise_steps：一共走多少步；None 表示一路走到 x_0。
        denoise_steps = (
            denoise_steps
            if denoise_steps is not None
            else num_inference_steps - denoise_from
        )
        # 注意是切片，不是重新 set_timesteps：这样「部分去噪」与完整
        # 去噪在相同步数下的数值行为完全一致。
        timesteps = timesteps[denoise_from : denoise_from + denoise_steps]
        if not self.silent:
            print(f"Selected timesteps: {timesteps}")

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                # CFG 要求 batch 里前半是无条件、后半是有条件，所以复制一份。
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                # 不同调度器对输入 latent 的缩放方式不同，必须交给调度器处理。
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                # 标准 CFG：无条件预测 + w × (有条件 - 无条件)。
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    # 默认 0.0，即不启用（zero terminal SNR 场景才需要）。
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                    )

                # compute the previous noisy sample x_t -> x_t-1
                # 去噪用正向 scheduler.step。
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # compute image
        # 隐空间 → 像素：先除以缩放因子还原 VAE 的输入尺度，再解码。
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        numimg = image.shape[0]
        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=[True] * numimg)
        image = self.image_processor.postprocess(image, output_type=output_type)

        # 返回约定与 diffusers 一致；nsfw 检测被跳过，统一填 False。
        if return_latent:
            if not return_dict:
                ret = (latents, image, [False] * numimg)
            else:
                ret = InversionPipelineOutput(latents=latents, images=image)
        else:
            if not return_dict:
                ret = (image, [False] * numimg)
            else:
                ret = StableDiffusionPipelineOutput(
                    images=image, nsfw_content_detected=[False] * numimg
                )

        # Offload last model to CPU
        # 显存不够时是「逐层上卡」模式，收尾要手动把最后一块搬回 CPU。
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        return ret

        # if not output_type == "latent":
        #     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        #     image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        #     has_nsfw_concept = [False for _ in has_nsfw_concept]
        # else:
        #     image = latents
        #     has_nsfw_concept = None

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # # Offload last model to CPU
        # if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        #     self.final_offload_hook.offload()

        # if not return_dict:
        #     return (image, has_nsfw_concept)

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        # 上面这段是被 return 提前截断的死代码，来自上游 diffusers 的原始实现，
        # 保留以便与官方版本对照。

    @torch.no_grad()
    def invert(
        self,
        prompt: Optional[str] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        num_inference_steps: int = 50,
        # invert_from: int = 0,
        invert_steps: Tuple[int] = None,
        extra_invert_steps: int = None,
        guidance_scale: float = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_guidance_amount: float = 0.1,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        lambda_auto_corr: float = 20.0,
        lambda_kl: float = 20.0,
        num_reg_steps: int = 5,
        num_auto_corr_rolls: int = 5,
    ):
        r"""
        Function used to generate inverted latents given a prompt and image.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch which will be used for conditioning. Can also accept
                image latents as `image`, if passing latents directly, it will not be encoded again.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 1):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            cross_attention_guidance_amount (`float`, defaults to 0.1):
                Amount of guidance needed from the reference cross-attention maps.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            lambda_auto_corr (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control auto correction
            lambda_kl (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control Kullback–Leibler divergence output
            num_reg_steps (`int`, *optional*, defaults to 5):
                Number of regularization loss steps
            num_auto_corr_rolls (`int`, *optional*, defaults to 5):
                Number of auto correction roll steps

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero.Pix2PixInversionPipelineOutput`] or
            `tuple`:
            [`~pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero.Pix2PixInversionPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is the inverted
            latents tensor and then second is the corresponding decoded image.
        """
        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # 反演默认 guidance_scale=1，即不做 CFG（反演要的是「忠实还原」，
        # 而不是「按提示词生成」）。
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Preprocess image
        # 统一到 [-1,1] 张量。
        image = self.image_processor.preprocess(image)

        # 4. Prepare latent variables
        # 编码到隐空间：反演的起点 x_0。
        latents = self.prepare_image_latents(
            image, batch_size, self.vae.dtype, device, generator
        )

        # 5. Encode input prompt
        num_images_per_prompt = 1
        # 这里用的是标准 encode_prompt，取 [0] 拿 prompt_embeds。
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )[0]

        # 4. Prepare timesteps
        # 关键：用 inverse_scheduler（DDIMInverseScheduler），时间步从小到大，
        # 是 DDIM 的逆过程。
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 6. Rejig the UNet so that we can obtain the cross-attenion maps and
        # use them for guiding the subsequent image generation.

        # 7. Denoising loop where we obtain the cross-attention maps.
        # invert_steps 支持一次反演、多个深度取点：
        #   单个 int  → 只返回该深度的结果（return_single=True）
        #   元组      → 把沿途这些深度的隐状态都存下来一起返回
        # 这样 compute_stepwise_reconstruction_distance 只需跑一次反演
        # 就能比较 k 步与 k+1 步两个中间状态。
        return_single = False
        if invert_steps is None:
            invert_steps = (num_inference_steps,)
        elif isinstance(invert_steps, int):
            invert_steps = (invert_steps,)
            return_single = True
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        )
        invert_from = 0
        # print(f"Normal timesteps: {timesteps}")
        # 只反演到所需的最深一步（max(invert_steps)），多余的不算。
        timesteps = timesteps[invert_from : invert_from + max(invert_steps)]
        if not self.silent:
            print(f"Selected timesteps: {timesteps}")
        # 索引 i 对应第 i 步之后的状态，所以取 timesteps[i-1] 作为「第 i 步」
        # 的时间步标签，用它当字典键。
        return_timesteps = [timesteps[i - 1] for i in invert_steps]
        if not self.silent:
            print(f"Return timesteps: {return_timesteps}")

        # 记录沿途需要返回的隐状态，键是时间步数值。
        ret_latents = {}

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.inverse_scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                # 注意这里没写 return_dict=False，所以取 .sample。
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # regularization of the noise prediction
                # 反演的核心技巧：DDIM 反演出的噪声会偏离标准正态，导致
                # 「反演再重建」不闭合。这里对噪声预测做几步梯度下降，
                # 把它拉回「接近 IID 标准正态」：
                #   - auto_corr_loss：抑制空间自相关（相邻像素不该相关）
                #   - kl_divergence：把分布拉向 N(0, I)
                # lambda_* 是学习率式的权重系数（20 比较大，但要配合
                # num_reg_steps/num_auto_corr_rolls 一起看）。
                with torch.enable_grad():
                    for _ in range(num_reg_steps):
                        if lambda_auto_corr > 0:
                            # 多次随机 roll 求平均梯度，降低单次 roll 的方差。
                            for _ in range(num_auto_corr_rolls):
                                var = torch.autograd.Variable(
                                    noise_pred.detach().clone(), requires_grad=True
                                )

                                # Derive epsilon from model output before regularizing to IID standard normal
                                # 正则化对象是「由模型输出推出的 ε」，而不是模型输出本身。
                                var_epsilon = self.get_epsilon(
                                    var, latent_model_input.detach(), t
                                )

                                l_ac = self.auto_corr_loss(
                                    var_epsilon, generator=generator
                                )
                                l_ac.backward()

                                # 梯度按 roll 次数平均，等价于对多次 roll 的
                                # 梯度取均值。
                                grad = var.grad.detach() / num_auto_corr_rolls
                                noise_pred = noise_pred - lambda_auto_corr * grad

                        if lambda_kl > 0:
                            var = torch.autograd.Variable(
                                noise_pred.detach().clone(), requires_grad=True
                            )

                            # Derive epsilon from model output before regularizing to IID standard normal
                            var_epsilon = self.get_epsilon(
                                var, latent_model_input.detach(), t
                            )

                            l_kld = self.kl_divergence(var_epsilon)
                            l_kld.backward()

                            grad = var.grad.detach()
                            noise_pred = noise_pred - lambda_kl * grad

                        # 每步正则化后切断计算图，避免梯度累积到下一轮。
                        noise_pred = noise_pred.detach()

                # compute the previous noisy sample x_t -> x_t-1
                # 反演用 inverse_scheduler，方向与去噪相反。
                latents = self.inverse_scheduler.step(
                    noise_pred, t, latents
                ).prev_sample

                # 需要的话把当前中间状态存下来（深拷贝在下面做）。
                if t in return_timesteps:
                    ret_latents[t.cpu().item()] = latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        rets = []
        # print(ret_latents.keys())
        for select_t in return_timesteps:
            # 时间步是张量，必须先 .cpu().item() 再当字典键（张量不可哈希）。
            latents = ret_latents[select_t.cpu().item()]
            # detach + clone：把中间状态从计算图里摘出来。
            inverted_latents = latents.detach().clone()

            # 8. Post-processing
            # 顺便解码出像素图，方便肉眼看反演到第 k 步长什么样。
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

            if not return_dict:
                rets.append((inverted_latents, image))
            else:
                rets.append(
                    InversionPipelineOutput(latents=inverted_latents, images=image)
                )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        # 只请求了一个深度就返回单个对象，避免调用方到处写 [0]。
        if len(rets) == 1 and return_single:
            return rets[0]
        else:
            return rets

    def compute_reconstruction(
        self,
        x0: PIL.Image.Image,
        reconstruction_steps: int,
        prompt: str = None,
        num_inference_steps=50,
    ):
        """反演 k 步再退回 k 步，得到 x0 的扩散重建结果。"""
        # 不传提示词就现编一个（BLIP/CLIP Interrogator）。
        if prompt is None:
            prompt = self.generate_caption(x0)
        # 第一步：x0 → 隐空间 → 反演 reconstruction_steps 步。
        x_inv = self.invert(
            prompt,
            x0,
            invert_steps=reconstruction_steps,
            num_inference_steps=num_inference_steps,
        ).latents
        # 第二步：从对应的时间步开始往回走同样的步数。
        # denoise_from = 总步数 - k，正好是反演到第 k 步所处的时刻。
        x_recon = self.denoise(
            prompt,
            latents=x_inv,
            denoise_from=num_inference_steps - reconstruction_steps,
            num_inference_steps=num_inference_steps,
        ).images
        return x_recon[0]

    def compute_reconstruction_distance(
        self,
        x0: PIL.Image.Image,
        reconstruction_steps: int,
        prompt: str = None,
        num_inference_steps=50,
        distance="l2",
    ):
        """原图与「k 步往返重建结果」的 L2 距离。"""
        xrecon = self.compute_reconstruction(
            x0,
            reconstruction_steps,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
        )
        if distance == "l2":
            # 先归一到 [-1,1]，再拉平成一维向量，最后用 cdist 算欧氏距离。
            # （这里的 [None, None] 把 (N,) 变成 (1,1,N)，是为了复用 cdist
            # 的批量接口拿一个标量结果。）
            x0_pt = pil_to_tensor(x0).float() / 127.5 - 1
            xrecon_pt = pil_to_tensor(xrecon).float() / 127.5 - 1
            x0_pt = x0_pt.flatten()[None, None]
            xrecon_pt = xrecon_pt.flatten()[None, None]
            dist = torch.cdist(x0_pt, xrecon_pt, p=2)[0, 0]
        else:
            raise Exception("Unsupported distance")
        return dist

    def compute_stepwise_reconstruction_distance(
        self,
        x0: PIL.Image.Image,
        reconstruction_steps: int,
        prompt: str = None,
        extra_steps: int = 1,
        num_inference_steps=50,
        use_latent=True,
        distance="l2",
    ):
        """reconstruction_steps specifies how many inference steps to go back to obtain x-tilde from paper,
        extra_steps specifies how many inference steps to go back and forth to obtain a reconstruction of x-tilde
        note that inference steps skip over multiple original DDPM training steps, rather than the original DDPM steps used in training.

        论文里的「逐步重建误差」：把图像反演到第 k 步得到 x̃（x-tilde），
        再只往前退 1 步得到 x̃ 的重建，两者之差就是第 k 步的局部往返误差。
        这比「走完全程再比」更能看出误差是在哪一段产生的。
        """
        if prompt is None:
            prompt = self.generate_caption(x0)
        # 一次反演取两个中间状态：第 k 步（x̃）和第 k+extra 步。
        ret = self.invert(
            prompt,
            x0,
            invert_steps=(reconstruction_steps, reconstruction_steps + extra_steps),
            num_inference_steps=num_inference_steps,
        )
        x_inv, x_inv_extra = ret
        # 从第 k+extra 步只往回退 extra_steps 步，得到 x̃ 的重建。
        # 注意必须传 denoise_steps 限制步数，否则会一路退到 x_0
        #（那就变成整幅图的重建，而不是「逐步」的了）。
        x_extra_recon = self.denoise(
            prompt,
            latents=x_inv_extra.latents,
            denoise_from=num_inference_steps - reconstruction_steps - extra_steps,
            denoise_steps=extra_steps,
            num_inference_steps=num_inference_steps,
            return_latent=True,
        )
        # print(x_inv.latents.shape, x_extra_recon.shape, x_inv.latents.min(), x_inv.latents.max(), x_inv.latents.mean())
        if distance == "l2":
            # use_latent=True 在隐空间比较（更敏感、无需解码）；
            # False 则解码到像素域再比（更贴近人眼看到的结果）。
            if use_latent:
                x_inv = x_inv.latents[0]
                x_extra_recon = x_extra_recon.latents[0]
                x_inv = x_inv.flatten()[None, None]
                x_extra_recon = x_extra_recon.flatten()[None, None]
                dist = torch.cdist(x_inv, x_extra_recon, p=2)[0, 0]
            else:
                x0_pt = pil_to_tensor(x_inv.images[0]).float() / 127.5 - 1
                xrecon_pt = pil_to_tensor(x_extra_recon.images[0]).float() / 127.5 - 1
                x0_pt = x0_pt.flatten()[None, None]
                xrecon_pt = xrecon_pt.flatten()[None, None]
                dist = torch.cdist(x0_pt, xrecon_pt, p=2)[0, 0]
        else:
            raise Exception("Unsupported distance")
        return dist

    def generate_caption(self, image):
        # 转发到创建管线时挂上的字幕器（BLIP 或 CLIP Interrogator）。
        return self.captioner.generate_caption(image)


def compute_diff(img1, img2):
    """两张图求差并偏移到中灰，方便直接当图看（论文定性分析用）。"""
    # offset=127 把差值映射到 [0,255] 中心，scale=1 不放大对比度。
    x_diff = PIL.ImageChops.subtract(img1, img2, offset=127, scale=1)
    return x_diff


def create_pipeline(
    sd_model_ckpt="runwayml/stable-diffusion-v1-5",
    blip_ckpt="Salesforce/blip-image-captioning-large",
    clip_interrogate_ckpt="ViT-L-14/openai",
    use_blip_only=False,
):
    """组装一条「反演 + 部分去噪」的管线（5.5 节深入重建实验用）。"""
    # 默认用 CLIP Interrogator 生成更丰富的提示词；use_blip_only=True
    # 时用 BLIP（更快，也是 compute_deeper_reconstructions 走的分支）。
    if use_blip_only:
        captioner = BLIPCaptioner(captioner_ckpt=blip_ckpt)
    else:
        captioner = CLIPInterrogator(clip_model_name=clip_interrogate_ckpt)

    pipeline = StableDiffusionPipelinePartialInversion.from_pretrained(
        resolve_model_source(sd_model_ckpt)
    )

    # 从同一份配置派生出一对调度器：正向（去噪）与逆向（反演）。
    # 两者必须来自同一 config，否则噪声表不一致，往返就不会闭合。
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline.enable_model_cpu_offload()

    # 把字幕器挂在管线上，便于 self.generate_caption 调用。
    pipeline.captioner = captioner
    return pipeline
