import torch
from typing import Optional, Union, List
from types import MethodType
from voyager.worldcache_voyager.cache_utils import cache_init
import loguru


def apply_worldcache_voyager_pipeline(pipeline):
    """
    Apply worldcache to Voyager pipeline.
    This function patches the pipeline's __call__ method to initialize cache
    and wrap the transformer to pass cache_dic and current.
    """

    original_call = pipeline.__call__

    @torch.no_grad()
    def __call_worldcache_voyager_pipeline(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        video_length: Optional[int] = None,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[dict] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Optional[tuple] = None,
        freqs_cis_cond: Optional[tuple] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        i2v_mode: bool = False,
        i2v_condition_type: Optional[str] = None,
        i2v_stability: bool = True,
        img_latents: Optional[torch.Tensor] = None,
        semantic_images=None,
        partial_cond=None,
        partial_mask=None,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate video with Worldcache acceleration.
        """
        if timesteps is not None:
            num_steps = len(timesteps)
        else:
            num_steps = num_inference_steps

        mode = kwargs.get('mode', getattr(self, '_worldcache_mode', 'worldcache'))
        if hasattr(self, 'worldcache_mode'):
            mode = self.worldcache_mode

        cache_dic, current = cache_init(num_steps, output_dir=output_dir, mode=mode)

        original_transformer = self.transformer
        original_transformer_forward = original_transformer.forward

        def transformer_forward_wrapper(*args, **kwargs):
            kwargs['cache_dic'] = cache_dic
            kwargs['current'] = current
            return original_transformer_forward(*args, **kwargs)

        original_transformer.forward = transformer_forward_wrapper

        try:
            self._worldcache_cache_dic = cache_dic
            self._worldcache_current = current

            step_tracker = {'last_t': None, 'current_step': -1}

            def transformer_forward_with_step(*args, **kwargs):
                if len(args) >= 2:
                    t_expand = args[1]
                    if isinstance(t_expand, torch.Tensor) and len(t_expand) > 0:
                        current_t = t_expand[0].item()
                        if step_tracker['last_t'] is None or current_t != step_tracker['last_t']:
                            step_tracker['current_step'] += 1
                            step_tracker['last_t'] = current_t
                        current['step'] = step_tracker['current_step']

                kwargs['cache_dic'] = cache_dic
                kwargs['current'] = current
                return original_transformer_forward(*args, **kwargs)

            original_transformer.forward = transformer_forward_with_step

            try:
                result = original_call(
                    prompt=prompt,
                    height=height,
                    width=width,
                    video_length=video_length,
                    data_type=data_type,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                    sigmas=sigmas,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    eta=eta,
                    generator=generator,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    attention_mask=attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_attention_mask=negative_attention_mask,
                    output_type=output_type,
                    return_dict=return_dict,
                    cross_attention_kwargs=cross_attention_kwargs,
                    guidance_rescale=guidance_rescale,
                    clip_skip=clip_skip,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    freqs_cis=freqs_cis,
                    freqs_cis_cond=freqs_cis_cond,
                    vae_ver=vae_ver,
                    enable_tiling=enable_tiling,
                    n_tokens=n_tokens,
                    embedded_guidance_scale=embedded_guidance_scale,
                    i2v_mode=i2v_mode,
                    i2v_condition_type=i2v_condition_type,
                    i2v_stability=i2v_stability,
                    img_latents=img_latents,
                    semantic_images=semantic_images,
                    partial_cond=partial_cond,
                    partial_mask=partial_mask,
                    output_dir=output_dir,
                    **kwargs,
                )
            finally:
                original_transformer.forward = original_transformer_forward

            return result

        finally:
            if hasattr(self, '_worldcache_cache_dic'):
                delattr(self, '_worldcache_cache_dic')
            if hasattr(self, '_worldcache_current'):
                delattr(self, '_worldcache_current')

    pipeline.__class__.__call__ = __call_worldcache_voyager_pipeline

    loguru.logger.info("Worldcache Voyager pipeline applied")
