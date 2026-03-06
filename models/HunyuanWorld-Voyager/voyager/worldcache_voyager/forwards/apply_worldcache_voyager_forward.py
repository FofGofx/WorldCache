import torch
from typing import Optional, Union, Dict
from types import MethodType
from voyager.worldcache_voyager.cache_utils import cal_type
from voyager.worldcache_voyager.worldcache_utils import derivative_approximation, taylor_formula, taylor_cache_init
from voyager.modules.attenion import get_cu_seqlens
import loguru


def apply_worldcache_voyager_forward(model):
    """
    Apply Worldcache Voyager forward.
    """

    def worldcache_voyager_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        # Text embedding for modulation.
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        freqs_cos_cond: Optional[torch.Tensor] = None,
        freqs_sin_cond: Optional[torch.Tensor] = None,
        # Guidance for modulation, should be cfg_scale x 1000.
        guidance: torch.Tensor = None,
        return_dict: bool = True,
        *,
        cache_dic=None,
        current=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the transformer with Worldcache acceleration.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        t : torch.Tensor
            Timestep tensor.
        text_states : torch.Tensor, optional
            Text embeddings.
        text_mask : torch.Tensor, optional
            Text mask.
        text_states_2 : torch.Tensor, optional
            Text embeddings 2 for modulation.
        freqs_cos, freqs_sin : torch.Tensor, optional
            Precomputed rotary embeddings.
        freqs_cos_cond, freqs_sin_cond : torch.Tensor, optional
            Precomputed rotary embeddings for condition.
        guidance : torch.Tensor, optional
            Guidance vector for distillation.
        return_dict : bool, optional
            Whether to return a dictionary.
        cache_dic : dict, optional
            Cache dictionary for Worldcache.
        current : dict, optional
            Current step information for Worldcache.

        Returns
        -------
        torch.Tensor or dict
            Output tensor or dictionary.
        """
        if cache_dic is None or current is None:
            return self._original_forward(
                x, t, text_states, text_mask, text_states_2,
                freqs_cos, freqs_sin, freqs_cos_cond, freqs_sin_cond,
                guidance, return_dict
            )

        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        vec = self.time_in(t)

        if self.i2v_condition_type == "token_replace":
            token_replace_t = torch.zeros_like(t)
            token_replace_vec = self.time_in(token_replace_t)
            frist_frame_token_num = th * tw
        else:
            token_replace_vec = None
            frist_frame_token_num = None

        vec_2 = self.vector_in(text_states_2)
        vec = vec + vec_2
        if self.i2v_condition_type == "token_replace":
            token_replace_vec = token_replace_vec + vec_2

        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            vec = vec + self.guidance_in(guidance)

        if self.use_context_block:
            condition = img.clone()
            height = (condition.shape[-2] - 2) // 2
            condition = condition[..., -height:, :]
            condition = self.condition_in(condition)

        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(
                txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        if self.use_context_block:
            cond_seq_len = condition.shape[1]
            cu_seqlens_q_cond = get_cu_seqlens(text_mask, cond_seq_len)
            cu_seqlens_kv_cond = cu_seqlens_q_cond
            max_seqlen_q_cond = cond_seq_len + txt_seq_len
            max_seqlen_kv_cond = max_seqlen_q_cond

            context_block_args = [
                condition,
                txt,
                vec,
                cu_seqlens_q_cond,
                cu_seqlens_kv_cond,
                max_seqlen_q_cond,
                max_seqlen_kv_cond,
                (freqs_cos_cond, freqs_sin_cond),
                self.i2v_condition_type,
                token_replace_vec,
                frist_frame_token_num,
            ]
            condition1, txt1 = self.context_block1(*context_block_args)

            condition2 = torch.cat((condition1, txt1), 1)
            context_block_args = [
                condition2,
                vec,
                txt_seq_len,
                cu_seqlens_q_cond,
                cu_seqlens_kv_cond,
                max_seqlen_q_cond,
                max_seqlen_kv_cond,
                (freqs_cos_cond, freqs_sin_cond),
                self.i2v_condition_type,
                token_replace_vec,
                frist_frame_token_num,
            ]
            condition2 = self.context_block2(*context_block_args)

            condition1 = self.zero_linear1(condition1)
            condition2 = self.zero_linear2(condition2)

            condition2 = torch.cat(
                (torch.zeros_like(img)[:, :-condition1.shape[1]], condition2), dim=1)
            condition1 = torch.cat(
                (torch.zeros_like(img)[:, :-condition1.shape[1]], condition1), dim=1)

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        if isinstance(t, torch.Tensor) and len(t) > 0:
            current['timestep'] = t[0].item() if hasattr(t[0], 'item') else float(t[0])
        else:
            current['timestep'] = None

        cal_type(cache_dic, current)
        current['stream'] = 'final'
        current['layer'] = 'final'
        current['module'] = 'final'
        taylor_cache_init(cache_dic, current)

        if current['type'] == 'full':
            for layer_num, block in enumerate(self.double_blocks):
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                    self.i2v_condition_type,
                    token_replace_vec,
                    frist_frame_token_num,
                ]

                if self.training and self.gradient_checkpoint and \
                        (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers):
                    img, txt = torch.utils.checkpoint.checkpoint(
                        ckpt_wrapper(block), *double_block_args, use_reentrant=False)
                    if self.use_context_block:
                        img += condition1
                else:
                    img, txt = block(*double_block_args)
                    if self.use_context_block:
                        img += condition1

            x = torch.cat((img, txt), 1)

            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                        self.i2v_condition_type,
                        token_replace_vec,
                        frist_frame_token_num,
                    ]

                    if self.training and self.gradient_checkpoint and \
                            (self.gradient_checkpoint_layers == -1 or \
                            layer_num + len(self.double_blocks) < self.gradient_checkpoint_layers):
                        x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(
                            block), *single_block_args, use_reentrant=False)
                        if self.use_context_block:
                            x += condition2
                    else:
                        x = block(*single_block_args)
                        if self.use_context_block:
                            x += condition2

            img = x[:, :img_seq_len, ...]

            img = self.final_layer(img, vec)

            derivative_approximation(cache_dic, current, img)

        elif current['type'] == 'worldcache':
            img = taylor_formula(cache_dic, current)

        img = self.unpatchify(img, tt, th, tw)

        if return_dict:
            out["x"] = img
            return out
        return img

    model._original_forward = model.forward
    model.forward = MethodType(worldcache_voyager_forward, model)
    loguru.logger.info("Worldcache Voyager forward applied")
