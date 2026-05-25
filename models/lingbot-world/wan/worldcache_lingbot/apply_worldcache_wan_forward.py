from types import MethodType

import logging
import torch
import torch.nn.functional as torch_F
from einops import rearrange

from ..modules.model import sinusoidal_embedding_1d
from .worldcache_utils import (
    compute_adaptive_slope,
    compute_curvature,
    compute_prediction_error,
    compute_token_groups,
    compute_velocity,
    update_history_buffer,
)

logger = logging.getLogger(__name__)


def _log_worldcache(message):
    print(message, flush=True)


def _update_full_state(cache_dic, current, feature):
    if cache_dic["mode"] != "worldcache":
        return

    history = cache_dic["worldcache_history"]
    config = cache_dic["worldcache_config"]
    branch = current.get("branch", "unknown")
    model_name = current.get("model_name", "unknown")
    step = current["step"]
    num_steps = current["num_steps"]

    update_history_buffer(history, feature, step, max_history=3)

    if len(history["outputs"]) >= 2 and len(history["steps"]) >= 2:
        feature_prev = history["outputs"][-2]
        feature_curr = history["outputs"][-1]
        step_prev = history["steps"][-2]
        step_curr = history["steps"][-1]
        velocity = compute_velocity(feature_curr, feature_prev, step_curr - step_prev)
        if velocity is not None:
            history["velocities"].append(velocity)
            history["velocities"] = history["velocities"][-2:]

    if len(history["outputs"]) >= 3:
        curvature = compute_curvature(history, eps=config["eps"])
        if curvature is not None:
            cache_dic["cached_curvature"] = curvature
            mask_stable, mask_linear, mask_chaotic = compute_token_groups(
                curvature,
                percentile_stable=config["percentile_stable"],
                percentile_chaotic=config["percentile_chaotic"],
            )
            cache_dic["cached_masks"] = {
                "mask_stable": mask_stable,
                "mask_linear": mask_linear,
                "mask_chaotic": mask_chaotic,
            }

            total_tokens = mask_stable.numel()
            _log_worldcache(
                f"[LingBot WorldCache][{model_name}][{branch}] "
                f"Curvature step={step}/{num_steps - 1} "
                f"mean={curvature.float().mean().item():.4f} "
                f"stable={mask_stable.sum().item()}/{total_tokens} "
                f"linear={mask_linear.sum().item()}/{total_tokens} "
                f"chaotic={mask_chaotic.sum().item()}/{total_tokens}"
            )
        else:
            _log_worldcache(
                f"[LingBot WorldCache][{model_name}][{branch}] "
                f"Curvature unavailable at step={step}/{num_steps - 1}"
            )
    else:
        _log_worldcache(
            f"[LingBot WorldCache][{model_name}][{branch}] "
            f"History warmup step={step}/{num_steps - 1} "
            f"history={len(history['outputs'])}/3"
        )

    config["chaotic_error_accumulated"] = 0.0


def _worldcache_predict(cache_dic, current):
    history = cache_dic["worldcache_history"]
    if not history["outputs"]:
        raise RuntimeError("WorldCache prediction requested without cached full outputs.")

    config = cache_dic["worldcache_config"]
    masks = cache_dic["cached_masks"]
    mask_stable = masks["mask_stable"]
    mask_linear = masks["mask_linear"]
    mask_chaotic = masks["mask_chaotic"]
    if mask_stable is None or mask_linear is None or mask_chaotic is None:
        raise RuntimeError("WorldCache prediction requested before token groups were initialized.")

    feature_latest = history["outputs"][-1]
    output = feature_latest.clone()
    last_full_step = current["activated_steps"][-1]
    k = current["step"] - last_full_step
    branch = current.get("branch", "unknown")
    model_name = current.get("model_name", "unknown")

    if history["velocities"] and mask_linear.any():
        velocity_curr = history["velocities"][-1]
        linear_pred = feature_latest + k * velocity_curr
        output = torch.where(mask_linear.unsqueeze(-1), linear_pred, output)

    if mask_chaotic.any() and len(history["velocities"]) >= 2:
        velocity_curr = history["velocities"][-1]
        velocity_prev = history["velocities"][-2]
        velocity_adapt = compute_adaptive_slope(
            velocity_curr,
            velocity_prev,
            k,
            config["n_max"],
            hermite_weights=config.get("hermite_weights"),
        )
        chaotic_pred = feature_latest + k * velocity_adapt
        output = torch.where(mask_chaotic.unsqueeze(-1), chaotic_pred, output)

        if cache_dic.get("cached_curvature") is not None:
            if k == 1:
                feature_prev = feature_latest
            else:
                velocity_prev_step = compute_adaptive_slope(
                    velocity_curr,
                    velocity_prev,
                    k - 1,
                    config["n_max"],
                    hermite_weights=config.get("hermite_weights"),
                )
                feature_prev = feature_latest + (k - 1) * velocity_prev_step

            error_details = compute_prediction_error(
                cache_dic["cached_curvature"],
                chaotic_pred,
                feature_prev,
                mask_chaotic,
                return_details=True,
            )
            config["chaotic_error_accumulated"] += error_details["error"]

    _log_worldcache(
        f"[LingBot WorldCache][{model_name}][{branch}] "
        f"Predict step={current['step']}/{current['num_steps'] - 1} "
        f"k={k} stable={mask_stable.sum().item()} "
        f"linear={mask_linear.sum().item()} chaotic={mask_chaotic.sum().item()} "
        f"error_accumulated={config.get('chaotic_error_accumulated', 0.0):.4f}"
    )
    return output


def apply_worldcache_wan_forward(model):
    if getattr(model, "_worldcache_forward_patched", False):
        return

    def worldcache_wan_forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
    ):
        runtime = getattr(self, "_worldcache_runtime", None)
        if runtime is None:
            return self._original_forward(
                x,
                t,
                context,
                seq_len,
                y=y,
                dit_cond_dict=dit_cond_dict,
            )

        cache_dic = runtime["cache_dic"]
        current = runtime["current"]
        current["branch"] = runtime["branch"]
        current["model_name"] = runtime["model_name"]

        if self.model_type == "i2v":
            assert y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            batch_t = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (batch_t, seq_len)).float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = [
                rearrange(
                    item,
                    "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                    c1=self.patch_size[0],
                    c2=self.patch_size[1],
                    c3=self.patch_size[2],
                )
                for item in dit_cond_dict["c2ws_plucker_emb"]
            ]
            c2ws_plucker_emb = torch.cat(c2ws_plucker_emb, dim=1)
            c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
            c2ws_hidden_states = self.c2ws_hidden_states_layer2(
                torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb))
            )
            dit_cond_dict = dict(dit_cond_dict)
            dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb + c2ws_hidden_states

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dit_cond_dict=dit_cond_dict,
        )

        if current["type"] == "full":
            for block in self.blocks:
                x = block(x, **kwargs)
            features = self.head(x, e)
            _update_full_state(cache_dic, current, features)
        elif current["type"] == "worldcache":
            features = _worldcache_predict(cache_dic, current)
        else:
            raise ValueError(f"Unsupported calculation type: {current['type']}")

        output = self.unpatchify(features, grid_sizes)
        return [item.float() for item in output]

    if not hasattr(model, "_original_forward"):
        model._original_forward = model.forward
    model.forward = MethodType(worldcache_wan_forward, model)
    model._worldcache_forward_patched = True
    logger.info("WorldCache LingBot forward applied")
