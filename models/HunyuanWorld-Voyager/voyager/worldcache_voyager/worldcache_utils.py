from typing import Dict
import torch
import math
from voyager.worldcache_voyager.cache_utils.worldcache_utils import (
    compute_curvature,
    compute_token_groups,
    compute_adaptive_slope,
    compute_prediction_error,
    update_history_buffer,
    compute_velocity,
)


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    In Worldcache mode, also computes curvature and updates token groups.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: Feature tensor (final layer output)
    """
    mode = cache_dic.get('mode', 'worldcache')

    if len(current['activated_steps']) >= 2:
        difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    else:
        difference_distance = 0

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    computed_orders = [0]

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
            computed_orders.append(i + 1)
        else:
            break

    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

    if mode == 'worldcache' and current.get('type') == 'full':
        worldcache_history = cache_dic['worldcache_history']
        worldcache_config = cache_dic['worldcache_config']
        current_step = current['step']
        num_steps = current['num_steps']

        history_count = len(worldcache_history['outputs'])
        print(f"[Worldcache Full Comp] Step: {current_step}/{num_steps} | History: {history_count} outputs")

        update_history_buffer(worldcache_history, feature, current_step, max_history=3)

        if len(worldcache_history['outputs']) >= 2 and len(worldcache_history['steps']) >= 2:
            F_t_minus_1 = worldcache_history['outputs'][-2]
            F_t = worldcache_history['outputs'][-1]
            step_t_minus_1 = worldcache_history['steps'][-2]
            step_t = worldcache_history['steps'][-1]
            dt = step_t - step_t_minus_1

            if dt > 0:
                v_t = compute_velocity(F_t, F_t_minus_1, dt)
                if v_t is not None:
                    worldcache_history['velocities'].append(v_t)
                    if len(worldcache_history['velocities']) > 2:
                        worldcache_history['velocities'] = worldcache_history['velocities'][-2:]

        if len(worldcache_history['outputs']) >= 3:
            curvature = compute_curvature(
                worldcache_history,
                current_step,
                eps=worldcache_config['eps']
            )

            if curvature is not None:
                cache_dic['cached_curvature'] = curvature

                curvature_flat = curvature.flatten()
                curvature_mean = curvature_flat.mean().item()
                curvature_min = curvature_flat.min().item()
                curvature_max = curvature_flat.max().item()
                curvature_p20 = torch.quantile(curvature_flat, 0.20).item()
                curvature_p80 = torch.quantile(curvature_flat, 0.80).item()

                print(f"[Worldcache Curvature] Step: {current_step}/{num_steps} | Mean: {curvature_mean:.4f} | Min: {curvature_min:.4f} | Max: {curvature_max:.4f} | P20: {curvature_p20:.4f} | P80: {curvature_p80:.4f}")

                mask_stable, mask_linear, mask_chaotic = compute_token_groups(
                    curvature,
                    percentile_stable=worldcache_config['percentile_stable'],
                    percentile_chaotic=worldcache_config['percentile_chaotic']
                )

                total_tokens = mask_stable.numel()
                stable_count = mask_stable.sum().item()
                linear_count = mask_linear.sum().item()
                chaotic_count = mask_chaotic.sum().item()
                stable_pct = (stable_count / total_tokens * 100) if total_tokens > 0 else 0.0
                linear_pct = (linear_count / total_tokens * 100) if total_tokens > 0 else 0.0
                chaotic_pct = (chaotic_count / total_tokens * 100) if total_tokens > 0 else 0.0

                print(f"[Worldcache Groups] Step: {current_step}/{num_steps} | Stable: {stable_count} ({stable_pct:.1f}%) | Linear: {linear_count} ({linear_pct:.1f}%) | Chaotic: {chaotic_count} ({chaotic_pct:.1f}%)")

                cache_dic['cached_masks'] = {
                    'mask_stable': mask_stable,
                    'mask_linear': mask_linear,
                    'mask_chaotic': mask_chaotic,
                }
            else:
                print(f"[Worldcache Curvature] Step: {current_step}/{num_steps} | Failed: Unable to compute curvature")
        else:
            print(f"[Worldcache Curvature] Step: {current_step}/{num_steps} | Failed: Insufficient history (need 3 outputs, have {len(worldcache_history['outputs'])})")

        cache_dic['worldcache_config']['chaotic_error_accumulated'] = 0.0


def worldcache_predict(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Worldcache prediction with differentiated strategies for different token groups.

    - Stable group: Reuse latest full computation output
    - Linear group: First-order Taylor prediction
    - Chaotic group: Adaptive damped Taylor prediction

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :return: Predicted feature tensor
    """
    worldcache_config = cache_dic['worldcache_config']
    worldcache_history = cache_dic['worldcache_history']
    cached_masks = cache_dic['cached_masks']
    current_step = current['step']
    num_steps = current['num_steps']

    last_full_step = current['activated_steps'][-1]
    k = current_step - last_full_step

    print(f"[Worldcache Predict] Step: {current_step}/{num_steps} | k: {k} | Last Full: {last_full_step}")

    mask_stable = cached_masks['mask_stable']
    mask_linear = cached_masks['mask_linear']
    mask_chaotic = cached_masks['mask_chaotic']

    total_tokens = mask_stable.numel()
    stable_count = mask_stable.sum().item()
    linear_count = mask_linear.sum().item()
    chaotic_count = mask_chaotic.sum().item()

    if len(worldcache_history['outputs']) > 0:
        F_latest = worldcache_history['outputs'][-1]
    else:
        F_latest = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][0]

    output = F_latest.clone()

    if mask_linear.any():
        taylor_factors = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
        F_0 = taylor_factors[0]

        if len(taylor_factors) > 1:
            F_1 = taylor_factors[1]
            linear_pred = F_0 + k * F_1

            if mask_linear.dim() < output.dim():
                mask_linear_expanded = mask_linear.unsqueeze(-1)
            else:
                mask_linear_expanded = mask_linear

            output = torch.where(mask_linear_expanded, linear_pred, output)

    print(f"[Worldcache Strategy] Step: {current_step}/{num_steps} | Stable: {stable_count} (reuse) | Linear: {linear_count} (taylor) | Chaotic: {chaotic_count} (adaptive)")

    if mask_chaotic.any() and len(worldcache_history['velocities']) >= 2:
        v_curr = worldcache_history['velocities'][-1]
        v_prev = worldcache_history['velocities'][-2] if len(worldcache_history['velocities']) >= 2 else v_curr

        v_adapt = compute_adaptive_slope(
            v_curr,
            v_prev,
            k,
            worldcache_config['n_max'],
            hermite_weights=worldcache_config.get('hermite_weights')
        )

        taylor_factors = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
        F_0 = taylor_factors[0]

        chaotic_pred = F_0 + k * v_adapt

        if mask_chaotic.dim() < output.dim():
            mask_chaotic_expanded = mask_chaotic.unsqueeze(-1)
        else:
            mask_chaotic_expanded = mask_chaotic

        output = torch.where(mask_chaotic_expanded, chaotic_pred, output)

        if cache_dic.get('cached_curvature') is not None:
            if k == 1:
                F_prev = F_0
            else:
                v_adapt_prev = compute_adaptive_slope(
                    v_curr,
                    v_prev,
                    k - 1,
                    worldcache_config['n_max'],
                    hermite_weights=worldcache_config.get('hermite_weights')
                )
                F_prev = F_0 + (k - 1) * v_adapt_prev

            error_details = compute_prediction_error(
                cache_dic['cached_curvature'],
                chaotic_pred,
                F_prev,
                mask_chaotic,
                eps=worldcache_config['eps'],
                return_details=True
            )

            error = error_details['error']

            print(f"[Worldcache Error Calc] Step: {current_step}/{num_steps} | ds_mean: {error_details['ds_mean']:.4f} | ds_max: {error_details['ds_max']:.4f} | ds_min: {error_details['ds_min']:.4f} | curvature_mean: {error_details['curvature_mean']:.4f} | curvature_max: {error_details['curvature_max']:.4f} | curvature_min: {error_details['curvature_min']:.4f} | error_tensor_mean: {error_details['error_tensor_mean']:.4f} | error_tensor_max: {error_details['error_tensor_max']:.4f} | error_tensor_min: {error_details['error_tensor_min']:.4f} | final_error: {error:.4f}")

            cache_dic['worldcache_config']['chaotic_error_accumulated'] += error

            accumulated_error = cache_dic['worldcache_config']['chaotic_error_accumulated']
            error_threshold = worldcache_config['error_threshold']
            status = "WARNING" if accumulated_error >= error_threshold * 0.8 else "OK"
            print(f"[Worldcache Error] Step: {current_step}/{num_steps} | Step Error: {error:.4f} | Accumulated: {accumulated_error:.4f} | Threshold: {error_threshold:.4f} | Status: {status}")

    return output


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion or Worldcache prediction.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    mode = cache_dic.get('mode', 'worldcache')

    if mode == 'worldcache' and current.get('type') == 'worldcache':
        return worldcache_predict(cache_dic, current)

    x = current['step'] - current['activated_steps'][-1]
    output = 0

    used_orders = []
    for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
        term = (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)
        output += term
        used_orders.append(i)

    return output


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}
