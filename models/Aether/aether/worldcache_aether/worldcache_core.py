from typing import Dict 
import torch
import math
import sys
from aether.worldcache_aether.cache_utils.worldcache_utils import (
    compute_curvature,
    compute_token_groups,
    compute_adaptive_slope,
    compute_prediction_error,
    update_history_buffer,
    compute_velocity,
    recombine_modalities,
    unflatten_from_tokens,
    split_modalities,
    flatten_to_tokens,
)

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    In Worldcache mode, also computes curvature and updates token groups.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: Feature tensor (output after norm_final, before norm_out and proj_out)
    """
    mode = cache_dic.get('mode', 'worldcache')
    
    # Calculate difference_distance only if we have at least 2 activated steps
    if len(current['activated_steps']) >= 2:
        difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    else:
        difference_distance = 0  # First step, no previous step to compare
    #difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    computed_orders = [0]  # Always compute 0th order (the feature itself)
    
    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
            computed_orders.append(i + 1)
        else:
            break
    
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors
    
    # Worldcache mode: compute curvature and update token groups
    # Note: For Worldcache mode with modality splitting, the history is updated in forward after unpatchify
    # So we skip curvature computation here and do it in forward after splitting
    if mode == 'worldcache' and current.get('type') == 'full':
        # The actual Worldcache history update and curvature computation happens in forward
        # after unpatchify and splitting. We just reset error accumulator here.
        cache_dic['worldcache_config']['chaotic_error_accumulated'] = 0.0

def worldcache_predict(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Worldcache prediction with differentiated strategies for different token groups.
    
    - Stable group: Reuse latest full computation output
    - Linear group: First-order Taylor prediction
    - Chaotic group: Adaptive damped Taylor prediction
    
    For modality splitting mode, this function:
    1. Predicts split tokens (RGB, Depth, Raymap separately)
    2. Recombines them back to spatial latents
    3. Returns the recombined latents for further processing
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :return: Predicted split tokens (batch, seq_len*3, 24) for modality splitting mode
    """
    worldcache_config = cache_dic['worldcache_config']
    worldcache_history = cache_dic['worldcache_history']
    cached_masks = cache_dic['cached_masks']
    
    # Get prediction step k (steps since last full computation)
    last_full_step = current['activated_steps'][-1]
    k = current['step'] - last_full_step
    
    # Get cached masks
    mask_stable = cached_masks['mask_stable']
    mask_linear = cached_masks['mask_linear']
    mask_chaotic = cached_masks['mask_chaotic']
    
    # Get latest full computation output (for stable group)
    # In modality splitting mode, this is split_tokens with shape (batch, seq_len*3, 24)
    if len(worldcache_history['outputs']) > 0:
        F_latest = worldcache_history['outputs'][-1]
    else:
        # Fallback: use cached feature
        F_latest = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][0]
    
    # Initialize output with latest full computation
    output = F_latest.clone()
    
    # 1. Stable group: Reuse latest full computation (already set above)
    # No modification needed for stable tokens
    
    # 2. Linear group: First-order Taylor prediction
    if mask_linear.any():
        # Get Taylor factors from cache
        # Note: In modality splitting mode, we need to compute Taylor factors from split tokens
        # For now, we use velocity from history
        if len(worldcache_history['velocities']) > 0:
            v_t = worldcache_history['velocities'][-1]
            # Linear prediction: F_k = F_0 + k * v_t
            linear_pred = F_latest + k * v_t
            
            # Apply linear mask
            # Expand mask to match tensor dimensions
            if mask_linear.dim() < output.dim():
                mask_linear_expanded = mask_linear.unsqueeze(-1)
            else:
                mask_linear_expanded = mask_linear
            
            # Update linear group tokens
            output = torch.where(mask_linear_expanded, linear_pred, output)
    
    # 3. Chaotic group: Adaptive damped Taylor prediction
    if mask_chaotic.any() and len(worldcache_history['velocities']) >= 2:
        # Get current and previous velocities
        v_curr = worldcache_history['velocities'][-1]  # Current velocity
        v_prev = worldcache_history['velocities'][-2] if len(worldcache_history['velocities']) >= 2 else v_curr
        
        # Compute adaptive slope
        v_adapt = compute_adaptive_slope(
            v_curr,
            v_prev,
            k,
            worldcache_config['n_max'],
            hermite_weights=worldcache_config.get('hermite_weights')
        )
        
        # Adaptive prediction: F_k = F_0 + k * v_adapt
        chaotic_pred = F_latest + k * v_adapt
        
        # Apply chaotic mask
        if mask_chaotic.dim() < output.dim():
            mask_chaotic_expanded = mask_chaotic.unsqueeze(-1)
        else:
            mask_chaotic_expanded = mask_chaotic
        
        # Update chaotic group tokens
        output = torch.where(mask_chaotic_expanded, chaotic_pred, output)
        
        # Compute and accumulate prediction error (scalar)
        if cache_dic.get('cached_curvature') is not None:
            # Get previous prediction for error calculation
            # For k=1, use F_latest (last full computation)
            # For k>1, approximate previous prediction using F_latest + (k-1) * v_adapt_prev
            if k == 1:
                F_prev = F_latest
            else:
                # Compute previous adaptive slope for k-1
                v_adapt_prev = compute_adaptive_slope(
                    v_curr,
                    v_prev,
                    k - 1,
                    worldcache_config['n_max'],
                    hermite_weights=worldcache_config.get('hermite_weights')
                )
                # Previous prediction: F_{k-1} = F_latest + (k-1) * v_adapt_prev
                F_prev = F_latest + (k - 1) * v_adapt_prev
            
            error = compute_prediction_error(
                cache_dic['cached_curvature'],
                chaotic_pred,
                F_prev,
                mask_chaotic,
                eps=worldcache_config['eps']
            )
            
            # Accumulate error
            cache_dic['worldcache_config']['chaotic_error_accumulated'] += error
    
    # Print Worldcache prediction statistics
    n_stable = mask_stable.sum().item() if isinstance(mask_stable, torch.Tensor) else int(mask_stable.sum())
    n_linear = mask_linear.sum().item() if isinstance(mask_linear, torch.Tensor) else int(mask_linear.sum())
    n_chaotic = mask_chaotic.sum().item() if isinstance(mask_chaotic, torch.Tensor) else int(mask_chaotic.sum())
    error_accumulated = cache_dic['worldcache_config'].get('chaotic_error_accumulated', 0.0)
    print(f"  -> Worldcache Prediction | k={k} | stable={n_stable} | linear={n_linear} | chaotic={n_chaotic} | error_accumulated={error_accumulated:.4f}", flush=True)
    
    # Return predicted split tokens (batch, seq_len*3, 24)
    return output


def worldcache_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Worldcache prediction.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :return: Predicted feature tensor (output after norm_final, before norm_out and proj_out)
    """
    mode = cache_dic.get('mode', 'worldcache')
    
    # Worldcache mode: use Worldcache prediction
    if mode == 'worldcache' and current.get('type') == 'worldcache':
        return worldcache_predict(cache_dic, current)
    
    # Fallback for unsupported modes
    raise ValueError(f"Unsupported mode '{mode}' or type '{current.get('type')}' for worldcache_formula. Use mode='worldcache' with type='worldcache'.")

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}

