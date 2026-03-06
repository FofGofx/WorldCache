"""
Worldcache utility functions.
Implements curvature computation, token grouping, and adaptive prediction strategies.
"""
import torch
from typing import Dict, Tuple, Optional


def compute_curvature(
    history: Dict,
    current_step: int,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute curvature for each token based on historical outputs.
    
    Curvature = ||a|| / (||v||² + eps)
    where:
    - v_t = (F_t - F_{t-1}) / Δt (velocity)
    - a_t = (v_t - v_{t-1}) / Δt (acceleration)
    
    Args:
        history: Dictionary containing 'outputs', 'steps', 'velocities'
        current_step: Current time step index
        eps: Small value for numerical stability
    
    Returns:
        curvature: Tensor of curvature values (one per token)
    """
    outputs = history['outputs']
    steps = history['steps']
    velocities = history['velocities']
    
    # Need at least 3 outputs to compute acceleration
    if len(outputs) < 3:
        return None
    
    # Get the last 3 outputs and steps
    F_t_minus_2 = outputs[-3]  # F_{t-2}
    F_t_minus_1 = outputs[-2]  # F_{t-1}
    F_t = outputs[-1]          # F_t
    
    step_t_minus_2 = steps[-3]
    step_t_minus_1 = steps[-2]
    step_t = steps[-1]
    
    # Compute time differences (using step indices, not timestep values)
    dt1 = step_t_minus_1 - step_t_minus_2  # Δt for first interval
    dt2 = step_t - step_t_minus_1          # Δt for second interval
    
    if dt1 <= 0 or dt2 <= 0:
        return None
    
    # Compute velocities
    # v_{t-1} = (F_{t-1} - F_{t-2}) / dt1
    v_t_minus_1 = (F_t_minus_1 - F_t_minus_2) / dt1
    
    # v_t = (F_t - F_{t-1}) / dt2
    v_t = (F_t - F_t_minus_1) / dt2
    
    # Compute acceleration: a_t = (v_t - v_{t-1}) / dt2
    # Use dt2 as the time interval for acceleration
    a_t = (v_t - v_t_minus_1) / dt2
    
    # Compute norms: flatten to compute per-token norm
    # Assuming outputs are of shape (batch, seq_len, features)
    # We compute norm along the feature dimension for each token
    v_norm = torch.norm(v_t, dim=-1)  # Shape: (batch, seq_len)
    a_norm = torch.norm(a_t, dim=-1)  # Shape: (batch, seq_len)
    
    # Compute curvature: ||a|| / (||v||² + eps)
    curvature = a_norm / (v_norm ** 2 + eps)
    
    return curvature


def compute_token_groups(
    curvature: torch.Tensor,
    percentile_stable: float = 0.20,
    percentile_chaotic: float = 0.80
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Group tokens into stable, linear, and chaotic groups based on curvature.
    
    Args:
        curvature: Tensor of curvature values (one per token)
        percentile_stable: Percentile threshold for stable group (default 20%)
        percentile_chaotic: Percentile threshold for chaotic group (default 80%)
    
    Returns:
        mask_stable: Boolean mask for stable tokens (curvature < 20% percentile)
        mask_linear: Boolean mask for linear tokens (20% ≤ curvature < 80% percentile)
        mask_chaotic: Boolean mask for chaotic tokens (curvature ≥ 80% percentile)
    """
    # Flatten curvature to compute percentiles across all tokens
    curvature_flat = curvature.flatten()
    
    # Compute percentiles
    # Convert to float32 since torch.quantile() only supports float or double dtype
    p_stable = torch.quantile(curvature_flat.float(), percentile_stable)
    p_chaotic = torch.quantile(curvature_flat.float(), percentile_chaotic)
    
    # Generate masks
    mask_stable = curvature < p_stable
    mask_chaotic = curvature >= p_chaotic
    mask_linear = ~(mask_stable | mask_chaotic)
    
    return mask_stable, mask_linear, mask_chaotic


def compute_adaptive_slope(
    v_curr: torch.Tensor,
    v_prev: torch.Tensor,
    k: int,
    n_max: int,
    hermite_weights: Optional[list] = None
) -> torch.Tensor:
    """
    Compute adaptive slope using Hermite interpolation.
    
    v_adapt = (1 - α_k) · v_curr + α_k · v_prev
    where α_k = 3x_k² - 2x_k³, x_k = min(k / n_max, 1.0)
    
    Args:
        v_curr: Current velocity (slope)
        v_prev: Previous velocity (slope)
        k: Prediction step number (1, 2, ..., n_max)
        n_max: Maximum prediction steps
        hermite_weights: Precomputed Hermite weights (optional)
    
    Returns:
        v_adapt: Adaptive slope
    """
    if hermite_weights is not None and 1 <= k <= len(hermite_weights):
        alpha_k = hermite_weights[k - 1]
    else:
        # Compute on the fly if weights not provided
        x_k = min(k / n_max, 1.0)
        alpha_k = 3 * x_k * x_k - 2 * x_k * x_k * x_k
    
    # Convert to tensor if needed
    if not isinstance(alpha_k, torch.Tensor):
        alpha_k = torch.tensor(alpha_k, device=v_curr.device, dtype=v_curr.dtype)
    
    # Compute adaptive slope
    v_adapt = (1 - alpha_k) * v_curr + alpha_k * v_prev
    
    return v_adapt


def compute_prediction_error(
    cached_curvature: torch.Tensor,
    x_t: torch.Tensor,
    x_t_minus_1: torch.Tensor,
    mask_chaotic: torch.Tensor,
    eps: float = 1e-8
) -> float:
    """
    Compute prediction error for chaotic group using scalar accumulation.
    
    error = cached_curvature × ds
    where ds = ||x_t - x_{t-1}|| (only for chaotic tokens)
    
    The error is computed as a scalar by taking .abs().mean() of the tensor.
    
    Args:
        cached_curvature: Cached curvature values (shape: [batch, seq_len] or [batch*seq_len])
        x_t: Current prediction (shape: [batch, seq_len, features])
        x_t_minus_1: Previous prediction (shape: [batch, seq_len, features])
        mask_chaotic: Boolean mask for chaotic tokens (shape: [batch, seq_len])
        eps: Small value for numerical stability
    
    Returns:
        error: Scalar error value
    """
    # Compute difference
    diff = x_t - x_t_minus_1  # Shape: [batch, seq_len, features]
    
    # Compute norm along feature dimension for each token
    ds = torch.norm(diff, dim=-1)  # Shape: [batch, seq_len]
    
    # Ensure mask_chaotic and cached_curvature have compatible shapes with ds
    # ds shape: [batch, seq_len]
    # mask_chaotic shape: [batch, seq_len]
    # cached_curvature shape: [batch, seq_len] or flattened
    
    # Flatten if needed for compatibility
    if cached_curvature.shape != ds.shape:
        # Reshape cached_curvature to match ds
        if cached_curvature.numel() == ds.numel():
            cached_curvature = cached_curvature.reshape(ds.shape)
        else:
            # If shapes don't match, use broadcasting
            pass
    
    # Apply chaotic mask to both curvature and ds
    curvature_chaotic = cached_curvature * mask_chaotic
    ds_chaotic = ds * mask_chaotic
    
    # Compute error = cached_curvature × ds (only for chaotic tokens)
    error_tensor = curvature_chaotic * ds_chaotic
    
    # Convert to scalar using .abs().mean() - average over all elements
    error = error_tensor.abs().mean().item()
    
    return error


def update_history_buffer(
    history: Dict,
    output: torch.Tensor,
    step: int,
    max_history: int = 3
):
    """
    Update history buffer, keeping only the most recent max_history outputs.
    
    Args:
        history: History dictionary to update
        output: New output tensor (output after norm_final, before norm_out and proj_out)
        step: Current step index
        max_history: Maximum number of outputs to keep (default 3)
    """
    history['outputs'].append(output.detach().clone())
    history['steps'].append(step)
    
    # Keep only the most recent max_history outputs
    if len(history['outputs']) > max_history:
        history['outputs'] = history['outputs'][-max_history:]
        history['steps'] = history['steps'][-max_history:]
        # Also trim velocities if present
        if len(history['velocities']) > max_history - 1:
            history['velocities'] = history['velocities'][-(max_history - 1):]


def compute_velocity(
    F_t: torch.Tensor,
    F_t_minus_1: torch.Tensor,
    dt: int
) -> torch.Tensor:
    """
    Compute velocity: v_t = (F_t - F_{t-1}) / Δt
    
    Args:
        F_t: Current output
        F_t_minus_1: Previous output
        dt: Time difference (in steps)
    
    Returns:
        v_t: Velocity tensor
    """
    if dt <= 0:
        return None
    return (F_t - F_t_minus_1) / dt


def split_modalities(
    latents: torch.Tensor,
    latent_channels: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split 56-channel latents into RGB, Depth, and Raymap modalities.
    
    Args:
        latents: Tensor of shape (batch, num_frames, 56, height, width)
        latent_channels: Number of channels per modality (default 16 for RGB/Depth)
    
    Returns:
        rgb_latents: RGB latents (channels 0-15), shape (batch, num_frames, 16, height, width)
        depth_latents: Depth latents (channels 16-31), shape (batch, num_frames, 16, height, width)
        raymap_latents: Raymap latents (channels 32-55), shape (batch, num_frames, 24, height, width)
    """
    rgb_latents = latents[:, :, :latent_channels]
    depth_latents = latents[:, :, latent_channels:latent_channels * 2]
    raymap_latents = latents[:, :, latent_channels * 2:]
    return rgb_latents, depth_latents, raymap_latents


def recombine_modalities(
    rgb_latents: torch.Tensor,
    depth_latents: torch.Tensor,
    raymap_latents: torch.Tensor
) -> torch.Tensor:
    """
    Recombine RGB, Depth, and Raymap latents back into 56-channel format.
    
    Args:
        rgb_latents: RGB latents (channels 0-15), shape (batch, num_frames, 16, height, width)
        depth_latents: Depth latents (channels 16-31), shape (batch, num_frames, 16, height, width)
        raymap_latents: Raymap latents (channels 32-55), shape (batch, num_frames, 24, height, width)
    
    Returns:
        latents: Combined latents, shape (batch, num_frames, 56, height, width)
    """
    return torch.cat([rgb_latents, depth_latents, raymap_latents], dim=2)


def flatten_to_tokens(
    latents: torch.Tensor,
    patch_size: int,
    patch_size_t: Optional[int] = None
) -> torch.Tensor:
    """
    Convert spatial latents to token sequence.
    Each spatial position becomes a token with channel features.
    
    Args:
        latents: Tensor of shape (batch, num_frames, channels, height, width)
        patch_size: Spatial patch size (for compatibility, not used in this implementation)
        patch_size_t: Temporal patch size (optional, for compatibility)
    
    Returns:
        tokens: Token sequence, shape (batch, seq_len, channels)
        where seq_len = num_frames * height * width
    """
    batch_size, num_frames, channels, height, width = latents.shape
    # Flatten spatial and temporal dimensions: (batch, num_frames * height * width, channels)
    tokens = latents.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames * height * width, channels)
    return tokens


def unflatten_from_tokens(
    tokens: torch.Tensor,
    patch_size: int,
    patch_size_t: Optional[int],
    num_frames: int,
    height: int,
    width: int,
    channels: int
) -> torch.Tensor:
    """
    Convert token sequence back to spatial latents.
    This is the inverse of flatten_to_tokens operation.
    
    Args:
        tokens: Token sequence, shape (batch, seq_len, channels)
        patch_size: Spatial patch size (for compatibility, not used)
        patch_size_t: Temporal patch size (optional, for compatibility)
        num_frames: Number of frames
        height: Height of output
        width: Width of output
        channels: Number of channels
    
    Returns:
        latents: Spatial latents, shape (batch, num_frames, channels, height, width)
    """
    batch_size = tokens.shape[0]
    # Reshape back: (batch, num_frames, height, width, channels) -> (batch, num_frames, channels, height, width)
    latents = tokens.reshape(batch_size, num_frames, height, width, channels)
    latents = latents.permute(0, 1, 4, 2, 3)
    return latents