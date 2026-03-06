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

    Curvature = ||a|| / (||v||^2 + eps)
    where:
    - v_t = (F_t - F_{t-1}) / dt (velocity)
    - a_t = (v_t - v_{t-1}) / dt (acceleration)

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

    if len(outputs) < 3:
        return None

    F_t_minus_2 = outputs[-3]
    F_t_minus_1 = outputs[-2]
    F_t = outputs[-1]

    step_t_minus_2 = steps[-3]
    step_t_minus_1 = steps[-2]
    step_t = steps[-1]

    dt1 = step_t_minus_1 - step_t_minus_2
    dt2 = step_t - step_t_minus_1

    if dt1 <= 0 or dt2 <= 0:
        return None

    v_t_minus_1 = (F_t_minus_1 - F_t_minus_2) / dt1
    v_t = (F_t - F_t_minus_1) / dt2

    a_t = (v_t - v_t_minus_1) / dt2

    v_norm = torch.norm(v_t, dim=-1)
    a_norm = torch.norm(a_t, dim=-1)

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
        mask_linear: Boolean mask for linear tokens (20% <= curvature < 80% percentile)
        mask_chaotic: Boolean mask for chaotic tokens (curvature >= 80% percentile)
    """
    curvature_flat = curvature.flatten()

    p_stable = torch.quantile(curvature_flat, percentile_stable)
    p_chaotic = torch.quantile(curvature_flat, percentile_chaotic)

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

    v_adapt = (1 - alpha_k) * v_curr + alpha_k * v_prev
    where alpha_k = 3*x_k^2 - 2*x_k^3, x_k = min(k / n_max, 1.0)

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
        x_k = min(k / n_max, 1.0)
        alpha_k = 3 * x_k * x_k - 2 * x_k * x_k * x_k

    if not isinstance(alpha_k, torch.Tensor):
        alpha_k = torch.tensor(alpha_k, device=v_curr.device, dtype=v_curr.dtype)

    v_adapt = (1 - alpha_k) * v_curr + alpha_k * v_prev

    return v_adapt


def compute_prediction_error(
    cached_curvature: torch.Tensor,
    x_t: torch.Tensor,
    x_t_minus_1: torch.Tensor,
    mask_chaotic: torch.Tensor,
    eps: float = 1e-8,
    return_details: bool = False
):
    """
    Compute prediction error for chaotic group using scalar accumulation.

    error = cached_curvature * ds
    where ds = ||x_t - x_{t-1}|| (only for chaotic tokens)

    The error is computed as a scalar by taking .abs().mean() of the tensor.

    Args:
        cached_curvature: Cached curvature values (shape: [batch, seq_len] or [batch*seq_len])
        x_t: Current prediction (shape: [batch, seq_len, features])
        x_t_minus_1: Previous prediction (shape: [batch, seq_len, features])
        mask_chaotic: Boolean mask for chaotic tokens (shape: [batch, seq_len])
        eps: Small value for numerical stability
        return_details: If True, return detailed statistics dictionary

    Returns:
        error: Scalar error value (if return_details=False)
        dict: Dictionary with error and detailed statistics (if return_details=True)
    """
    diff = x_t - x_t_minus_1
    ds = torch.norm(diff, dim=-1)

    if cached_curvature.shape != ds.shape:
        if cached_curvature.numel() == ds.numel():
            cached_curvature = cached_curvature.reshape(ds.shape)
        else:
            pass

    curvature_chaotic = cached_curvature * mask_chaotic
    ds_chaotic = ds * mask_chaotic

    error_tensor = curvature_chaotic * ds_chaotic

    error = error_tensor.abs().mean().item()

    if return_details:
        ds_chaotic_nonzero = ds_chaotic[ds_chaotic != 0]
        ds_mean = ds_chaotic_nonzero.mean().item() if len(ds_chaotic_nonzero) > 0 else 0.0
        ds_max = ds_chaotic_nonzero.max().item() if len(ds_chaotic_nonzero) > 0 else 0.0
        ds_min = ds_chaotic_nonzero.min().item() if len(ds_chaotic_nonzero) > 0 else 0.0

        curvature_chaotic_nonzero = curvature_chaotic[curvature_chaotic != 0]
        curvature_mean = curvature_chaotic_nonzero.mean().item() if len(curvature_chaotic_nonzero) > 0 else 0.0
        curvature_max = curvature_chaotic_nonzero.max().item() if len(curvature_chaotic_nonzero) > 0 else 0.0
        curvature_min = curvature_chaotic_nonzero.min().item() if len(curvature_chaotic_nonzero) > 0 else 0.0

        error_tensor_abs = error_tensor.abs()
        error_tensor_nonzero = error_tensor_abs[error_tensor_abs != 0]
        error_tensor_mean = error_tensor_nonzero.mean().item() if len(error_tensor_nonzero) > 0 else 0.0
        error_tensor_max = error_tensor_nonzero.max().item() if len(error_tensor_nonzero) > 0 else 0.0
        error_tensor_min = error_tensor_nonzero.min().item() if len(error_tensor_nonzero) > 0 else 0.0

        return {
            'error': error,
            'ds_mean': ds_mean,
            'ds_max': ds_max,
            'ds_min': ds_min,
            'curvature_mean': curvature_mean,
            'curvature_max': curvature_max,
            'curvature_min': curvature_min,
            'error_tensor_mean': error_tensor_mean,
            'error_tensor_max': error_tensor_max,
            'error_tensor_min': error_tensor_min,
        }

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
        output: New output tensor (final layer output)
        step: Current step index
        max_history: Maximum number of outputs to keep (default 3)
    """
    history['outputs'].append(output.detach().clone())
    history['steps'].append(step)

    if len(history['outputs']) > max_history:
        history['outputs'] = history['outputs'][-max_history:]
        history['steps'] = history['steps'][-max_history:]
        if len(history['velocities']) > max_history - 1:
            history['velocities'] = history['velocities'][-(max_history - 1):]


def compute_velocity(
    F_t: torch.Tensor,
    F_t_minus_1: torch.Tensor,
    dt: int
) -> torch.Tensor:
    """
    Compute velocity: v_t = (F_t - F_{t-1}) / dt

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
