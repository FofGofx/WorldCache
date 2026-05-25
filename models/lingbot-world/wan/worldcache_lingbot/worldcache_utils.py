import torch


def compute_curvature(history, eps=1e-8):
    outputs = history["outputs"]
    steps = history["steps"]

    if len(outputs) < 3 or len(steps) < 3:
        return None

    feature_t2 = outputs[-3]
    feature_t1 = outputs[-2]
    feature_t0 = outputs[-1]

    step_t2 = steps[-3]
    step_t1 = steps[-2]
    step_t0 = steps[-1]

    dt_prev = step_t1 - step_t2
    dt_curr = step_t0 - step_t1
    if dt_prev <= 0 or dt_curr <= 0:
        return None

    velocity_prev = (feature_t1 - feature_t2) / dt_prev
    velocity_curr = (feature_t0 - feature_t1) / dt_curr
    acceleration = (velocity_curr - velocity_prev) / dt_curr

    velocity_norm = torch.norm(velocity_curr, dim=-1)
    acceleration_norm = torch.norm(acceleration, dim=-1)
    return acceleration_norm / (velocity_norm.square() + eps)


def compute_token_groups(curvature, percentile_stable=0.30, percentile_chaotic=0.60):
    curvature_flat = curvature.float().flatten()
    stable_threshold = torch.quantile(curvature_flat, percentile_stable)
    chaotic_threshold = torch.quantile(curvature_flat, percentile_chaotic)

    mask_stable = curvature < stable_threshold
    mask_chaotic = curvature >= chaotic_threshold
    mask_linear = ~(mask_stable | mask_chaotic)
    return mask_stable, mask_linear, mask_chaotic


def compute_adaptive_slope(v_curr, v_prev, k, n_max, hermite_weights=None):
    if hermite_weights is not None and 1 <= k <= len(hermite_weights):
        alpha = hermite_weights[k - 1]
    else:
        x_k = min(k / max(n_max, 1), 1.0)
        alpha = 3 * x_k * x_k - 2 * x_k * x_k * x_k

    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=v_curr.dtype, device=v_curr.device)

    return (1 - alpha) * v_curr + alpha * v_prev


def compute_prediction_error(
    cached_curvature,
    x_t,
    x_prev,
    mask_chaotic,
    return_details=False,
):
    delta = x_t - x_prev
    delta_norm = torch.norm(delta, dim=-1)

    if cached_curvature.shape != delta_norm.shape and cached_curvature.numel() == delta_norm.numel():
        cached_curvature = cached_curvature.reshape(delta_norm.shape)

    curvature_chaotic = cached_curvature * mask_chaotic
    delta_chaotic = delta_norm * mask_chaotic
    error_tensor = curvature_chaotic * delta_chaotic
    error_value = error_tensor.abs().mean().item()

    if not return_details:
        return error_value

    nonzero_delta = delta_chaotic[delta_chaotic != 0]
    nonzero_curvature = curvature_chaotic[curvature_chaotic != 0]
    nonzero_error = error_tensor.abs()[error_tensor != 0]

    return {
        "error": error_value,
        "delta_mean": nonzero_delta.mean().item() if nonzero_delta.numel() else 0.0,
        "delta_max": nonzero_delta.max().item() if nonzero_delta.numel() else 0.0,
        "curvature_mean": nonzero_curvature.mean().item() if nonzero_curvature.numel() else 0.0,
        "curvature_max": nonzero_curvature.max().item() if nonzero_curvature.numel() else 0.0,
        "error_tensor_mean": nonzero_error.mean().item() if nonzero_error.numel() else 0.0,
        "error_tensor_max": nonzero_error.max().item() if nonzero_error.numel() else 0.0,
    }


def update_history_buffer(history, output, step, max_history=3):
    history["outputs"].append(output.detach().clone())
    history["steps"].append(step)

    if len(history["outputs"]) > max_history:
        history["outputs"] = history["outputs"][-max_history:]
        history["steps"] = history["steps"][-max_history:]
        history["velocities"] = history["velocities"][-(max_history - 1):]


def compute_velocity(feature_t, feature_prev, dt):
    if dt <= 0:
        return None
    return (feature_t - feature_prev) / dt
