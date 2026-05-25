import os


def _build_worldcache_config(worldcache_config=None):
    config = {
        "percentile_stable": float(os.environ.get("WORLDCACHE_PERCENTILE_STABLE", "0.30")),
        "percentile_chaotic": float(os.environ.get("WORLDCACHE_PERCENTILE_CHAOTIC", "0.60")),
        "n_max": int(os.environ.get("WORLDCACHE_N_MAX", "2")),
        "error_threshold": float(os.environ.get("WORLDCACHE_ERROR_THRESHOLD", "0.2")),
        "eps": 1e-8,
        "warmup_steps": 5,
    }
    if worldcache_config is not None:
        config.update(worldcache_config)

    hermite_weights = []
    for k in range(1, config["n_max"] + 1):
        x_k = min(k / max(config["n_max"], 1), 1.0)
        hermite_weights.append(3 * x_k * x_k - 2 * x_k * x_k * x_k)
    config["hermite_weights"] = hermite_weights
    config["chaotic_error_accumulated"] = 0.0
    return config


def cache_init(num_steps, mode=None, worldcache_config=None):
    if mode is None:
        mode = os.environ.get("WORLDCACHE_MODE", "worldcache")
    if mode not in {"original", "worldcache"}:
        raise ValueError(f"Unsupported WORLDCACHE_MODE: {mode}")

    cache_dic = {
        "mode": mode,
        "cache_counter": 0,
        "worldcache_history": {
            "outputs": [],
            "steps": [],
            "velocities": [],
        },
        "cached_curvature": None,
        "cached_masks": {
            "mask_stable": None,
            "mask_linear": None,
            "mask_chaotic": None,
        },
    }

    if mode == "worldcache":
        cache_dic["worldcache_config"] = _build_worldcache_config(worldcache_config)

    current = {
        "step": 0,
        "num_steps": num_steps,
        "type": None,
        "branch": None,
        "model_name": None,
        "segment_start_step": 0,
        "activated_steps": [],
    }
    return cache_dic, current
