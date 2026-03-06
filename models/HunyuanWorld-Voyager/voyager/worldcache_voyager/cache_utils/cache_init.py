import os
import torch


def cache_init(num_steps, output_dir=None, mode='worldcache'):
    """
    Initialization for cache.

    Args:
        num_steps: Number of denoising steps
        output_dir: Output directory for log file (optional)
        mode: Cache mode - 'worldcache' or 'original'
    """
    cache_dic = {}
    cache = {}
    cache[-1] = {}

    cache[-1]['double_stream'] = {}
    cache[-1]['single_stream'] = {}
    cache_dic['cache_counter'] = 0

    cache[-1]['final'] = {}
    cache[-1]['final']['final'] = {}
    cache[-1]['final']['final']['final'] = {}

    cache_dic['cache'] = cache
    cache_dic['mode'] = mode

    if mode == 'worldcache':
        # Worldcache mode uses adaptive prediction based on curvature.
        # cache_interval is kept for backward compatibility.
        cache_dic['cache_interval'] = 6
        cache_dic['max_order'] = 1
        cache_dic['first_enhance'] = 1

        cache_dic['worldcache_config'] = {
            'percentile_stable': float(os.environ.get('WORLDCACHE_PERCENTILE_STABLE', '0.30')),
            'percentile_chaotic': float(os.environ.get('WORLDCACHE_PERCENTILE_CHAOTIC', '0.60')),
            'n_max': int(os.environ.get('WORLDCACHE_N_MAX', '6')),
            'error_threshold': float(os.environ.get('WORLDCACHE_ERROR_THRESHOLD', '1.0')),
            'eps': 1e-8,
            'warmup_steps': 5,
        }

        cache_dic['worldcache_history'] = {
            'outputs': [],
            'steps': [],
            'velocities': [],
        }

        cache_dic['cached_curvature'] = None
        cache_dic['cached_masks'] = {
            'mask_stable': None,
            'mask_linear': None,
            'mask_chaotic': None,
        }

        cache_dic['worldcache_config']['chaotic_error_accumulated'] = 0.0

        n_max = cache_dic['worldcache_config']['n_max']
        hermite_weights = []
        for k in range(1, n_max + 1):
            x_k = min(k / n_max, 1.0)
            alpha_k = 3 * x_k * x_k - 2 * x_k * x_k * x_k
            hermite_weights.append(alpha_k)
        cache_dic['worldcache_config']['hermite_weights'] = hermite_weights

    elif mode == 'original':
        # Original mode: no acceleration, always use full computation.
        cache_dic['cache_interval'] = 1
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 0

    cache_dic['taylor_cache'] = True

    current = {}
    current['activated_steps'] = [0]

    current['step'] = 0
    current['num_steps'] = num_steps

    return cache_dic, current
