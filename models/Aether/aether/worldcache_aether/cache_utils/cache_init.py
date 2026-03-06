import os

def cache_init(num_steps, output_dir=None, mode=None, worldcache_config=None):   
    '''
    Initialization for cache.
    
    Args:
        num_steps: Number of denoising steps
        output_dir: Output directory for log file (optional)
        mode: Cache mode - 'worldcache' or 'original'
              If None, reads from WORLDCACHE_MODE environment variable or defaults to 'worldcache'
        worldcache_config: Optional dict with Worldcache configuration parameters. If None, uses default values.
                     Expected keys: 'percentile_stable', 'percentile_chaotic', 'n_max', 'error_threshold'
                     Optional keys: 'eps', 'warmup_steps' (defaults: 1e-8, 3)
    '''
    cache_dic = {}
    cache = {}
    cache[-1]={}

    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    cache[-1]['final'] = {}
    cache[-1]['final']['final'] = {}
    cache[-1]['final']['final']['final'] = {}

    cache_dic['cache'] = cache

    # Get mode from parameter, environment variable, or default
    if mode is None:
        mode = os.environ.get('WORLDCACHE_MODE', 'worldcache')
    cache_dic['mode'] = mode

    if mode == 'worldcache':
        # Worldcache mode uses adaptive prediction based on curvature
        # For Worldcache, we still need cache_interval for backward compatibility
        # but the actual decision logic is in cal_type
        cache_dic['cache_interval'] = 1  # Not used in Worldcache mode
        cache_dic['max_order'] = 1
        cache_dic['first_enhance'] = 3
        
        # Worldcache configuration - use provided config or default values
        if worldcache_config is not None:
            # Use provided config, with defaults for optional parameters
            cache_dic['worldcache_config'] = {
                'percentile_stable': worldcache_config.get('percentile_stable', 0.30),
                'percentile_chaotic': worldcache_config.get('percentile_chaotic', 0.60),
                'n_max': worldcache_config.get('n_max', 2),
                'error_threshold': worldcache_config.get('error_threshold', 0.2),
                'eps': worldcache_config.get('eps', 1e-8),                    # 数值稳定性
                'warmup_steps': worldcache_config.get('warmup_steps', 10),      # Warm-up步数
            }
        else:
            # Default Worldcache configuration (backward compatible)
            cache_dic['worldcache_config'] = {
                'percentile_stable': 0.30,      # 稳态组阈值（20%分位数）
                'percentile_chaotic': 0.60,      # 混沌组阈值（80%分位数）
                'n_max': 2,                     # 最大预测步数
                'error_threshold': 0.2,          # 误差阈值
                'eps': 1e-8,                    # 数值稳定性
                'warmup_steps': 10,              # Warm-up步数
            }
        
        # Worldcache history buffer (最多保留3次完整计算)
        cache_dic['worldcache_history'] = {
            'outputs': [],      # 存储final layer输出 [F_t-2, F_t-1, F_t]
            'steps': [],        # 对应的步数 [step_t-2, step_t-1, step_t]
            'velocities': [],   # 速度 [v_t-1, v_t] (用于计算加速度)
        }
        
        # Cached curvature and masks (只在full computation时更新)
        cache_dic['cached_curvature'] = None
        cache_dic['cached_masks'] = {
            'mask_stable': None,
            'mask_linear': None,
            'mask_chaotic': None,
        }
        
        # 混沌组累加误差（标量）
        cache_dic['worldcache_config']['chaotic_error_accumulated'] = 0.0
        
        # 预计算Hermite权重表
        n_max = cache_dic['worldcache_config']['n_max']
        hermite_weights = []
        for k in range(1, n_max + 1):
            x_k = min(k / n_max, 1.0)
            alpha_k = 3 * x_k * x_k - 2 * x_k * x_k * x_k
            hermite_weights.append(alpha_k)
        cache_dic['worldcache_config']['hermite_weights'] = hermite_weights
    
    elif mode == 'original':
        # Original mode: no acceleration, full computation at every step
        # Still goes through Worldcache path for logging functionality
        cache_dic['cache_interval'] = 1  # Every step is full computation
        cache_dic['max_order'] = 0  # For derivative calculation (logging only, not used for prediction)
        cache_dic['first_enhance'] = 0  # Can compute derivative from first step
    
    else:
        raise ValueError(f"Unsupported cache mode: {mode}. Use 'worldcache' or 'original'.")
    
    cache_dic['taylor_cache'] = True
    
    current = {}
    current['activated_steps'] = [0]

    current['step'] = 0
    current['num_steps'] = num_steps

    return cache_dic, current
