import sys

def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    mode = cache_dic.get('mode', 'worldcache')
    
    # Worldcache mode decision logic
    if mode == 'worldcache':
        worldcache_config = cache_dic['worldcache_config']
        warmup_steps = worldcache_config['warmup_steps']
        num_steps = current['num_steps']
        current_step = current['step']
        
        # Calculate k (prediction steps since last full computation)
        k = 0
        if len(current['activated_steps']) > 0:
            last_full_step = current['activated_steps'][-1]
            k = current_step - last_full_step
        error_accumulated = cache_dic['worldcache_config'].get('chaotic_error_accumulated', 0.0)
        
        # 1. Warm-up阶段（前warmup_steps步）：强制full computation
        if current_step < warmup_steps:
            current['type'] = 'full'
            cache_dic['cache_counter'] = 0
            current['activated_steps'].append(current['step'])
            print(f"  -> Calculation Type: full | Reason: warm-up (step {current_step} < {warmup_steps}) | k=0 | error={error_accumulated:.4f}", flush=True)
        
        # 2. 最后一步（num_steps-1）：强制full computation
        elif current_step == num_steps - 1:
            current['type'] = 'full'
            cache_dic['cache_counter'] = 0
            current['activated_steps'].append(current['step'])
            # Reset error accumulator
            cache_dic['worldcache_config']['chaotic_error_accumulated'] = 0.0
            print(f"  -> Calculation Type: full | Reason: final step (step {current_step} == {num_steps-1}) | k={k} | error={error_accumulated:.4f}", flush=True)
        
        # 3. 曲率未计算时：full computation
        elif cache_dic.get('cached_curvature') is None:
            current['type'] = 'full'
            cache_dic['cache_counter'] = 0
            current['activated_steps'].append(current['step'])
            print(f"  -> Calculation Type: full | Reason: curvature not computed yet | k={k} | error={error_accumulated:.4f}", flush=True)
        
        # 4. 误差超标：full computation（重置误差）
        elif cache_dic['worldcache_config']['chaotic_error_accumulated'] >= worldcache_config['error_threshold']:
            current['type'] = 'full'
            cache_dic['cache_counter'] = 0
            current['activated_steps'].append(current['step'])
            # Reset error accumulator
            cache_dic['worldcache_config']['chaotic_error_accumulated'] = 0.0
            print(f"  -> Calculation Type: full | Reason: error threshold exceeded (error {error_accumulated:.4f} >= {worldcache_config['error_threshold']:.4f}) | k={k} | error={error_accumulated:.4f}", flush=True)
        
        # 5. 预测步数超过N_max：full computation
        # 计算从上次full computation到当前步的预测步数
        elif len(current['activated_steps']) > 0:
            last_full_step = current['activated_steps'][-1]
            prediction_steps = current_step - last_full_step
            if prediction_steps > worldcache_config['n_max']:
                current['type'] = 'full'
                cache_dic['cache_counter'] = 0
                current['activated_steps'].append(current['step'])
                # Reset error accumulator
                cache_dic['worldcache_config']['chaotic_error_accumulated'] = 0.0
                print(f"  -> Calculation Type: full | Reason: prediction steps exceeded (k={prediction_steps} > {worldcache_config['n_max']}) | k={prediction_steps} | error={error_accumulated:.4f}", flush=True)
            else:
                # 6. 其他情况：使用Worldcache预测
                cache_dic['cache_counter'] += 1
                current['type'] = 'worldcache'
                print(f"  -> Calculation Type: worldcache | Reason: using Worldcache prediction | k={prediction_steps} | error={error_accumulated:.4f}", flush=True)
        else:
            # Fallback: should not happen, but use full computation
            current['type'] = 'full'
            cache_dic['cache_counter'] = 0
            current['activated_steps'].append(current['step'])
            print(f"  -> Calculation Type: full | Reason: fallback (no activated steps) | k=0 | error={error_accumulated:.4f}", flush=True)
    
    # Original mode: always use full computation (no acceleration)
    elif mode == 'original':
        # Always perform full computation at every step
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        print(f"  -> Calculation Type: full | Reason: original mode (no acceleration) | step={current['step']}", flush=True)
    
    else:
        raise ValueError(f"Unsupported cache mode: {mode}. Use 'worldcache' or 'original'.")
