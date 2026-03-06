def cal_type(cache_dic, current):
    """
    Determine calculation type for this step.
    """
    mode = cache_dic.get('mode', 'worldcache')

    # Original mode: always use full computation (no acceleration)
    if mode == 'original':
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        return

    if mode == 'worldcache':
        worldcache_config = cache_dic['worldcache_config']
        warmup_steps = worldcache_config['warmup_steps']
        num_steps = current['num_steps']
        current_step = current['step']

        last_full_step = current['activated_steps'][-1] if len(current['activated_steps']) > 0 else None
        k = current_step - last_full_step if last_full_step is not None else 0
        accumulated_error = cache_dic['worldcache_config'].get('chaotic_error_accumulated', 0.0)

        if current_step < warmup_steps:
            current['type'] = 'full'
            cache_dic['cache_counter'] = 0
            current['activated_steps'].append(current['step'])
            print(f"[Worldcache Decision] Step: {current_step}/{num_steps} | Last Full: {last_full_step if last_full_step is not None else 'N/A'} | k: {k} | Error: {accumulated_error:.4f} | Decision: FULL | Reason: WARMUP")

        elif current_step == num_steps - 1:
            current['type'] = 'full'
            cache_dic['cache_counter'] = 0
            current['activated_steps'].append(current['step'])
            cache_dic['worldcache_config']['chaotic_error_accumulated'] = 0.0
            print(f"[Worldcache Decision] Step: {current_step}/{num_steps} | Last Full: {last_full_step if last_full_step is not None else 'N/A'} | k: {k} | Error: {accumulated_error:.4f} | Decision: FULL | Reason: LAST_STEP | Predicted Steps: {k}")

        elif cache_dic.get('cached_curvature') is None:
            current['type'] = 'full'
            cache_dic['cache_counter'] = 0
            current['activated_steps'].append(current['step'])
            print(f"[Worldcache Decision] Step: {current_step}/{num_steps} | Last Full: {last_full_step if last_full_step is not None else 'N/A'} | k: {k} | Error: {accumulated_error:.4f} | Decision: FULL | Reason: NO_CURVATURE | Predicted Steps: {k}")

        elif cache_dic['worldcache_config']['chaotic_error_accumulated'] >= worldcache_config['error_threshold']:
            current['type'] = 'full'
            cache_dic['cache_counter'] = 0
            current['activated_steps'].append(current['step'])
            cache_dic['worldcache_config']['chaotic_error_accumulated'] = 0.0
            print(f"[Worldcache Decision] Step: {current_step}/{num_steps} | Last Full: {last_full_step if last_full_step is not None else 'N/A'} | k: {k} | Error: {accumulated_error:.4f} | Decision: FULL | Reason: ERROR_EXCEEDED | Threshold: {worldcache_config['error_threshold']:.4f} | Predicted Steps: {k}")

        elif len(current['activated_steps']) > 0:
            last_full_step = current['activated_steps'][-1]
            prediction_steps = current_step - last_full_step
            if prediction_steps > worldcache_config['n_max']:
                current['type'] = 'full'
                cache_dic['cache_counter'] = 0
                current['activated_steps'].append(current['step'])
                cache_dic['worldcache_config']['chaotic_error_accumulated'] = 0.0
                print(f"[Worldcache Decision] Step: {current_step}/{num_steps} | Last Full: {last_full_step} | k: {prediction_steps} | Error: {accumulated_error:.4f} | Decision: FULL | Reason: K_EXCEEDED | N_max: {worldcache_config['n_max']} | Predicted Steps: {prediction_steps}")
            else:
                cache_dic['cache_counter'] += 1
                current['type'] = 'worldcache'
                print(f"[Worldcache Decision] Step: {current_step}/{num_steps} | Last Full: {last_full_step} | k: {prediction_steps} | Error: {accumulated_error:.4f} | Decision: WORLDCACHE | Reason: NORMAL_PREDICT")
        else:
            current['type'] = 'full'
            cache_dic['cache_counter'] = 0
            current['activated_steps'].append(current['step'])
            print(f"[Worldcache Decision] Step: {current_step}/{num_steps} | Last Full: N/A | k: 0 | Error: {accumulated_error:.4f} | Decision: FULL | Reason: FALLBACK")

        return

    raise ValueError("Unsupported cache mode")
