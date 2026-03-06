from types import MethodType
from aether.worldcache_aether.cache_utils import cache_init, cal_type
import loguru
import os
import sys

def apply_worldcache_aether_pipeline(pipeline):
    """
    Apply Worldcache to Aether pipeline.
    This function modifies the pipeline's __call__ method to integrate caching.
    """
    
    # Store original __call__ method at class level to avoid issues with multiple instances
    pipeline_class = pipeline.__class__
    if not hasattr(pipeline_class, '_original_call_worldcache'):
        # Save the original method from the class, not the instance
        # This ensures we get the original method even if it was already replaced
        pipeline_class._original_call_worldcache = pipeline_class.__call__
    
    # Store original transformer forward if not already stored
    if not hasattr(pipeline.transformer, '_original_forward'):
        pipeline.transformer._original_forward = pipeline.transformer.forward
    
    def _transformer_call_with_cache(self, *args, **kwargs):
        """
        Wrapper for transformer forward that injects cache_dic and current.
        """
        # Get cache_dic and current from pipeline
        if hasattr(pipeline, '_worldcache_cache_dic') and hasattr(pipeline, '_worldcache_current'):
            cache_dic = pipeline._worldcache_cache_dic
            current = pipeline._worldcache_current
            
            # Update step in current (this will be set by the loop)
            # The step should be updated before calling transformer
            # For now, we rely on the pipeline setting it
            
            # Inject cache_dic and current into kwargs
            kwargs['cache_dic'] = cache_dic
            kwargs['current'] = current
        
        # Call the patched forward method (which handles cache logic)
        return pipeline.transformer.forward(*args, **kwargs)
    
    def __call_worldcache(
        self,
        task=None,
        image=None,
        video=None,
        goal=None,
        raymap=None,
        height=None,
        width=None,
        num_frames=None,
        num_inference_steps=None,
        timesteps=None,
        guidance_scale=None,
        use_dynamic_cfg=False,
        num_videos_per_prompt=1,
        eta=0.0,
        generator=None,
        return_dict=True,
        attention_kwargs=None,
        fps=None,
        output_dir=None,
    ):
        """
        Wrapper for __call__ method that adds Worldcache caching.
        This intercepts the original __call__ and modifies the denoising loop.
        """
        # Debug: Confirm this function is being called
        print(f"[Worldcache] DEBUG: __call_worldcache called", flush=True)
        print(f"[DEBUG] __call_worldcache invoked", file=sys.stderr, flush=True)
        
        # Determine num_inference_steps if not provided
        if num_inference_steps is None:
            if task is None:
                if video is not None:
                    task = "reconstruction"
                elif goal is not None:
                    task = "planning"
                else:
                    task = "prediction"
            num_inference_steps = self._default_num_inference_steps.get(task, 50)
        
        # Get mode from environment variable or default to 'worldcache'
        mode = os.environ.get('WORLDCACHE_MODE', 'worldcache')
        # Output to both stdout and stderr to ensure visibility
        print(f"[Worldcache] Mode read from env: {mode}", flush=True)
        print(f"[DEBUG] WORLDCACHE_MODE read from env: {mode}", file=sys.stderr, flush=True)
        
        # Read Worldcache configuration from environment variables (if provided)
        worldcache_config = None
        if mode == 'worldcache':
            # Check if any Worldcache parameters are provided via environment variables
            percentile_stable_env = os.environ.get('WORLDCACHE_PERCENTILE_STABLE')
            percentile_chaotic_env = os.environ.get('WORLDCACHE_PERCENTILE_CHAOTIC')
            n_max_env = os.environ.get('WORLDCACHE_N_MAX')
            error_threshold_env = os.environ.get('WORLDCACHE_ERROR_THRESHOLD')
            
            # Build worldcache_config dict if any parameter is provided
            if percentile_stable_env is not None or percentile_chaotic_env is not None or \
               n_max_env is not None or error_threshold_env is not None:
                worldcache_config = {}
                if percentile_stable_env is not None:
                    worldcache_config['percentile_stable'] = float(percentile_stable_env)
                if percentile_chaotic_env is not None:
                    worldcache_config['percentile_chaotic'] = float(percentile_chaotic_env)
                if n_max_env is not None:
                    worldcache_config['n_max'] = int(n_max_env)
                if error_threshold_env is not None:
                    worldcache_config['error_threshold'] = float(error_threshold_env)
        
        # Initialize cache with mode and optional worldcache_config
        cache_dic, current = cache_init(num_inference_steps, output_dir=output_dir, mode=mode, worldcache_config=worldcache_config)
        
        # Diagnostic print for cache_dic
        cache_mode = cache_dic.get('mode', 'unknown')
        has_worldcache_config = 'worldcache_config' in cache_dic
        print(f"[Worldcache] Cache initialized - mode: {cache_mode}, has worldcache_config: {has_worldcache_config}", flush=True)
        print(f"[DEBUG] cache_dic mode: {cache_mode}, has worldcache_config: {has_worldcache_config}", file=sys.stderr, flush=True)
        
        # Print Worldcache mode initialization info if in worldcache mode
        if mode == 'worldcache' and 'worldcache_config' in cache_dic:
            worldcache_config_val = cache_dic['worldcache_config']
            worldcache_msg = f"[Worldcache Mode] Initialized - percentile_stable={worldcache_config_val['percentile_stable']:.2f}, percentile_chaotic={worldcache_config_val['percentile_chaotic']:.2f}, n_max={worldcache_config_val['n_max']}, error_threshold={worldcache_config_val['error_threshold']:.4f}, warmup_steps={worldcache_config_val['warmup_steps']}"
            print(worldcache_msg, flush=True)
            print(worldcache_msg, file=sys.stderr, flush=True)
        
        # Store as attributes so transformer forward can access them
        self._worldcache_cache_dic = cache_dic
        self._worldcache_current = current
        self._worldcache_step_counter = -1  # Will be incremented in forward
        
        try:
            # Get the original __call__ method from the class
            # Use class-level storage to avoid recursion issues
            original_call = self.__class__._original_call_worldcache
            
            # Safety check: ensure we're not calling ourselves
            if original_call is __call_worldcache:
                raise RuntimeError("Recursion detected: _original_call_worldcache points to __call_worldcache itself!")
            
            # Call original __call__ method
            # The transformer forward will automatically get cache_dic and current from pipeline
            result = original_call(
                self,
                task=task,
                image=image,
                video=video,
                goal=goal,
                raymap=raymap,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                use_dynamic_cfg=use_dynamic_cfg,
                num_videos_per_prompt=num_videos_per_prompt,
                eta=eta,
                generator=generator,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                fps=fps,
                output_dir=output_dir,
            )
        finally:
            # Clean up attributes to prevent state leakage between calls
            if hasattr(self, '_worldcache_cache_dic'):
                del self._worldcache_cache_dic
            if hasattr(self, '_worldcache_current'):
                del self._worldcache_current
            if hasattr(self, '_worldcache_step_counter'):
                del self._worldcache_step_counter
        
        return result
    
    # Replace __call__ method
    # Note: Python looks up __call__ in the class, not the instance
    # So we need to replace it at the class level
    # Only replace if not already replaced to avoid issues
    if pipeline_class.__call__ is not __call_worldcache:
        pipeline_class.__call__ = __call_worldcache
    
    loguru.logger.info("Worldcache Aether pipeline applied")
    
    # Store reference to pipeline in transformer for forward to access
    # This needs to be set at initialization time, not during each __call__
    if not hasattr(pipeline.transformer, '_worldcache_pipeline_ref'):
        pipeline.transformer._worldcache_pipeline_ref = pipeline
    
    # Print mode confirmation after applying
    default_mode = os.environ.get('WORLDCACHE_MODE', 'worldcache')
    print(f"[Worldcache] Pipeline applied - Default mode: {default_mode}", flush=True)