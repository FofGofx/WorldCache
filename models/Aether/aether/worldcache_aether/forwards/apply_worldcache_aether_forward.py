import torch
from typing import Optional, Union, Dict, Tuple, Any
from types import MethodType
from aether.worldcache_aether.cache_utils import cal_type
from aether.worldcache_aether.worldcache_core import derivative_approximation, worldcache_formula, taylor_cache_init
from aether.worldcache_aether.cache_utils.worldcache_utils import (
    split_modalities,
    recombine_modalities,
    flatten_to_tokens,
    unflatten_from_tokens,
    compute_curvature,
    compute_token_groups,
    compute_velocity,
    update_history_buffer,
)
import loguru
import sys

def apply_worldcache_aether_forward(model):
    """
    Apply Worldcache Aether forward.
    """
    
    def worldcache_aether_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        cache_dic=None,
        current=None,
    ) -> Union[Tuple[torch.Tensor], Any]:
        """
        Forward pass for the transformer with Worldcache acceleration.
        """
        # Try to get cache_dic and current from pipeline if not provided directly
        if cache_dic is None or current is None:
            # Print warning if cache_dic is not available yet (shouldn't happen but helps debug)
            if not hasattr(self, '_worldcache_pipeline_ref'):
                print(f"[Worldcache] Warning: No pipeline ref found, using original forward", flush=True)
            # Try to get from pipeline reference stored in transformer
            if hasattr(self, '_worldcache_pipeline_ref'):
                pipeline = self._worldcache_pipeline_ref
                if hasattr(pipeline, '_worldcache_cache_dic') and hasattr(pipeline, '_worldcache_current'):
                    cache_dic = pipeline._worldcache_cache_dic
                    current = pipeline._worldcache_current
                    # Increment step counter and update current['step']
                    if hasattr(pipeline, '_worldcache_step_counter'):
                        pipeline._worldcache_step_counter += 1
                        current['step'] = pipeline._worldcache_step_counter
                        # Diagnostic print on first call only
                        if pipeline._worldcache_step_counter == 0:
                            cache_mode_from_dic = cache_dic.get('mode', 'unknown') if cache_dic else 'None'
                            print(f"[Worldcache] Forward: Retrieved cache_dic (mode={cache_mode_from_dic}), step={current.get('step', 'unknown')}", flush=True)
                            print(f"[DEBUG] Successfully retrieved cache_dic and current from pipeline", file=sys.stderr, flush=True)
                else:
                    # Debug: pipeline exists but cache_dic/current not set
                    print(f"[Worldcache] Debug: Pipeline ref exists but cache_dic/current not available", flush=True)
            else:
                # Debug: pipeline ref not found
                print(f"[Worldcache] Debug: Pipeline ref not found on transformer", flush=True)
        
        # If still not available, use original forward
        if cache_dic is None or current is None:
            # Debug print before falling back to original forward
            print(f"[Worldcache] Debug: cache_dic={cache_dic is not None}, current={current is not None}, using original forward", flush=True)
            # Call original forward method
            return self._original_forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                timestep_cond=timestep_cond,
                ofs=ofs,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
            )
        
        # Debug: Successfully got cache_dic and current
        if current.get('step', -1) == 0:
            cache_mode = cache_dic.get('mode', 'unknown')
            print(f"[Worldcache] Forward: Successfully using cache_dic (mode={cache_mode}), step={current.get('step', 'unknown')}", flush=True)
        
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        try:
            from diffusers.models.lora import scale_lora_layers, unscale_lora_layers
            USE_PEFT_BACKEND = True
        except ImportError:
            USE_PEFT_BACKEND = False

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # Determine calculation type for this step (before setting cache keys)
        cal_type(cache_dic, current)
        
        # Set cache keys for Aether (simplified structure)
        current['stream'] = 'final'
        current['layer'] = 'final'
        current['module'] = 'final'
        taylor_cache_init(cache_dic, current)

        if current['type'] == 'full':
            calc_type_info = f"  -> Executing: full (mode={cache_dic.get('mode', 'unknown')})"
            print(calc_type_info, flush=True)
            # 3. Transformer blocks
            for i, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        attention_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                    )

            hidden_states = self.norm_final(hidden_states)

            # Save derivative after norm_final (before norm_out and proj_out)
            # This allows preserving timestep conditioning information in acceleration modes
            derivative_approximation(cache_dic, current, hidden_states)

            # 4. Final block
            hidden_states = self.norm_out(hidden_states, temb=emb)
            hidden_states = self.proj_out(hidden_states)
            
            # 5. Unpatchify (for full mode)
            p = self.config.patch_size
            p_t = self.config.patch_size_t

            if p_t is None:
                output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
                output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
            else:
                output = hidden_states.reshape(
                    batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
                )
                output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
        
        elif current['type'] == 'worldcache':
            calc_type_info = f"  -> Executing: worldcache (mode={cache_dic.get('mode', 'unknown')})"
            print(calc_type_info, flush=True)
            
            mode = cache_dic.get('mode', 'worldcache')
            # Check if we're in modality splitting mode (Worldcache with split info)
            has_modality_split = 'modality_split_info' in cache_dic
            
            if mode == 'worldcache' and has_modality_split and current.get('type') == 'worldcache':
                # Modality splitting mode: worldcache_predict returns split tokens
                # We need to recombine them back to spatial latents
                split_tokens = worldcache_formula(cache_dic, current)  # (batch, seq_len*3, 24)
                
                # Get split info
                split_info = cache_dic['modality_split_info']
                seq_len = split_info['seq_len']
                latent_channels = split_info['latent_channels']
                num_frames = split_info['num_frames']
                height = split_info['height']
                width = split_info['width']
                p = split_info['patch_size']
                p_t = split_info['patch_size_t']
                
                # Split tokens back into RGB, Depth, Raymap
                # Remove padding: RGB and Depth were padded to 24 channels
                rgb_tokens_padded = split_tokens[:, :seq_len, :]  # (batch, seq_len, 24)
                depth_tokens_padded = split_tokens[:, seq_len:2*seq_len, :]  # (batch, seq_len, 24)
                raymap_tokens = split_tokens[:, 2*seq_len:3*seq_len, :]  # (batch, seq_len, 24)
                
                # Remove padding from RGB and Depth
                rgb_tokens = rgb_tokens_padded[:, :, :latent_channels]  # (batch, seq_len, 16)
                depth_tokens = depth_tokens_padded[:, :, :latent_channels]  # (batch, seq_len, 16)
                
                # Unflatten tokens back to spatial latents
                rgb_latents = unflatten_from_tokens(rgb_tokens, p, p_t, num_frames, height, width, latent_channels)
                depth_latents = unflatten_from_tokens(depth_tokens, p, p_t, num_frames, height, width, latent_channels)
                raymap_latents = unflatten_from_tokens(raymap_tokens, p, p_t, num_frames, height, width, 24)
                
                # Recombine modalities
                output = recombine_modalities(rgb_latents, depth_latents, raymap_latents)
                # output shape: (batch, num_frames, 56, height, width)
                
                # Skip the rest of the processing and go to return
                if USE_PEFT_BACKEND:
                    unscale_lora_layers(self, lora_scale)
                
                if not return_dict:
                    return (output,)
                
                try:
                    from diffusers.models.transformer_2d import Transformer2DModelOutput
                    return Transformer2DModelOutput(sample=output)
                except ImportError:
                    return (output,)
            else:
                # Non-split mode: worldcache_formula returns the output after norm_final
                hidden_states = worldcache_formula(cache_dic, current)
                # Note: For worldcache mode, we skip transformer blocks and directly use cached values
                # The cached value is after norm_final, so we need to apply norm_out and proj_out
                # to preserve timestep conditioning information
                
                # Apply norm_out and proj_out to preserve timestep conditioning
                hidden_states = self.norm_out(hidden_states, temb=emb)
                hidden_states = self.proj_out(hidden_states)
                
                # 5. Unpatchify
                p = self.config.patch_size
                p_t = self.config.patch_size_t
                
                if p_t is None:
                    output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
                    output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
                else:
                    output = hidden_states.reshape(
                        batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
                    )
                    output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
                
                # Continue with normal processing for non-split mode
                # output is already set from unpatchify above
        
        # For Worldcache mode in full computation: split modalities and update cache
        mode = cache_dic.get('mode', 'worldcache')
        if mode == 'worldcache' and current.get('type') == 'full':
            # Get latent_channels from pipeline if available
            latent_channels = 16  # Default value
            if hasattr(self, '_worldcache_pipeline_ref'):
                pipeline = self._worldcache_pipeline_ref
                if hasattr(pipeline, 'vae') and hasattr(pipeline.vae, 'config'):
                    latent_channels = pipeline.vae.config.latent_channels
            
            # Split modalities: (batch, num_frames, 56, height, width) -> RGB, Depth, Raymap
            rgb_latents, depth_latents, raymap_latents = split_modalities(output, latent_channels)
            
            # Flatten to tokens: each spatial position becomes a token
            rgb_tokens = flatten_to_tokens(rgb_latents, p, p_t)  # (batch, seq_len, 16)
            depth_tokens = flatten_to_tokens(depth_latents, p, p_t)  # (batch, seq_len, 16)
            raymap_tokens = flatten_to_tokens(raymap_latents, p, p_t)  # (batch, seq_len, 24)
            
            # Store metadata for reconstruction
            cache_dic['modality_split_info'] = {
                'latent_channels': latent_channels,
                'patch_size': p,
                'patch_size_t': p_t,
                'num_frames': num_frames,
                'height': height,
                'width': width,
                'seq_len': rgb_tokens.shape[1],
            }
            
            # Store split tokens in cache for Worldcache history
            # We need to pad RGB and Depth to 24 channels to have uniform shape
            max_channels = 24
            rgb_tokens_padded = torch.nn.functional.pad(rgb_tokens, (0, max_channels - rgb_tokens.shape[-1]))
            depth_tokens_padded = torch.nn.functional.pad(depth_tokens, (0, max_channels - depth_tokens.shape[-1]))
            split_tokens = torch.cat([rgb_tokens_padded, depth_tokens_padded, raymap_tokens], dim=1)
            # Shape: (batch, seq_len*3, 24)
            
            # Update Worldcache history with split tokens
            worldcache_history = cache_dic['worldcache_history']
            worldcache_config = cache_dic['worldcache_config']
            current_step = current['step']
            
            update_history_buffer(worldcache_history, split_tokens, current_step, max_history=3)
            
            # Compute velocity if we have at least 2 outputs
            if len(worldcache_history['outputs']) >= 2 and len(worldcache_history['steps']) >= 2:
                F_t_minus_1 = worldcache_history['outputs'][-2]
                F_t = worldcache_history['outputs'][-1]
                step_t_minus_1 = worldcache_history['steps'][-2]
                step_t = worldcache_history['steps'][-1]
                dt = step_t - step_t_minus_1
                
                if dt > 0:
                    v_t = compute_velocity(F_t, F_t_minus_1, dt)
                    if v_t is not None:
                        worldcache_history['velocities'].append(v_t)
                        # Keep only recent velocities
                        if len(worldcache_history['velocities']) > 2:
                            worldcache_history['velocities'] = worldcache_history['velocities'][-2:]
            
            # Compute curvature if we have at least 3 outputs
            if len(worldcache_history['outputs']) >= 3:
                curvature = compute_curvature(
                    worldcache_history,
                    current_step,
                    eps=worldcache_config['eps']
                )
                
                if curvature is not None:
                    # Update cached curvature
                    cache_dic['cached_curvature'] = curvature
                    
                    # Compute token groups (all tokens treated equally)
                    mask_stable, mask_linear, mask_chaotic = compute_token_groups(
                        curvature,
                        percentile_stable=worldcache_config['percentile_stable'],
                        percentile_chaotic=worldcache_config['percentile_chaotic']
                    )
                    
                    # Update cached masks
                    cache_dic['cached_masks'] = {
                        'mask_stable': mask_stable,
                        'mask_linear': mask_linear,
                        'mask_chaotic': mask_chaotic,
                    }
                    
                    # Print curvature and token group statistics
                    curvature_mean = curvature.float().mean().item() if isinstance(curvature, torch.Tensor) else float(curvature)
                    n_stable = mask_stable.sum().item() if isinstance(mask_stable, torch.Tensor) else int(mask_stable.sum())
                    n_linear = mask_linear.sum().item() if isinstance(mask_linear, torch.Tensor) else int(mask_linear.sum())
                    n_chaotic = mask_chaotic.sum().item() if isinstance(mask_chaotic, torch.Tensor) else int(mask_chaotic.sum())
                    total_tokens = n_stable + n_linear + n_chaotic
                    print(f"  -> Curvature computed | mean={curvature_mean:.4f} | stable={n_stable}({n_stable/total_tokens*100:.1f}%) | linear={n_linear}({n_linear/total_tokens*100:.1f}%) | chaotic={n_chaotic}({n_chaotic/total_tokens*100:.1f}%)", flush=True)
            
            # Store original tokens info for reconstruction (without padding)
            cache_dic['split_tokens_info'] = {
                'rgb_tokens': rgb_tokens,
                'depth_tokens': depth_tokens,
                'raymap_tokens': raymap_tokens,
            }

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        
        try:
            from diffusers.models.transformer_2d import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output)
        except ImportError:
            return (output,)

    # Store original forward method
    if not hasattr(model, '_original_forward'):
        model._original_forward = model.forward
    
    # Replace forward method
    model.forward = MethodType(worldcache_aether_forward, model)
    loguru.logger.info("Worldcache Aether forward applied")

