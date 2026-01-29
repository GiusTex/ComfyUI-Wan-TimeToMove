import torch
import comfy
from comfy.model_patcher import ModelPatcher
from comfy.samplers import (sampling_function, process_conds, cast_to_load_options,
                            preprocess_conds_hooks, get_total_hook_groups_in_conds,
                            filter_registered_hooks_on_conds)
from .utils import add_noise_at_step


class TTMGuider:
    def __init__(self, model_patcher: ModelPatcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg
    
    def set_ttm_options(self, ttm_options):
        self.ttm_reference_latents = ttm_options["ttm_reference_latents"]
        self.ttm_start_step = ttm_options["ttm_start_step"]
        self.ttm_end_step = ttm_options["ttm_end_step"]
        self.latent_image = ttm_options["latent_image"]
        self.motion_mask = ttm_options["motion_mask"]
        self.start_sampler_step = ttm_options["start_sampler_step"]

    def inner_set_conds(self, conds):
        for k in conds:
            self.original_conds[k] = comfy.sampler_helpers.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.outer_predict_noise(*args, **kwargs)

    def outer_predict_noise(self, x, timestep, model_options={}, seed=None):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self.predict_noise,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, self.model_options, is_model_options=True)
        ).execute(x, timestep, model_options, seed)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        #---------------------------------------------------------
        sigmas = model_options["sigmas"]
        noise = model_options["noise"]
        i = torch.argmin(torch.abs(sigmas - timestep)).item()
        
        ttm_ref_latent = model_options["ttm_reference_latents"]
        ttm_start_step = model_options["ttm_start_step"]
        ttm_end_step = model_options["ttm_end_step"]
        ttm_mask = model_options["motion_mask"]
        # Time-to-move (TTM)
        if (i + ttm_start_step) < ttm_end_step:
            if i + ttm_start_step < len(sigmas):
                sigma_next = sigmas[i + ttm_start_step]
                noisy_latents = add_noise_at_step(ttm_ref_latent, 
                                            noise,
                                            sigma_next.to(x.device)
                                        ).to(x)
                x = x * (1 - ttm_mask) + noisy_latents * ttm_mask
            else:
                x = x * (1 - ttm_mask) + ttm_ref_latent * ttm_mask
        #---------------------------------------------------------
        return sampling_function(self.inner_model, x, timestep, 
                                 self.conds.get("negative", None), 
                                 self.conds.get("positive", None), 
                                 self.cfg[i],
                                 model_options=model_options, seed=seed)

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}
        
        #---------------------------------------------------------
        skipped_sigmas = sigmas[self.start_sampler_step:]
              # 4   <    5
        if len(skipped_sigmas) < len(sigmas): # sampler doesn't have start_step option
            sigmas = skipped_sigmas
              # 4   ==   4
        elif len(skipped_sigmas) == len(sigmas): # sampler already has option
            pass # we don't want another sigma less
        steps = len(sigmas)-1
        extra_args["model_options"]["steps"] = steps
        #---------------------------------------------------------
        # Pass ttm options to KSAMPLER.sample
        ttm_start_step = max(self.ttm_start_step - self.start_sampler_step, 0)
        ttm_end_step = self.ttm_end_step - self.start_sampler_step

        extra_args["model_options"]["ttm_reference_latents"] = self.ttm_reference_latents.to(noise.device)
        extra_args["model_options"]["ttm_start_step"] = ttm_start_step
        extra_args["model_options"]["ttm_end_step"] = ttm_end_step
        extra_args["model_options"]["motion_mask"] = self.motion_mask.to(noise.device)
        extra_args["model_options"]["sigmas"] = sigmas
        extra_args["model_options"]["noise"] = noise

        if ttm_start_step > steps:
            raise ValueError("TTM start step is beyond the total number of steps")

        if ttm_end_step > ttm_start_step:
            print("Using Time-to-move (TTM)")
            print(f"TTM reference latents shape: {self.ttm_reference_latents.shape}")
            print(f"TTM motion mask shape: {self.motion_mask.shape}")
            print(f"Applying TTM from step {ttm_start_step} to {ttm_end_step}")
        #---------------------------------------------------------
        # Cfg schedule taken from Kijai WanVideo-Wrapper
        if isinstance(self.cfg, list):
            if steps < len(self.cfg):
                print(f"Received {len(self.cfg)} cfg values, but only {steps} steps. Slicing cfg list to match steps.")
                self.cfg = self.cfg[:steps]
            elif steps > len(self.cfg):
                print(f"Received only {len(self.cfg)} cfg values, but {steps} steps. Extending cfg list to match steps.")
                self.cfg.extend([self.cfg[-1]] * (steps - len(self.cfg)))
            print(f"Using per-step cfg list: {self.cfg}")
        else:
            self.cfg = [self.cfg] * (steps + 1)
        #---------------------------------------------------------

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        
        # run steps and get final samples
        samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
        device = self.model_patcher.load_device

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)
        cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            self.model_patcher.cleanup()

        comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        if sigmas.shape[-1] == 0:
            return latent_image

        if latent_image.is_nested:
            latent_image, latent_shapes = comfy.utils.pack_latents(latent_image.unbind())
            noise, _ = comfy.utils.pack_latents(noise.unbind())
        else:
            latent_shapes = [latent_image.shape]

        if denoise_mask is not None:
            if denoise_mask.is_nested:
                denoise_masks = denoise_mask.unbind()
                denoise_masks = denoise_masks[:len(latent_shapes)]
            else:
                denoise_masks = [denoise_mask]

            for i in range(len(denoise_masks), len(latent_shapes)):
                denoise_masks.append(torch.ones(latent_shapes[i]))

            for i in range(len(denoise_masks)):
                denoise_masks[i] = comfy.sampler_helpers.prepare_mask(denoise_masks[i], latent_shapes[i], self.model_patcher.load_device)

            if len(denoise_masks) > 1:
                denoise_mask, _ = comfy.utils.pack_latents(denoise_masks)
            else:
                denoise_mask = denoise_masks[0]

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        preprocess_conds_hooks(self.conds)

        try:
            orig_model_options = self.model_options
            self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
            orig_hook_mode = self.model_patcher.hook_mode
            if get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
            comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            filter_registered_hooks_on_conds(self.conds, self.model_options)
            executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
            )
            output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode
            self.model_patcher.restore_hook_patches()

        del self.conds

        if len(latent_shapes) > 1:
            output = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(output, latent_shapes))
        return output
