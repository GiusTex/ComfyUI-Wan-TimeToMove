import torch
from comfy.samplers import Sampler
from comfy.extra_samplers import uni_pc

from .k_diffusion import sampling as k_diffusion_sampling


class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
    # Add ttm_options to extra_args
    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None,
                 ttm_reference_latents=None, ttm_start_step=None,
                 ttm_end_step=None, latent_image=None, motion_mask=None):
        
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.scale_latent_inpaint(x=x, sigma=sigma, noise=self.noise, latent_image=self.latent_image) * latent_mask
        model_options["ttm_reference_latents"] = ttm_reference_latents
        model_options["ttm_start_step"] = ttm_start_step
        model_options["ttm_end_step"] = ttm_end_step
        model_options["latent_image"] = latent_image
        model_options["motion_mask"] = motion_mask
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out
    

class KSAMPLER(Sampler):
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)
        
        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples
    

def ksampler(sampler_name, ttm_options, extra_options={}, inpaint_options={}):
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable, **extra_options):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable, **extra_options)
        sampler_function = dpm_adaptive_function
    elif sampler_name == "lcm":
        def lcm_function(model, noise, sigmas, extra_args, callback, disable, **extra_options):
            extra_args["ttm_reference_latents"] = ttm_options["ttm_reference_latents"]
            extra_args["ttm_start_step"] = ttm_options["ttm_start_step"]
            extra_args["ttm_end_step"] = ttm_options["ttm_end_step"]
            extra_args["latent_image"] = ttm_options["latent_image"]
            extra_args["motion_mask"] = ttm_options["motion_mask"]
            return k_diffusion_sampling.sample_lcm(model, noise, sigmas, extra_args=extra_args, callback=callback, disable=disable, **extra_options)
        sampler_function = lcm_function
    else:
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))

    return KSAMPLER(sampler_function, extra_options, inpaint_options)


def sampler_object(name, ttm_options):
    if name == "uni_pc":
        sampler = KSAMPLER(uni_pc.sample_unipc)
    elif name == "uni_pc_bh2":
        sampler = KSAMPLER(uni_pc.sample_unipc_bh2)
    elif name == "ddim":
        sampler = ksampler("euler", inpaint_options={"random": True})
    elif name == "lcm":
        sampler = ksampler(name, ttm_options)
    else:
        sampler = ksampler(name)
    return sampler
