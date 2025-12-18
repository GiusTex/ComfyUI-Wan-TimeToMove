import torch
import logging
from comfy_api.latest import io
from comfy.utils import PROGRESS_BAR_ENABLED
import torch.nn.functional as F
import latent_preview
import comfy
from nodes import VAEDecodeTiled, PreviewImage, VAEDecode
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
from comfy.samplers import SAMPLER_NAMES
from PIL import Image
from .utils import (pil2tensor, warning, set_preview_method, sample_custom_ultra, 
                    global_preview_method, store_ksampler_results, globals_cleanup, 
                    add_noise_at_step, add_noise_to_reference_video)
from .samplers import sampler_object


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# Copied from ComfyUI Wanvideo Wrapper
class WanVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("VAE",),
                    "image": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                    "optional": {
                        "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for leapfusion I2V where some noise can add motion and give sharper results"}),
                        "latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for leapfusion I2V where lower values allow for more motion"}),
                        "mask": ("MASK"),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("reference_latents",)
    FUNCTION = "encode"
    CATEGORY = "Wan22 TimeToMove"

    def encode(self, vae, image, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, noise_aug_strength=0.0, latent_strength=1.0, mask=None):
        image = image.clone()

        if image.shape[-1] == 4:
            image = image[..., :3]

        if noise_aug_strength > 0.0:
            image = add_noise_to_reference_video(image, ratio=noise_aug_strength)
        
        if enable_vae_tiling:
            latents = vae.encode_tiled(image[:,:,:,:3] * 2.0 - 1.0, tile_size=(tile_x//vae.upscale_ratio, tile_y//vae.upscale_ratio), tile_stride=(tile_stride_x//vae.upscale_ratio, tile_stride_y//vae.upscale_ratio))
        else:
            latents = vae.encode(image[:,:,:,:3] * 2.0 - 1.0)

        if latent_strength != 1.0:
            latents *= latent_strength

        log.info(f"WanVideo Encode: Encoded latents shape {latents.shape}")
 
        return ({"samples": latents, "noise_mask": mask},)
    

# Copied from ComfyUI Wanvideo Wrapper
class AddTTMLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT", {"tooltip": "wanvideo latent"}),
                    "reference_latents": ("LATENT", {"tooltip": "Reference image to encode"}),
                    "start_step": ("INT", {"default": 0, "min": -1, "max": 1000, "step": 1, "tooltip": "Start step for whole denoising process"}),
                    "end_step": ("INT", {"default": 2, "min": 1, "max": 1000, "step": 1, "tooltip": "The step to stop applying TTM"}),
                    "ref_masks": ("MASK", {"tooltip": "Reference mask to encode"}),
                }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "add"
    CATEGORY = "Wan22 TimeToMove"

    def add(self, latent, reference_latents, start_step, end_step, ref_masks):        
        if end_step < max(0, start_step):
            raise ValueError(f"`end_step` ({end_step}) must be >= `start_step` ({start_step}).")
        
        mask_sampled = ref_masks[::4]
        mask_sampled = mask_sampled.unsqueeze(1).unsqueeze(0)  # [1, T, 1, H, W]
        
        vae_upscale_factor = 8
        if reference_latents["samples"].shape[1] == 48:
            vae_upscale_factor = 16
        
        # Upsample spatially to latent resolution
        H_latent = mask_sampled.shape[-2] // vae_upscale_factor
        W_latent = mask_sampled.shape[-1] // vae_upscale_factor
        mask_latent = F.interpolate(
            mask_sampled.float(),
            size=(mask_sampled.shape[2], H_latent, W_latent),
            mode="nearest"
        )

        latent["ttm_reference_latents"] = reference_latents["samples"].squeeze(0) # [16, T, H, W]
        latent["ttm_mask"] =  mask_latent.squeeze(0).movedim(1, 0)  # [1, T, H, W]
        latent["ttm_start_step"] = start_step
        latent["ttm_end_step"] = end_step

        return (latent,)


class TTMKSamplerSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TTMKSamplerSelect",
            category="Wan Animate End Reference",
            inputs=[
                io.Combo.Input("sampler_name", options=SAMPLER_NAMES, default="lcm"),
                io.Latent.Input("latent"),
                ],
            outputs=[
                io.Sampler.Output(),
                ]
        )

    @classmethod
    def execute(cls, sampler_name, latent) -> io.NodeOutput:
        ttm_options = {}
        ttm_options["ttm_reference_latents"] = latent.get("ttm_reference_latents", None)
        ttm_options["ttm_start_step"] = latent["ttm_start_step"]
        ttm_options["ttm_end_step"] = latent["ttm_end_step"]
        ttm_options["latent_image"] = latent["samples"]
        ttm_options["motion_mask"] = latent["ttm_mask"]

        sampler = sampler_object(sampler_name, ttm_options)
        return io.NodeOutput(sampler)

    get_sampler = execute


class WanVideoSamplerCustomUltraAdvancedEfficient:
    # Image Preview code taken from jags111's efficiency-nodes (TSC_KSampler)
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": ("BOOLEAN", {"default": False}),
                    "preview_method": (["auto", "latent2rgb", "taesd", "vae_decoded_only", "none"],),
                    "vae_decode": (["true", "true (tiled)", "false"],),
                    },
                    "optional": {
                        "optional_vae": ("VAE",),
                    },
                    "hidden": {
                        "prompt": "PROMPT", 
                        "extra_pnginfo": "EXTRA_PNGINFO", 
                        "my_unique_id": "UNIQUE_ID",
                    },
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "SAMPLER", "SIGMAS", "LATENT","LATENT", "IMAGE", "VAE",)
    RETURN_NAMES = ("model", "positive", "negative", "sampler", "sigmas", "output", "denoised_output", "image", "vae", )
    FUNCTION = "sample"
    CATEGORY = "Wan22 TimeToMove"

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent, start_at_step, end_at_step, return_with_leftover_noise, preview_method, vae_decode, optional_vae=(None,), prompt=None, extra_pnginfo=None, my_unique_id=None):
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image
        
        # Rename the vae variable
        vae = optional_vae
        # If vae is not connected, disable vae decoding
        if vae == (None,) and vae_decode != "false":
            print(f"{warning('Sampler Custom Ultra Advanced Warning:')} No vae input detected, proceeding as if vae_decode was false.\n")
            vae_decode = "false"
        
        # ------------------------------------------------------------------------------------------------------
        def vae_decode_latent(vae, out, vae_decode):
            return VAEDecodeTiled().decode(vae,out,320)[0] if "tiled" in vae_decode else VAEDecode().decode(vae,out)[0]
        # ---------------------------------------------------------------------------------------------------------------

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]
        
        def process_latents():
            x0_output = {}
            # Initialize output variables
            out = out_denoised = images = preview = previous_preview_method = None

            if not add_noise:
                noise = Noise_EmptyNoise().generate_noise(latent)
            else:
                noise = Noise_RandomNoise(noise_seed).generate_noise(latent)
                
                #Time-to-move (TTM)
                ttm_start_step = 0
                ttm_reference_latents = latent.get("ttm_reference_latents", None)
                if ttm_reference_latents is not None:
                    motion_mask = latent["ttm_mask"].to(latent_image.device, latent_image.dtype)
                    ttm_start_step = max(latent["ttm_start_step"] - start_at_step, 0)
                    ttm_end_step = latent["ttm_end_step"] - start_at_step
    
                    if ttm_start_step > end_at_step:
                        raise ValueError("TTM start step is beyond the total number of steps")
                    
                    sigma = sigmas[ttm_start_step]

                    if ttm_end_step > ttm_start_step:
                        log.info("Using Time-to-move (TTM)")
                        log.info(f"TTM reference latents shape: {ttm_reference_latents.shape}")
                        log.info(f"TTM motion mask shape: {motion_mask.shape}")
                        log.info(f"Applying TTM from step {ttm_start_step} to {ttm_end_step}")

                        noise = add_noise_at_step(ttm_reference_latents,
                                            noise, 
                                            sigma
                        ).to(latent_image.device, latent_image.dtype)
                #--------------------------------------------------------------
                
            try:
                # Change the global preview method (temporarily)
                set_preview_method(preview_method)
                
                x0_output = {}
                callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

                disable_pbar = not PROGRESS_BAR_ENABLED
        
                disable_noise = False
                if not add_noise:
                    disable_noise = True
                                
                # Prepare noise for img specified by batch_inds
                if disable_noise:
                    noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
                else:
                    batch_inds = latent["batch_index"] if "batch_index" in latent else None
                    noise = comfy.sample.prepare_noise(latent_image, noise_seed, batch_inds)
                  
                force_full_denoise = True
                if return_with_leftover_noise:
                    force_full_denoise = False
        
                device = comfy.model_management.intermediate_device()
                model_options = model.model_options
                start_step = start_at_step
                last_step = end_at_step
                denoise_mask = noise_mask

                samples = sample_custom_ultra(model, device, 
                                        noise, 
                                        sampler, 
                                        positive, negative, 
                                        cfg, model_options, 
                                        latent_image, 
                                        start_step, last_step, 
                                        force_full_denoise, denoise_mask, 
                                        sigmas, 
                                        callback, disable_pbar, noise_seed)
                
                samples = samples.to(comfy.model_management.intermediate_device())
                
                out = latent.copy()
                out["samples"] = samples
                if "x0" in x0_output:
                    out_denoised = latent.copy()
                    out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
                else:
                    out_denoised = out
    
                previous_preview_method = global_preview_method()

                # ---------------------------------------------------------------------------------------------------------------
                # Decode image if not yet decoded
                if "true" in vae_decode:
                    if images is None:
                        images = vae_decode_latent(vae, out, vae_decode)
                        # Store decoded image as base image of no script is detected
                        store_ksampler_results("image", my_unique_id, images)
    
                # Define preview images
                if preview_method == "none" or (preview_method == "vae_decoded_only" and vae_decode == "false"):
                    preview = {"images": list()}
                elif images is not None:
                    preview = PreviewImage().save_images(images, prompt=prompt, extra_pnginfo=extra_pnginfo)["ui"]
        
                # Define a dummy output image
                if images is None and vae_decode == "false":
                    images = WanVideoSamplerCustomUltraAdvancedEfficient.empty_image

            finally:
                # Restore global changes
                set_preview_method(previous_preview_method)
              
            return out, out_denoised, preview, images
        
        # ---------------------------------------------------------------------------------------------------------------
        # Clean globally stored objects of non-existant nodes
        globals_cleanup(prompt)
        # ---------------------------------------------------------------------------------------------------------------
        out, out_denoised, preview, images = process_latents()

        result = (model, positive, negative, sampler, sigmas, 
                  out, out_denoised, images, vae,)

        if preview is None:
            return {"result": result}
        else:
            return {"ui": preview, "result": result}
