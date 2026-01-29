import torch.nn.functional as F
from .samplers import TTMGuider
from .utils import add_noise_to_reference_video


# Copied from ComfyUI Wanvideo Wrapper
class EncodeWanVideo:
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
                
        print(f"WanVideo Encode: Encoded latents shape {latents.shape}")
 
        return ({"samples": latents, "noise_mask": mask},)


# Copied from ComfyUI Wanvideo Wrapper
class TTMLatentAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT", {"tooltip": "wanvideo latent"}),
                    "reference_latents": ("LATENT", {"tooltip": "Reference image to encode"}),
                    "ttm_start_step": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Start step to apply TTM latent guide"}),
                    "ttm_end_step": ("INT", {"default": 3, "min": 1, "max": 1000, "step": 1, "tooltip": "The step to stop applying TTM"}),
                    "ref_masks": ("MASK", {"tooltip": "Reference mask to encode"}),
                }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "add"
    CATEGORY = "Wan22 TimeToMove"

    def add(self, latent, reference_latents, ttm_start_step, ttm_end_step, ref_masks):  
              
        if ttm_end_step < max(0, ttm_start_step):
            raise ValueError(f"`ttm_end_step` ({ttm_end_step}) must be >= `ttm_start_step` ({ttm_start_step}).")
        
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

        latent["ttm_reference_latents"] = reference_latents["samples"]
        latent["ttm_mask"] =  mask_latent.movedim(2, 1)
        latent["ttm_start_step"] = ttm_start_step
        latent["ttm_end_step"] = ttm_end_step
        
        return (latent,)


class TimeToMoveGuider:    
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Works with a list of floats too (one cfg float per step)"}),
                    "latent": ("LATENT", {"tooltip": "You can connect here the latent from TTM Latent Add, to pass reference video and ttm options"}),
                    "start_sampler_step": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Start step of the whole sampling process. It will automatically skip the selected number of sigmas (starting from the first ones); if the sampler has a start_step option and you changed its value, set the same here"}),
                    },
                }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "guide"
    CATEGORY = "Wan22 TimeToMove"

    def guide(cls, model, positive, negative, cfg, latent, start_sampler_step):
        guider = TTMGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        ttm_options = {}
        ttm_options["ttm_reference_latents"] = latent.get("ttm_reference_latents", None)
        ttm_options["ttm_start_step"] = latent["ttm_start_step"]
        ttm_options["ttm_end_step"] = latent["ttm_end_step"]
        ttm_options["latent_image"] = latent["samples"]
        ttm_options["motion_mask"] = latent["ttm_mask"]
        ttm_options["start_sampler_step"] = start_sampler_step
        guider.set_ttm_options(ttm_options)

        return (guider,)


# Taken from kijai WanVideo-Wrapper
class CFGFloatListScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 30, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of steps to schedule cfg for"} ),
            "cfg_scale_start": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "cfg_scale_end": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "interpolation": (["linear", "ease_in", "ease_out"], {"default": "linear", "tooltip": "Interpolation method to use for the cfg scale"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "Start percent of the steps to apply cfg"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "End percent of the steps to apply cfg"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("float_list",)
    FUNCTION = "process"
    CATEGORY = "Wan22 TimeToMove"
    DESCRIPTION = "Helper node to generate a list of floats that can be used to schedule cfg scale for the steps, outside the set range cfg is set to 1.0. Taken from Kijai WanVideo-Wrapper"

    def process(self, steps, cfg_scale_start, cfg_scale_end, interpolation, start_percent, end_percent, unique_id):

        # Create a list of floats for the cfg schedule
        cfg_list = [1.0] * steps
        start_idx = min(int(steps * start_percent), steps - 1)
        end_idx = min(int(steps * end_percent), steps - 1)

        for i in range(start_idx, end_idx + 1):
            if i >= steps:
                break

            if end_idx == start_idx:
                t = 0
            else:
                t = (i - start_idx) / (end_idx - start_idx)

            if interpolation == "linear":
                factor = t
            elif interpolation == "ease_in":
                factor = t * t
            elif interpolation == "ease_out":
                factor = t * (2 - t)

            cfg_list[i] = round(cfg_scale_start + factor * (cfg_scale_end - cfg_scale_start), 2)

        # If start_percent > 0, always include the first step
        if start_percent > 0:
            cfg_list[0] = 1.0

        return (cfg_list,)
