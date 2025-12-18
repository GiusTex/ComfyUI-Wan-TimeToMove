import torch
from PIL import Image
import numpy as np
import latent_preview
from comfy.cli_args import args
from comfy.samplers import sample


# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def format_message(text, color_code):
    RESET_COLOR = "\033[0m"
    return f"{color_code}{text}{RESET_COLOR}"

WARNING_COLOR = "\033[93m"  # Yellow

def warning(text):
    return format_message(text, WARNING_COLOR)


# Set global preview_method
def set_preview_method(method):
    if method == 'auto' or method == 'LatentPreviewMethod.Auto':
        args.preview_method = latent_preview.LatentPreviewMethod.Auto
    elif method == 'latent2rgb' or method == 'LatentPreviewMethod.Latent2RGB':
        args.preview_method = latent_preview.LatentPreviewMethod.Latent2RGB
    elif method == 'taesd' or method == 'LatentPreviewMethod.TAESD':
        args.preview_method = latent_preview.LatentPreviewMethod.TAESD
    else:
        args.preview_method = latent_preview.LatentPreviewMethod.NoPreviews


def sample_custom_ultra(model, device, noise, sampler, positive, negative, cfg, model_options={}, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    if last_step is not None and last_step < (len(sigmas) - 1):
        sigmas = sigmas[:last_step + 1]
        if force_full_denoise:
            sigmas[-1] = 0

    if start_step is not None:
        if start_step < (len(sigmas) - 1):
            sigmas = sigmas[start_step:]
        else:
            if latent_image is not None:
                return latent_image
            else:
                return torch.zeros_like(noise)
    
    return sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)


# Extract global preview_method
def global_preview_method():
    return args.preview_method


# Cache for Efficiency Node models
loaded_objects = {
    "ckpt": [], # (ckpt_name, ckpt_model, clip, bvae, [id])
    "refn": [], # (ckpt_name, ckpt_model, clip, bvae, [id])
    "vae": [],  # (vae_name, vae, [id])
    "lora": []  # ([(lora_name, strength_model, strength_clip)], ckpt_name, lora_model, clip_lora, [id])
}

# Cache for Efficient Ksamplers
last_helds = {
    "latent": [],   # (latent, [parameters], id)    # Base sampling latent results
    "image": [],    # (image, id)                   # Base sampling image results
    "cnet_img": []  # (cnet_img, [parameters], id)  # HiRes-Fix control net preprocessor image results
}

def store_ksampler_results(key: str, my_unique_id, value, parameters_list=None):
    global last_helds

    for i, data in enumerate(last_helds[key]):
        id_ = data[-1]  # ID will always be the last in the tuple
        if id_ == my_unique_id:
            # Check if parameters_list is provided or not
            updated_data = (value, parameters_list, id_) if parameters_list is not None else (value, id_)
            last_helds[key][i] = updated_data
            return True

    # If parameters_list is given
    if parameters_list is not None:
        last_helds[key].append((value, parameters_list, my_unique_id))
    else:
        last_helds[key].append((value, my_unique_id))
    return True


# This function cleans global variables associated with nodes that are no longer detected on UI
def globals_cleanup(prompt):
    global loaded_objects
    global last_helds

    # Step 1: Clean up last_helds
    for key in list(last_helds.keys()):
        original_length = len(last_helds[key])
        last_helds[key] = [
            (*values, id_)
            for *values, id_ in last_helds[key]
            if str(id_) in prompt.keys()
        ]

    # Step 2: Clean up loaded_objects
    for key in list(loaded_objects.keys()):
        for i, tup in enumerate(list(loaded_objects[key])):
            # Remove ids from id array in each tuple that don't exist in prompt
            id_array = [id for id in tup[-1] if str(id) in prompt.keys()]
            if len(id_array) != len(tup[-1]):
                if id_array:
                    loaded_objects[key][i] = tup[:-1] + (id_array,)
                    #print(f'Updated tuple at index {i} in {key} in loaded_objects: {loaded_objects[key][i]}')
                else:
                    # If id array becomes empty, delete the corresponding tuple
                    loaded_objects[key].remove(tup)
                    #print(f'Deleted tuple at index {i} in {key} in loaded_objects because its id array became empty.')


# Copied from ComfyUI Wanvideo Wrapper
def add_noise_to_reference_video(image, ratio=None):
    sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio
    image_noise = torch.randn_like(image) * sigma[:, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image


# Copied from Kijai Wanvideo Wrapper
def add_noise_at_step(
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    sigma: torch.IntTensor,
) -> torch.FloatTensor:
    
    sigma = sigma.view(sigma.shape + (1,) * (len(noise.shape)-1))

    return (1 - sigma) * original_samples + sigma * noise
