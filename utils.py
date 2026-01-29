import torch


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
