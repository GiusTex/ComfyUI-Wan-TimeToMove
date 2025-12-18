from .nodes import (WanVideoEncode,
                    TTMKSamplerSelect,
                    AddTTMLatent,
                    WanVideoSamplerCustomUltraAdvancedEfficient)

NODE_CLASS_MAPPINGS = {
    "WanVideoEncode": WanVideoEncode,
    "TTMKSamplerSelect": TTMKSamplerSelect,
    "AddTTMLatent": AddTTMLatent,
    "WanVideoSamplerCustomUltraAdvancedEfficient": WanVideoSamplerCustomUltraAdvancedEfficient,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoEncode": "WanVideo Encode",
    "TTMKSamplerSelect": "TimeToMove KSampler Select",
    "AddTTMLatent": "Add TTM Latent",
    "WanVideoSamplerCustomUltraAdvancedEfficient": "WanVideoSampler Custom Ultra Advanced Efficient",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']