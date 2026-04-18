from .nodes import (EncodeWanVideo,
                    TTMLatentAdd,
                    TimeToMoveGuider)

NODE_CLASS_MAPPINGS = {
    "EncodeWanVideo": EncodeWanVideo,
    "TTMLatentAdd": TTMLatentAdd,
    "TimeToMoveGuider": TimeToMoveGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EncodeWanVideo": "Encode WanVideo",
    "TTMLatentAdd": "TTM Latent Add",
    "TimeToMoveGuider": "TimeToMove Guider",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
