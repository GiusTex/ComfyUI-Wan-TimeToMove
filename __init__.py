from .nodes import (EncodeWanVideo,
                    TTMLatentAdd,
                    TimeToMoveGuider,
                    CFGFloatListScheduler)

NODE_CLASS_MAPPINGS = {
    "EncodeWanVideo": EncodeWanVideo,
    "TTMLatentAdd": TTMLatentAdd,
    "TimeToMoveGuider": TimeToMoveGuider,
    "CFGFloatListScheduler": CFGFloatListScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EncodeWanVideo": "Encode WanVideo",
    "TTMLatentAdd": "TTM Latent Add",
    "TimeToMoveGuider": "TimeToMove Guider",
    "CFGFloatListScheduler": "CFGFloatListScheduler",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
