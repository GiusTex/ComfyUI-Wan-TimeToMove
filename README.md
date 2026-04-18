# ComfyUI-Wan-TimeToMove
A native comfyui port of Kijai's WanVideo-Wrapper TimeToMove

<img width="763" height="449" alt="ComfyUI-TTM-nodes" src="https://github.com/user-attachments/assets/af01cab0-5ccf-435c-bab0-f138cb227f1c" />

https://github.com/user-attachments/assets/0b201e7a-d3c6-417f-8293-e20e8d2872fb

### Updates:
- Moved `CFGFloatListScheduler` to [ComfyUI-MoreEfficientSamplers](https://github.com/GiusTex/ComfyUI-MoreEfficientSamplers).

### Nodes
The custom node contains 3 new nodes:
- `Encode WanVideo`: taken from wanvideo-wrapper, it encodes the reference video.
- `TTM Latent Add`: taken from wanvideo-wrapper, it embeds in the latent the reference to the driving video.
- `Timove To Move Guider`: this node adds the ttm latent to the latent noise before passing it to the sampling function. This node removes the necessity of a dedicated sampler.

### Other custom nodes used:
- The scheduler used is [this](https://github.com/BigStationW/flowmatch_scheduler-comfyui), useful for models using lightx loras. You can still use other samplers/schedulers.
- If you don't have kijai wan-videowrapper and need a cfg scheduler, you can find `CFGFloatListScheduler` here [ComfyUI-MoreEfficientSamplers](https://github.com/GiusTex/ComfyUI-MoreEfficientSamplers).

### Download
To install ComfyUI-Wan-TimeToMove, follow these steps:
- Go in the ComfyUI `custom_nodes` folder, then download the repository or clone it here: `git clone https://github.com/GiusTex/ComfyUI-Wan-TimeToMove.git`.
- Restart ComfyUI.
