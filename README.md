# ComfyUI-Wan-TimeToMove
A native comfyui port of Kijai's WanVideo-Wrapper TimeToMove

<img width="782" height="458" alt="WanTTM nodes (updated)" src="https://github.com/user-attachments/assets/a2f81804-442e-431a-b5cd-34f71f3c01a5" />

### Updates:
- Solved color issue.

### WIP
- Working to fix bug on longer sequences (around 49 frames seems to work, around 81 frames duplicates the subject)

[VIDEO...]

### Nodes
The custom node contains 4 new nodes:
- `Encode WanVideo`: taken from wanvideo-wrapper, it encodes the reference video.
- `TTM Latent Add`: taken from wanvideo-wrapper, it embeds in the latent the reference to the driving video.
- `Timove To Move Guider`: this node adds the ttm latent to the latent noise before passing it to the sampling function. This node removes the necessity of a dedicated sampler.
- `CFG Float List Scheduler`: taken from wanvideo-wrapper, it creates a list of cfg values, and submits them step by step, making possible using different cfg values at different steps.

### Other custom nodes used:
- The advanced sampler used in the [second workflow](https://github.com/GiusTex/ComfyUI-Wan-TimeToMove/blob/TTM-v2/wanvideo_2_2_I2V_A14B_TimeToMove_workflow2.json) can be found [here](https://github.com/GiusTex/ComfyUI-MoreEfficientSamplers). You can still use the native comfyui `sampler custom advanced` using [this](https://github.com/GiusTex/ComfyUI-Wan-TimeToMove/blob/TTM-v2/wanvideo_2_2_I2V_A14B_TimeToMove_workflow1.json) workflow.
- The scheduler used is [this](https://github.com/BigStationW/flowmatch_scheduler-comfyui), useful for models using lightx loras. You can still use other samplers/schedulers.

### Download
To install ComfyUI-Wan-TimeToMove, follow these steps:
- Go in the ComfyUI `custom_nodes` folder, then download the repository or clone it here: `git clone https://github.com/GiusTex/ComfyUI-Wan-TimeToMove.git`.
- Restart ComfyUI.
