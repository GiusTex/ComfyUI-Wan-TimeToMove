# ComfyUI-Wan-TimeToMove
A native comfyui port of kijai's WanVideo-Wrapper TimeToMove

<img width="1577" height="691" alt="WanVideo TTM nodes image" src="https://github.com/user-attachments/assets/0a7b34a6-805d-4c4a-80f3-ed2143907068" />

https://github.com/user-attachments/assets/551eac0d-c5fe-49a8-b1a2-3884d0ece746

**This node is still WIP.** For now only the [lcm sampler](https://github.com/GiusTex/ComfyUI-Wan-TimeToMove/blob/main/k_diffusion/sampling.py#L1020) supports TimeToMove, and the generated frames are a bit dark (this color difference is seen especially when a first frame is passed).

The second sampler can be found here: `https://github.com/GiusTex/ComfyUI-MoreEfficientSamplers` but you can change it, and the scheduler used is this: `https://github.com/BigStationW/flowmatch_scheduler-comfyui`, useful when you use lightx loras.

### Download
To install ComfyUI-Wan-TimeToMove, follow these steps:
- Go in the ComfyUI `custom_nodes` folder, then download the repository or clone it here: `git clone https://github.com/GiusTex/ComfyUI-Wan-TimeToMove.git`.
- Restart ComfyUI.
