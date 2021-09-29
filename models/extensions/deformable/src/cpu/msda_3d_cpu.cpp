#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


at::Tensor msda_3d_cpu_forward(
    const at::Tensor &feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws)
{
    AT_ERROR("Not implemented on CPU.");
}

std::vector<at::Tensor> msda_3d_cpu_backward(
    const at::Tensor &feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws,
    const at::Tensor &grad_output)
{
    AT_ERROR("Not implemented on CPU.");
}
