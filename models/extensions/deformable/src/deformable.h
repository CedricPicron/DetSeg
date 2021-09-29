#pragma once

#include "cpu/msda_3d_cpu.h"

#ifdef WITH_CUDA
#include "cuda/msda_3d_cuda.h"
#endif


at::Tensor msda_3d_forward(
    const at::Tensor &feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws)
{
    if (feats.type().is_cuda())
    {
        #ifdef WITH_CUDA
        return msda_3d_cuda_forward(feats, map_hw, map_offs, sample_xyz, attn_ws);

        #else
        AT_ERROR("Not compiled with GPU support.");
        #endif
    }
    return msda_3d_cpu_forward(feats, map_hw, map_offs, sample_xyz, attn_ws);
}

std::vector<at::Tensor> msda_3d_backward(
    const at::Tensor &feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws,
    const at::Tensor &grad_output)
{
    if (feats.type().is_cuda())
    {
        #ifdef WITH_CUDA
        return msda_3d_cuda_backward(feats, map_hw, map_offs, sample_xyz, attn_ws, grad_output);

        #else
        AT_ERROR("Not compiled with GPU support.");
        #endif
    }
    return msda_3d_cpu_backward(feats, map_hw, map_offs, sample_xyz, attn_ws, grad_output);
}