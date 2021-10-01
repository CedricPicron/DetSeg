#include <vector>
#include "cuda/msda_3d_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace at::cuda::detail;

at::Tensor msda_3d_cuda_forward(
    const at::Tensor &in_feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws)
{
    AT_ASSERTM(in_feats.device().is_cuda(), "in_feats must be a CUDA tensor");
    AT_ASSERTM(map_hw.device().is_cuda(), "map_hw must be a CUDA tensor");
    AT_ASSERTM(map_offs.device().is_cuda(), "map_offs must be a CUDA tensor");
    AT_ASSERTM(sample_xyz.device().is_cuda(), "sample_xyz must be a CUDA tensor");
    AT_ASSERTM(attn_ws.device().is_cuda(), "attn_ws must be a CUDA tensor");

    auto batch_size = in_feats.size(0);
    auto num_heads = in_feats.size(2);
    auto channels = in_feats.size(3);

    auto num_out_feats = sample_xyz.size(1);
    auto num_pts = sample_xyz.size(3);

    auto out_feats = at::zeros({batch_size, num_out_feats, num_heads, channels}, in_feats.options());
    int64_t num_kernels = batch_size * num_out_feats * num_heads * num_pts;

    if (num_kernels > 0)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feats.scalar_type(), "msda_3d_cuda_forward", [&] {
            msda_3d_cuda_forward_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(num_kernels),
            getTensorInfo<scalar_t, int>(in_feats),
            getTensorInfo<int64_t, int>(map_hw),
            getTensorInfo<int64_t, int>(map_offs),
            getTensorInfo<scalar_t, int>(sample_xyz),
            getTensorInfo<scalar_t, int>(attn_ws),
            getTensorInfo<scalar_t, int>(out_feats));
        });
    }

    return out_feats;
}

std::vector<at::Tensor> msda_3d_cuda_backward(
    const at::Tensor &in_feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws,
    const at::Tensor &grad_out_feats)
{
    AT_ERROR("Currently not implemented.");
}