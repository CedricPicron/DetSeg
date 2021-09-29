#include <vector>
#include "cuda/msda_3d_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <cuda.h>
#include <cuda_runtime.h>


at::Tensor msda_3d_cuda_forward(
    const at::Tensor &in_feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws)
{
    /*
    Args:
        in_feats (FloatTensor): Features to sample from of shape [batch_size, num_in_feats, num_heads, channels].
        map_hw (LongTensor): Feature map sizes in (H, W) format of shape [num_maps, 2].
        map_offs (LongTensor): Feature map offsets of shape [num_maps].
        sample_xyz (FloatTensor): Sample locations of shape [batch_size, num_out_feats, num_heads, num_pts, 3].
        attn_ws (FloatTensor): Attention weights of shape [batch_size, num_out_feats, num_heads, num_pts].

    Returns:
        out_feats (FloatTensor): Weighted sampled features of shape [batch_size, num_out_feats, num_heads, channels].
    */

    AT_ASSERTM(in_feats.type().is_cuda(), "in_feats must be a CUDA tensor");
    AT_ASSERTM(map_hw.type().is_cuda(), "map_hw must be a CUDA tensor");
    AT_ASSERTM(map_offs.type().is_cuda(), "map_offs must be a CUDA tensor");
    AT_ASSERTM(sample_xyz.type().is_cuda(), "sample_xyz must be a CUDA tensor");
    AT_ASSERTM(attn_ws.type().is_cuda(), "attn_ws must be a CUDA tensor");

    auto batch_size = in_feats.size(0);
    auto num_heads = in_feats.size(2);
    auto channels = in_feats.size(3);

    auto num_out_feats = sample_xyz.size(1);
    auto num_pts = sample_xyz.size(3);

    auto out_feats = at::empty({batch_size, num_out_feats, num_heads, channels}, in_feats.options());
    int num_kernels = static_cast<int>(batch_size * num_out_feats * num_heads * num_pts);

    if (num_kernels > 0)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feats.scalar_type(), "msda_3d_cuda_forward", [&] {
            msda_3d_cuda_forward_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_kernels,
            at::cuda::detail::getTensorInfo<scalar_t, int>(in_feats),
            at::cuda::detail::getTensorInfogetTensorInfo<int64_t, int>(map_hw),
            at::cuda::detail::getTensorInfogetTensorInfo<int64_t, int>(map_offs),
            at::cuda::detail::getTensorInfo<scalar_t, int>(sample_xyz),
            at::cuda::detail::getTensorInfo<scalar_t, int>(attn_ws),
            at::cuda::detail::getTensorInfo<scalar_t, int>(out_feats));
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
    const at::Tensor &grad_out_feats,
    const int im2col_step)
{
    AT_ASSERTM(in_feats.is_contiguous(), "in_feats tensor has to be contiguous");
    AT_ASSERTM(map_hw.is_contiguous(), "map_hw tensor has to be contiguous");
    AT_ASSERTM(map_offs.is_contiguous(), "map_offs tensor has to be contiguous");
    AT_ASSERTM(sample_xyz.is_contiguous(), "sample_xyz tensor has to be contiguous");
    AT_ASSERTM(attn_ws.is_contiguous(), "attn_ws tensor has to be contiguous");
    AT_ASSERTM(grad_out_feats.is_contiguous(), "grad_out_feats tensor has to be contiguous");

    AT_ASSERTM(in_feats.type().is_cuda(), "in_feats must be a CUDA tensor");
    AT_ASSERTM(map_hw.type().is_cuda(), "map_hw must be a CUDA tensor");
    AT_ASSERTM(map_offs.type().is_cuda(), "map_offs must be a CUDA tensor");
    AT_ASSERTM(sample_xyz.type().is_cuda(), "sample_xyz must be a CUDA tensor");
    AT_ASSERTM(attn_ws.type().is_cuda(), "attn_ws must be a CUDA tensor");
    AT_ASSERTM(grad_out_feats.type().is_cuda(), "grad_out_feats must be a CUDA tensor");

    const int batch_size = in_feats.size(0);
    const int num_in_feats = in_feats.size(1);
    const int num_heads = in_feats.size(2);
    const int channels = in_feats.size(3);

    const int num_maps = map_hw.size(0);

    const int num_out_feats = sample_xyz.size(1);
    const int num_pts = sample_xyz.size(4);

    const int im2col_step_ = std::min(batch_size, im2col_step);

    AT_ASSERTM(batch_size % im2col_step_ == 0, "batch_size(%d) must divide im2col_step(%d)", batch_size, im2col_step_);

    auto grad_in_feats = at::zeros_like(in_feats);
    auto grad_sample_xyz = at::zeros_like(sample_xyz);
    auto grad_attn_ws = at::zeros_like(attn_ws);

    const int batch_size_n = im2col_step_;
    auto per_in_feats_size = num_in_feats * num_heads * channels;
    auto per_sample_loc_size = num_out_feats * num_heads * num_maps * num_pts * 2;
    auto per_attn_ws_size = num_out_feats * num_heads * num_maps * num_pts;
    auto grad_out_feats_n = grad_out_feats.view({batch_size/im2col_step_, batch_size_n, num_out_feats, num_heads, channels});
    
    for (int n = 0; n < batch_size/im2col_step_; ++n)
    {
        auto grad_out_feats_g = grad_out_feats_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES(in_feats.type(), "ms_deform_attn_backward_cuda", ([&] {
            ms_deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                                    grad_out_feats_g.data<scalar_t>(),
                                    in_feats.data<scalar_t>() + n * im2col_step_ * per_in_feats_size,
                                    map_hw.data<int64_t>(),
                                    map_offs.data<int64_t>(),
                                    sample_xyz.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    attn_ws.data<scalar_t>() + n * im2col_step_ * per_attn_ws_size,
                                    batch_size_n, num_in_feats, num_heads, channels, num_maps, num_out_feats, num_pts,
                                    grad_in_feats.data<scalar_t>() +  n * im2col_step_ * per_in_feats_size,
                                    grad_sample_xyz.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    grad_attn_ws.data<scalar_t>() + n * im2col_step_ * per_attn_ws_size);

        }));
    }

    return {
        grad_in_feats, grad_sample_xyz, grad_attn_ws
    };
}