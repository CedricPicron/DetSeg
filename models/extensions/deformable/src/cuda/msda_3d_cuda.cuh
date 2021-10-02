#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/macros/Macros.h>

using namespace at::cuda::detail;

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void msda_3d_cuda_forward_kernel(
  const int num_kernels,
  TensorInfo<scalar_t, int> in_feats,
  TensorInfo<int64_t, int> map_hw,
  TensorInfo<int64_t, int> map_offs,
  TensorInfo<scalar_t, int> sample_xyz,
  TensorInfo<scalar_t, int> attn_ws,
  TensorInfo<scalar_t, int> out_feats)
{
  // Get sizes
  const int H = in_feats.sizes[2];
  const int C = in_feats.sizes[3];
  const int M = map_hw.sizes[0];
  const int P = sample_xyz.sizes[3];
  const int F = out_feats.sizes[1];

  // Get strides
  const int in_sN = in_feats.strides[0];
  const int in_sF = in_feats.strides[1];
  const int in_sH = in_feats.strides[2];
  const int in_sC = in_feats.strides[3];

  const int map_sM = map_hw.strides[0];
  const int map_sC = map_hw.strides[1];

  const int sample_sN = sample_xyz.strides[0];
  const int sample_sF = sample_xyz.strides[1];
  const int sample_sH = sample_xyz.strides[2];
  const int sample_sP = sample_xyz.strides[3];
  const int sample_sC = sample_xyz.strides[4];

  const int attn_sN = attn_ws.strides[0];
  const int attn_sF = attn_ws.strides[1];
  const int attn_sH = attn_ws.strides[2];
  const int attn_sP = attn_ws.strides[3];

  const int out_sN = out_feats.strides[0];
  const int out_sF = out_feats.strides[1];
  const int out_sH = out_feats.strides[2];
  const int out_sC = out_feats.strides[3];

  CUDA_KERNEL_LOOP(index, num_kernels)
  {
    // Get kernel-specific indices
    const int c = index % C;
    const int p = (index / C) % P;
    const int h = (index / (C * P)) % H;
    const int f = (index / (C * P * H)) % F;
    const int n = index / (C * P * H * F);

    // Get normalized sample locations
    const int sample_offset = n * sample_sN + f * sample_sF + h * sample_sH + p * sample_sP;
    scalar_t ix = sample_xyz.data[sample_offset];
    scalar_t iy = sample_xyz.data[sample_offset + sample_sC];
    scalar_t iz = sample_xyz.data[sample_offset + 2 * sample_sC];

    // Clip normalized sample locations between 0 and 1
    float ix_f = ::min(::max(static_cast<float>(ix), 0.f), 1.f);
    float iy_f = ::min(::max(static_cast<float>(iy), 0.f), 1.f);
    float iz_f = ::min(::max(static_cast<float>(iz), 0.f), 1.f);

    // Get unnormalized Z sample locations
    iz_f = iz_f * (M-1);
    const int iz_0 = static_cast<int>(::min(::floor(iz_f), static_cast<float>(M-2)));
    const int iz_1 = iz_0 + 1;

    // Get map widths and heights
    int64_t h_0 = map_hw.data[iz_0 * map_sM];
    int64_t w_0 = map_hw.data[iz_0 * map_sM + map_sC];
    int64_t h_1 = map_hw.data[iz_1 * map_sM];
    int64_t w_1 = map_hw.data[iz_1 * map_sM + map_sC];

    // Get unnormalized X and Y sample locations
    float ix_f0 = ix_f * static_cast<float>(w_0 - 1);
    float ix_f1 = ix_f * static_cast<float>(w_1 - 1);
    float iy_f0 = iy_f * static_cast<float>(h_0 - 1);
    float iy_f1 = iy_f * static_cast<float>(h_1 - 1);

    scalar_t ix_0 = static_cast<scalar_t>(ix_f0);
    scalar_t ix_1 = static_cast<scalar_t>(ix_f1);
    scalar_t iy_0 = static_cast<scalar_t>(iy_f0);
    scalar_t iy_1 = static_cast<scalar_t>(iy_f1);

    const int ix_00 = static_cast<int>(::min(::floor(ix_f0), static_cast<float>(w_0-2)));
    const int ix_10 = static_cast<int>(::min(::floor(ix_f1), static_cast<float>(w_1-2)));
    const int iy_00 = static_cast<int>(::min(::floor(iy_f0), static_cast<float>(h_0-2)));
    const int iy_10 = static_cast<int>(::min(::floor(iy_f1), static_cast<float>(h_1-2)));

    const int ix_01 = ix_00 + 1;
    const int ix_11 = ix_10 + 1;
    const int iy_01 = iy_00 + 1;
    const int iy_11 = iy_10 + 1;

    // Get X and Y sampling weights
    scalar_t wx_00 = ix_01 - ix_0;
    scalar_t wy_00 = iy_01 - iy_0;
    scalar_t wx_10 = ix_11 - ix_1;
    scalar_t wy_10 = iy_11 - iy_1;
    scalar_t wx_01 = ix_0 - ix_00;
    scalar_t wy_01 = iy_0 - iy_00;
    scalar_t wx_11 = ix_1 - ix_10;
    scalar_t wy_11 = iy_1 - iy_10;

    // Get Z sampling weights
    iz = static_cast<scalar_t>(iz_f);
    scalar_t wz_0 = iz_1 - iz;
    scalar_t wz_1 = iz - iz_0;

    // Get XYZ sampling weights
    scalar_t w_000 = wx_00 * wy_00 * wz_0;
    scalar_t w_001 = wx_10 * wy_10 * wz_1;
    scalar_t w_010 = wx_00 * wy_01 * wz_0;
    scalar_t w_011 = wx_10 * wy_11 * wz_1;
    scalar_t w_100 = wx_01 * wy_00 * wz_0;
    scalar_t w_101 = wx_11 * wy_10 * wz_1;
    scalar_t w_110 = wx_01 * wy_01 * wz_0;
    scalar_t w_111 = wx_11 * wy_11 * wz_1;

    // Get attention weight
    scalar_t w = attn_ws.data[n * attn_sN + f * attn_sF + h * attn_sH + p * attn_sP];

    // Get map offsets
    int64_t off_0 = map_offs.data[iz_0];
    int64_t off_1 = map_offs.data[iz_1];

    // Get initial input pointers
    auto in_ptr = in_feats.data + n * in_sN + h * in_sH + c * in_sC;
    auto in_ptr_000 = in_ptr + (off_0 + w_0 * iy_00 + ix_00) * in_sF;
    auto in_ptr_001 = in_ptr + (off_1 + w_1 * iy_10 + ix_10) * in_sF;
    auto in_ptr_010 = in_ptr + (off_0 + w_0 * iy_01 + ix_00) * in_sF;
    auto in_ptr_011 = in_ptr + (off_1 + w_1 * iy_11 + ix_10) * in_sF;
    auto in_ptr_100 = in_ptr + (off_0 + w_0 * iy_00 + ix_01) * in_sF;
    auto in_ptr_101 = in_ptr + (off_1 + w_1 * iy_10 + ix_11) * in_sF;
    auto in_ptr_110 = in_ptr + (off_0 + w_0 * iy_01 + ix_01) * in_sF;
    auto in_ptr_111 = in_ptr + (off_1 + w_1 * iy_11 + ix_11) * in_sF;

    // Get initial output pointer
    auto out_ptr = out_feats.data + n * out_sN + f * out_sF + h * out_sH + c * out_sC;

    // Add weighted sampled inputs to output
    atomicAdd(out_ptr, *in_ptr_000 * w_000 * w);
    atomicAdd(out_ptr, *in_ptr_001 * w_001 * w);
    atomicAdd(out_ptr, *in_ptr_010 * w_010 * w);
    atomicAdd(out_ptr, *in_ptr_011 * w_011 * w);
    atomicAdd(out_ptr, *in_ptr_100 * w_100 * w);
    atomicAdd(out_ptr, *in_ptr_101 * w_101 * w);
    atomicAdd(out_ptr, *in_ptr_110 * w_110 * w);
    atomicAdd(out_ptr, *in_ptr_111 * w_111 * w);
  }
}
