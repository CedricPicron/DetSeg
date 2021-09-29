#include <algorithm>
#include <cstdio>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/macros/Macros.h>

#include <THC/THCAtomics.cuh>

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}


template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(const scalar_t* &bottom_data, 
                                                   const int &height, const int &width, const int &nheads, const int &head_size,
                                                   const scalar_t &h, const scalar_t &w, const int &m, const int &c)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * head_size;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * head_size + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear(const scalar_t* &bottom_data, 
                                                   const int &height, const int &width, const int &nheads, const int &head_size,
                                                   const scalar_t &h, const scalar_t &w, const int &m, const int &c,
                                                   const scalar_t &top_grad,
                                                   const scalar_t &attn_weight,
                                                   scalar_t* &grad_value, 
                                                   scalar_t* grad_sampling_loc,
                                                   scalar_t* grad_attn_weight)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * head_size;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * head_size + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value+ptr1, w1*top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value+ptr2, w2*top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value+ptr3, w3*top_grad_value); 
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value+ptr4, w4*top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}


template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_gm(const scalar_t* &bottom_data, 
                                                   const int &height, const int &width, const int &nheads, const int &head_size,
                                                   const scalar_t &h, const scalar_t &w, const int &m, const int &c,
                                                   const scalar_t &top_grad,
                                                   const scalar_t &attn_weight,
                                                   scalar_t* &grad_value, 
                                                   scalar_t* grad_sampling_loc,
                                                   scalar_t* grad_attn_weight)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * head_size;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * head_size + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value+ptr1, w1*top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value+ptr2, w2*top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value+ptr3, w3*top_grad_value); 
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value+ptr4, w4*top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_attn_weight, top_grad * val); 
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}


template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void msda_3d_cuda_forward_kernel(
  const int num_kernels,
  at::cuda::detail::TensorInfo<scalar_t, int> in_feats,
  at::cuda::detail::TensorInfo<int64_t, int> map_hw,
  at::cuda::detail::TensorInfo<int64_t, int> map_offs,
  at::cuda::detail::TensorInfo<scalar_t, int> sample_xyz,
  at::cuda::detail::TensorInfo<scalar_t, int> attn_ws,
  at::cuda::detail::TensorInfo<scalar_t, int> out_feats)
{
  H = in_feats.sizes[2];
  C = in_feats.sizes[3];
  P = sample_xyz.sizes[3];
  F = out_feats.sizes[1];

  int in_sN = in_feats.strides[0];
  int in_sF = in_feats.strides[1];
  int in_sH = in_feats.strides[2];
  int in_sC = in_feats.strides[3];

  int sample_sN = sample_xyz.strides[0];
  int sample_sF = sample_xyz.strides[1];
  int sample_sH = sample_xyz.strides[2];
  int sample_sP = sample_xyz.strides[3];
  int sample_sC = sample_xyz.strides[4];

  int attn_sN = attn_ws.strides[0];
  int attn_sF = attn_ws.strides[1];
  int attn_sH = attn_ws.strides[2];
  int attn_sP = attn_ws.strides[3];

  int out_sN = out_feats.strides[0];
  int out_sF = out_feats.strides[1];
  int out_sH = out_feats.strides[2];
  int out_sC = out_feats.strides[3];

  CUDA_KERNEL_LOOP(index, num_kernels)
  {
    const int p = index % P;
    const int h = (index / P) % H;
    const int f = (index / (P * H)) % F;
    const int n = index / (P * H * F);

    const int sample_offset = n * sample_sN + f * sample_sF + h * sample_sH + p * sample_sP;
    scalar_t ix = sample_xyz.data[sample_offset]
    scalar_t iy = sample_xyz.data[sample_offset + sample_sC]
    scalar_t iz = sample_xyz.data[sample_offset + 2 * sample_sC]

    const int H1 = 0
    const int W1 = 0
    const int H2 = 0
    const int W2 = 0

    auto in_ptr_NC = in_feats.data + n * in_sN;
    auto out_ptr_NFHC = out_feats.data + n * out_sN + f * out_sF + h * out_sH;

    for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NFHC += out_sC) 
    {

    }

  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int head_size, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % head_size;
    _temp /= head_size;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * head_size;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, head_size, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_w=cache_grad_sampling_loc[0], _grad_h=cache_grad_sampling_loc[1], _grad_a=cache_grad_attn_weight[0];
          int sid=2;
          for (unsigned int tid = 1; tid < blockSize; ++tid)
          {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }
          
          
          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int head_size, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % head_size;
    _temp /= head_size;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * head_size;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, head_size, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();

        for (unsigned int s=blockSize/2; s>0; s>>=1)
        {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
          }
          __syncthreads();
        }

        if (tid == 0)
        { 
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int head_size, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % head_size;
    _temp /= head_size;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * head_size;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, head_size, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_w=cache_grad_sampling_loc[0], _grad_h=cache_grad_sampling_loc[1], _grad_a=cache_grad_attn_weight[0];
          int sid=2;
          for (unsigned int tid = 1; tid < blockDim.x; ++tid)
          {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }
          
          
          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int head_size, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % head_size;
    _temp /= head_size;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * head_size;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, head_size, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1)
        {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre)
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            } 
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int head_size, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % head_size;
    _temp /= head_size;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * head_size;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, head_size, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1)
        {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre)
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          atomicAdd(grad_sampling_loc, cache_grad_sampling_loc[0]);
          atomicAdd(grad_sampling_loc + 1, cache_grad_sampling_loc[1]);
          atomicAdd(grad_attn_weight, cache_grad_attn_weight[0]);
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int head_size, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % head_size;
    _temp /= head_size;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * head_size;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear_gm(
            data_value_ptr, spatial_h, spatial_w, num_heads, head_size, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            grad_sampling_loc, grad_attn_weight);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
void ms_deformable_col2im_cuda(cudaStream_t stream,
                              const scalar_t* grad_col,
                              const scalar_t* data_value,
                              const int64_t * data_spatial_shapes,
                              const int64_t * data_level_start_index,
                              const scalar_t * data_sampling_loc,
                              const scalar_t * data_attn_weight,
                              const int batch_size, 
                              const int spatial_size, 
                              const int num_heads,
                              const int head_size, 
                              const int num_levels,
                              const int num_query,
                              const int num_point, 
                              scalar_t* grad_value,
                              scalar_t* grad_sampling_loc,
                              scalar_t* grad_attn_weight)
{
  const int num_threads = (head_size > CUDA_NUM_THREADS)?CUDA_NUM_THREADS:head_size;
  const int num_kernels = batch_size * num_query * num_heads * head_size;
  const int num_actual_kernels = batch_size * num_query * num_heads * head_size;
  if (head_size > 1024)
  {
    if ((head_size & 1023) == 0)
    {
      ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        head_size, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
    }
    else
    {
      ms_deformable_col2im_gpu_kernel_gm<scalar_t>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
    }
  }
  else{
    switch(head_size)
    {
      case 1:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 2:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 4:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 8:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 16:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 32:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 64:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 128:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 256:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 512:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 1024:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      head_size, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      default:
        if (head_size < 64)
        {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        head_size, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        }
        else
        {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        head_size, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        }
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}