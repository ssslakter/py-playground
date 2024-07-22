#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

const int BLOCK_SIZE = 16;

__global__ void mode_filter2d_k(int *input, int *output, int w, int h, int r, int k)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= h)
    {
        return;
    }
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ int count[];
    for (int i = thread_id; i < BLOCK_SIZE * BLOCK_SIZE * k; i += BLOCK_SIZE * BLOCK_SIZE)
    {
        count[i] = 0;
    }
    __syncthreads();
    for (int i = x - r; i <= x + r; i++)
    {
        for (int j = y - r; j <= y + r; j++)
        {
            if (i >= 0 && i < w && j >= 0 && j < h)
            {
                &count[thread_id*k + input[i * h + j]]++;
            }
        }
    }
    __syncthreads();
    int max = 0;
    int idx_max = 0;
    for (int i = 0; i < k; i++)
    {
        auto m = count[thread_id*k + i];
        if (m > max)
        {
            idx_max = i;
            max = m;
        }
    }
    output[y*w + x] = idx_max;
}

torch::Tensor mode_filter2d(torch::Tensor input, int r)
{
    CHECK_INPUT(input);

    auto output = torch::empty_like(input);
    auto k = std::get<0>(torch::_unique(input)).size(0);
    TORCH_CHECK(input.max().item<int>() == k - 1 && input.min().item<int>() == 0,
                "input must be a tensor of integers from 0 to k-1");
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((input.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, (input.size(1) + BLOCK_SIZE - 1) / BLOCK_SIZE);

    mode_filter2d_k<<<grid, block, BLOCK_SIZE * BLOCK_SIZE * k * sizeof(int)>>>(
        input.data_ptr<int>(), output.data_ptr<int>(), input.size(0), input.size(1), r, k);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mode_filter2d", &mode_filter2d, "Mode sliding filter");
}