#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

const int BLOCK_SIZE = 16;

__global__ void mode_filter2d_k(uint *input, uint *output, int w, int h, int r, int k)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= h)
    {
        return;
    }
    extern __shared__ uint count[];

    for (int i = x - r; i <= x + r; i++)
    {
        for (int j = y - r; j <= y + r; j++)
        {
            if (i >= 0 && i < w && j >= 0 && j < h)
            {
                count[threadIdx.x * k + input[i * k + j]]++;
            }
        }
    }
    uint max = 0;
    for (int i = 0; i < k; i++)
    {
        if (count[i] > max)
        {
            max = count[i];
        }
    }
    output[x * k + y] = max;
}

torch::Tensor mode_filter2d(torch::Tensor input, int r)
{
    CHECK_INPUT(input);

    auto output = torch::empty_like(input);
    auto k = std::get<0>(torch::_unique(input)).size(0);

    mode_filter2d_k<<<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * k * sizeof(uint)>>>(input.data_ptr<uint>(), output.data_ptr<uint>(), input.size(0), input.size(1), r, k);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mode_filter", &mode_filter2d, "Mode sliding filter");
}