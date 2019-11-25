import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


mod = SourceModule("""
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int i = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
""")


def main():
    try:
        # n = 1000000000
        # n = 50000
        n = 100000000
        a, b, c = np.random.randn(1, n), np.random.randn(1, n), np.empty(n)
        a, b, c = a.astype(np.float32), b.astype(np.float32), c.astype(np.float32)
        a_gpu, b_gpu, c_gpu = cuda.mem_alloc(a.nbytes), cuda.mem_alloc(b.nbytes), cuda.mem_alloc(c.nbytes)
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)
        cuda.memcpy_htod(c_gpu, c)
        func = mod.get_function("vectorAdd")
        block_dim_z = 4
        threads_per_block = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK) / block_dim_z
        block_dim_x, block_dim_y = int(math.sqrt(threads_per_block / block_dim_z)), int(math.sqrt(threads_per_block / block_dim_z))
        max_blocks_per_grid = block_dim_x * block_dim_y * block_dim_z
        grid_dim_z = 4
        blocks_per_grid = max_blocks_per_grid / grid_dim_z
        grid_dim_x, grid_dim_y = int(math.sqrt(blocks_per_grid / grid_dim_z)), int(math.sqrt(blocks_per_grid / grid_dim_z))
        func(a_gpu, b_gpu, c_gpu, np.int32(n), block=(block_dim_x, block_dim_y, block_dim_z), grid=(grid_dim_x, grid_dim_y, grid_dim_z))
        c_doubled = np.empty_like(c)
        cuda.memcpy_dtoh(c_doubled, c_gpu)
        print(a)
        print(b)
        print(c_doubled)
    except cuda.Error as e:
        print(e)


if __name__ == "__main__":
    main()
