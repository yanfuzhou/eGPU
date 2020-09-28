# reference: http://web.mit.edu/pocky/www/cudaworkshop/Matrix/MatrixAdd.cu
# reference: https://stackoverflow.com/questions/35799478/how-to-implement-a-nxm-cuda-matrix-multiplication

import numpy
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

m, n = 60, 30
block_x, block_y = 32, 32
a = numpy.random.randn(m, n).astype(numpy.float32)
b = numpy.random.randn(m, n).astype(numpy.float32)
c = numpy.zeros_like(a, dtype=a.dtype)

threadsPerBlock = (block_x, block_y, 1)
blocksPerGrid = (m // block_x + 1, n // block_y + 1, 1)

mod = SourceModule("""
__global__ void matrixAdd(const float *a, 
                          const float *b, 
                          float *c, 
                          int M, int N)
{
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int k = col + row * N;
if(col < N && row < M)
        c[k] = a[k] + b[k];
}
""")

vec_add = mod.get_function("matrixAdd")
vec_add(cuda.In(a), cuda.In(b), cuda.Out(c), numpy.int32(m), numpy.int32(n),
        block=threadsPerBlock, grid=blocksPerGrid)

print(a)
print(b)
print(c)
