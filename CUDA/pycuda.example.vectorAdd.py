import numpy
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

num = 50000
a = numpy.random.randn(1, num).astype(numpy.float32)
b = numpy.random.randn(1, num).astype(numpy.float32)
c = numpy.empty_like(a, dtype=a.dtype)

threadsPerBlock = 256
blocksPerGrid = int((num + threadsPerBlock - 1) / threadsPerBlock)

mod = SourceModule("""
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
""")

vec_add = mod.get_function("vectorAdd")
vec_add(cuda.In(a), cuda.In(b), cuda.Out(c), numpy.int32(num),
        block=(threadsPerBlock, 1, 1), grid=(blocksPerGrid, 1, 1), shared=0)

print(a)
print(b)
print(c)
