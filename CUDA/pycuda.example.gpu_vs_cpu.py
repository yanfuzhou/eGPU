import time
import numpy
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# test number of times
n = 10
# max for single run: math.pow(2, 27) = 134217728
num = 130000000
threads_per_block = 1024
ia = numpy.random.randn(1, num).astype(numpy.float32)
ib = numpy.random.randn(1, num).astype(numpy.float32)


def cpu():
    return numpy.sin(ia) + numpy.sin(ib)


def gpu():
    oc = numpy.empty_like(ia, dtype=ia.dtype)
    blocks_per_grid = int((num + threads_per_block - 1) / threads_per_block)
    mod = SourceModule("""
    __global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < numElements)
        {
            C[i] = sinf(A[i]) + sinf(B[i]);
        }
    }
    """)
    vec_add = mod.get_function("vectorAdd")
    vec_add(cuda.In(ia), cuda.In(ib), cuda.Out(oc), numpy.int32(num),
            block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1, 1), shared=0)
    return oc


t1s, t2s = list(), list()
for i in range(n):
    # Run on CPU
    start = time.time()
    test1 = cpu()
    t1 = round(time.time() - start, 3)
    t1s.append(t1)
    # Run on GPU
    start = time.time()
    test2 = gpu()
    t2 = round(time.time() - start, 3)
    t2s.append(t2)

print('CPU time: %s' % round((sum(t1s) / n), 3))
print('GPU time: %s' % round((sum(t2s) / n), 3))
