import numpy
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

m, n = 4, 4
a = numpy.random.randn(m, n).astype(numpy.float32)
b = numpy.random.randn(m, n).astype(numpy.float32)
c = numpy.zeros_like(a, dtype=a.dtype)

threadsPerBlock = (m, n, 1)
blocksPerGrid = (1, 1, 1)

mod = SourceModule("""
__global__ void matrixAdd(const float a[""" + str(m) + """][""" + str(n) + """], 
                          const float b[""" + str(m) + """][""" + str(n) + """], 
                          float c[""" + str(m) + """][""" + str(n) + """])
{
int i = threadIdx.x;
int j = threadIdx.y;
c[i][j] = a[i][j] + b[i][j];
}
""")

vec_add = mod.get_function("matrixAdd")
vec_add(cuda.In(a), cuda.In(b), cuda.Out(c),
        block=threadsPerBlock, grid=blocksPerGrid)

print(a)
print(b)
print(c)
