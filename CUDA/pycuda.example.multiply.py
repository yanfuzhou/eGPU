import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

# mod = SourceModule("""
#     __global__ void multiply(float **dest) {
#         const int i = threadIdx.x;
#         const int j = threadIdx.y;
#         dest[i][j] = 2.0*dest[i][j];
#     }
# """)
#
# a = np.random.randn(32, 32).astype(np.float32)
# multiply = mod.get_function("multiply")
# multiply(drv.InOut(a), block=(32, 32, 1), grid=(1, 1))
#
# print(a)

mod = SourceModule("""
    __global__ void multiply(float *dest, int lda) {
        const int i = threadIdx.x;
        const int j = threadIdx.y;
        float *p = &dest[i * lda + j]; // row major order
        *p *= 2.0f;
    }
""")

N = 8
a = np.random.randn(N, N).astype(np.float32)
print(a)
multiply = mod.get_function("multiply")
lda = np.int32(N)
multiply(drv.InOut(a), lda, block=(N, N, 1), grid=(1, 1))
print(a)
