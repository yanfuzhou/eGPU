# https://towardsdatascience.com/heres-how-to-use-cupy-to-make-numpy-700x-faster-4b920dda1f56
import numpy as np
import cupy as cp
import time


# Numpy and CPU
s = time.time()
x_cpu = np.ones((1000, 1000, 500))
e = time.time()
print(e - s)

# CuPy and GPU
s = time.time()
x_gpu = cp.ones((1000, 1000, 500))
cp.cuda.Stream.null.synchronize()
e = time.time()
print(e - s)
