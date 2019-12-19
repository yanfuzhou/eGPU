import time
import cupy as cp
import numpy as np

n = 10
num = 13000000
a = np.random.randn(1, num).astype(np.float64)
b = np.random.randn(1, num).astype(np.float64)


def cpu():
    c = np.sin(a) + np.cos(b)
    return c


def gpu():
    ga = cp.array(a)
    gb = cp.array(b)
    gpu_c = cp.sin(ga) + cp.cos(gb)
    gc = cp.asnumpy(gpu_c)
    return gc


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
