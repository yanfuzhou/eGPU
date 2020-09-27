import time
import cupy as cp
import numpy as np

# test number of times
n = 10
# max for single run: math.pow(2, 27) = 134217728
size = (30, 1000000)

a = np.random.random(size).astype(np.float64)
b = np.random.random(size).astype(np.float64)


def cpu():
    c = np.einsum('ij,ij->i', a, b)
    return c


def gpu():
    cp.cuda.Stream.null.synchronize()
    ga = cp.asarray(a)
    gb = cp.asarray(b)
    gpu_c = cp.einsum('ij,ij->i', ga, gb)
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
