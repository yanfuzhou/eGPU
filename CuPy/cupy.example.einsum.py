import cupy as cp
import numpy as np
with cp.cuda.Device(0):
    a = np.random.randn(1, 1000000).astype(np.float64)
    b = np.random.randn(1, 1000000).astype(np.float64)
    ga = cp.asarray(a)
    gb = cp.asarray(b)
    c = cp.einsum('ij,ij->i', ga, gb)
    _c = cp.asnumpy(c)
    print(_c)

cp.cuda.Device(0).use()
a = np.random.randn(1, 1000000).astype(np.float64)
b = np.random.randn(1, 1000000).astype(np.float64)
ga = cp.asarray(a)
gb = cp.asarray(b)
c = cp.einsum('ij,ij->i', ga, gb)
_c = cp.asnumpy(c)
print(_c)
