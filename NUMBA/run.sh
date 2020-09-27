#!/bin/bash
source ../venv/bin/activate
export NUMBA_CUDALIB="/usr/lib/cuda"
export NUMBA_LIBDEVICE="/usr/lib/cuda/nvvm/libdevice"
export NUMBA_NVVM="/usr/lib/x86_64-linux-gnu/libnvvm.so"
#export NUMBA_NVVM="/usr/lib/cuda/nvvm"
python numba.example.vecadd.py
