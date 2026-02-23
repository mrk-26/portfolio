import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
import pycuda.compiler
import pycuda.gpuarray
import numpy as np
from pycuda import gpuarray

print("PyCUDA and CUDA are working!")

# Verify CUDA and cuDNN availability
try:
    import cupy as cp
    print("CuPy is available, which means cuDNN should be accessible.")
except ImportError:
    print("CuPy is not available. Make sure cuDNN is correctly installed.")
