from numba import cuda
import numpy as np

# Define a CUDA kernel
@cuda.jit
def vector_add(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

# Initialize arrays
N = 1024
a = np.ones(N, dtype=np.float32)
b = np.ones(N, dtype=np.float32)
c = np.zeros(N, dtype=np.float32)

# Transfer data to the GPU
a_device = cuda.to_device(a)
b_device = cuda.to_device(b)
c_device = cuda.device_array_like(c)

# Launch the kernel
threads_per_block = 128
blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block
vector_add[blocks_per_grid, threads_per_block](a_device, b_device, c_device)

# Copy the result back to the host
c_device.copy_to_host(c)

# Verify the result
print(c)  # Should print an array of twos
