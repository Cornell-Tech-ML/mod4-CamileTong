# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for CUDA."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a function."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """See `tensor_ops.py`"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function that applies fn to each element.

    Each CUDA thread processes one element, handling broadcasting between shapes.

    Args:
    ----
        fn: Function to apply to each element (float -> float)

    Returns:
    -------
        CUDA kernel function that performs the mapping

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        thread_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Only process if thread is within output size
        if thread_idx >= out_size:
            return
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        to_index(thread_idx, out_shape, out_index)

        broadcast_index(out_index, out_shape, in_shape, in_index)

        in_position = index_to_position(in_index, in_strides)

        out[thread_idx] = fn(in_storage[in_position])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zip function that applies fn to pairs of elements.

    Each CUDA thread processes one element pair, handling broadcasting between shapes.

    Args:
    ----
        fn: Function to apply to each pair of elements (float, float -> float)

    Returns:
    -------
        CUDA kernel function that performs the element-wise operation

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        thread_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Only process if thread is within output size
        if thread_idx >= out_size:
            return

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert linear thread index to output tensor index
        to_index(thread_idx, out_shape, out_index)

        # Handle broadcasting from output shape to input shapes
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        a_position = index_to_position(a_index, a_strides)
        b_position = index_to_position(b_index, b_strides)

        out[thread_idx] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """CUDA kernel that performs parallel reduction sum using tree-based approach.

    Each block reduces BLOCK_DIM elements to a single sum using shared memory.
    The reduction is performed in log2(BLOCK_DIM) steps for efficiency.

    Requirements:
    1. Use shared memory for intermediate calculations
    2. Each thread loads exactly one element from global memory
    3. Perform reduction in log2(BLOCK_DIM) steps
    4. Handle cases where input size isn't perfectly divisible by BLOCK_DIM

    Args:
    ----
        out (Storage): Output storage array (size = ⌈n/BLOCK_DIM⌉)
        a (Storage): Input storage array (size = n)
        size (int): Length of input array

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Calculate thread indices
    thread_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_idx = cuda.threadIdx.x

    # Load input data into shared memory
    if thread_idx < size:
        cache[local_idx] = float(a[thread_idx])
    else:
        cache[local_idx] = 0.0

    cuda.syncthreads()  # Ensure all data is loaded

    # Perform tree reduction in log2(BLOCK_DIM) steps
    # BLOCK_DIM = 32 = 2^5, so we need 5 steps
    for step in range(5):  # log2(32) = 5
        stride = 1 << step  # 2^step: 1, 2, 4, 8, 16
        if local_idx % (2 * stride) == 0:
            cache[local_idx] += cache[local_idx + stride]
        cuda.syncthreads()  # Ensure all threads complete current step

    # Write final result for this block
    if local_idx == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """See `tensor_ops.py`"""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        thread_pos = cuda.threadIdx.x
        block_pos = cuda.blockIdx.x

        # Initialize shared memory with reduction value
        cache[thread_pos] = reduce_value

        if block_pos < out_size:
            # Convert block index to output index
            to_index(block_pos, out_shape, out_index)

            # Map thread to position in reduction dimension
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + thread_pos

            # Load input data if within bounds
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                a_pos = index_to_position(out_index, a_strides)
                cache[thread_pos] = a_storage[a_pos]

            cuda.syncthreads()

            # Tree reduction within block
            for stride in range(1, 11):  # log2(1024) = 10
                if thread_pos % (2**stride) == 0:
                    cache[thread_pos] = fn(
                        cache[thread_pos], cache[thread_pos + 2 ** (stride - 1)]
                    )
                cuda.syncthreads()

            # Write result
            if thread_pos == 0:
                out[block_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32

    # Allocate shared memory for input matrices
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Get thread indices (row, col) in the block
    row = cuda.threadIdx.x
    col = cuda.threadIdx.y

    # Exit if thread is outside matrix bounds
    if row >= size or col >= size:
        return

    # Load input matrices into shared memory
    a_shared[row, col] = a[size * row + col]
    b_shared[row, col] = b[size * row + col]

    # Ensure all data is loaded
    cuda.syncthreads()

    # Compute dot product for position (row, col)
    result = 0.0
    for k in range(size):
        result += a_shared[row, k] * b_shared[k, col]

    # Write result to global memory
    out[size * row + col] = result


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """See `tensor_ops.py`"""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # Matrix dimensions
    N, M, K = a_shape[1], b_shape[2], a_shape[2]  # A: N×K, B: K×M

    # Compute C[i,j] = sum(A[i,k] * B[k,j])
    result = 0.0
    for block_idx in range(0, K, BLOCK_DIM):
        # 1. Load data into shared memory
        if i < N and block_idx + pj < K:
            a_idx = (
                batch * a_batch_stride
                + i * a_strides[1]
                + (block_idx + pj) * a_strides[2]
            )
            a_shared[pi, pj] = a_storage[a_idx]
        else:
            a_shared[pi, pj] = 0.0

        if j < M and block_idx + pi < K:
            b_idx = (
                batch * b_batch_stride
                + (block_idx + pi) * b_strides[1]
                + j * b_strides[2]
            )
            b_shared[pi, pj] = b_storage[b_idx]
        else:
            b_shared[pi, pj] = 0.0

        cuda.syncthreads()

        # 2. Compute partial dot product for this block
        for k in range(min(BLOCK_DIM, K - block_idx)):
            result += a_shared[pi, k] * b_shared[k, pj]

        cuda.syncthreads()

    # 3. Write final result to global memory
    if i < N and j < M:
        out_idx = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_idx] = result


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
