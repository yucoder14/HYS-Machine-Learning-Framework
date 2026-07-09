from numba import cuda, types
import cupy as cp
import math

MAX_BATCH_DIM = 4

BATCH_DIM_NUM_THREADS = 1
TPB = 16

def _verify_shapes(A, B):
    k = A.shape[-1]
    k_prime = B.shape[-2]

    if k != k_prime:
        raise ValueError(f"Invalid shapes: {A.shape} and {B.shape}")

    for i in range(-min(len(A.shape), len(B.shape)), -2):
        if A.shape[i] != B.shape[i] and A.shape[i] != 1 and B.shape[i] != 1:
            raise ValueError(f"Invalid shapes: {A.shape} and {B.shape}")

# treating the tensor as stacks of matrices
def _broadcast_shape(A, B):
    ndim = max(len(A.shape), len(B.shape))

    # matrix shape
    mat_shape = (A.shape[-2], B.shape[-1])

    batch_shape = ()
    # batch shape
    if ndim > 2:
        A_shape = (1,) * (ndim - len(A.shape)) + A.shape
        B_shape = (1,) * (ndim - len(B.shape)) + B.shape
        for i in range(ndim - 2):
            batch_shape += (max(A_shape[i], B_shape[i]),)

    return batch_shape, mat_shape

def _broadcast_stride(A_strides, A_shape, C_batch_dim):
    A_batch_strides = A_strides[:-2] if len(A_strides) > 2 else []
    A_batch_shape = A_shape[:-2] if len(A_shape) > 2 else []

    pad_size = C_batch_dim - len(A_batch_strides) if C_batch_dim > len(A_batch_strides) else 0
    zero_pad = cp.zeros(pad_size, dtype=cp.int32)

    strides = cp.hstack(
        (zero_pad, A_batch_strides),
    )

    # standard broadcasting rule for batches
    # set stride to 0 whenever shape is 1
    for i in range(-len(A_batch_strides), 0):
        if A_batch_shape[i] == 1:
            strides[i] = 0

    return strides

@cuda.jit(device=True)
def _dot(a, b, dim):
    # I have to explicitly cast it in float32 because float64 is used by default with no casting
    # this reduced compute be a lot
    tmp = cp.float32(0)
    for i in range(dim):
      tmp += a[i] * b[i]

    return tmp

@cuda.jit
def _matmul_kernel(
    A_batch_flat, B_batch_flat, C_batch_flat,
    A_batch_strides, B_batch_strides, C_batch_strides,
    batch_shape, batch_dim, K,
    num_batches, N, M
):
    x, y, z = cuda.grid(3)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    # shared array to store offsets
    shared_coords = cuda.shared.array(shape=(BATCH_DIM_NUM_THREADS, MAX_BATCH_DIM), dtype=types.int32)
    shared_offsets = cuda.shared.array(shape=(BATCH_DIM_NUM_THREADS, 2), dtype=types.int32)
    
    # for tiling matrix multiplication; this reduces memory look ups and therefore 
    # decreasing memory latency when doing matrix multiplication; it's kind of like 
    # caching smaller blocks from the matrix and doing matrix multiplication on them 
    # accumulating the total sum 
    s_A = cuda.shared.array(shape=(BATCH_DIM_NUM_THREADS, TPB, TPB), dtype=types.float32)
    s_B = cuda.shared.array(shape=(BATCH_DIM_NUM_THREADS, TPB, TPB), dtype=types.float32)

    if tx == 0 and ty == 0 and z < num_batches:
        remainder = z
        for i in range(batch_dim):
            shared_coords[tz, i] = remainder // C_batch_strides[i]
            remainder %= C_batch_strides[i]

        # Calculate the base pointers once and store them in shared memory too
        a_off = _dot(shared_coords[tz], A_batch_strides, batch_dim)
        shared_offsets[tz, 0] = a_off

        b_off = _dot(shared_coords[tz], B_batch_strides, batch_dim)
        shared_offsets[tz, 1] = b_off
        
    cuda.syncthreads()

    A_batch_offset = shared_offsets[tz, 0]
    B_batch_offset = shared_offsets[tz, 1]
    tmp = cp.float32(0.0)

    num_tiles = math.ceil(K / TPB)
    for t in range(num_tiles):
        # this divergence is yet another source of bottleneck 
        if z < num_batches and y < N and (t * TPB + tx) < K:
            s_A[tz, ty, tx] = A_batch_flat[A_batch_offset, y, t * TPB + tx]
        else:
            s_A[tz, ty, tx] = 0.0

        if z < num_batches and (t * TPB + ty) < K and x < M:
            s_B[tz, ty, tx] = B_batch_flat[B_batch_offset, x, t * TPB + ty]
        else:
            s_B[tz, ty, tx] = 0.0

        # Sync to guarantee the entire tile is loaded by all threads in the block
        cuda.syncthreads()

        # Multiply the elements of the tile completely out of Shared Memory
        if z < num_batches and y < N and x < M:
            for k in range(TPB):
                tmp += s_A[tz, ty, k] * s_B[tz, k, tx]

        # Sync to make sure math is done before we overwrite the tile in the next iteration
        cuda.syncthreads()

    # Write final result back out to Global Memory
    if z < num_batches and y < N and x < M:
        C_batch_flat[z, y, x] = tmp

def matmul_3D(A, B):
    # coalese the shapes to be at least 2 dimensional
    squeeze_left = A.ndim < 2
    squeeze_right = B.ndim < 2
    A = A if A.ndim > 1 else A.reshape(1, -1)
    B = B if B.ndim > 1 else B.reshape(-1, 1)

    _verify_shapes(A, B)

    batch_shape, mat_shape = _broadcast_shape(A, B)
    C = cp.zeros(batch_shape + mat_shape, dtype=cp.float32)

    N, M = mat_shape
    K = int(A.shape[-1])
    # change the dimensions for ease of access inside the kernel
    B = B.swapaxes(-2, -1)
    B = cp.ascontiguousarray(B)

    A = A.astype(cp.float32)
    B = B.astype(cp.float32)

    # stride calculation
    A_strides = cp.array(A.strides, dtype=cp.int32) // A.itemsize
    B_strides = cp.array(B.strides, dtype=cp.int32) // B.itemsize
    C_strides = cp.array(C.strides, dtype=cp.int32) // C.itemsize

    A_mat_size = A.shape[-2] * A.shape[-1]
    B_mat_size = B.shape[-2] * B.shape[-1]
    C_mat_size = N * M

    A_batch_strides = _broadcast_stride(A_strides, A.shape, len(batch_shape)) // A_mat_size
    B_batch_strides = _broadcast_stride(B_strides, B.shape, len(batch_shape)) // B_mat_size
    C_batch_strides = C_strides[:len(batch_shape)] // C_mat_size

    # flatten the batch dimensions
    A_batch_flat = A.reshape(-1, *A.shape[-2:])
    B_batch_flat = B.reshape(-1, *B.shape[-2:])
    C_batch_flat = C.reshape(-1, *C.shape[-2:])

    # launch kernel
    batch_dim = cp.float32(len(batch_shape))
    batch_num = cp.float32(C_batch_flat.shape[0])
    threadsperblock = (TPB, TPB, BATCH_DIM_NUM_THREADS) # maximum 1024 threads per block
    blockspergrid_x = math.ceil(M / TPB) #
    blockspergrid_y = math.ceil(N / TPB) #
    blockspergrid_z = math.ceil(C_batch_flat.shape[0] / BATCH_DIM_NUM_THREADS) # batch dimension
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    _matmul_kernel[blockspergrid, threadsperblock](
      A_batch_flat, B_batch_flat, C_batch_flat,
      A_batch_strides, B_batch_strides, C_batch_strides,
      batch_shape, batch_dim, K,
      batch_num, N, M
    )
    cuda.synchronize()

    C = C.reshape(batch_shape + mat_shape)

    if squeeze_left:
        return C.squeeze(axis=0)
    elif squeeze_right:
        return C.squeeze(axis=-1)
    else:
        return C
