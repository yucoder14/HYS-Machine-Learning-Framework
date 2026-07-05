import numpy as np
import time

def matmul_2D_naive(A, B):
    _verify_shapes(A.shape, B.shape)

    C_shape = (A.shape[0], B.shape[1])
    C = np.zeros(C_shape)

    for i in range(C_shape[0]):
        for j in range(C_shape[1]):
            C[i , j] = np.dot(A[i, :], B[:, j])

    return C

def matmul_2D_naive_strides(A, B):
    C_shape = (A.shape[0], B.shape[1])
    C = np.zeros(C_shape)

    A_strides = np.array(A.strides) // A.itemsize
    B_strides = np.array(B.strides) // B.itemsize
    C_strides = np.array(C.strides) // C.itemsize

    # row order contiguous array
    A_flat = A.ravel()
    # col order contiguous array
    B_flat = B.ravel(order="F")
    C_flat = C.ravel()

    for i in range(C_flat.size):
        a_row = i // C_strides[0]
        b_col = i % C_strides[0]
        a_slice = slice(a_row * C_strides[0], (a_row + 1) * C_strides[0])
        b_slice = slice(b_col * C_strides[0], (b_col + 1) * C_strides[0])
        C_flat[i] = np.dot(A_flat[a_slice], B_flat[b_slice])

    return C_flat.reshape(C_shape)

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
    zero_pad = np.zeros(pad_size, dtype=np.int32)

    strides = np.hstack(
        (zero_pad, A_batch_strides),
    )

    # standard broadcasting rule for batches 
    # set stride to 0 whenever shape is 1
    for i in range(-len(A_batch_strides), 0): 
        if A_batch_shape[i] == 1:
            strides[i] = 0  

    return strides

def matmul(A, B): 
    # force things into matrix forms
    squeeze_left = A.ndim < 2 
    squeeze_right = B.ndim < 2
    A = A if A.ndim > 1 else A.reshape(1, -1)
    B = B if B.ndim > 1 else B.reshape(-1, 1)

    _verify_shapes(A, B)
    K = A.shape[-1]

    # get new shape for the product
    batch_shape, mat_shape = _broadcast_shape(A, B)
    C = np.zeros(batch_shape + mat_shape) 
    
    # strides
    A_strides = np.array(A.strides, dtype=np.int32) // A.itemsize
    B_strides = np.array(B.strides, dtype=np.int32) // B.itemsize
    C_strides = np.array(C.strides, dtype=np.int32) // C.itemsize

    A_batch_strides = _broadcast_stride(A_strides, A.shape, len(batch_shape))
    A_mat_strides = A_strides[-2:] if A.ndim > 2 else A_strides

    B_batch_strides = _broadcast_stride(B_strides, B.shape, len(batch_shape)) 
    B_mat_strides = B_strides[-2:] if B.ndim > 2 else B_strides

    C_batch_strides = C_strides[:len(batch_shape)]
    C_mat_strides = C_strides[len(batch_shape):]

    # from here, this should be the kernel that does this
    # pointers
    A_flat = A.ravel() 
    B_flat = B.swapaxes(-2, -1).ravel() 
    C_flat = C.ravel()

    coord_calc_time = []
    dot_product_time = []

    for i in range(C.size): 
        # calculate the batch and matrix coordinates 
        # these will be used to determine where i corresponds
        # to original tensor C
        # this is a very heavy operation!!
        start = time.perf_counter()
        C_batch_coord = (i // C_batch_strides) % batch_shape 
        C_mat_coord = (i // C_mat_strides) % mat_shape
        end = time.perf_counter()
        coord_calc_time.append(end - start)

        # using strides to determine batch coordinates of 
        # input matrices; flat index gives the batch offset 
        A_batch_offset = int((C_batch_coord * A_batch_strides).sum())
        B_batch_offset = int((C_batch_coord * B_batch_strides).sum())

        # get row and column of the matrix 
        row = C_mat_coord[0] 
        col = C_mat_coord[1] 

        start = time.perf_counter()
        # slices for the correct row and column
        A_slice = slice(A_batch_offset + row * K, A_batch_offset + (row + 1) * K)
        B_slice = slice(B_batch_offset + col * K, B_batch_offset + (col + 1) * K)

        # Take the dot product
        C_flat[i] = (A_flat[A_slice] * B_flat[B_slice]).sum()
        end = time.perf_counter()
        dot_product_time.append(end - start)
    print(np.array(coord_calc_time).mean(), np.array(dot_product_time).mean())

    if squeeze_left: 
        return C.squeeze(axis=0)
    elif squeeze_right:
        return C.squeeze(axis=-1)
    else:
        return C

if __name__ == "__main__":
    shapes = [ 
        [(2, 1, 3, 2), (1, 4, 2, 5)],
        [(22,), (22, 22)],
        [(22, 22), (22,)],
        [(22,), (22,)],
        [(32,64), (64, 32)],
        [(24, 1, 32,64), (53, 64, 32)]
    ]

    for a_shape, b_shape in shapes:
        A = np.random.rand(*a_shape)
        B = np.random.rand(*b_shape)

        C = matmul(A, B)

        assert np.allclose(A @ B, C)
