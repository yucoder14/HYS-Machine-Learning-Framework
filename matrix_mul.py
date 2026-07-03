import numpy as np

def _verify_shapes(A_shape, B_shape):
    k = A_shape[-1]
    k_prime = B_shape[-2] if len(B_shape) > 1 else B_shape[-1]

    if k != k_prime:
        raise ValueError(f"Invalid shapes: {A_shape} and {B_shape}")

    for i in range(-min(len(A_shape), len(B_shape)), -2): 
        if A_shape[i] != B_shape[i]:  
            raise ValueError(f"Invalid shapes: {A_shape} and {B_shape}")
    

# treating the tensor as stacks of matrices
def _broadcast_shape(A_shape, B_shape): 
    ndim = max(len(A_shape), len(B_shape)) 
    k = A_shape[-1]
    k_prime = B_shape[-2] if len(B_shape) > 1 else B_shape[-1]

    def _pad_ones(shape, left=True):
        if len(shape) == 1: 
            if left: 
                shape = (1,) + shape
            else: 
                shape = shape + (1,)
        return np.array(shape)
    
    A_shape = _pad_ones(A_shape)
    B_shape = _pad_ones(B_shape, False)
    C_shape = np.zeros(ndim, dtype=np.int32)
    
    for i in range(-len(A_shape), -1):
        C_shape[i] = A_shape[i]

    for i in range(-len(B_shape), 0): 
        if not C_shape[i]:
            C_shape[i] = B_shape[i]

    return C_shape
     

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

def matmul_strides(A, B):
    _verify_shapes(A.shape, B.shape)

    C_shape = _broadcast_shape(A.shape, B.shape)
    C = np.zeros(C_shape)

    A_flat = A.ravel()
    B_flat = B.ravel(order="F")
    C_flat = C.ravel()

    A_strides = np.array(A.strides, dtype=np.int32) // A.itemsize
    B_strides = np.array(B.strides, dtype=np.int32) // B.itemsize
    C_strides = np.array(C.strides, dtype=np.int32) // C.itemsize

    print(C.shape)
    print(A_strides, B_strides, C_strides)
    k = C_strides[-2]  

    for i in range(C_flat.size):
        print((i // k) + i //A.shape[-2], i)

    return C.reshape(C_shape)

if __name__ == "__main__": 
    A_shape = (2, 1, 4)
    B_shape = (2, 4, 2)

    A = np.random.rand(*A_shape)
    B = np.random.rand(*B_shape)

    C = matmul_strides(A, B)

    assert np.allclose(A @ B, C)
    
    #assert np.allclose(A @ B, matmul_2D_naive_strides(A, B))
    #assert np.allclose(A @ B, matmul_2D_naive(A, B))
