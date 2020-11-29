import numpy as np

def pivoting(A, k):
    max_row_offset = np.argmax(np.abs(A[k:, k]))
    if max_row_offset != k:
        A[[k, k+max_row_offset]] = A[[k+max_row_offset, k]]


def gauss_elim_row(A):
    result = np.array(A, dtype=np.float64)
    if result.shape[0] != result.shape[1]:
        raise Exception

    n = result.shape[0]
    P = np.zeros((n, n))
    for k in range(n-1):
        pivoting(result, k)
        Akk = result[k,k]
        result[k,k] = 1
        result[k,k+1:] = result[k,k+1:]/Akk
        for j in range(k+1, n):
            result[j,k+1:] -= result[k,k+1:] * result[j,k]
    
    result[n-1,n-1] = 1
    return result


if __name__ == "__main__":
    matrix = np.random.randint(5, 10, (4, 4))
    print(matrix)
    print(np.array2string(gauss_elim_row(matrix), precision=2))
    lu = gauss_elim_row(matrix)
    L = lu.copy()
    U = lu.copy()
    for i in range(4):
        L[:i,i] = 0
        U[i,:i] = 0
    
    print(L @ U)

