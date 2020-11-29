import numpy as np
import time

def rank1_updateT(C, A_column, B_row):
    m = A_column.shape[0]
    n = B_row.shape[0]
    for j in range(n):
        for i in range(m):
            C[i, j] += A_column[i] * B_row[j]

def matrix_multp_6b(A, B):
    m, k = A.shape
    _, n = B.shape
    C = np.zeros([m, n])
    for p in range(k):
        rank1_updateT(C, A[:, p], B[p, :])
    return C

if __name__ == "__main__":
    n = 100
    k = 200
    for m in [10, 100, 1000]:
        A = np.random.randint(0, 20, (m,k))
        B = np.random.randint(0, 20, (k,n))
        print(f"A: {m}x{k}, B: {k}x{n}; time:", end=" ")
        start = time.time()
        matrix_multp_6b(A, B)
        stop = time.time()
        print(f"{stop-start} s")
