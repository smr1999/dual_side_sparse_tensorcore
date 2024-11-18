import numpy as np


def mma_cpu(matrixA: np.ndarray, matrixB: np.ndarray, matrixC: np.ndarray, 
            M: int, N: int, K: int):
    
    result: np.ndarray = np.zeros([M,N], dtype=float)
    
    for m in range(M):
        for n in range(N):
            result[m,n] = matrixC[m,n]
            for k in range(K):
                result[m,n] += matrixA[m,k] * matrixB[k,n] 
    
    return result