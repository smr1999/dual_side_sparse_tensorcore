import numpy as np
from project.units.subcore import Subcore


def dense_mma_gpu(matrixA: np.ndarray, matrixB: np.ndarray, matrixC: np.ndarray,
                  M: int, N: int, K: int,
                  m_stride: int, n_stride: int, k_stride: int,
                  subcore: Subcore
                  ) -> tuple[np.ndarray, int]:
    num_OHMMA: int = 0
    matrixD_GPU: np.ndarray = matrixC.copy()

    for k in range(0, K, k_stride):
        for i in range(0, M, m_stride):
            for j in range(0, N, n_stride):
                num_OHMMA += 1

                subcore.set_values(
                    matrixA[i:i+8, k],
                    matrixB[k, j:j+16],
                    matrixD_GPU[i:i+8, j:j+16]
                )
                matrixD_GPU[i:i+8, j:j+16] = subcore.execute()

    return matrixD_GPU, num_OHMMA
