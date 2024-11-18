import math
import numpy as np
from project.utils.commons import popc
from project.units.subcore import Subcore


def gather(matrix: np.ndarray, bitmap: np.ndarray,
           row_size: int, col_size: int) -> np.ndarray:
    result: list = []

    for i in range(row_size):
        if popc(bitmap[i, :]) == 0:
            continue
        temp: list = []
        for j in range(col_size):
            if bitmap[i, j] == 1:  # Only load specific values
                temp.append(matrix[i][j])
        result.append(temp)

    return np.array([]) if len(result) == 0 else np.vstack(result)


def accumulate(first_matrix: np.ndarray, second_matrix: np.ndarray,
               row_size: int, col_size: int) -> np.ndarray:
    result: np.ndarray = np.zeros([row_size, col_size])

    for i in range(row_size):
        for j in range(col_size):
            result[i, j] = first_matrix[i, j] + second_matrix[i, j]

    return result


def scatter(matrix: np.ndarray, value_matrix: np.ndarray, bitmap: np.ndarray,
            row_size: int, col_size: int) -> np.ndarray:
    x_index: int = 0
    y_index: int = 0

    for i in range(row_size):
        if popc(bitmap[i, :]) != 0:
            for j in range(col_size):
                if bitmap[i, j] == 1:
                    matrix[i, j] = value_matrix[x_index, y_index]

                    y_index += 1

            y_index = 0
            x_index += 1

    return matrix


def sparse_mma_gpu(matrixA_condense: np.ndarray, matrixB_condense: np.ndarray, matrixC: np.ndarray,
                   matrixA_bitmap: np.ndarray, matrixB_bitmap: np.ndarray,
                   M: int, N: int, K: int,
                   m_stride: int, n_stride: int, k_stride: int,
                   subcore: Subcore
                   ) -> tuple[np.ndarray, int]:
    num_OHMMA: int = 0
    matrixD_GPU: np.ndarray = matrixC.copy()

    for k in range(0, K, k_stride):
        vectorA_bitmap: np.ndarray = matrixA_bitmap[:, k]
        vectorB_bitmap: np.ndarray = matrixB_bitmap[k, :]
        mult_bitmaps: np.ndarray = np.dot(
            vectorA_bitmap.reshape(M, 1),
            vectorB_bitmap.reshape(1, N)
        )

        # 1. Gather step
        partial_matrixD: np.ndarray = \
            gather(matrixD_GPU, mult_bitmaps,
                   M, N)

        temp_matrix: np.ndarray = np.zeros([M, N], dtype=float)
        for i in range(0, math.ceil(popc(vectorA_bitmap)/m_stride) * m_stride, m_stride):
            for j in range(0, math.ceil(popc(vectorB_bitmap)/n_stride) * n_stride, n_stride):
                num_OHMMA += 1
                subcore.set_values(
                    matrixA_condense[i:i+m_stride, k],
                    matrixB_condense[k, j:j+n_stride],
                    temp_matrix[i:i+m_stride, j:j+n_stride]
                )

                temp_matrix[i:i+m_stride, j:j+n_stride] = subcore.execute()

        if partial_matrixD.shape[0] != 0:
            # 2. Accumulate
            partial_matrixD: np.ndarray = \
                accumulate(partial_matrixD, temp_matrix,
                           partial_matrixD.shape[0],
                           partial_matrixD.shape[1]
                           )

            # 3. Scatter
            matrixD_GPU: np.ndarray = scatter(
                matrixD_GPU, partial_matrixD, mult_bitmaps,
                M, N
            )

    return matrixD_GPU, num_OHMMA
