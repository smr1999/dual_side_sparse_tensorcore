import numpy as np


def popc(vector: np.ndarray, val: int = 1) -> int:
    count: int = 0
    for i in vector:
        if i == val:
            count += 1

    return count


def is_equal(first_matrix: np.ndarray, second_matrix: np.ndarray,
             row_size: int, col_size: int, eps: float = 1e-4) -> bool:
    for i in range(row_size):
        for j in range(col_size):
            if abs(first_matrix[i, j] - second_matrix[i, j]) > eps:
                return False

    return True


def print_matrix(matrix: np.ndarray):
    for row in matrix:
        for item in row:
            print(round(item, 2), end=" ")
        print()
