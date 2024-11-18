import random
import numpy as np
from project.enums.matrix_layout import MatrixLayout


class Matrix:
    def __init__(self, row_size: int, col_size: int,
                 layout: MatrixLayout,
                 sparse_ratio: float,
                 minimum: float = 10.0, maximum: float = 50.0) -> None:
        self.row_size: int = row_size
        self.col_size: int = col_size

        self.minimum: float = minimum
        self.maximum: float = maximum

        self.layout: MatrixLayout = layout
        self.sparse_ratio: float = sparse_ratio

        self.__matrix: np.ndarray = None

        self.__condense_matrix: np.ndarray = np.zeros(
            [self.row_size, self.col_size])
        self.__bitmap_matrix: np.ndarray = np.zeros(
            [self.row_size, self.col_size],
            dtype=int
        )

    @property
    def matrix(self) -> np.ndarray:
        return self.__matrix

    @property
    def condense_matrix(self) -> np.ndarray:
        return self.__condense_matrix

    @property
    def bitmap_matrix(self) -> np.ndarray:
        return self.__bitmap_matrix

    def create_matrix(self) -> None:
        matrix: list = []
        for i in range(self.row_size):
            row: list = []
            for j in range(self.col_size):
                row.append(random.uniform(self.minimum, self.maximum))
            matrix.append(row)

        self.__matrix = np.vstack(matrix)

    def make_sparse(self) -> None:
        assert self.sparse_ratio >= 0 and self.sparse_ratio <= 1

        flattened_matrix: np.ndarray = self.__matrix.flatten()
        flattened_matrix[np.random.choice(
            flattened_matrix.size,
            int(flattened_matrix.size * self.sparse_ratio),
            replace=False)
        ] = 0

        self.__matrix = flattened_matrix.reshape(
            self.row_size, self.col_size
        )

    def make_dense_with_bitmap(self) -> None:
        temp: int = 0
        if self.layout == MatrixLayout.row_major:
            for i in range(self.row_size):
                temp = 0
                for j in range(self.col_size):
                    if self.__matrix[i, j] != 0:
                        self.__bitmap_matrix[i, j] = 1
                        self.__condense_matrix[i, temp] = self.__matrix[i, j]
                        temp += 1

        elif self.layout == MatrixLayout.col_major:
            for j in range(self.col_size):
                temp = 0
                for i in range(self.row_size):
                    if self.__matrix[i, j] != 0:
                        self.__bitmap_matrix[i, j] = 1
                        self.__condense_matrix[temp, j] = self.__matrix[i, j]
                        temp += 1
