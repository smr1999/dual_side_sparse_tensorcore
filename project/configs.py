from project.enums.matrix_layout import MatrixLayout


class Configs:
    # do not change these settings
    M: int = 32
    N: int = 32
    K: int = 16  # we can set any value for K because each grid computes 32*32*K

    # do not change these settings
    m_stride = 8
    n_stride = 16
    k_stride = 1

    # do not change these settings
    matrixA_layout: MatrixLayout = MatrixLayout.col_major
    matrixB_layout: MatrixLayout = MatrixLayout.row_major
    matrixC_layout: MatrixLayout = MatrixLayout.row_major

    # Adaptively change this prarmeters (0 <= sparsity_ratio <= 1)
    matrixA_sparsity_ratio: float = 0.9
    matrixB_sparsity_ratio: float = 0.9
    matrixC_sparsity_ratio: float = 0
