import numpy as np
from project.configs import Configs
from project.data_structures.matrix import Matrix
from project.units.subcore import Subcore
from project.utils.commons import is_equal

from project.utils.cpu_mma import mma_cpu
from project.utils.dense_gpu_mma import dense_mma_gpu
from project.utils.sparse_gpu_mma import sparse_mma_gpu


class Main:

    def __init__(self) -> None:
        self.create_matrices()
        self.sparsify_matrices()
        self.condense_matrices()
        self.execute()

    def create_matrices(self):
        self.matrixA: Matrix = Matrix(Configs.M, Configs.K,
                                      Configs.matrixA_layout,
                                      Configs.matrixA_sparsity_ratio)
        self.matrixB: Matrix = Matrix(Configs.K, Configs.N,
                                      Configs.matrixB_layout,
                                      Configs.matrixB_sparsity_ratio)
        self.matrixC: Matrix = Matrix(Configs.M, Configs.N,
                                      Configs.matrixC_layout, 
                                      Configs.matrixC_sparsity_ratio)

        self.matrixA.create_matrix()
        self.matrixB.create_matrix()
        self.matrixC.create_matrix()

    def sparsify_matrices(self):
        self.matrixA.make_sparse()
        self.matrixB.make_sparse()
        self.matrixC.make_sparse()

    def condense_matrices(self):
        self.matrixA.make_dense_with_bitmap()
        self.matrixB.make_dense_with_bitmap()
        self.matrixC.make_dense_with_bitmap()

    def execute(self, debug: bool = True):
        self.subcore = Subcore()
        
        matrixD_CPU = mma_cpu(
            self.matrixA.matrix, self.matrixB.matrix, self.matrixC.matrix,
            Configs.M, Configs.N, Configs.K
        )

        matrixD_GPU_dense, GPU_OHMMA_dense = dense_mma_gpu(
            self.matrixA.matrix, self.matrixB.matrix, self.matrixC.matrix,
            Configs.M, Configs.N, Configs.K,
            Configs.m_stride, Configs.n_stride, Configs.k_stride,
            self.subcore
        )

        matrixD_GPU_sparse, GPU_OHMMA_sparse = sparse_mma_gpu(
            self.matrixA.condense_matrix, self.matrixB.condense_matrix, self.matrixC.matrix,
            self.matrixA.bitmap_matrix, self.matrixB.bitmap_matrix,
            Configs.M, Configs.N, Configs.K,
            Configs.m_stride, Configs.n_stride, Configs.k_stride,
            self.subcore
        )

        if debug:
            if is_equal(matrixD_CPU, matrixD_GPU_dense, Configs.M, Configs.N):
                print("CPU implementation and GPU dense outer product are the same.")
            else:
                print("CPU implementation and GPU dense outer product are different.")

            if is_equal(matrixD_CPU, matrixD_GPU_sparse, Configs.M, Configs.N):
                print("CPU implementation and GPU sparse outer product are the same.")
            else:
                print("CPU implementation and GPU sparse outer product are different.")

            if is_equal(matrixD_GPU_dense, matrixD_GPU_sparse, Configs.M, Configs.N):
                print(
                    "GPU dense outer product and GPU sparse outer product are the same.")
                print(f"Speedup sparse over dense is: {GPU_OHMMA_dense}/{GPU_OHMMA_sparse} = {"INF" if GPU_OHMMA_sparse == 0 else round(GPU_OHMMA_dense/GPU_OHMMA_sparse * 100, 2)}%.")
            else:
                print(
                    "GPU dense outer product and GPU sparse outer product are different.")
                
            
