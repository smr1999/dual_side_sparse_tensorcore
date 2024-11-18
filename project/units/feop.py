import numpy as np


class FEOP:
    def __init__(self) -> None:
        self.vecterA: np.ndarray = None
        self.vectorB: np.ndarray = None
        self.vectorC: np.ndarray = None

    def set_values(self, vecterA: np.ndarray, vectorB: np.ndarray, vectorC: np.ndarray):
        assert len(vectorB) == 4 and len(vectorC) == 4

        self.vecterA = vecterA
        self.vectorB = vectorB
        self.vectorC = vectorC

    def execute(self) -> np.ndarray:
        result: np.ndarray = np.zeros([4])

        for i in range(4):
            result[i] = self.vecterA * self.vectorB[i] + self.vectorC[i]

        return result
