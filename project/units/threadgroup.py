import numpy as np
from project.units.feop import FEOP


class ThreadGroup:
    def __init__(self) -> None:
        self.vectorA: np.ndarray = None
        self.vectorB: np.ndarray = None
        self.vectorC: np.ndarray = None

        self.feops: list[FEOP] = []
        for _ in range(4):
            self.feops.append(FEOP())

    def set_values(self, vectorA: np.ndarray, vectorB: np.ndarray, vectorC: np.ndarray):
        assert len(vectorA) == 4 and len(vectorB) == 4 and len(vectorC) == 4

        self.vectorA = vectorA
        self.vectorB = vectorB
        self.vectorC = vectorC

    def execute(self) -> np.ndarray:
        result: list = []

        for i in range(4):
            self.feops[i].set_values(
                self.vectorA[i], self.vectorB, self.vectorC[i]
            )

            result.append(self.feops[i].execute())

        return np.vstack(result)