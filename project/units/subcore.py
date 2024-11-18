import numpy as np
from project.units.tensorcore import Tensorcore


class Subcore:
    def __init__(self) -> None:
        self.vectorA: np.ndarray = None
        self.vectorB: np.ndarray = None
        self.vectorC: np.ndarray = None

        self.tensorcores: list[Tensorcore] = []
        for i in range(2):
            self.tensorcores.append(Tensorcore())

    def set_values(self, vectorA: np.ndarray, vectorB: np.ndarray, vectorC: np.ndarray):
        assert len(vectorA) == 8 and len(vectorB) == 16 and len(vectorC) == 8

        self.vectorA = vectorA
        self.vectorB = vectorB
        self.vectorC = vectorC

    def execute(self) -> np.ndarray:
        result: list = []

        for i in range(2):
            self.tensorcores[i].set_values(
                self.vectorA,
                self.vectorB[i*8:(i+1)*8],
                self.vectorC[:, i*8:(i+1)*8]
            )

            result.append(self.tensorcores[i].execute())

        return np.hstack(result)
