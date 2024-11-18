import numpy as np
from project.units.octet import Octet


class Tensorcore:
    def __init__(self) -> None:
        self.vectorA: np.ndarray = None
        self.vectorB: np.ndarray = None
        self.vectorC: np.ndarray = None

        self.octets: list[Octet] = []
        for i in range(2):
            self.octets.append(Octet())

    def set_values(self, vectorA: np.ndarray, vectorB: np.ndarray, vectorC: np.ndarray):
        assert len(vectorA) == 8 and len(vectorB) == 8 and len(vectorC) == 8

        self.vectorA = vectorA
        self.vectorB = vectorB
        self.vectorC = vectorC

    def execute(self) -> np.ndarray:
        result: list = []

        for i in range(2):
            self.octets[i].set_values(
                self.vectorA,
                self.vectorB[i*4:(i+1)*4],
                self.vectorC[:, i*4:(i+1)*4]
            )

            result.append(self.octets[i].execute())

        return np.hstack(result)
