import numpy as np
from project.units.threadgroup import ThreadGroup


class Octet:
    def __init__(self) -> None:
        self.vectorA: np.ndarray = None
        self.vectorB: np.ndarray = None
        self.vectorC: np.ndarray = None

        self.threadgroups: list[ThreadGroup] = []
        for i in range(2):
            self.threadgroups.append(ThreadGroup())

    def set_values(self, vectorA: np.ndarray, vectorB: np.ndarray, vectorC: np.ndarray):
        assert len(vectorA) == 8 and len(vectorB) == 4 and len(vectorC) == 8

        self.vectorA = vectorA
        self.vectorB = vectorB
        self.vectorC = vectorC

    def execute(self) -> np.ndarray:
        result: list = []

        for i in range(2):
            self.threadgroups[i].set_values(
                self.vectorA[i*4:(i+1)*4],
                self.vectorB,
                self.vectorC[i*4:(i+1)*4, :]
            )

            result.append(self.threadgroups[i].execute())

        return np.vstack(result)