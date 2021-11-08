"""
    Types
"""

from typing import Tuple, Union
from abc import ABC, abstractmethod

from numpy import ndarray
from torch import Tensor, device

_OBS_TYPE = Union[Tensor, ndarray]
_ACT_TYPE = Union[int, float, list, ndarray]
_VAL_TYPE = Union[float, Tensor]

class PolicyType(ABC):

    @abstractmethod
    def currentState(self) -> dict:
        pass

    @abstractmethod
    def loadState(self, state: dict) -> None:
        pass

    @abstractmethod
    def update(self, *batchDict: Tuple[dict,...]):
        pass

    @abstractmethod
    def getAction(self, observation: Union[Tensor, ndarray]) -> Union[int, float, list, ndarray]:
        pass

    @abstractmethod
    def getActions(self, observation: Union[Tensor, ndarray]) -> Union[list, ndarray]:
        pass

    @abstractmethod
    def getRndAction(self) -> Union[int, float, list, ndarray]:
        pass

    @abstractmethod
    def getValue(self, observation: Union[Tensor, ndarray], action) -> float:
        pass

    @abstractmethod
    def getActionWVal(self, observation: Union[Tensor, ndarray]) -> tuple:
        pass

    @abstractmethod
    def getAVP(self, observation: Union[Tensor, ndarray]) -> tuple:
        pass

    @abstractmethod
    def getProb4Action(self, observation: Union[Tensor, ndarray]) -> tuple:
        pass

    @property
    @abstractmethod
    def device(self) -> device:
        pass

    @property
    @abstractmethod
    def test(self) -> bool:
        pass
    
    @test.setter
    @abstractmethod
    def test(self, flag: bool):
        pass

    @property
    def train(self) -> bool:
        return not self.test

    @train.setter
    def train(self, flag: bool):
        self.test = not flag
    

class AgentType(ABC):

    @abstractmethod
    def currentState(self) -> dict:
        pass

    @abstractmethod
    def loadState(self, state: dict) -> None:
        pass
    
    @abstractmethod
    def rndAction(self) -> Union[int, float, list, ndarray]:
        pass

    @abstractmethod
    def getBatch(self, size: int,proportion: float = 1.0, random:bool = False) -> dict:
        pass

    @abstractmethod
    def getEpisode(self, random: bool, device = None) -> dict:
        pass

    @abstractmethod
    def fullStep(self) -> dict:
        pass

    @abstractmethod
    def envStep(self, action: Union[int, float, list, ndarray]) -> dict:
        pass

    @abstractmethod
    def test(self, iters:int) -> dict:
        pass

    @abstractmethod
    def reset(self) -> dict:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
    
    @property
    @abstractmethod
    def device(self) -> device:
        pass
