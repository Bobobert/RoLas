"""
    Types
"""

from typing import Union
import numpy as np

class AgentType:
    config, policy, tbw = {}, None, None
    def initAgent(self, **kwargs):
        pass
    def currentState(self) -> dict:
        pass
    def loadState(self, state:dict):
        pass
    def getBatch(self, size: int,proportion: float = 1.0, random = False,) -> dict:
        pass
    def getEpisode(self, random) -> dict:
        pass
    def fullStep(self) -> dict:
        pass
    def envStep(self, action) -> dict:
        pass
    def test(self, iters:int) -> dict:
        pass
    def reset(self) -> dict:
        pass

class PolicyType:
    config, tbw = {}, None
    _test = True
    def initPolicy(self, **kwargs):
        pass
    def currentState(self) -> dict:
        pass
    def loadState(self, state:dict):
        pass
    def getAction(self, observation) -> Union[int, float, list, np.ndarray]:
        pass
    def getActions(self, observation) -> Union[list, np.ndarray]:
        pass
    def getValue(self, observation, action) -> float:
        pass
    def update(self, batchDict: dict):
        pass
    @property
    def test(self):
        return self._test
    @property
    def train(self):
        return not self._test

    