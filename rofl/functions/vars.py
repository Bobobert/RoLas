"""
Scheduled variables, this can be moved each time they are called or by a moving method
"""
from abc import ABC
from .const import math

class Variable(ABC):
    """
    Abstract class for a variable with some schedule
    """
    _opvalue_ = None
    _value_ = None
    def __call__(self):
        self._step_()
        return self.value
        
    @property
    def value(self):
        """
        Main property for the class to output its value.
        To advance one step first call the variable

        returns value
        """
        return self._value_

    def _step_(self):
        """
        Method to update the variable value of the object 
        each time is called. Can be passed to have a manual
        step calling
        """
        pass

    def step(self):
        """
        Method to update the variable in a manual manner.
        """
        raise NotImplementedError

    def restore(self, step:int = -1):
        """
        Restore the intial value to the variable or if a step 
        is given restores from that step

        returns None
        """
        if step <= 0:
            self._i_ = 0
            self._value_ = self._opvalue_
        else:
            self._i_ = step -1
            self.step()

    def __eq__(self, other):
        self._step_()
        return self._value_.__eq__(other)
    def __le__(self, other):
        self._step_()
        return self._value_.__le__(other)
    def __lt__(self, other):
        self._step_()
        return self._value_.__lt__(other)
    def __gt__(self, other):
        self._step_()
        return self._value_.__gt__(other)
    def __ge__(self, other):
        self._step_()
        return self._value_.__ge__(other)
    def __add__(self, other):
        return self._value_.__add__(other)
    def __mul__(self, other):
        self._step_()
        return self._value_.__mul__(other)
    def __repr__(self):
        return self._value_.__repr__()
    def __float__(self):
        self._step_()
        return self._value_.__float__()
    def __div__(self, other):
        self._step_()
        return self._value_.__div__(other)
    def __truediv__(self, other):
        self._step_()
        return self._value_.__truediv__(other)
    def __floordiv__(self, other):
        self._step_()
        return self._value_.__floordiv__(other)
    def __neg__(self):
        self._step_()
        return self._value_.__neg__()

def updateVar(config):
    vrs = config.get("variables", [])
    for v in vrs:
        v.step()

class linearSchedule(Variable):
    def __init__(self, initValue, life:int, minValue = None, maxValue = None):
        assert (minValue is not None) or (maxValue is not None), \
            "At least one of these must be not None to describe behavior"
        if minValue is not None:
            assert initValue >= minValue, "Initial value is less than minimal"
            self._last_ = minValue
            self._F = max
        elif maxValue is not None:
            assert initValue <= maxValue, "Initial value is more than maximum"
            self._last_ = maxValue
            self._F = min
        self._opvalue_ = initValue
        self._value_ = initValue
        self._diff_ = self._last_ - self._opvalue_
        self._i_ = 0
        assert life > 0, "Life of the variable must be positive. Live Chill"
        self._life_ = life
    
    def step(self):
        xm = self._diff_ * self._i_ / self._life_
        y = self._opvalue_ + xm
        self._i_ += 1
        self._value_ = self._F(self._last_, y)

    def __repr__(self):
        s = "{} linearSchedule: init {}, last {}, life {}".format(self._value_, self._opvalue_, self._last_, self._life_)
        return s

class runningStat():
    """
    Custom class to keep track of a single value running
    statistic. 

    Defaults to zeros, usual method to add the values with the 
    operator +=, eg

        x = runningStat()\\
        x += 2\\
        x += 3\\
        x()

    Based from http://www.johndcook.com/blog/standard_deviation/
    """
    def __init__(self):
        self.__mn__ = 0.0
        self.__sn__ = 0.0
        self.__n__ = 0

    def __call__(self):
        return (self.mean, self.std)

    def __iadd__(self, other):
        if self.__n__ == 0:
            self.__mn__ = other
            self.__n__ += 1
        else:
            self.__n__ += 1
            oldMN = self.__mn__
            self.__mn__ += (other - oldMN) / self.__n__
            self.__sn__ += (other - oldMN) * (other - self.__mn__)
        return self

    def add(self, x):
        self += x

    @property
    def mean(self):
        return self.__mn__

    @property
    def var(self):
        return self.__sn__ / (self.__n__ -  1) if self.__n__ > 1 else math.pow(self.__sn__, 2)

    @property
    def std(self):
        return math.sqrt(self.var)
    
    def reset(self):
        self.__init__()

    def __repr__(self):
        return "mean {:.4f} std {:.4f}".format(*self())
