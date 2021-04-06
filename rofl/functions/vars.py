"""
Scheduled variables, this can be moved each time they are called or by a moving method
"""
from abc import ABC

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
            self._step_()
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
    def __radd__(self, other):
        self._step_()
        return self._value_.__radd__(other)
    def __mul__(self, other):
        self._step_()
        return self._value_.__mul__(other)
    def __rmul__(self, other):
        self._step_()
        return self._value_.__rmul__(other)
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

    def _step_(self):
        xm = self._diff_ * self._i_ / self._life_
        y = self._opvalue_ + xm
        self._i_ += 1
        self._value_ = self._F(self._last_, y)
