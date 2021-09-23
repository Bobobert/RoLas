import numpy as np
import math

class Stack:
    """
    Dict stack working in a FIFO manner
    """
    stack = dict()
    def __init__(self):
        self.min = 0
        self.actual = 0
    def add(self, obj, **kwargs):
        self.stack[self.actual] = obj
        self.actual += 1
    def pop(self):
        popped = self.stack[self.min]
        self.stack.pop(self.min)
        self.min += 1
        return popped
    def __len__(self):
        return len(self.stack)

class minHeap():
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        return self.heap.__repr__()

    @property
    def empty(self):
        return (np.inf, None)

    def __getitem__(self, i):
        if i < 1:
            raise ValueError("index must be greater than 0")
        if i > len(self):
            return self.empty
        return self.heap[i-1]
    
    def __setitem__(self, i, obj):
        self.heap[i - 1] = obj

    def parent(self, i):
        return max(math.floor(i / 2), 1)

    def right(self, i):
        return 2*i + 1

    def left(self, i):
        return 2*i

    def add(self, obj = None, value = np.inf):
        """
        parameters
        -----------
        value: int, float
            Any real number to compare the objects between them.
        obj: Any
            Any object to store in the heaps position.
        """
        self.heap.append((value, obj))
        if len(self) > 1:
            self.heapUp(len(self))
    
    def min(self):
        return self[1]

    def heapUp(self, i):
        parentIx = self.parent(i)
        if parentIx == i:
            return
        parent, actual = self[parentIx], self[i]
        if actual[0] < parent[0]:
            self[parentIx] = actual
            self[i] = parent
            self.heapUp(parentIx)
    
    def heapDown(self, i):
        left, right = self.left(i), self.right(i)
        n, j = len(self), left
        if left > n:
            return
        elif left < n:
            j = right if self[left][0] > self[right][0] else left
        iTuple, jTuple = self[i], self[j]
        if jTuple[0] < iTuple[0]:
            self[i] = jTuple
            self[j] = iTuple
            self.heapDown(j)
    
    def extractMin(self):
        root, n = self[1], len(self)
        if n < 1:
            return self.empty
        elif n == 1:
            self.heap = []
            return root
        self[1] = self[n]
        self.heap.pop()
        self.heapDown(1)
        return root
    
    def pop(self):
        return self.extractMin()[1]

class maxHeap(minHeap):
    egg = 'You were expecting a minHeap\nBut it was me!\nMaxHeap'
    def min(self):
        print(self.egg)
    def max(self):
        return self._convert(self[1])
    @staticmethod
    def _convert(tup):
        return (-tup[0], tup[1])
    @property
    def empty(self):
        return self._convert(super().empty)
    def add(self, obj = None, value = -np.inf):
        super().add(-value, obj)
    def extractMin(self):
        print(self.egg)
    def extractMax(self):
        return self._convert(super().extractMin())
    def pop(self):
        return self.extractMax()[1]
