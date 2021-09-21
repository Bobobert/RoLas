"""
    You encountered the Dummy 
    A cotton heart and a button eye
"""

class dummyTBW:
    egg = "You talked to the DUMMY . . ."
    def __init__(self, *args):
        None
    def __call__(self, *args):
        return None
    def __eq__(self, x):
        if x is None:
            return True
        return False
    def __repr__(self):
        return self.egg
    def close(self):
        pass
