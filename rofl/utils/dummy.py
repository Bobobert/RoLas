"""
    You encountered the Dummy 
    A cotton heart and a button eye
"""

class dummyTBW:
    egg = "You talked to the DUMMY beside the chalkboard. . ."
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

class dummySaver():
    egg = 'This dummy smells like flowers, someone stored a bunch inside it'
    def start(self):
        pass
    def check(self, results = None):
        pass
    def addObj(self, *args, **kwargs):
        pass
    def saveAll(self, results = None):
        pass
    def load(self):
        print(self.egg)
    def __repr__(self) -> str:
        return self.egg
