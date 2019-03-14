from abc import ABC, abstractmethod

class ConfoundDetection(ABC):
    """
    Interface to describe a class that detects the possible confounding variables
    to control for.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self, d):
        pass
        