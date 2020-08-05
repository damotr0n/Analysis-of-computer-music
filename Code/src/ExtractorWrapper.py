from abc import ABC, abstractmethod, abstractproperty

class ExtractorWrapper(ABC):
    """Interface for an extractor wrapper class"""

    features = ["tempo"]

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def extract_features(self, input_filename:str):
        """Extract features from sound given by the filename"""
        pass
