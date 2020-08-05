from abc import ABC, abstractmethod

class SynthWrapper(ABC):
    """Interface for a synthesizer wrapper class"""

    # Example parameter list of dictionaries
    # Has to be implemented for all parameters that are required to be swept
    parameters = [
        {
            "name":         "parameter 1",
            "min":          0,
            "max":          10,
            "increment":    1
        },
        {
            "name":         "parameter 2",
            "min":          2.5,
            "max":          23.7,
            "increment":    2.5
        }
    ]

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initializes the synthesizer, returns true if successful"""
        pass

    @abstractmethod
    def generate_sound(self, file_name: str, parameters: list) -> bool:
        """Generates the sound made by the synth given a list of parameters
        
        file_name: name of the output file
        parameters: list of parameters to apply to the synthesizer
        """
        pass
