from src import SynthWrapper as SW
from include import ctcsound as cs

class Csound_SynthWrapper(SW.SynthWrapper):

    def __init__(self):

        # Defines an orchestra of instruments that are run
        self.orc = """

            sr = 44100
            ksmps = 32
            nchnls = 1
            0dbfs  = 1

            instr 1
                out(oscili(p4/2 + oscili:k(p4/2, p6), p5))
            endin

        """

        # Setup dictionaries of parameters
        self.parameters = [
            {
                "name": "sound frequency",
                "min": 27.5,
                "max": 7040.0,
                "increment": 125.0
            },
            {
                "name": "LFO frequency",
                "min": 2.5,
                "max": 7.0,
                "increment": 0.125
            }
        ]

    def initialize(self) -> bool:
        pass

    def generate_sound(self, file_name:str, parameters:list) -> bool:
        self.synth = cs.Csound()

        sco_beg = "i 1 0 5 0.5 "
        sco_end = ""

        params = ""
        for p in parameters:
            params = params + str(p) + " "
        
        sco = sco_beg + params + sco_end

        self.synth.compileOrc(self.orc)
        self.synth.readScore(sco)
        self.synth.setOption('-o' + file_name + '.wav')
        self.synth.start()
        self.synth.perform()
        self.synth.stop()
        self.synth.reset()

        return True
