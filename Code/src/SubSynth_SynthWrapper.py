from src import SynthWrapper as SW
from include import ctcsound as cs

class SubSynth_SynthWrapper(SW.SynthWrapper):
    # based on https://write.flossmanuals.net/csound/b-subtractive-synthesis/
    
    def __init__(self):
        
        self.orc = """

            sr = 44100
            ksmps = 4
            nchnls = 2
            0dbfs = 1

            initc7 1,1,0.8                 ;set initial controller position

            prealloc 1, 10

            instr 1

                ; asign p-fields to variables
                iCPS   =            cpsmidinn(p4) ;convert from note number to cps
                kAmp1  =            p5
                iType1 =            p6
                kPW1   =            p7
                kOct1  =            octave(p8) ;convert from octave displacement to multiplier
                kTune1 =            cent(p9)   ;convert from cents displacement to multiplier
                kAmp2  =            p10
                iType2 =            p11
                kPW2   =            p12
                kOct2  =            octave(p13)
                kTune2 =            cent(p14)
                iCF    =            p15
                iFAtt  =            p16
                iFDec  =            p17
                iFSus  =            p18
                iFRel  =            p19
                kRes   =            p20
                iAAtt  =            p21
                iADec  =            p22
                iASus  =            p23
                iARel  =            p24

                ;oscillator 1
                ;if type is sawtooth or square...
                if iType1==1||iType1==2 then
                ;...derive vco2 'mode' from waveform type
                    iMode1 = (iType1=1?0:2)
                    aSig1  vco2   kAmp1,iCPS*kOct1*kTune1,iMode1,kPW1;VCO audio oscillator
                else                                   ;otherwise...
                    aSig1  noise  kAmp1, 0.5              ;...generate white noise
                endif

                ;oscillator 2 (identical in design to oscillator 1)
                if iType2==1||iType2==2 then
                    iMode2  =  (iType2=1?0:2)
                    aSig2  vco2   kAmp2,iCPS*kOct2*kTune2,iMode2,kPW2
                else
                    aSig2 noise  kAmp2,0.5
                endif

                ;mix oscillators
                aMix       sum          aSig1,aSig2
                ;lowpass filter
                kFiltEnv   expsegr      0.0001,iFAtt,iCPS*iCF,iFDec,iCPS*iCF*iFSus,iFRel,0.0001
                aOut       moogladder   aMix, kFiltEnv, kRes

                ;amplitude envelope
                aAmpEnv    expsegr      0.0001,iAAtt,1,iADec,iASus,iARel,0.0001
                aOut       =            aOut*aAmpEnv
                        outs         aOut,aOut
            endin

        """

        self.parameters = [
            {
                # Defined as a numbered note 
                # that gets translated by the instrument definition into a Hz value
                "name": "input key",
                "min": 10,
                "max": 80,
                "increment": 1
            },
            # ===========================================================
            # Oscillator 1
            {
                "name": "o1 amplitude",
                "min": 0.2,
                "max": 1,
                "increment": 0.2
            },
            {
                "name": "o1 type",
                "min": 2,
                "max": 2,
                "increment": 2
            },
            {
                # Values of 0 and 1 mean nothing as they create no sound
                "name": "o1 pulse width modulation",
                "min": 0.2,
                "max": 0.8,
                "increment": 0.2
            },
            # ===========================================================
            # Oscillator 2
            {
                "name": "o2 amplitude",
                "min": 0,
                "max": 1,
                "increment": 0.2
            },
            {
                "name": "o2 type",
                "min": 2,
                "max": 2,
                "increment": 2
            },
            {
                # Values of 0 and 1 mean nothing as they create no sound
                "name": "o2 pulse width modulation",
                "min": 0.2,
                "max": 0.8,
                "increment": 0.2
            }
        ]

    
    def initialize(self) -> bool:
        pass

    def generate_sound(self, file_name:str, parameters:list) -> bool:
        self.synth = cs.Csound()

        sco_beg = "i 1 0 5 "

        sco_inter_1 = "0 0 "
        sco_inter_2 = " 0 0 "

        # defines global low pass filter envelope
        # set to be non intrusive, at 20kHz
        # also defines the global amplitude envelope, 
        # set to be static for all sounds to make sure the sound is 5s long
        # attack decay sustain release
        sco_end = "20000 0.05 0.95 1 1 0 0.05 0.95 1 1"

        params = ""
        for p in parameters[:-3]:
            params = params + str(p) + " "
        
        params_2 = ""
        for p in parameters[-3:]:
            params_2 = params_2 + str(p) + " "
        
        sco = sco_beg + params + sco_inter_1 + params_2 + sco_inter_2 + sco_end

        self.synth.compileOrc(self.orc)
        self.synth.readScore(sco)
        self.synth.setOption('-o' + file_name + '.wav')
        self.synth.start()
        self.synth.perform()
        self.synth.stop()
        self.synth.reset()

        return True