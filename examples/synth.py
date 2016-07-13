import muser.iodata as iodata
import muser.utils as utils
import math
import time

synth = iodata.SynthClient(channels=2)

def sine_tone(amp, freq, phase=0):
    def tone(t):
        return amp * math.sin(2 * math.pi * freq * t + phase)
    return tone

synth.add_synth_function(sine_tone(0.75, 440))
synth.activate()
synth.connect(synth.outports[0], 'system:playback_1')
synth.connect(synth.outports[1], 'system:playback_2')

for i in range(5):
    synth.toggle()
    time.sleep(0.5)
    synth.toggle()
    time.sleep(0.5)

synth.outports.clear()
synth.deactivate()
synth.close()
