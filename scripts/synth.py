import muser.live as live
import muser.utils as utils
import math
import time

synth = live.SynthClient(channels=2)

def sine_tone(amp, freq, phase=0):
    def tone(t):
        return amp * math.sin(2 * math.pi * freq * t + phase)
    return tone

def synth_on(synth, duration, pause=0):
    synth.toggle()
    time.sleep(duration)
    synth.toggle()
    time.sleep(pause)

synth.add_synth_function(sine_tone(0.75, 440))
synth.activate()
synth.connect(synth.outports[0], 'system:playback_1')
synth.connect(synth.outports[1], 'system:playback_2')

for i in range(2):
    synth_on(synth, duration=1, pause=0.5)

synth.add_synth_function(sine_tone(0.75, 220))

for i in range(2):
    synth_on(synth, duration=1, pause=0.5)

synth.deactivate()
synth.outports.clear()
synth.close()
